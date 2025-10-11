"""Typed configuration for document ingestion sources.

The canonical source of truth lives in ``scripts/docs_config_data.yaml``.
Each top-level key is a human-friendly document ID. The value is a mapping with
the following required keys (hyphenated to mirror external config naming):

- 'document-url': str — public/source URL of the document
- 'document-path': str — local repository path to the document file
- 'pinecone-namespace': str — target Pinecone namespace for upserts
- 'input-format': str — one of {markdown, text, pdf, json, yaml}

When ``input-format`` is ``text`` the referenced ``document-path`` must end in
``.txt`` so downstream tooling can rely on plain-text files.

A runtime validator and small helper utilities are provided to enforce
shape/values and to catch common mistakes (e.g., malformed URL, missing file,
or mismatched file extension for the given input-format) early.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast
from urllib.parse import urlparse

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping

DocConfig = TypedDict(
    "DocConfig",
    {
        "document-url": str,
        "document-path": str,
        "pinecone-namespace": str,
        "input-format": str,
    },
)

DocsConfig = dict[str, DocConfig]

# Supported input formats for split_text.py
ALLOWED_INPUT_FORMATS: frozenset[str] = frozenset({"markdown", "text", "pdf", "json", "yaml"})

CONFIG_DATA_PATH: Path = Path(__file__).resolve().parent / "docs_config_data.yaml"
"""Default location of the document configuration YAML file."""

# Internal cache container to avoid re-reading from disk on every access.
_CONFIG_STATE: dict[str, DocsConfig | None] = {"cache": None}


def _is_valid_url(value: str) -> bool:
    """Return True if value looks like an absolute HTTP(S) URL."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class InvalidDocsConfigError(Exception):
    """Raised when the ``docs_config`` mapping fails validation.

    The exception message contains a bullet list of human-readable issues.
    """

    def __init__(self, issues: list[str]) -> None:
        """Initialize the error with a bullet-list of issues.

        Args:
            issues: A list of human-readable validation errors.

        """
        msg = "Invalid docs_config:\n" + "\n".join(f"- {e}" for e in issues)
        super().__init__(msg)


def _validate_required_strings(doc_id: str, entry: DocConfig, required_keys: set[str]) -> list[str]:
    errs: list[str] = []
    missing = required_keys.difference(entry.keys())
    if missing:
        errs.append(f"{doc_id}: missing keys: {sorted(missing)}")
    for key in required_keys.intersection(entry.keys()):
        val = entry.get(key, "")
        if not isinstance(val, str) or not val.strip():
            errs.append(f"{doc_id}: '{key}' must be a non-empty string")
    return errs


def _validate_input_format(doc_id: str, entry: DocConfig) -> list[str]:
    errs: list[str] = []
    fmt = entry.get("input-format", "")
    if not fmt.strip():
        errs.append(f"{doc_id}: 'input-format' is required and must be a non-empty string")
        return errs
    fmt_norm = fmt.strip().lower()
    if fmt_norm not in ALLOWED_INPUT_FORMATS:
        errs.append(
            f"{doc_id}: 'input-format' must be one of {sorted(ALLOWED_INPUT_FORMATS)}; got {fmt!r}",
        )
    return errs


def _validate_url_field(doc_id: str, entry: DocConfig) -> list[str]:
    errs: list[str] = []
    url = entry.get("document-url", "")
    if url and not _is_valid_url(url):
        errs.append(f"{doc_id}: 'document-url' must be an absolute http(s) URL; got {url!r}")
    return errs


def _validate_path_and_suffix(doc_id: str, entry: DocConfig, base_dir: Path) -> list[str]:
    errs: list[str] = []
    path_str = entry.get("document-path", "")
    fmt = entry.get("input-format", "")
    if path_str:
        path = (base_dir / path_str) if not Path(path_str).is_absolute() else Path(path_str)
        if not path.exists():
            errs.append(f"{doc_id}: 'document-path' does not exist: {path}")
        suffix = path.suffix.lower()
        fmt_norm = fmt.strip().lower()
        if fmt_norm == "pdf" and suffix != ".pdf":
            errs.append(f"{doc_id}: expected a .pdf file for input-format=pdf; got {path.name}")
        elif fmt_norm == "json" and suffix != ".json":
            errs.append(f"{doc_id}: expected a .json file for input-format=json; got {path.name}")
        elif fmt_norm == "yaml" and suffix not in {".yml", ".yaml"}:
            errs.append(f"{doc_id}: expected a .yml/.yaml file for input-format=yaml; got {path.name}")
        elif fmt_norm == "text" and suffix != ".txt":
            errs.append(f"{doc_id}: expected a .txt file for input-format=text; got {path.name}")
        elif fmt_norm == "markdown" and suffix not in {".md", ".markdown", ".mdown", ".mkd"}:
            errs.append(
                f"{doc_id}: expected a markdown file (.md/.markdown/.mdown/.mkd) for input-format=markdown; got {path.name}",
            )
    return errs


def _validate_namespace(doc_id: str, entry: DocConfig) -> list[str]:
    errs: list[str] = []
    ns = entry.get("pinecone-namespace", "")
    if not ns.strip():
        errs.append(f"{doc_id}: 'pinecone-namespace' must be non-empty")
    return errs


def _validate_entry(
    doc_id: str,
    entry: DocConfig,
    base_dir: Path,
    required_keys: set[str],
) -> list[str]:
    """Validate a single document config entry and return a list of error strings."""
    errs: list[str] = []
    errs.extend(_validate_required_strings(doc_id, entry, required_keys))
    errs.extend(_validate_input_format(doc_id, entry))
    errs.extend(_validate_url_field(doc_id, entry))
    errs.extend(_validate_path_and_suffix(doc_id, entry, base_dir))
    errs.extend(_validate_namespace(doc_id, entry))
    return errs


def validate_docs_config(config: DocsConfig, *, base_dir: Path | None = None) -> None:
    """Validate the docs_config mapping: keys, types, URL, file existence, and format sanity.

    Args:
        config: Mapping from document-id to its configuration dict.
        base_dir: Base directory to resolve relative ``document-path`` values against.
            Defaults to the repository root (parent directory of this module).

    Raises:
        InvalidDocsConfigError: If any entry is missing required keys, has the wrong types,
            contains an invalid URL, the referenced file does not exist, or the file extension
            does not match the declared ``input-format``.

    """
    required_keys = {"document-url", "document-path", "pinecone-namespace", "input-format"}

    # Default base directory: repository root (parent of this module)
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    errors: list[str] = []

    for doc_id, entry in config.items():
        if not doc_id:
            errors.append(f"invalid document id: {doc_id!r} (must be non-empty str)")
            continue
        errors.extend(_validate_entry(doc_id, entry, base_dir, required_keys))

    if errors:
        raise InvalidDocsConfigError(errors)


def _coerce_docs_config(raw: object) -> DocsConfig:
    """Convert an arbitrary object into a ``DocsConfig`` mapping.

    Args:
        raw: Parsed object from the configuration data file.

    Returns:
        A ``DocsConfig`` dictionary.

    Raises:
        InvalidDocsConfigError: If the raw object is not a mapping of document ids to mapping entries.

    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = "configuration file must contain a mapping at the top level"
        raise InvalidDocsConfigError([msg])

    raw_mapping = cast("dict[object, object]", raw)

    config: DocsConfig = {}
    for key_obj, value in raw_mapping.items():
        if not isinstance(key_obj, str):
            msg = f"invalid document id: {key_obj!r} (must be a non-empty string)"
            raise InvalidDocsConfigError([msg])
        key = key_obj
        if not key.strip():
            msg = f"invalid document id: {key!r} (must be a non-empty string)"
            raise InvalidDocsConfigError([msg])
        if not isinstance(value, dict):
            msg = f"{key}: entry must be a mapping of configuration values"
            raise InvalidDocsConfigError([msg])
        value_mapping = cast("dict[str, object]", value)
        config[key] = _validate_and_build_doc_config(key, value_mapping)
    return config


def _validate_and_build_doc_config(key: str, value: Mapping[str, object]) -> DocConfig:
    """Validate and construct a DocConfig from a raw dict, raising if invalid."""
    required_keys = ["document-url", "document-path", "pinecone-namespace", "input-format"]
    errors: list[str] = []
    doc_config: dict[str, str] = {}
    for k in required_keys:
        if k not in value:
            errors.append(f"{key}: missing required key '{k}'")
            continue
        val_obj = value[k]
        if not isinstance(val_obj, str):
            errors.append(f"{key}: key '{k}' must be a string")
        else:
            doc_config[k] = val_obj
    if errors:
        raise InvalidDocsConfigError(errors)
    return cast("DocConfig", doc_config)


def _set_cache(config: DocsConfig) -> None:
    """Store a deep copy of the given config in the module cache."""
    _CONFIG_STATE["cache"] = copy.deepcopy(config)


def load_docs_config(
    *,
    path: Path | None = None,
    base_dir: Path | None = None,
    validate: bool = True,
) -> DocsConfig:
    """Load the docs configuration from disk, optionally validating it.

    Args:
        path: Optional explicit config file path. Defaults to ``CONFIG_DATA_PATH``.
        base_dir: Base directory for validation of ``document-path`` fields.
        validate: When True (default), run :func:`validate_docs_config` on the loaded data.

    Returns:
        The loaded configuration mapping.

    Raises:
        InvalidDocsConfigError: If validation fails.
        FileNotFoundError: If the config file path does not exist.

    """
    config_path = path or CONFIG_DATA_PATH
    if not config_path.exists():
        msg = f"configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _coerce_docs_config(raw)

    if validate:
        validate_docs_config(config, base_dir=base_dir)

    _set_cache(config)
    return copy.deepcopy(config)


def get_docs_config(
    *,
    refresh: bool = False,
    path: Path | None = None,
    base_dir: Path | None = None,
) -> DocsConfig:
    """Return the cached docs config, reloading from disk when requested."""
    cached = _CONFIG_STATE["cache"]
    if refresh or cached is None:
        return load_docs_config(path=path, base_dir=base_dir)
    return copy.deepcopy(cached)


def save_docs_config(
    config: DocsConfig,
    *,
    path: Path | None = None,
    base_dir: Path | None = None,
    validate: bool = True,
) -> None:
    """Persist the provided configuration mapping back to disk.

    Args:
        config: Mapping of document ids to configuration entries.
        path: Optional override for the config file location.
        base_dir: Base directory for validation (if enabled).
        validate: When True (default), run validation before writing.

    Raises:
        InvalidDocsConfigError: If validation fails.

    """
    if validate:
        validate_docs_config(config, base_dir=base_dir)

    config_path = path or CONFIG_DATA_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    _set_cache(config)


def refresh_docs_config(*, path: Path | None = None, base_dir: Path | None = None) -> DocsConfig:
    """Force a reload of the docs configuration from disk."""
    return get_docs_config(refresh=True, path=path, base_dir=base_dir)


def get_doc_entry(
    doc_id: str,
    *,
    refresh: bool = False,
    path: Path | None = None,
    base_dir: Path | None = None,
) -> DocConfig:
    """Return a single document configuration entry by id."""
    config = get_docs_config(refresh=refresh, path=path, base_dir=base_dir)
    if doc_id not in config:
        msg = f"unknown document id: {doc_id!r}"
        raise KeyError(msg)
    return copy.deepcopy(config[doc_id])


def upsert_doc_entry(
    doc_id: str,
    entry: DocConfig,
    *,
    path: Path | None = None,
    base_dir: Path | None = None,
) -> DocsConfig:
    """Add or update a document entry and persist it to disk."""
    if not doc_id:
        msg = "document id must be a non-empty string"
        raise ValueError(msg)
    config = get_docs_config(refresh=True, path=path, base_dir=base_dir)
    config[doc_id] = entry
    save_docs_config(config, path=path, base_dir=base_dir)
    return get_docs_config(path=path, base_dir=base_dir)


def remove_doc_entry(
    doc_id: str,
    *,
    path: Path | None = None,
    base_dir: Path | None = None,
) -> DocsConfig:
    """Remove a document configuration entry by id and persist the change."""
    config = get_docs_config(refresh=True, path=path, base_dir=base_dir)
    if doc_id not in config:
        msg = f"unknown document id: {doc_id!r}"
        raise KeyError(msg)
    del config[doc_id]
    save_docs_config(config, path=path, base_dir=base_dir)
    return get_docs_config(path=path, base_dir=base_dir)


def iter_docs(config: Mapping[str, DocConfig] | None = None) -> list[tuple[str, DocConfig]]:
    """Return configured documents as a list of ``(doc_id, entry)`` pairs."""
    source = get_docs_config() if config is None else config
    return [(doc_id, copy.deepcopy(entry)) for doc_id, entry in source.items()]


def resolve_document_path(entry: DocConfig, *, base_dir: Path | None = None) -> Path:
    """Return an absolute Path to the entry's ``document-path``.

    Args:
        entry: A DocConfig entry.
        base_dir: Base directory to resolve relative paths against. Defaults to the
            repository root (parent directory of this module).

    Returns:
        Absolute Path to the document file.

    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    p = Path(entry["document-path"])
    return p if p.is_absolute() else (base_dir / p)


def build_split_cli_args(
    doc_id: str,
    *,
    base_dir: Path | None = None,
    dry_run: bool = True,
    output: str | None = None,
) -> list[str]:
    """Build CLI arguments for ``scripts/split_text.py`` for a given document id.

    This is a convenience for orchestrator scripts that will invoke the splitter
    with consistent, validated parameters.

    Args:
        doc_id: Key into ``docs_config``.
        base_dir: Base directory to resolve relative paths. Defaults to the repository
            root (parent directory of this module) if not provided.
        dry_run: Include ``--dry-run`` to write JSON instead of upserting.
        output: Optional output path (used only when dry_run=True).

    Returns:
        A list of CLI-style arguments (e.g., ["--document-id", "fastmcp-docs", ...]).

    """
    config = get_docs_config()
    if doc_id not in config:
        msg = f"unknown document id: {doc_id!r}"
        raise KeyError(msg)

    entry = config[doc_id]
    # Ensure entry validates (will raise helpful error otherwise)
    validate_docs_config({doc_id: entry}, base_dir=base_dir)

    path = resolve_document_path(entry, base_dir=base_dir)
    args: list[str] = [
        "--document-id",
        doc_id,
        "--document-url",
        entry["document-url"],
        "--document-path",
        str(path),
        "--input-format",
        entry["input-format"],
        "--pinecone-namespace",
        entry["pinecone-namespace"],
    ]
    if dry_run:
        args.insert(0, "--dry-run")
        if output:
            args.extend(["--output", output])
    return args
