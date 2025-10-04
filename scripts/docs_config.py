"""Typed configuration for document ingestion sources.

Each top-level key is a human-friendly document ID. The value is a mapping with
the following required keys (hyphenated to mirror external config naming):

- 'document-url': str — public/source URL of the document
- 'document-path': str — local repository path to the document file
- 'pinecone-namespace': str — target Pinecone namespace for upserts
- 'input-format': str — one of {markdown, text, pdf, json, yaml}

A runtime validator and small helper utilities are provided to enforce
shape/values and to catch common mistakes (e.g., malformed URL, missing file,
or mismatched file extension for the given input-format) early.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

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
            Defaults to the directory containing this module.

    Raises:
        InvalidDocsConfigError: If any entry is missing required keys, has the wrong types,
            contains an invalid URL, the referenced file does not exist, or the file extension
            does not match the declared ``input-format``.

    """
    required_keys = {"document-url", "document-path", "pinecone-namespace", "input-format"}

    # Default base directory: project root (this file's directory)
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    errors: list[str] = []

    for doc_id, entry in config.items():
        if not doc_id:
            errors.append(f"invalid document id: {doc_id!r} (must be non-empty str)")
            continue
        errors.extend(_validate_entry(doc_id, entry, base_dir, required_keys))

    if errors:
        raise InvalidDocsConfigError(errors)


def iter_docs(config: DocsConfig | None = None) -> list[tuple[str, DocConfig]]:
    """Return configured documents as a list of (doc_id, entry) pairs.

    Args:
        config: Optional alternative mapping to iterate; defaults to module-level ``docs_config``.

    Returns:
        List of (document-id, DocConfig) tuples for each configured document.

    """
    cfg = docs_config if config is None else config
    return list(cfg.items())


def resolve_document_path(entry: DocConfig, *, base_dir: Path | None = None) -> Path:
    """Return an absolute Path to the entry's ``document-path``.

    Args:
        entry: A DocConfig entry.
        base_dir: Base directory to resolve relative paths against. Defaults to the
            directory containing this module.

    Returns:
        Absolute Path to the document file.

    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    p = Path(entry["document-path"])  # type: ignore[index]
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
        base_dir: Base directory to resolve relative paths. Defaults to the directory
            containing this module if not provided.
        dry_run: Include ``--dry-run`` to write JSON instead of upserting.
        output: Optional output path (used only when dry_run=True).

    Returns:
        A list of CLI-style arguments (e.g., ["--document-id", "fastmcp-docs", ...]).

    """
    if doc_id not in docs_config:
        msg = f"unknown document id: {doc_id!r}"
        raise KeyError(msg)

    entry = docs_config[doc_id]
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


docs_config: DocsConfig = {
    "fastmcp-docs": {
        "document-url": "https://gofastmcp.com/llms-full.txt",
        "document-path": "docs/fastmcp-llms-full.txt.md",
        "pinecone-namespace": "fastmcp",
        "input-format": "markdown",
    },
    "cribl-api": {
        "document-url": "https://docs.cribl.io/api/",
        "document-path": "docs/cribl-api.txt",
        "pinecone-namespace": "cribl",
        "input-format": "text",
    },
    "cribl-api-docs": {
        "document-url": "https://cdn.cribl.io/dl/4.13.3/cribl-apidocs-4.13.3-3c6c5bfd.yml",
        "document-path": "docs/cribl-apidocs-4.13.3-3c6c5bfd.yml",
        "pinecone-namespace": "cribl",
        "input-format": "yaml",
    },
    "cribl-stream-docs": {
        "document-url": "https://cdn.cribl.io/dl/4.13.3/cribl-stream-docs-4.13.3.pdf",
        "document-path": "docs/cribl-stream-docs-4.13.3.pdf",
        "pinecone-namespace": "cribl",
        "input-format": "pdf",
    },
    "cribl-edge-docs": {
        "document-url": "https://cdn.cribl.io/dl/4.13.3/cribl-edge-docs-4.13.3.pdf",
        "document-path": "docs/cribl-edge-docs-4.13.3.pdf",
        "pinecone-namespace": "cribl",
        "input-format": "pdf",
    },
}
