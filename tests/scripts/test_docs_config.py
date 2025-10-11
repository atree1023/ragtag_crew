"""Tests for the ``scripts.docs_config`` helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from scripts import docs_config

if TYPE_CHECKING:
    from pathlib import Path


def _mk_entry(
    *,
    document_url: str = "https://example.com/source",
    document_path: str = "sample.md",
    namespace: str = "example-ns",
    input_format: str = "markdown",
) -> docs_config.DocConfig:
    return {
        "document-url": document_url,
        "document-path": document_path,
        "pinecone-namespace": namespace,
        "input-format": input_format,
    }


def fail_unless(*, condition: bool, message: str) -> None:
    """Fail the current test if the condition is not satisfied."""
    if not condition:
        pytest.fail(message)


def test_validate_docs_config_accepts_valid_entry(tmp_path: Path) -> None:
    """Valid configuration entries should pass validation."""
    source = tmp_path / "sample.md"
    source.write_text("content", encoding="utf-8")
    config = {"sample": _mk_entry()}
    docs_config.validate_docs_config(config, base_dir=tmp_path)


@pytest.mark.parametrize(
    ("input_format", "suffix"),
    [
        ("pdf", ".pdf"),
        ("json", ".json"),
        ("yaml", ".yaml"),
        ("markdown", ".md"),
        ("text", ".txt"),
    ],
)
def test_validate_docs_config_enforces_suffix(tmp_path: Path, input_format: str, suffix: str) -> None:
    """Ensure suffix validation matches format expectations."""
    source = tmp_path / f"source{suffix}"
    source.write_text("content", encoding="utf-8")
    entry = _mk_entry(document_path=source.name, input_format=input_format)
    docs_config.validate_docs_config({"doc": entry}, base_dir=tmp_path)


def test_validate_docs_config_rejects_missing_keys(tmp_path: Path) -> None:
    """Missing required fields should trigger validation errors."""
    entry = cast(
        "docs_config.DocConfig",
        {
        "document-url": "https://example.com",
        "document-path": "missing.md",
        },
    )
    with pytest.raises(docs_config.InvalidDocsConfigError) as exc_info:
        docs_config.validate_docs_config({"doc": entry}, base_dir=tmp_path)
    fail_unless(condition="missing keys" in str(exc_info.value), message=str(exc_info.value))


def test_validate_docs_config_rejects_bad_url(tmp_path: Path) -> None:
    """Reject entries containing unsupported URL schemes."""
    source = tmp_path / "sample.md"
    source.write_text("content", encoding="utf-8")
    entry = _mk_entry(document_url="ftp://invalid", document_path=source.name)
    with pytest.raises(docs_config.InvalidDocsConfigError) as exc_info:
        docs_config.validate_docs_config({"doc": entry}, base_dir=tmp_path)
    fail_unless(condition="http(s)" in str(exc_info.value), message=str(exc_info.value))


def test_validate_docs_config_rejects_suffix_mismatch(tmp_path: Path) -> None:
    """Detect suffix mismatches for declared formats."""
    source = tmp_path / "sample.md"
    source.write_text("content", encoding="utf-8")
    entry = _mk_entry(document_path=source.name, input_format="pdf")
    with pytest.raises(docs_config.InvalidDocsConfigError) as exc_info:
        docs_config.validate_docs_config({"doc": entry}, base_dir=tmp_path)
    fail_unless(condition="expected a .pdf" in str(exc_info.value), message=str(exc_info.value))


def test_validate_docs_config_allows_missing_paths_when_disabled(tmp_path: Path) -> None:
    """Missing files may be tolerated when path existence checks are disabled."""
    entry = _mk_entry(document_path="missing.md")
    with pytest.raises(docs_config.InvalidDocsConfigError):
        docs_config.validate_docs_config({"doc": entry}, base_dir=tmp_path)
    docs_config.validate_docs_config(
        {"doc": entry},
        base_dir=tmp_path,
        require_paths_exist=False,
    )


def test_load_and_save_docs_config_roundtrip(tmp_path: Path, config_path: Path) -> None:
    """Persisted configuration should load identically."""
    source = tmp_path / "sample.md"
    source.write_text("content", encoding="utf-8")
    cfg = {"sample": _mk_entry(document_path=source.name)}
    docs_config.save_docs_config(cfg, path=config_path, base_dir=tmp_path)
    loaded = docs_config.load_docs_config(path=config_path, base_dir=tmp_path)
    fail_unless(condition=loaded == cfg, message=str(loaded))


def test_get_doc_entry_returns_copy(tmp_path: Path, config_path: Path) -> None:
    """Mutating fetched entries should not alter stored config."""
    source = tmp_path / "sample.md"
    source.write_text("content", encoding="utf-8")
    cfg = {"sample": _mk_entry(document_path=source.name)}
    docs_config.save_docs_config(cfg, path=config_path, base_dir=tmp_path)
    entry = docs_config.get_doc_entry("sample", path=config_path, base_dir=tmp_path)
    entry["document-url"] = "https://changed"
    reloaded = docs_config.get_doc_entry("sample", path=config_path, base_dir=tmp_path)
    fail_unless(condition=reloaded["document-url"] == "https://example.com/source", message=str(reloaded))


def test_upsert_doc_entry_persists_and_returns_updated_config(tmp_path: Path, config_path: Path) -> None:
    """Upsert should persist entries and update the cached config."""
    source = tmp_path / "upsert.md"
    source.write_text("content", encoding="utf-8")
    entry = _mk_entry(document_path=str(source))
    docs_config.save_docs_config({}, path=config_path, validate=False)
    updated = docs_config.upsert_doc_entry("doc", entry, path=config_path)
    fail_unless(condition="doc" in updated, message=str(updated))
    stored = docs_config.get_doc_entry("doc", path=config_path)
    fail_unless(condition=stored["document-path"] == str(source), message=str(stored))


def test_remove_doc_entry_deletes_and_persists(tmp_path: Path, config_path: Path) -> None:
    """Removing an entry should persist the deletion and refresh the cache."""
    source = tmp_path / "keep.md"
    source.write_text("content", encoding="utf-8")
    entry_keep = _mk_entry(document_path=str(source))
    entry_drop = _mk_entry(document_path=str(source))
    docs_config.save_docs_config({"keep": entry_keep, "drop": entry_drop}, path=config_path, validate=False)
    remaining = docs_config.remove_doc_entry("drop", path=config_path)
    fail_unless(condition="drop" not in remaining, message=str(remaining))
    with pytest.raises(KeyError):
        docs_config.get_doc_entry("drop", path=config_path)


def test_get_docs_config_reloads_when_paths_required(tmp_path: Path, config_path: Path) -> None:
    """Requesting configs with strict path requirements should trigger revalidation."""
    entry = _mk_entry(document_path="missing.md")
    docs_config.save_docs_config({"doc": entry}, path=config_path, validate=False)
    loaded = docs_config.get_docs_config(
        path=config_path,
        base_dir=tmp_path,
        require_paths_exist=False,
    )
    fail_unless(condition="doc" in loaded, message=str(loaded))
    with pytest.raises(docs_config.InvalidDocsConfigError):
        docs_config.get_docs_config(path=config_path, base_dir=tmp_path, require_paths_exist=True)


def test_iter_docs_returns_deep_copies() -> None:
    """iter_docs should not expose the original mapping by reference."""
    config = {
        "sample": _mk_entry(),
    }
    docs = docs_config.iter_docs(config)
    fail_unless(condition=len(docs) == 1, message=str(docs))
    docs[0][1]["document-url"] = "https://changed"
    fail_unless(condition=config["sample"]["document-url"] == "https://example.com/source", message=str(config))


def test_build_split_cli_args_constructs_expected_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure CLI args include required switches and values."""
    source = tmp_path / "sample.md"
    source.write_text("content", encoding="utf-8")
    entry = _mk_entry(document_path=source.name)
    monkeypatch.setattr(docs_config, "get_docs_config", lambda: {"sample": entry})
    args = docs_config.build_split_cli_args("sample", base_dir=tmp_path, dry_run=True, output="out.json")
    fail_unless(condition=args[0] == "--dry-run", message=str(args))
    fail_unless(condition="--document-id" in args, message=str(args))
    fail_unless(condition="sample" in args, message=str(args))
    fail_unless(condition="--output" in args, message=str(args))
    fail_unless(condition="out.json" in args, message=str(args))


def test_resolve_document_path_handles_relative_and_absolute(tmp_path: Path) -> None:
    """Resolve relative entries and preserve absolute paths."""
    relative_entry = _mk_entry(document_path="docs/sample.md")
    absolute_path = tmp_path / "docs" / "sample.md"
    absolute_entry = _mk_entry(document_path=str(absolute_path))
    resolved_relative = docs_config.resolve_document_path(relative_entry, base_dir=tmp_path)
    resolved_absolute = docs_config.resolve_document_path(absolute_entry, base_dir=tmp_path)
    fail_unless(condition=resolved_relative == absolute_path, message=str(resolved_relative))
    fail_unless(condition=resolved_absolute == absolute_path, message=str(resolved_absolute))
