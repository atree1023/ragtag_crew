"""Tests for the ``scripts.docs_config`` helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    entry = {
        "document-url": "https://example.com",
        "document-path": "missing.md",
    }
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
