"""Tests for the ``scripts.doc_dwnld`` module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.error import URLError

import pytest

from scripts import doc_dwnld

if TYPE_CHECKING:
    from pathlib import Path


def fail_unless(*, condition: bool, message: str) -> None:
    """Fail the current test if the condition is not satisfied."""
    if not condition:
        pytest.fail(message)


EXIT_SUCCESS = 0
EXIT_SELECTION_ERROR = 2
EXIT_DOWNLOAD_FAILURE = 3


def test_iter_selected_docs_all_returns_every_entry() -> None:
    """Selecting all documents should yield each mapping."""
    config: doc_dwnld.DocsConfig = {
        "a": {
            "document-url": "https://example.com/a",
            "document-path": "docs/a.txt",
            "pinecone-namespace": "ns",
            "input-format": "text",
        },
        "b": {
            "document-url": "https://example.com/b",
            "document-path": "docs/b.pdf",
            "pinecone-namespace": "ns",
            "input-format": "pdf",
        },
    }
    results = list(doc_dwnld.iter_selected_docs(config, all_docs=True, single_id=None))
    fail_unless(condition=len(results) == len(config), message=str(results))


def test_iter_selected_docs_single_id() -> None:
    """Selecting by ID should return exactly that entry."""
    config: doc_dwnld.DocsConfig = {
        "only": {
            "document-url": "https://example.com/only",
            "document-path": "docs/only.txt",
            "pinecone-namespace": "ns",
            "input-format": "text",
        },
    }
    results = list(doc_dwnld.iter_selected_docs(config, all_docs=False, single_id="only"))
    fail_unless(condition=len(results) == 1, message=str(results))
    fail_unless(condition=results[0][0] == "only", message=str(results))


def test_iter_selected_docs_missing_choice_raises() -> None:
    """Omitting selection flags should raise a ValueError."""
    with pytest.raises(ValueError, match="--all or --id"):
        list(doc_dwnld.iter_selected_docs({}, all_docs=False, single_id=None))


def test_iter_selected_docs_unknown_id_raises_key_error() -> None:
    """Selecting a missing document id should raise KeyError."""
    config: doc_dwnld.DocsConfig = {
        "a": {
            "document-url": "https://example.com/a",
            "document-path": "docs/a.txt",
            "pinecone-namespace": "ns",
            "input-format": "text",
        },
    }
    with pytest.raises(KeyError, match="unknown document id"):
        list(doc_dwnld.iter_selected_docs(config, all_docs=False, single_id="missing"))


def test_download_one_converts_html_to_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HTML content for text documents should be converted to plain text."""
    root = tmp_path
    docs_path = root / "docs"

    def fake_project_root() -> Path:
        return root

    def fake_docs_dir() -> Path:
        return docs_path

    def fake_read_url_bytes(url: str, *, user_agent: str = "") -> tuple[bytes, dict[str, str]]:
        del user_agent
        fail_unless(condition=url == "https://example.com/page", message=url)
        html = b"<html><body><h1>Title</h1><p>Body</p></body></html>"
        headers = {"content-type": "text/html; charset=utf-8"}
        return html, headers

    monkeypatch.setattr(doc_dwnld, "project_root", fake_project_root)
    monkeypatch.setattr(doc_dwnld, "docs_dir", fake_docs_dir)
    monkeypatch.setattr(doc_dwnld, "_read_url_bytes", fake_read_url_bytes)

    entry: doc_dwnld.DocConfig = {
        "document-url": "https://example.com/page",
        "document-path": "docs/original.txt",
        "pinecone-namespace": "ns",
        "input-format": "text",
    }

    destination = doc_dwnld.download_one("sample", entry)
    fail_unless(condition=destination.exists(), message=str(destination))
    payload = destination.read_text(encoding="utf-8")
    fail_unless(condition="Title" in payload and "Body" in payload, message=payload)
    fail_unless(condition=destination.name == "sample.txt", message=destination.name)


def test_download_one_saves_binary_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-text content should be written as retrieved bytes."""
    root = tmp_path

    def fake_project_root() -> Path:
        return root

    def fake_read_url_bytes(url: str, *, user_agent: str = "") -> tuple[bytes, dict[str, str]]:
        del user_agent
        fail_unless(condition=url == "https://example.com/file.md", message=url)
        data = b"binary-data"
        headers = {"content-type": "application/octet-stream"}
        return data, headers

    monkeypatch.setattr(doc_dwnld, "project_root", fake_project_root)
    monkeypatch.setattr(doc_dwnld, "_read_url_bytes", fake_read_url_bytes)

    entry: doc_dwnld.DocConfig = {
        "document-url": "https://example.com/file.md",
        "document-path": "docs/file.md",
        "pinecone-namespace": "ns",
        "input-format": "markdown",
    }

    destination = doc_dwnld.download_one("sample", entry)
    fail_unless(condition=destination.exists(), message=str(destination))
    fail_unless(condition=destination.read_bytes() == b"binary-data", message=str(destination.read_bytes()))
    fail_unless(condition=destination.suffix == ".md", message=destination.suffix)


def test_download_one_rejects_non_http_urls(tmp_path: Path) -> None:
    """Only http(s) URLs should be accepted."""
    entry: doc_dwnld.DocConfig = {
        "document-url": "ftp://example.com/file.txt",
        "document-path": str(tmp_path / "file.txt"),
        "pinecone-namespace": "ns",
        "input-format": "text",
    }
    with pytest.raises(ValueError, match="URL must start with http"):
        doc_dwnld.download_one("doc", entry)


def test_main_list_outputs_all_entries(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """--list should emit tab-separated rows for each config entry."""
    config: doc_dwnld.DocsConfig = {
        "doc": {
            "document-url": "https://example.com/doc",
            "document-path": "docs/doc.txt",
            "pinecone-namespace": "ns",
            "input-format": "text",
        },
    }

    def fake_get_docs_config(
        *,
        _require_paths_exist: bool = True,
        **_kwargs: object,
    ) -> doc_dwnld.DocsConfig:
        del _require_paths_exist, _kwargs
        return config

    monkeypatch.setattr(doc_dwnld, "get_docs_config", fake_get_docs_config)
    exit_code = doc_dwnld.main(["--list"])
    fail_unless(condition=exit_code == EXIT_SUCCESS, message=str(exit_code))
    captured = capsys.readouterr()
    fail_unless(condition="doc\ttext\thttps://example.com/doc" in captured.out, message=captured.out)


def test_main_without_selection_returns_code_two(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing selection flags should produce exit code 2 and error output."""
    config: doc_dwnld.DocsConfig = {}

    def fake_get_docs_config(
        *,
        _require_paths_exist: bool = True,
        **_kwargs: object,
    ) -> doc_dwnld.DocsConfig:
        del _require_paths_exist, _kwargs
        return config

    monkeypatch.setattr(doc_dwnld, "get_docs_config", fake_get_docs_config)
    exit_code = doc_dwnld.main([])
    fail_unless(condition=exit_code == EXIT_SELECTION_ERROR, message=str(exit_code))
    captured = capsys.readouterr()
    fail_unless(condition="ERROR" in captured.err, message=captured.err)


def test_main_download_failure_sets_exit_code_three(monkeypatch: pytest.MonkeyPatch) -> None:
    """Download failures should produce exit code 3."""
    config: doc_dwnld.DocsConfig = {
        "doc": {
            "document-url": "https://example.com/doc",
            "document-path": "docs/doc.txt",
            "pinecone-namespace": "ns",
            "input-format": "text",
        },
    }

    def fake_download_one(doc_id: str, entry: doc_dwnld.DocConfig) -> Path:
        del doc_id, entry
        error_message = "boom"
        raise URLError(error_message)

    def fake_get_docs_config(
        *,
        _require_paths_exist: bool = True,
        **_kwargs: object,
    ) -> doc_dwnld.DocsConfig:
        del _require_paths_exist, _kwargs
        return config

    monkeypatch.setattr(doc_dwnld, "get_docs_config", fake_get_docs_config)
    monkeypatch.setattr(doc_dwnld, "download_one", fake_download_one)

    exit_code = doc_dwnld.main(["--all"])
    fail_unless(condition=exit_code == EXIT_DOWNLOAD_FAILURE, message=str(exit_code))
