"""Tests for the ``scripts.doc_dwnld`` module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts import doc_dwnld

if TYPE_CHECKING:
    from pathlib import Path


def fail_unless(*, condition: bool, message: str) -> None:
    """Fail the current test if the condition is not satisfied."""
    if not condition:
        pytest.fail(message)


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
