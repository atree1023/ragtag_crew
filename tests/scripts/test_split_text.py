"""Tests for the ``scripts.split_text`` helpers."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from scripts import split_text

if TYPE_CHECKING:
    from pathlib import Path


API_KEY = "key"
DOCUMENT_DATE = "2025-10-10"
BATCH_SIZE = 2
EXPECTED_SECTION = "Intro"


def fail_unless(*, condition: bool, message: str) -> None:
    """Fail the current test when the provided condition is false."""
    if not condition:
        pytest.fail(message)


def test_metadata_values_to_section_id_orders_by_header_level() -> None:
    """Metadata should be converted into an ordered section identifier."""
    meta = {"Header_2": "Second", "Header_1": EXPECTED_SECTION}
    section_id = split_text.metadata_values_to_section_id(meta)
    fail_unless(condition=section_id == f"{EXPECTED_SECTION}|Second", message=section_id)


def test_build_document_chunks_markdown_produces_section_ids() -> None:
    """Markdown input should create chunks with section identifiers."""
    markdown = "# Intro\n\n## Details\n\nParagraph text."
    chunks = split_text.build_document_chunks(
        markdown,
        document_id="doc",
        document_url="https://example.com",
        document_date=DOCUMENT_DATE,
        input_format="markdown",
    )
    fail_unless(condition=bool(chunks), message="expected markdown chunks")
    first_chunk = chunks[0]
    fail_unless(condition=first_chunk.chunk_section_id is not None, message=str(first_chunk))
    fail_unless(condition=EXPECTED_SECTION in (first_chunk.chunk_section_id or ""), message=str(first_chunk))


def test_build_document_chunks_text_omits_section_ids() -> None:
    """Plain text input should not populate section identifiers."""
    chunks = split_text.build_document_chunks(
        "Plain body text without headers.",
        document_id="doc",
        document_url="https://example.com",
        document_date=DOCUMENT_DATE,
        input_format="text",
    )
    fail_unless(condition=bool(chunks), message="expected text chunks")
    fail_unless(condition=all(chunk.chunk_section_id is None for chunk in chunks), message=str(chunks))


def test_create_document_chunks_from_path_json(tmp_path: Path) -> None:
    """JSON documents should be split using RecursiveJsonSplitter."""
    payload = {"title": "Example", "items": [1, 2, 3]}
    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    chunks = split_text.create_document_chunks_from_path(
        document_path=json_path,
        input_format="json",
        document_id="doc",
        document_url="https://example.com",
        document_date=DOCUMENT_DATE,
    )
    fail_unless(condition=bool(chunks), message="expected json chunks")
    fail_unless(
        condition=any("items" in chunk.chunk_content for chunk in chunks),
        message=str([c.chunk_content for c in chunks]),
    )


def test_create_document_chunks_from_path_yaml(tmp_path: Path) -> None:
    """YAML documents should be parsed and chunked similarly to JSON."""
    yaml_text = """\
    title: Example
    list:
      - a
      - b
    """
    yaml_path = tmp_path / "sample.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    chunks = split_text.create_document_chunks_from_path(
        document_path=yaml_path,
        input_format="yaml",
        document_id="doc",
        document_url="https://example.com",
        document_date=DOCUMENT_DATE,
    )
    fail_unless(condition=bool(chunks), message="expected yaml chunks")
    fail_unless(
        condition=any("Example" in chunk.chunk_content for chunk in chunks),
        message=str([c.chunk_content for c in chunks]),
    )


def test_create_document_chunks_from_path_pdf_uses_extract_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PDF inputs should rely on extract_pdf_text and produce text chunks."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    def fake_extract(path: Path) -> str:
        fail_unless(condition=path == pdf_path, message=str(path))
        return "Page 1 content"

    monkeypatch.setattr(split_text, "extract_pdf_text", fake_extract)
    chunks = split_text.create_document_chunks_from_path(
        document_path=pdf_path,
        input_format="pdf",
        document_id="doc",
        document_url="https://example.com",
        document_date=DOCUMENT_DATE,
    )
    fail_unless(condition=bool(chunks), message="expected pdf chunks")
    fail_unless(condition=chunks[0].chunk_section_id is None, message=str(chunks[0]))
    fail_unless(condition="Page 1 content" in chunks[0].chunk_content, message=str(chunks[0]))


def test_create_document_chunks_from_path_json_invalid(tmp_path: Path) -> None:
    """Invalid JSON content should raise a RuntimeError with context."""
    json_path = tmp_path / "bad.json"
    json_path.write_text("{not: json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid JSON input"):
        split_text.create_document_chunks_from_path(
            document_path=json_path,
            input_format="json",
            document_id="doc",
            document_url="https://example.com",
            document_date=DOCUMENT_DATE,
        )


def test_create_document_chunks_from_path_yaml_invalid(tmp_path: Path) -> None:
    """Invalid YAML should raise a RuntimeError surfaced to callers."""
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(": bad", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid YAML input"):
        split_text.create_document_chunks_from_path(
            document_path=yaml_path,
            input_format="yaml",
            document_id="doc",
            document_url="https://example.com",
            document_date=DOCUMENT_DATE,
        )


def test_write_json_records_creates_file(tmp_path: Path) -> None:
    """Dry-run output helper should emit JSON to disk."""
    target = tmp_path / "records.json"
    records: list[dict[str, str | None]] = [{"_id": "doc:chunk0", "chunk_content": "data"}]
    bytes_written = split_text.write_json_records(target, records)
    fail_unless(condition=target.exists(), message=str(target))
    fail_unless(condition=bytes_written == len(target.read_bytes()), message=str(bytes_written))


def test_summarize_upsert_response_prefers_known_fields() -> None:
    """Upsert response summarizer should surface expected attributes."""

    @dataclass
    class DummyResponse:
        upserted_count: int = 3
        status: str = "ok"

    dummy = DummyResponse()
    summary = split_text.summarize_upsert_response(dummy)
    fail_unless(condition=summary["upserted_count"] == dummy.upserted_count, message=str(summary))
    fail_unless(condition=summary["status"] == dummy.status, message=str(summary))


def test_upsert_records_batches_and_returns_last_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upsert helper should batch payloads and return the final response."""

    class FakeIndex:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[dict[str, str | None]]]] = []

        def upsert_records(self, namespace: str, records: list[dict[str, str | None]]) -> dict[str, int]:
            self.calls.append((namespace, records))
            return {"upserted_count": len(records)}

    class FakePinecone:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.index = FakeIndex()

        def Index(self, host: str) -> FakeIndex:  # noqa: N802 - match real client API
            self.host = host
            return self.index

    fake_client = FakePinecone(API_KEY)
    monkeypatch.setenv("PINECONE_API_KEY", API_KEY)

    def fake_pinecone_factory(api_key: str) -> FakePinecone:
        fail_unless(condition=api_key == API_KEY, message=api_key)
        return fake_client

    monkeypatch.setattr(split_text, "Pinecone", fake_pinecone_factory)

    records: list[dict[str, str | None]] = [{"_id": f"doc:chunk{i}", "chunk_content": "data"} for i in range(BATCH_SIZE + 1)]
    result = split_text.upsert_records(records, namespace="ns", host="https://host", batch_size=BATCH_SIZE)

    expected_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    last_batch_size = len(records) - (expected_batches - 1) * BATCH_SIZE

    fail_unless(condition=result == {"upserted_count": last_batch_size}, message=str(result))
    fail_unless(condition=len(fake_client.index.calls) == expected_batches, message=str(fake_client.index.calls))
    fail_unless(condition=fake_client.index.calls[0][0] == "ns", message=str(fake_client.index.calls))
    fail_unless(condition=len(fake_client.index.calls[0][1]) == BATCH_SIZE, message=str(fake_client.index.calls[0]))
    fail_unless(condition=len(fake_client.index.calls[-1][1]) == last_batch_size, message=str(fake_client.index.calls[-1]))


def test_upsert_records_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing API key should raise a RuntimeError."""
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        split_text.upsert_records([{"_id": "doc:chunk0", "chunk_content": "data"}], namespace="ns", host="https://host")
    fail_unless(condition="PINECONE_API_KEY" in str(exc_info.value), message=str(exc_info.value))


def test_upsert_records_returns_none_for_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty payload should return ``None`` without calling Pinecone."""
    monkeypatch.setenv("PINECONE_API_KEY", API_KEY)
    result = split_text.upsert_records([], namespace="ns", host="https://host")
    fail_unless(condition=result is None, message=str(result))


def test_upsert_records_retries_on_429_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Pinecone returns 429 with Retry-After header, helper should wait and retry then succeed."""

    class RateLimitError(Exception):
        def __init__(self) -> None:
            self.status_code = 429
            self.headers = {"Retry-After": "0.01"}
            super().__init__("Too Many Requests")

    calls: list[str] = []
    expected_calls = 2  # first attempt rate-limited, second attempt succeeds

    class FakeIndex:
        def __init__(self) -> None:
            self.calls = calls

        def upsert_records(self, namespace: str, records: list[dict[str, str | None]]) -> dict[str, int]:
            self.calls.append(namespace)
            # First call fails, second succeeds
            if len(self.calls) == 1:
                raise RateLimitError
            return {"upserted_count": len(records)}

    class FakePinecone:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.index = FakeIndex()

        def Index(self, host: str) -> FakeIndex:  # noqa: N802 - mirror SDK API
            self.host = host
            return self.index

    monkeypatch.setenv("PINECONE_API_KEY", API_KEY)

    def fake_factory(api_key: str) -> FakePinecone:
        return FakePinecone(api_key)

    monkeypatch.setattr(split_text, "Pinecone", fake_factory)

    records: list[dict[str, str | None]] = [{"_id": "doc:chunk0", "chunk_content": "x"}]
    policy = split_text.RetryPolicy(max_retries=3, base_backoff=0.001, max_backoff=0.01)
    result = split_text.upsert_records(records, namespace="ns", host="https://host", retry_policy=policy)

    fail_unless(condition=result == {"upserted_count": 1}, message=str(result))
    fail_unless(condition=len(calls) == expected_calls, message=str(calls))


def test_upsert_records_raises_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Helper should raise a RuntimeError after exceeding max_retries for 429 errors."""

    class RateLimitError(Exception):
        def __init__(self) -> None:
            self.status_code = 429
            self.headers = {"Retry-After": "0"}
            super().__init__("Too Many Requests")

    class FakeIndex:
        def upsert_records(self, _namespace: str, _records: list[dict[str, str | None]]) -> dict[str, int]:
            raise RateLimitError

    class FakePinecone:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def Index(self, _host: str) -> FakeIndex:  # noqa: N802 - mirror SDK API
            return FakeIndex()

    monkeypatch.setenv("PINECONE_API_KEY", API_KEY)

    def fake_factory(api_key: str) -> FakePinecone:
        return FakePinecone(api_key)

    monkeypatch.setattr(split_text, "Pinecone", fake_factory)

    policy = split_text.RetryPolicy(max_retries=2, base_backoff=0.0, max_backoff=0.0)
    bad_records: list[dict[str, str | None]] = [{"_id": "doc:chunk0", "chunk_content": "x"}]
    with pytest.raises(RuntimeError, match="rate-limited"):
        split_text.upsert_records(
            bad_records,
            namespace="ns",
            host="https://host",
            retry_policy=policy,
        )


def test_render_docs_table_outputs_expected_columns(tmp_path: Path) -> None:
    """Rendering the docs table should include headers and resolved paths."""
    entry: split_text.DocConfig = {
        "document-url": "https://example.com/doc.md",
        "document-path": "docs/doc.md",
        "pinecone-namespace": "ns",
        "input-format": "markdown",
    }
    table = split_text.render_docs_table([("doc", entry)], base_dir=tmp_path)
    fail_unless(condition="Document" in table.splitlines()[0], message=table)
    fail_unless(condition="doc" in table, message=table)
    fail_unless(condition="docs/doc.md" in table, message=table)


def test_prepare_config_document_downloads_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing documents should trigger a download before validation."""
    entry: split_text.DocConfig = {
        "document-url": "https://example.com/doc.txt",
        "document-path": "docs/doc.txt",
        "pinecone-namespace": "ns",
        "input-format": "text",
    }

    monkeypatch.setattr(split_text, "get_docs_config", lambda _require_paths_exist=False: {"doc": entry})

    download_calls: list[str] = []

    def fake_download(doc_id: str, _doc_entry: split_text.DocConfig) -> Path:
        fail_unless(condition=doc_id == "doc", message=doc_id)
        download_calls.append(doc_id)
        target = tmp_path / "docs" / "doc.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("content", encoding="utf-8")
        return target

    monkeypatch.setattr(split_text, "download_one", fake_download)

    def fake_validate(_config: split_text.DocsConfig, *, _base_dir: Path | None = None) -> None:
        fail_unless(condition=(tmp_path / "docs" / "doc.txt").exists(), message="expected download")

    monkeypatch.setattr(split_text, "validate_docs_config", fake_validate)

    logger = logging.getLogger("test.prepare")
    entry_copy, path = split_text.prepare_config_document("doc", logger=logger, base_dir=tmp_path)

    fail_unless(condition=entry_copy["document-url"] == entry["document-url"], message=str(entry_copy))
    fail_unless(condition=path == tmp_path / "docs" / "doc.txt", message=str(path))
    fail_unless(condition=download_calls == ["doc"], message=str(download_calls))


def test_prepare_config_document_skips_download_when_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing documents should not be downloaded again."""
    entry: split_text.DocConfig = {
        "document-url": "https://example.com/doc.txt",
        "document-path": "docs/doc.txt",
        "pinecone-namespace": "ns",
        "input-format": "text",
    }

    existing = tmp_path / "docs" / "doc.txt"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("ready", encoding="utf-8")

    monkeypatch.setattr(split_text, "get_docs_config", lambda _require_paths_exist=False: {"doc": entry})

    def fake_download(doc_id: str, _doc_entry: split_text.DocConfig) -> Path:
        pytest.fail(f"download should not run for {doc_id}")

    monkeypatch.setattr(split_text, "download_one", fake_download)

    def fake_validate(_config: split_text.DocsConfig, *, _base_dir: Path | None = None) -> None:
        fail_unless(condition=existing.exists(), message="expected existing file")

    monkeypatch.setattr(split_text, "validate_docs_config", fake_validate)

    logger = logging.getLogger("test.prepare")
    _, path = split_text.prepare_config_document("doc", logger=logger, base_dir=tmp_path)

    fail_unless(condition=path == existing, message=str(path))


def test_build_execution_context_manual_mode(tmp_path: Path) -> None:
    """Manual mode should resolve paths and preserve provided metadata."""
    args = argparse.Namespace(
        process=None,
        document_id="doc",
        document_url="https://example.com",
        document_path=str(tmp_path / "doc.md"),
        input_format=split_text.DEFAULT_INPUT_FORMAT,
        pinecone_namespace="ns",
    )

    logger = logging.getLogger("test.context.manual")
    context = split_text.build_execution_context(args, logger=logger, base_dir=tmp_path)

    fail_unless(condition=context.mode == "manual", message=str(context))
    fail_unless(condition=context.document_id == "doc", message=str(context))
    fail_unless(condition=context.document_path.name == "doc.md", message=str(context.document_path))
    fail_unless(condition=context.input_format == "markdown", message=str(context.input_format))


def test_parse_args_defaults_input_format(tmp_path: Path) -> None:
    """Manual CLI parsing should fall back to markdown when not supplied."""
    argv = [
        "--dry-run",
        "--document-id",
        "doc",
        "--document-url",
        "https://example.com",
        "--document-path",
        str(tmp_path / "doc.md"),
        "--pinecone-namespace",
        "ns",
        "--host",
        "https://host",
    ]

    args = split_text.parse_args(argv)
    fail_unless(condition=args.input_format == split_text.DEFAULT_INPUT_FORMAT, message=str(args.input_format))


def test_build_execution_context_config_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Config mode should defer to prepare_config_document for resolution."""
    args = argparse.Namespace(
        process="doc",
        document_id=None,
        document_url=None,
        document_path=None,
        input_format=None,
        pinecone_namespace=None,
    )

    entry: split_text.DocConfig = {
        "document-url": "https://example.com/doc.md",
        "document-path": "docs/doc.md",
        "pinecone-namespace": "ns",
        "input-format": "markdown",
    }
    resolved = tmp_path / "docs" / "doc.md"

    def fake_prepare(doc_id: str, *, _logger: logging.Logger, _base_dir: Path | None) -> tuple[split_text.DocConfig, Path]:
        fail_unless(condition=doc_id == "doc", message=doc_id)
        return entry, resolved

    monkeypatch.setattr(split_text, "prepare_config_document", fake_prepare)

    logger = logging.getLogger("test.context.config")
    context = split_text.build_execution_context(args, logger=logger, base_dir=tmp_path)

    fail_unless(condition=context.mode == "config", message=str(context))
    fail_unless(condition=context.document_path == resolved, message=str(context.document_path))
    fail_unless(condition=context.document_url == entry["document-url"], message=str(context.document_url))
