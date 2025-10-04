"""Split a document (markdown, text, pdf, json, or yaml) into chunks and upsert records to Pinecone.

This script:
- For markdown: parses a document into header-aware sections, then further splits into token-friendly chunks
- For plain text: uses only RecursiveCharacterTextSplitter.split_text on the full text
- For pdf: extracts text from pages (images discarded), then treats it as plain text splitting
- For json: uses RecursiveJsonSplitter.split_text to produce JSON-string chunks
- For yaml: converts YAML to Python structures via safe_load, then processes as json above
- Builds an array of DocumentChunk objects
- Converts them to JSON-ready dicts
- Optionally calls Pinecone Index.upsert_records with those records

Runtime configuration (CLI):
- ``--document-id``: identifier stored with each chunk (required)
- ``--document-url``: source URL for the document metadata (required)
- ``--document-path``: path to the input file (required)
- ``--pinecone-namespace`` (alias ``--namespace``): Pinecone namespace for upserts (required)
- ``--input-format``: one of {markdown, text, pdf, json, yaml} controlling how splitting occurs (default: markdown)
- ``--dry-run``: skip Pinecone upsert and write JSON to a file
 - ``--host``: Pinecone index host URL; defaults to the ``PINECONE_HOST`` env var (required unless env set)

Environment:
- Requires PINECONE_API_KEY to be present in the environment when performing upserts
- The Pinecone host is taken from ``--host`` or ``PINECONE_HOST``; the value must be provided even for --dry-run for
    consistency and logging.
"""

import argparse
import datetime
import json
import logging
import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)
from pinecone import Pinecone
from pypdf import PdfReader

PINECONE_HOST = os.getenv("PINECONE_HOST")


max_chunk_size = 1792
chunk_overlap = 128


def current_date_str() -> str:
    """Return the current UTC date as an ISO string (YYYY-MM-DD).

    This is used for both per-run logging filenames and the document_date
    recorded in each chunk's metadata, ensuring they are consistent for a run.
    """
    return datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")


@dataclass
class DocumentChunk:
    """A single chunk of a larger document for Pinecone upsert_records.

    Fields map directly to the record schema expected by upsert_records, where
    the "_id" is the unique identifier for the record and other fields are
    arbitrary metadata and content used by downstream embedding/search.
    """

    _id: str
    document_id: str
    document_url: str
    document_date: str
    chunk_content: str
    # Optional for non-markdown inputs; present for markdown inputs
    chunk_section_id: str | None = None


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF file, concatenating page texts with double newlines.

    Images and non-text content are discarded. Missing page text is treated as empty.

    Args:
        path: Filesystem path to a PDF document.

    Returns:
        The combined plain text content of all pages.

    """
    with path.open("rb") as fh:
        reader = PdfReader(fh)
        texts: list[str] = []
        for page in reader.pages:
            # extract_text can return None
            page_text = page.extract_text() or ""
            texts.append(page_text.strip())
        return "\n\n".join(t for t in texts if t)


def metadata_values_to_section_id(meta: Mapping[str, object], sep: str = "|") -> str:
    """Convert header metadata into a joined section path string.

    Args:
        meta: Mapping of header keys ('Header_1', 'Header_2', ...) to titles.
        sep: Separator to place between titles.

    Returns:
        Joined section path string.

    Example:
        metadata_values_to_section_id({'Header_1': 'Changelog', 'Header_2': 'New Contributors'})
        -> 'Changelog:New Contributors'

    """
    pattern = re.compile(r"(\d+)$")

    def level(key: str) -> int:
        m = pattern.search(key)
        return int(m.group(1)) if m else 10

    items = [(k, v) for k, v in meta.items() if isinstance(v, str) and v]
    items.sort(key=lambda kv: level(kv[0]))
    return sep.join(v.strip() for _, v in items)


def build_document_chunks(
    text: str,
    document_id: str,
    document_url: str,
    document_date: str,
    *,
    input_format: str = "markdown",
) -> list[DocumentChunk]:
    """Split input text into chunks and create records.

    Args:
        text: The full text of the document.
        document_id: Identifier to associate with all chunks from this document.
        document_url: Source URL recorded in chunk metadata.
        document_date: ISO date (YYYY-MM-DD) to record with each chunk.
        input_format: 'markdown' (default) or 'text' to control splitting.

    Returns:
        A list of DocumentChunk instances ready to be JSON-serialized for
        Pinecone upsert_records.

    """
    # Plain text path: split raw text with RecursiveCharacterTextSplitter.split_text
    if input_format.lower() == "text":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
        text_chunks = text_splitter.split_text(text)
        records: list[DocumentChunk] = []
        for i, chunk in enumerate(text_chunks):
            records.append(
                DocumentChunk(
                    _id=f"{document_id}:chunk{i}",
                    document_id=document_id,
                    document_url=document_url,
                    document_date=document_date,
                    chunk_content=chunk,
                ),
            )
        return records

    # Markdown path (default): header-aware splitting, unchanged behavior
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("####", "Header_4"),
        ("#####", "Header_5"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
    )

    header_docs = splitter.split_text(text)
    output_chunks = chunk_splitter.split_documents(header_docs)

    records: list[DocumentChunk] = []
    for i, output_chunk in enumerate(output_chunks):
        section_id = metadata_values_to_section_id(cast("Mapping[str, object]", output_chunk.metadata))  # type: ignore[arg-type]
        records.append(
            DocumentChunk(
                _id=f"{document_id}:chunk{i}",
                document_id=document_id,
                document_url=document_url,
                document_date=document_date,  # Use the updated date
                chunk_content=output_chunk.page_content,
                chunk_section_id=section_id,
            ),
        )

    return records


def create_document_chunks_from_path(
    *,
    document_path: Path,
    input_format: str,
    document_id: str,
    document_url: str,
    document_date: str,
) -> list[DocumentChunk]:
    """Create `DocumentChunk` objects from a file on disk given an input format.

    This consolidates the per-format branching to keep `main()` small and tidy.

    Args:
        document_path: Path to the source document file.
        input_format: One of {markdown, text, pdf, json, yaml}.
        document_id: ID attached to each chunk.
        document_url: Source URL for metadata.
        document_date: ISO date string recorded on each chunk.

    Returns:
        A list of `DocumentChunk` instances.

    """
    fmt = input_format.lower()

    # PDF -> extract text then split as plain text
    if fmt == "pdf":
        text = extract_pdf_text(document_path)
        return build_document_chunks(
            text,
            document_id=document_id,
            document_url=document_url,
            document_date=document_date,
            input_format="text",
        )

    # JSON -> load and split as JSON strings
    if fmt == "json":
        raw = document_path.read_text(encoding="utf-8")
        try:
            json_data = json.loads(raw)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON input: {e}"
            raise RuntimeError(msg) from e

        splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)
        json_text_chunks: list[str] = splitter.split_text(json_data=json_data)
        return [
            DocumentChunk(
                _id=f"{document_id}:chunk{i}",
                document_id=document_id,
                document_url=document_url,
                document_date=document_date,
                chunk_content=chunk_text,
            )
            for i, chunk_text in enumerate(json_text_chunks)
        ]

    # YAML -> parse then treat like JSON
    if fmt == "yaml":
        yaml_text = document_path.read_text(encoding="utf-8")
        try:
            yaml_data = yaml.safe_load(yaml_text)
        except Exception as e:  # surface YAML parsing errors with context
            msg = f"Invalid YAML input: {e}"
            raise RuntimeError(msg) from e

        splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)
        json_text_chunks = splitter.split_text(json_data=yaml_data)
        return [
            DocumentChunk(
                _id=f"{document_id}:chunk{i}",
                document_id=document_id,
                document_url=document_url,
                document_date=document_date,
                chunk_content=chunk_text,
            )
            for i, chunk_text in enumerate(json_text_chunks)
        ]

    # markdown or plain text
    text = document_path.read_text(encoding="utf-8")
    return build_document_chunks(
        text,
        document_id=document_id,
        document_url=document_url,
        document_date=document_date,
        input_format=fmt,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for behavior, metadata, and input path.

    Returns:
        argparse.Namespace containing:
        - dry_run: bool
        - output: str
        - document_id: str
        - document_url: str
        - document_path: str
        - input_format: str
        - pinecone_namespace: str
        - host: str

    """
    parser = argparse.ArgumentParser(description="Split documents and upsert/write Pinecone records")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, skip Pinecone upsert and write JSON records to a file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/document_chunks.json",
        help="Output file path for --dry-run (JSON array)",
    )
    parser.add_argument(
        "--document-id",
        type=str,
        required=True,
        help="Document identifier to store with each chunk",
    )
    parser.add_argument(
        "--document-url",
        type=str,
        required=True,
        help="Source URL recorded in chunk metadata",
    )
    parser.add_argument(
        "--document-path",
        type=str,
        required=True,
        help="Path to the input file",
    )
    parser.add_argument(
        "--input-format",
        choices=["markdown", "text", "pdf", "json", "yaml"],
        default="markdown",
        help=(
            "Input file format. 'markdown' uses header-aware splitting; 'text' uses simple text splitting; "
            "'pdf' extracts text from pages (images discarded) and then splits as plain text; "
            "'json' uses RecursiveJsonSplitter; 'yaml' converts YAML to JSON-like Python structures then splits as JSON."
        ),
    )
    parser.add_argument(
        "--pinecone-namespace",
        "--namespace",
        dest="pinecone_namespace",
        type=str,
        required=True,
        help="Pinecone namespace to upsert into",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=PINECONE_HOST,
        help=(
            "Pinecone index host URL (e.g., https://<index>.svc.<project>.pinecone.io). "
            "Defaults to the PINECONE_HOST environment variable."
        ),
    )
    args = parser.parse_args()
    # Enforce presence of a host (either via CLI or env) for consistent logging and explicitness
    if not args.host:
        parser.error("--host is required (or set PINECONE_HOST in the environment)")
    return args


def write_json_records(path: Path, records: list[dict[str, str]]) -> int:
    """Write the JSON array of record dicts to the given path.

    Returns:
        Number of bytes written to disk.

    """
    data = json.dumps(records, ensure_ascii=False, indent=2)
    # Ensure parent directories exist for convenience in --dry-run
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return len(data.encode("utf-8"))


def _truncate(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summarize_upsert_response(resp: object) -> dict[str, object]:
    """Create a safe, compact summary of the upsert response for logging."""
    summary: dict[str, object] = {}
    try:
        if resp is None:
            return {"response": None}
        # Mapping-like (dict) responses
        if isinstance(resp, dict):
            for key in ("upserted_count", "status", "usage", "namespace", "error"):
                if key in resp:
                    summary[key] = resp[key]
            if not summary:
                summary["keys"] = [str(k) for k in cast("dict[str, Any]", resp)]
            return summary
        # Object-like
        for attr in ("upserted_count", "status", "usage", "namespace", "error"):
            if hasattr(resp, attr):
                summary[attr] = getattr(resp, attr)
        if not summary:
            summary["repr"] = _truncate(repr(resp))
    except Exception as e:  # noqa: BLE001 - defensive logging only
        summary = {"summary_error": str(e), "type": type(resp).__name__}  # type: ignore[arg-type]
    return summary


def upsert_records(records_payload: list[dict[str, str]], namespace: str, *, host: str, batch_size: int = 64) -> object:
    """Upsert record dicts to Pinecone in batches using ``upsert_records``.

    Args:
        records_payload: List of record dictionaries to upsert.
        namespace: Pinecone namespace to upsert into.
        host: Pinecone index host URL to connect to (not control-plane URL).
        batch_size: Number of records to send per request (default 64).

    Environment:
        Requires ``PINECONE_API_KEY`` to be present in the environment when there are
        records to upsert.

    Returns:
        The last response object returned by the Pinecone client for the final batch,
        or ``None`` if there were no records.

    """
    # Fast exit for empty payloads (also helps with unit testing without credentials).
    if not records_payload:
        return None

    if batch_size <= 0:
        batch_size = 64

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        msg = "PINECONE_API_KEY is not set in the environment"
        raise RuntimeError(msg)

    if not host:
        msg = "Pinecone host must be provided"
        raise RuntimeError(msg)

    pc_any: Any = Pinecone(api_key=api_key)
    # Any type used here to account for Pinecone SDK versions with and without type hints
    index: Any = pc_any.Index(host=host)

    last_resp: object | None = None
    for start in range(0, len(records_payload), batch_size):
        batch = records_payload[start : start + batch_size]
        last_resp = index.upsert_records(namespace=namespace, records=batch)

    return last_resp


def main() -> None:
    """Build DocumentChunk records and either upsert to Pinecone or write JSON."""
    args = parse_args()
    # Compute a single date string for the entire run (for logs and chunk metadata)
    run_date = current_date_str()
    # Configure logging to append to logs/split_text.<YYYY-MM-DD>.log and also stream to console
    log_path = Path(__file__).resolve().parent.parent / "logs" / f"split_text.{run_date}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    start_ts = time.perf_counter()
    logger.info(
        "startup: dry_run=%s output=%s namespace=%s host=%s document_id=%s document_url=%s document_path=%s input_format=%s",
        args.dry_run,
        args.output,
        args.pinecone_namespace,
        args.host,
        args.document_id,
        args.document_url,
        args.document_path,
        args.input_format,
    )
    # Resolve document path from CLI
    document_path = Path(args.document_path).resolve()

    try:
        logger.info("reading document: %s (format=%s)", document_path, args.input_format)

        document_chunks = create_document_chunks_from_path(
            document_path=document_path,
            input_format=args.input_format,
            document_id=args.document_id,
            document_url=args.document_url,
            document_date=run_date,
        )

        logger.info("chunks built: count=%d", len(document_chunks))

        # Convert dataclass objects to plain dicts (JSON-serializable)
        records_payload: list[dict[str, str]] = []
        for dc in document_chunks:
            d = asdict(dc)
            # For text inputs, omit chunk_section_id when None
            if d.get("chunk_section_id") is None:
                d.pop("chunk_section_id", None)
            records_payload.append({k: v for k, v in d.items() if v is not None})

        if args.dry_run:
            # Write to working directory (or provided path) instead of upserting
            output_path = Path(args.output).resolve()
            bytes_written = write_json_records(output_path, records_payload)
            logger.info("dry-run: wrote JSON records to %s (%d bytes)", output_path, bytes_written)
        else:
            # Upsert the records into Pinecone
            logger.info(
                "upserting records: count=%d namespace=%s host=%s",
                len(records_payload),
                args.pinecone_namespace,
                args.host,
            )
            resp = upsert_records(records_payload, namespace=args.pinecone_namespace, host=args.host)
            summary = summarize_upsert_response(resp)
            logger.info("upsert complete: summary=%s", summary)
    except Exception:
        logger.exception("execution failed")
        sys.exit(1)
    finally:
        elapsed = time.perf_counter() - start_ts
        logger.info("finished in %.2fs", elapsed)


if __name__ == "__main__":
    main()
