"""Split documents into Pinecone-friendly chunks using manual or configured inputs.

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
- ``--document-id``: identifier stored with each chunk (required for manual mode)
- ``--document-url``: source URL for the document metadata (required for manual mode)
- ``--document-path``: path to the input file (required for manual mode)
- ``--pinecone-namespace`` (alias ``--namespace``): Pinecone namespace for upserts (required for manual mode)
- ``--input-format``: one of {markdown, text, pdf, json, yaml} controlling how splitting occurs (default: markdown)
- ``--dry-run``: skip Pinecone upsert and write JSON to a file
 - ``--host``: Pinecone index host URL; defaults to the ``PINECONE_HOST`` env var (required unless env set)
- ``--list``: show configured documents in a table
- ``--process <doc_id>``: process a document by id using configuration defaults (downloads the source when missing)

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)
from pinecone import Pinecone
from pypdf import PdfReader

if TYPE_CHECKING:
    from collections.abc import Mapping

from .doc_dwnld import download_one
from .docs_config import DocConfig, DocsConfig, get_docs_config, resolve_document_path, validate_docs_config

PINECONE_HOST = os.getenv("PINECONE_HOST")
DEFAULT_INPUT_FORMAT = "markdown"


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


@dataclass(frozen=True)
class ExecutionContext:
    """Resolved runtime configuration for manual or config-driven runs."""

    document_id: str
    document_url: str
    input_format: str
    pinecone_namespace: str
    document_path: Path
    mode: str


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for manual usage and configuration-driven modes."""
    parser = argparse.ArgumentParser(description="Split documents and upsert/write Pinecone records")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured documents and exit",
    )
    parser.add_argument(
        "--process",
        type=str,
        metavar="DOC_ID",
        help="Process a configured document using docs_config defaults",
    )
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
        help="Document identifier to store with each chunk",
    )
    parser.add_argument(
        "--document-url",
        type=str,
        help="Source URL recorded in chunk metadata",
    )
    parser.add_argument(
        "--document-path",
        type=str,
        help="Path to the input file",
    )
    parser.add_argument(
        "--input-format",
        choices=["markdown", "text", "pdf", "json", "yaml"],
        default=None,
        help=(
            "Input file format. 'markdown' uses header-aware splitting; 'text' uses simple text splitting; "
            "'pdf' extracts text from pages (images discarded) and then splits as plain text; "
            "'json' uses RecursiveJsonSplitter; 'yaml' converts YAML to JSON-like Python structures then splits as JSON. "
            "Defaults to 'markdown' in manual mode when omitted."
        ),
    )
    parser.add_argument(
        "--pinecone-namespace",
        "--namespace",
        dest="pinecone_namespace",
        type=str,
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
    args = parser.parse_args(argv)

    if args.list and args.process:
        parser.error("--list cannot be combined with --process")

    manual_fields = [
        ("document_id", "--document-id"),
        ("document_url", "--document-url"),
        ("document_path", "--document-path"),
        ("input_format", "--input-format"),
        ("pinecone_namespace", "--pinecone-namespace"),
    ]

    if args.list:
        return args

    if args.process:
        for attr, flag in manual_fields:
            if getattr(args, attr) is not None:
                parser.error(f"{flag} cannot be used with --process")
        if not args.host:
            parser.error("--host is required (or set PINECONE_HOST in the environment)")
        return args

    if args.input_format is None:
        args.input_format = DEFAULT_INPUT_FORMAT

    missing = [flag for attr, flag in manual_fields if getattr(args, attr) is None]
    if missing:
        parser.error(
            "the following arguments are required when not using --process: " + ", ".join(missing),
        )

    if not args.host:
        parser.error("--host is required (or set PINECONE_HOST in the environment)")

    return args


def render_docs_table(entries: list[tuple[str, DocConfig]], *, base_dir: Path | None = None) -> str:
    """Return an ASCII table summarizing configured documents."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    if not entries:
        return "No documents configured."

    headers = ["Document", "Format", "Namespace", "Path", "URL"]
    rows: list[list[str]] = []

    for doc_id, entry in sorted(entries, key=lambda pair: pair[0]):
        path = resolve_document_path(entry, base_dir=base_dir)
        try:
            display_path = str(path.relative_to(base_dir))
        except ValueError:
            display_path = str(path)
        rows.append(
            [
                doc_id,
                entry["input-format"],
                entry["pinecone-namespace"],
                display_path,
                entry["document-url"],
            ],
        )

    matrix = [headers, *rows]
    column_widths = [max(len(row[idx]) for row in matrix) for idx in range(len(headers))]

    lines: list[str] = []
    for idx, row in enumerate(matrix):
        formatted = " | ".join(value.ljust(column_widths[col_idx]) for col_idx, value in enumerate(row))
        lines.append(formatted)
        if idx == 0:
            separator = "-+-".join("-" * column_widths[col_idx] for col_idx in range(len(headers)))
            lines.append(separator)

    return "\n".join(lines)


def list_configured_documents() -> None:
    """Print configured documents from docs_config in a table."""
    config = get_docs_config(require_paths_exist=False)
    table = render_docs_table(list(config.items()))
    sys.stdout.write(f"{table}\n")


def prepare_config_document(
    doc_id: str,
    *,
    logger: logging.Logger,
    base_dir: Path | None = None,
) -> tuple[DocConfig, Path]:
    """Return a validated docs_config entry and ensure its document exists."""
    config: DocsConfig = get_docs_config(require_paths_exist=False)
    if doc_id not in config:
        msg = f"unknown document id: {doc_id!r}"
        raise KeyError(msg)

    entry = config[doc_id]
    resolved_path = resolve_document_path(entry, base_dir=base_dir)
    if not resolved_path.exists():
        logger.info("document missing locally; downloading: id=%s", doc_id)
        download_one(doc_id, entry)
    else:
        logger.info("document already present: id=%s path=%s", doc_id, resolved_path)

    validate_docs_config({doc_id: entry}, base_dir=base_dir)
    final_path = resolve_document_path(entry, base_dir=base_dir)
    return cast("DocConfig", dict(entry)), final_path


def build_execution_context(
    args: argparse.Namespace,
    *,
    logger: logging.Logger,
    base_dir: Path,
) -> ExecutionContext:
    """Construct the runtime execution context from CLI arguments."""
    if args.process:
        entry, document_path = prepare_config_document(args.process, logger=logger, base_dir=base_dir)
        return ExecutionContext(
            document_id=args.process,
            document_url=entry["document-url"],
            input_format=entry["input-format"],
            pinecone_namespace=entry["pinecone-namespace"],
            document_path=document_path,
            mode="config",
        )

    manual_values = {
        "document-id": args.document_id,
        "document-url": args.document_url,
        "document-path": args.document_path,
        "input-format": args.input_format,
        "pinecone-namespace": args.pinecone_namespace,
    }
    missing_manual = [key for key, value in manual_values.items() if value is None]
    if missing_manual:
        msg = "missing required arguments for manual mode: " + ", ".join(missing_manual)
        raise RuntimeError(msg)

    return ExecutionContext(
        document_id=cast("str", args.document_id),
        document_url=cast("str", args.document_url),
        input_format=cast("str", args.input_format),
        pinecone_namespace=cast("str", args.pinecone_namespace),
        document_path=Path(cast("str", args.document_path)).resolve(),
        mode="manual",
    )


def write_json_records(path: Path, records: list[dict[str, str | None]]) -> int:
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


def upsert_records(records_payload: list[dict[str, str | None]], namespace: str, *, host: str, batch_size: int = 64) -> object:
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


def main(argv: list[str] | None = None) -> None:
    """Build DocumentChunk records and either upsert to Pinecone or write JSON."""
    args = parse_args(argv)

    if args.list:
        list_configured_documents()
        return

    # Compute a single date string for the entire run (for logs and chunk metadata)
    run_date = current_date_str()
    base_dir = Path(__file__).resolve().parent.parent

    # Configure logging to append to logs/split_text.<YYYY-MM-DD>.log and also stream to console
    log_path = base_dir / "logs" / f"split_text.{run_date}.log"
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

    context = build_execution_context(args, logger=logger, base_dir=base_dir)

    logger.info(
        (
            "startup: mode=%s dry_run=%s output=%s namespace=%s host=%s "
            "document_id=%s document_url=%s document_path=%s input_format=%s"
        ),
        context.mode,
        args.dry_run,
        args.output,
        context.pinecone_namespace,
        args.host,
        context.document_id,
        context.document_url,
        context.document_path,
        context.input_format,
    )

    try:
        logger.info("reading document: %s (format=%s)", context.document_path, context.input_format)

        document_chunks = create_document_chunks_from_path(
            document_path=context.document_path,
            input_format=context.input_format,
            document_id=context.document_id,
            document_url=context.document_url,
            document_date=run_date,
        )

        logger.info("chunks built: count=%d", len(document_chunks))

        records_payload: list[dict[str, str | None]] = []
        for dc in document_chunks:
            d = asdict(dc)
            if d.get("chunk_section_id") is None:
                d.pop("chunk_section_id", None)
            records_payload.append(d)

        if args.dry_run:
            output_path = Path(args.output).resolve()
            bytes_written = write_json_records(output_path, records_payload)
            logger.info("dry-run: wrote JSON records to %s (%d bytes)", output_path, bytes_written)
        else:
            logger.info(
                "upserting records: count=%d namespace=%s host=%s",
                len(records_payload),
                context.pinecone_namespace,
                args.host,
            )
            resp = upsert_records(records_payload, namespace=context.pinecone_namespace, host=args.host)
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
