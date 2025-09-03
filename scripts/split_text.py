"""Split a markdown doc into chunks and upsert records to Pinecone.

This script:
- Parses a markdown document into header-aware sections
- Further splits into token-friendly chunks
- Builds an array of DocumentChunk objects
- Converts them to JSON-ready dicts
- Optionally calls Pinecone Index.upsert_records with those records

Runtime configuration (CLI):
- ``--document-id``: identifier stored with each chunk (required)
- ``--document-url``: source URL for the document metadata (required)
- ``--document-path``: path to the input markdown file (required)
- ``--pinecone-namespace`` (alias ``--namespace``): Pinecone namespace for upserts (required)
- ``--dry-run``: skip Pinecone upsert and write JSON to a file

Environment:
- Requires PINECONE_API_KEY to be present in the environment when performing upserts
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

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pinecone import Pinecone

pinecone_host = "https://ragtag-db-f059e7z.svc.aped-4627-b74a.pinecone.io"
chunk_size = 1024
chunk_overlap = 64


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
    chunk_section_id: str


def metadata_values_to_section_id(meta: Mapping[str, str], sep: str = "|") -> str:
    """Convert header metadata like into a string.

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
) -> list[DocumentChunk]:
    """Split the input markdown text into header-aware chunks and create records.

    Args:
        text: The full markdown text of the document.
        document_id: Identifier to associate with all chunks from this document.
        document_url: Source URL recorded in chunk metadata.
        document_date: ISO date (YYYY-MM-DD) to record with each chunk.

    Returns:
        A list of DocumentChunk instances ready to be JSON-serialized for
        Pinecone upsert_records.

    """
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("####", "Header_4"),
        ("#####", "Header_5"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    header_docs = splitter.split_text(text)
    output_chunks = chunk_splitter.split_documents(header_docs)

    records: list[DocumentChunk] = []
    for i, output_chunk in enumerate(output_chunks):
        section_id = metadata_values_to_section_id(output_chunk.metadata)
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for behavior, metadata, and input path.

    Returns:
        argparse.Namespace containing:
        - dry_run: bool
        - output: str
        - document_id: str
        - document_url: str
        - document_path: str
        - pinecone_namespace: str

    """
    parser = argparse.ArgumentParser(description="Split markdown and upsert/write Pinecone records")
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
        help="Path to the input markdown file",
    )
    parser.add_argument(
        "--pinecone-namespace",
        "--namespace",
        dest="pinecone_namespace",
        type=str,
        required=True,
        help="Pinecone namespace to upsert into",
    )
    return parser.parse_args()


def write_json_records(path: Path, records: list[dict]) -> int:
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


def summarize_upsert_response(resp: object) -> dict:
    """Create a safe, compact summary of the upsert response for logging."""
    summary: dict = {}
    try:
        if resp is None:
            return {"response": None}
        # Mapping-like (dict) responses
        if isinstance(resp, dict):
            for key in ("upserted_count", "status", "usage", "namespace", "error"):
                if key in resp:
                    summary[key] = resp[key]
            if not summary:
                summary["keys"] = list(resp.keys())
            return summary
        # Object-like
        for attr in ("upserted_count", "status", "usage", "namespace", "error"):
            if hasattr(resp, attr):
                summary[attr] = getattr(resp, attr)
        if not summary:
            summary["repr"] = _truncate(repr(resp))
    except Exception as e:  # noqa: BLE001 - defensive logging only
        summary = {"summary_error": str(e), "type": type(resp).__name__}
    return summary


def upsert_records(records_payload: list[dict], namespace: str, batch_size: int = 64) -> object:
    """Upsert record dicts to Pinecone in batches using ``upsert_records``.

    Args:
        records_payload: List of record dictionaries to upsert.
        namespace: Pinecone namespace to upsert into.
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

    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=pinecone_host)

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
        "startup: dry_run=%s output=%s namespace=%s host=%s document_id=%s document_url=%s document_path=%s",
        args.dry_run,
        args.output,
        args.pinecone_namespace,
        pinecone_host,
        args.document_id,
        args.document_url,
        args.document_path,
    )
    # Resolve document path from CLI
    document_path = Path(args.document_path).resolve()

    try:
        logger.info("reading document: %s", document_path)
        with document_path.open(encoding="utf-8") as file:
            text = file.read()

        document_chunks: list[DocumentChunk] = build_document_chunks(
            text,
            document_id=args.document_id,
            document_url=args.document_url,
            document_date=run_date,
        )
        logger.info("chunks built: count=%d", len(document_chunks))

        # Convert dataclass objects to plain dicts (JSON-serializable)
        records_payload = [asdict(dc) for dc in document_chunks]

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
                pinecone_host,
            )
            resp = upsert_records(records_payload, namespace=args.pinecone_namespace)
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
