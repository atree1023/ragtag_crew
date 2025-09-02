"""Split a markdown doc into chunks and upsert records to Pinecone.

This script:
- Parses the FastMCP markdown into header-aware sections
- Further splits into token-friendly chunks
- Builds an array of DocumentChunk objects
- Converts them to JSON-ready dicts
- Calls Pinecone Index.upsert_records with those records

Environment:
- Requires PINECONE_API_KEY to be present in the environment
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

from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

pinecone_namespace = "fastmcp"
pinecone_host = "https://ragtag-db-f059e7z.svc.aped-4627-b74a.pinecone.io"


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


def build_document_chunks(text: str) -> list[DocumentChunk]:
    """Split the input markdown text into header-aware chunks and create records.

    Args:
        text: The full markdown text of the document.

    Returns:
        A list of DocumentChunk instances ready to be JSON-serialized for
        Pinecone upsert_records.

    """
    document_id = "fastmcp_documentation"
    document_url = "https://gofastmcp.com/llms-full.txt"
    document_date = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("####", "Header_4"),
        ("#####", "Header_5"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
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
                document_date=document_date,
                chunk_content=output_chunk.page_content,
                chunk_section_id=section_id,
            ),
        )

    return records


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dry-run behavior and output file path."""
    parser = argparse.ArgumentParser(description="Split markdown and upsert/write Pinecone records")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, skip Pinecone upsert and write JSON records to a file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="document_chunks.json",
        help="Output file path for --dry-run (JSON array)",
    )
    return parser.parse_args()


def write_json_records(path: Path, records: list[dict]) -> int:
    """Write the JSON array of record dicts to the given path.

    Returns:
        Number of bytes written to disk.

    """
    data = json.dumps(records, ensure_ascii=False, indent=2)
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


def upsert_records(records_payload: list[dict]) -> object:
    """Upsert record dicts to Pinecone using upsert_records.

    Requires PINECONE_API_KEY in the environment.

    Returns:
        The response object returned by the Pinecone client.

    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        msg = "PINECONE_API_KEY is not set in the environment"
        raise RuntimeError(msg)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=pinecone_host)
    return index.upsert_records(namespace=pinecone_namespace, records=records_payload)


def main() -> None:
    """Build DocumentChunk records and either upsert to Pinecone or write JSON."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    logger = logging.getLogger(__name__)

    start_ts = time.perf_counter()
    logger.info(
        "startup: dry_run=%s output=%s namespace=%s host=%s",
        args.dry_run,
        args.output,
        pinecone_namespace,
        pinecone_host,
    )
    # Build path to ../docs/fastmcp-llms-full.txt relative to this script
    document_path = Path(__file__).resolve().parent.parent / "docs" / "fastmcp-llms-full.txt.md"

    try:
        logger.info("reading document: %s", document_path)
        with document_path.open(encoding="utf-8") as file:
            text = file.read()

        document_chunks: list[DocumentChunk] = build_document_chunks(text)
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
                pinecone_namespace,
                pinecone_host,
            )
            resp = upsert_records(records_payload)
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
