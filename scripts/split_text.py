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
import os
import re
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


def write_json_records(path: Path, records: list[dict]) -> None:
    """Write the JSON array of record dicts to the given path."""
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_records(records_payload: list[dict]) -> None:
    """Upsert record dicts to Pinecone using upsert_records.

    Requires PINECONE_API_KEY in the environment.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        msg = "PINECONE_API_KEY is not set in the environment"
        raise RuntimeError(msg)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=pinecone_host)
    index.upsert_records(namespace=pinecone_namespace, records=records_payload)


def main() -> None:
    """Build DocumentChunk records and either upsert to Pinecone or write JSON."""
    args = parse_args()
    # Build path to ../docs/fastmcp-llms-full.txt relative to this script
    document_path = Path(__file__).resolve().parent.parent / "docs" / "fastmcp-llms-full.txt.md"

    with document_path.open(encoding="utf-8") as file:
        text = file.read()

    document_chunks: list[DocumentChunk] = build_document_chunks(text)

    # Convert dataclass objects to plain dicts (JSON-serializable)
    records_payload = [asdict(dc) for dc in document_chunks]

    if args.dry_run:
        # Write to working directory (or provided path) instead of upserting
        output_path = Path(args.output).resolve()
        write_json_records(output_path, records_payload)
    else:
        # Upsert the records into Pinecone
        upsert_records(records_payload)


if __name__ == "__main__":
    main()
