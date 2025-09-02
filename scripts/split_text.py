import datetime
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_namespace = "fastmcp"
pinecone_host = "https://ragtag-db-f059e7z.svc.aped-4627-b74a.pinecone.io"

pc = Pinecone(api_key=pinecone_api_key)
pinecone_index = pc.Index(host=pinecone_host)


@dataclass
class DocumentChunk:
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


# Build path to ../docs/fastmcp-llms-full.txt relative to this script
document_path = Path(__file__).resolve().parent.parent / "docs" / "fastmcp-llms-full.txt.md"
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

with document_path.open(encoding="utf-8") as file:
    text = file.read()

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunk_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128,
)

chunks = splitter.split_text(text)
output_chunks = chunk_splitter.split_documents(chunks)

for i, output_chunk in enumerate(output_chunks):
    line = (
        f"_id={document_id}:chunk{i}, "
        f"document_id={document_id}, "
        f"document_url={document_url}, "
        f"document_date={document_date}, "
        f"chunk_content={output_chunk.page_content}, "
        f"chunk_section_id={metadata_values_to_section_id(output_chunk.metadata)}"
    )
    print(line)

pinecone_index.upsert_records(
    pinecone_namespace,
    [],
)
