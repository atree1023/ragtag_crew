# ragtag-crew

Tools for creating and maintaining Pinecone vector DB lookups for up‑to‑date documentation and code examples.

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.13-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-00A77F?logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![LangChain](https://img.shields.io/badge/Text%20Splitters-LangChain-1C3C3C)](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

> [!NOTE]
> This repository currently focuses on ingest tooling: splitting docs, building records, and upserting to a Pinecone index. Agent access (MCP tools) is planned; see AGENTS.md.

## Overview

AI coding agents work best with fresh, searchable technical context. ragtag-crew provides a small, scriptable toolchain to:

- Create a Pinecone index configured for text embedding
- Split markdown docs into header-aware, token-friendly chunks
- Upsert rich records with document metadata into Pinecone namespaces

The goal is a reliable pipeline to collect, update, and access documentation and example code as vector search context.

## Features

- Header-aware markdown parsing using LangChain text splitters
- Tunable chunk size and overlap (defaults: 1024 / 64)
- Record schema optimized for retrieval with clear metadata fields
- Batch upserts to Pinecone with simple logging and dry-run mode
- Idempotent index creation helper

> [!TIP]
> Keep each documentation source in its own Pinecone namespace to simplify updates and deletions without cross-talk.

## How it works

Two scripts power the pipeline today:

- `scripts/db_create.py` – Creates the Pinecone index `ragtag-db` with an integrated embedding model and field map `{"text": "chunk_content"}`.
- `scripts/split_text.py` – Splits a markdown document into chunks and either upserts records to Pinecone or writes them to JSON (dry run).

### Data model (record schema)

Each chunk becomes a record shaped like:

```jsonc
{
  "_id": "<document_id>:chunk<idx>",
  "document_id": "<your-doc-id>",
  "document_url": "https://…",
  "document_date": "YYYY-MM-DD", // run date
  "chunk_content": "…", // text used for embedding (per index field map)
  "chunk_section_id": "Header_1|Header_2|…" // joined header path
}
```

> [!IMPORTANT]
> The index created by `db_create.py` maps the embedding model’s `text` field to `chunk_content`. If you change field names, update the index configuration and the ingestion code together.

## Prerequisites

- Python 3.13 or newer (see `pyproject.toml`)
- A Pinecone account and API key

Environment variable:

- `PINECONE_API_KEY` – required for any operation that talks to Pinecone

## Installation

Install minimal dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install langchain-text-splitters pinecone
```

> [!NOTE] > `ruff` line length is configured to 128 via `pyproject.toml`, but `ruff` itself isn’t pinned as a dependency.

## Configuration

- Set your API key in the environment: `export PINECONE_API_KEY=…`
- The Pinecone index name used by `db_create.py` is `ragtag-db` (AWS / us-east-1).
- The ingestion script currently uses a fixed host:

  ```python
  pinecone_host = "https://ragtag-db-f059e7z.svc.aped-4627-b74a.pinecone.io"
  ```

  > [!WARNING]
  > If you create a new index in a different project/region, update `pinecone_host` in `scripts/split_text.py` to your index’s host, or refactor to read from an environment variable.

## Create the index

This is idempotent; it’s safe to re-run.

```bash
export PINECONE_API_KEY=…
python -m scripts.db_create
# or
python scripts/db_create.py
```

## Ingest a document (split and upsert)

`scripts/split_text.py` CLI flags:

- `--document-id` (required)
- `--document-url` (required)
- `--document-path` (required, path to a markdown file)
- `--pinecone-namespace` or `--namespace` (required)
- `--dry-run` (optional) to write JSON instead of upserting
- `--output` (optional, JSON path for `--dry-run`, default `logs/document_chunks.json`)

Example (dry run):

```bash
export PINECONE_API_KEY=…
python -m scripts.split_text \
  --dry-run \
  --document-id cribl-fastmcp \
  --document-url https://example.com/fastmcp \
  --document-path docs/fastmcp-llms-full.txt.md \
  --namespace cribl

# Inspect output
wc -l logs/document_chunks.json
```

Example (upsert to Pinecone):

```bash
export PINECONE_API_KEY=…
python -m scripts.split_text \
  --document-id cribl-fastmcp \
  --document-url https://example.com/fastmcp \
  --document-path docs/fastmcp-llms-full.txt.md \
  --namespace cribl
```

> [!TIP]
> Check logs at `logs/split_text.<YYYY-MM-DD>.log` for per-run details and a compact summary of the upsert response.

## Updating or re-importing docs

Use a consistent `document_id` and the same namespace. Since records use `_id = "<document_id>:chunk<idx>"`, re-running ingestion with the updated document will overwrite per-chunk records. For large structural changes, consider clearing the namespace first.

Planned improvements (see `TODO.md`):

- Config file for sources and variables
- Web crawl, JSON, and PDF readers
- Logging of per-chunk status
- Delete and re-import helper for existing docs

## Development

Run scripts in place; no build step required.

Optional linting (if you have ruff installed):

```bash
ruff check .
```

## Troubleshooting

- Missing `PINECONE_API_KEY`

  - Symptom: RuntimeError("PINECONE_API_KEY is not set in the environment")
  - Fix: `export PINECONE_API_KEY=…` and retry.

- Wrong Pinecone host

  - Symptom: Connection errors or upserts going to the wrong index.
  - Fix: Update `pinecone_host` in `scripts/split_text.py` to the host of your `ragtag-db` index (or make it configurable).

- Empty or tiny chunk count
  - Symptom: Few records produced.
  - Fix: Adjust `chunk_size` / `chunk_overlap` in `scripts/split_text.py`.

## References

- `AGENTS.md` – background, goals, and future MCP agent access
- `docs/` – example input documents (markdown/PDF/YAML)
- LangChain Text Splitters – https://python.langchain.com/docs/modules/data_connection/document_transformers/
- Pinecone Python SDK – https://docs.pinecone.io/reference/overview

---

<sub>Status: early-stage; interfaces may change. Feedback and issues welcome.</sub>
