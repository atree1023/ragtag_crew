# ragtag-crew

Tools for creating and maintaining Pinecone vector DB lookups for up‑to‑date documentation and code examples.

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.13-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-00A77F?logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![LangChain](https://img.shields.io/badge/Text%20Splitters-LangChain-1C3C3C)](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

> [!NOTE]
> This repository currently focuses on ingest tooling: splitting docs, building records, and upserting to a Pinecone index. Agent access (MCP tools) is planned.

## Overview

AI coding agents work best with fresh, searchable technical context. ragtag-crew provides a small, scriptable toolchain to:

- Create a Pinecone index configured for text embedding
- Split markdown docs into header-aware, token-friendly chunks
- Split PDFs by extracting text (images discarded), then chunking as plain text
- Split JSON using `RecursiveJsonSplitter` (chunks are JSON strings)
- Parse YAML and split as JSON (via `yaml.safe_load`)
- Upsert rich records with document metadata into Pinecone namespaces
- Delete all records in a Pinecone namespace when you want to fully refresh it

The goal is a reliable pipeline to collect, update, and access documentation and example code as vector search context.

## Features

- Header-aware markdown parsing using LangChain text splitters
- PDF-to-text extraction using pypdf (no OCR)
- Tunable chunk size and overlap (defaults: 1792 / 128)
- Record schema optimized for retrieval with clear metadata fields
- Batch upserts to Pinecone with simple logging and dry-run mode
- Idempotent index creation helper
- Namespace delete utility to cleanly reset a namespace
- Document downloader to fetch sources into `docs/` from `scripts/docs_config.py`

> [!TIP]
> Keep each documentation source in its own Pinecone namespace to simplify updates and deletions without cross-talk.

## How it works

Four scripts power the pipeline today:

- `scripts/db_create.py` – Creates the Pinecone index `ragtag-db` with an integrated embedding model and field map `{"text": "chunk_content"}`.
- `scripts/split_text.py` – Splits a document (markdown, text, pdf, json, yaml) into chunks and either upserts records to Pinecone or writes them to JSON (dry run).
- `scripts/ns_delete.py` – Deletes all records from a specified Pinecone namespace on the configured index host.
- `scripts/doc_dwnld.py` – Downloads configured documents to the local `docs/` folder using the entries in `scripts/docs_config.py`.

### Data model (record schema)

Each chunk becomes a record shaped like:

```jsonc
{
  "_id": "<document_id>:chunk<idx>",
  "document_id": "<your-doc-id>",
  "document_url": "https://…",
  "document_date": "YYYY-MM-DD", // run date
  "chunk_content": "…", // text used for embedding (per index field map)
  "chunk_section_id": "Header_1|Header_2|…" // joined header path (present for markdown; omitted for non-markdown)
}
```

> [!IMPORTANT]
> The index created by `db_create.py` maps the embedding model’s `text` field to `chunk_content`. If you change field names, update the index configuration and the ingestion code together.
>
> For non-markdown inputs (text, pdf, json, yaml), `chunk_section_id` is not included in records.

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
pip install langchain-text-splitters pinecone pypdf PyYAML

# Alternatively, install from this repo (uses pyproject.toml dependencies):
pip install -e .
```

## Configuration

- Set your API key in the environment: `export PINECONE_API_KEY=…` or using a tool such as Doppler
- The Pinecone index name used by `db_create.py` is `ragtag-db` (AWS / us-east-1).
- Provide your Pinecone index host in one of two ways:
  - Preferred: set an environment variable `export PINECONE_HOST=https://<index>.svc.<project>.pinecone.io`
  - Or pass `--host https://<index>.svc.<project>.pinecone.io` to the CLI

> [!IMPORTANT]
> All scripts accept or default to the Pinecone host via `--host` or `PINECONE_HOST`. This ensures the tools can target
> the correct index across projects and regions without editing code.

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
- `--document-path` (required, path to a file)
- `--input-format` (optional, one of `markdown`, `text`, `pdf`, `json`, `yaml`; default `markdown`)
- `--pinecone-namespace` or `--namespace` (required)
- `--host` (required unless `PINECONE_HOST` is set) – Pinecone index host URL
- `--dry-run` (optional) to write JSON instead of upserting
- `--output` (optional, JSON path for `--dry-run`, default `logs/document_chunks.json`)

Example (dry run, markdown):

```bash
export PINECONE_API_KEY=…
export PINECONE_HOST=https://your-index.svc.your-project.pinecone.io
python -m scripts.split_text \
  --dry-run \
  --document-id cribl-fastmcp \
  --document-url https://example.com/fastmcp \
  --document-path docs/fastmcp-llms-full.txt.md \
  --input-format markdown \
  --namespace cribl

# Inspect output
wc -l logs/document_chunks.json
```

Or pass host explicitly:

```bash
python -m scripts.split_text \
  --dry-run \
  --document-id cribl-fastmcp \
  --document-url https://example.com/fastmcp \
  --document-path docs/fastmcp-llms-full.txt.md \
  --input-format markdown \
  --namespace cribl \
  --host https://your-index.svc.your-project.pinecone.io
```

Example (upsert to Pinecone):

```bash
export PINECONE_API_KEY=…
export PINECONE_HOST=https://your-index.svc.your-project.pinecone.io
python -m scripts.split_text \
  --document-id cribl-fastmcp \
  --document-url https://example.com/fastmcp \
  --document-path docs/fastmcp-llms-full.txt.md \
  --input-format markdown \
  --namespace cribl
```

### Ingest a PDF

PDFs are converted to text first (no OCR; images are ignored), then split as plain text.

Dry run:

```bash
python -m scripts.split_text \
  --dry-run \
  --document-id cribl-edge-pdf \
  --document-url https://example.com/edge-pdf \
  --document-path docs/cribl-edge-docs-4.13.3.pdf \
  --input-format pdf \
  --namespace cribl \
  --host https://your-index.svc.your-project.pinecone.io
```

Upsert:

```bash
python -m scripts.split_text \
  --document-id cribl-edge-pdf \
  --document-url https://example.com/edge-pdf \
  --document-path docs/cribl-edge-docs-4.13.3.pdf \
  --input-format pdf \
  --namespace cribl \
  --host https://your-index.svc.your-project.pinecone.io
```

> [!TIP]
> Check logs at `logs/split_text.<YYYY-MM-DD>.log` for per-run details and a compact summary of the upsert response.

## Download documents

Use the downloader to fetch the sources defined in `scripts/docs_config.py` into your local `docs/` directory.

List available document ids:

```bash
python -m scripts.doc_dwnld --list
```

Download a single document by id:

```bash
python -m scripts.doc_dwnld --id fastmcp-docs
```

Download all configured documents:

```bash
python -m scripts.doc_dwnld --all
```

Notes:

- If an entry has `input-format` set to `text` and its `document-url` does not end with `.txt`, the page is fetched as HTML and converted to plain text by stripping tags, then saved as `docs/<document-id>.txt` (for example, `cribl-api` becomes `docs/cribl-api.txt`).
- For other formats, the file is saved to the configured `document-path` (typically under `docs/`).

## Delete a namespace

Use this when you want to fully refresh a namespace before re-importing.

Using environment variables for host:

```bash
export PINECONE_API_KEY=…
export PINECONE_HOST=https://your-index.svc.your-project.pinecone.io
python -m scripts.ns_delete --namespace cribl
```

Or pass host explicitly:

```bash
python -m scripts.ns_delete \
  --namespace cribl \
  --host https://your-index.svc.your-project.pinecone.io
```

Programmatic usage:

```python
from scripts.ns_delete import delete_namespace_records

delete_namespace_records(
    namespace="cribl",
    host="https://your-index.svc.your-project.pinecone.io",
)
```

### Ingest JSON

Dry run:

```bash
python -m scripts.split_text \
  --dry-run \
  --document-id my-json \
  --document-url https://example.com/my-json \
  --document-path docs/sample.json \
  --input-format json \
  --namespace myns \
  --host https://your-index.svc.your-project.pinecone.io
```

### Ingest YAML

Dry run:

```bash
python -m scripts.split_text \
  --dry-run \
  --document-id my-yaml \
  --document-url https://example.com/my-yaml \
  --document-path docs/sample.yaml \
  --input-format yaml \
  --namespace myns \
  --host https://your-index.svc.your-project.pinecone.io
```

## Updating or re-importing docs

Use a consistent `document_id` and the same namespace. Since records use `_id = "<document_id>:chunk<idx>"`, re-running ingestion with the updated document will overwrite per-chunk records. For large structural changes, consider clearing the namespace first using `scripts/ns_delete.py`.

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

- Missing or wrong Pinecone host

  - Symptom: CLI exits with `--host is required (or set PINECONE_HOST…)` or connection errors to Pinecone.
  - Fix: Set `export PINECONE_HOST=…` or pass `--host …` to the scripts. Double-check the host URL for your index.

- Empty or tiny chunk count
  - Symptom: Few records produced.
  - Fix: Adjust `chunk_size` / `chunk_overlap` in `scripts/split_text.py`.

## References

- `docs/` – example input documents (markdown/PDF/YAML/Text)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Pinecone Python SDK](https://docs.pinecone.io/reference/overview)

---

_Status: early-stage; interfaces may change. Feedback and issues welcome._
