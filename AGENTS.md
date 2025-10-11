# AGENT PLAYBOOK: ragtag-crew

This repository contains a Python 3.13 toolchain for maintaining Pinecone vector database content for documentation-oriented AI assistants. The goal of this playbook is to help `gpt-5-codex` (and similar agentic tools) navigate, extend, and operate the project safely.

## Repository Snapshot

- **Domain:** Documentation ingestion and Pinecone index management.
- **Primary language:** Python (>=3.13) with a thin scripts package under `scripts/`.
- **Dependencies:** `langchain-text-splitters`, `pinecone`, `pypdf`, `PyYAML`; optional `ruff` for linting (see `pyproject.toml`).
- **Key files:**
  - `scripts/db_create.py` – idempotent Pinecone index creator for `ragtag-db`.
  - `scripts/split_text.py` – multi-format document splitter + Pinecone upsert orchestrator.
  - `scripts/ns_delete.py` – namespace delete helper (script + importable API).
  - `scripts/doc_dwnld.py` – downloads sources declared in `scripts/docs_config_data.yaml` into `docs/` via the `docs_config` helpers.
  - `scripts/docs_config.py` – typed config interface, validators, and helpers for orchestrating document ingestion.
  - `scripts/docs_config_data.yaml` – YAML manifest of document sources consumed by `docs_config.py`.
- **Supporting artifacts:** `README.md` (user-facing guide), `uv.lock` (dependency lock), `logs/` directory is created on demand, `docs/` paths are expected but not committed.

## Environment & Tooling Setup

- Use Python 3.13+. Recommended quick start:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -e .  # installs dependencies from pyproject.toml
  ```

- Required environment variables when calling Pinecone:
  - `PINECONE_API_KEY` – mandatory for control-/data-plane calls.
  - `PINECONE_HOST` – preferred way to supply the index host (`https://<index>.svc.<project>.pinecone.io`). CLI flags `--host` fall back to this.
- Optional tooling:
  - `ruff check .` (line length 128; configured to ignore `D203`, `D213`).
  - `pyright` strict type checking (`pyright` from the repo root).
  - Pytest suite lives under `tests/` and exercises the downloader, config helpers, namespace deletion, and text splitting logic (`pytest` from the repo root).

## Core Workflows

1. **Create the Pinecone index** (`scripts/db_create.py`): Idempotent; safe to rerun. Uses `pinecone.Pinecone(...).create_index_for_model` with the `llama-text-embed-v2` preset (`field_map={"text": "chunk_content"}`) targeting AWS `us-east-1`.

2. **Download source documents** (`scripts/doc_dwnld.py`): Configuration lives in `scripts/docs_config_data.yaml` (loaded through `scripts/docs_config.py`). Key commands include `python -m scripts.doc_dwnld --list` for tab-separated inventory, `--id <doc_id>` to download a single entry, and `--all` for the full manifest. Text downloads auto-convert HTML to plain text unless the URL already ends with `.txt`, and files land under `docs/` (created on demand) or at the configured path relative to the repo root.

3. **Split & ingest documents** (`scripts/split_text.py`): Required flags include `--document-id`, `--document-url`, `--document-path`, `--pinecone-namespace` (`--namespace` alias), and a Pinecone host (`--host` or `PINECONE_HOST`). Supported formats are `markdown`, `text`, `pdf`, `json`, and `yaml`. Chunk tuning constants live at module top (`max_chunk_size=1792`, `chunk_overlap=128`). Dry runs write JSON records to `logs/document_chunks.json` (override with `--output`); production runs upsert in batches of 64 and log summaries to `logs/split_text.<YYYY-MM-DD>.log` alongside console output.

4. **Clean a namespace** (`scripts/ns_delete.py`): Provides both CLI usage (`python -m scripts.ns_delete --namespace <name> [--host ...]`) and the `delete_namespace_records(namespace, host, api_key=None)` helper. Errors raise `NamespaceDeleteError` when inputs are missing.

5. **Orchestrate from config** (`scripts/docs_config.py` helpers): `load_docs_config(...)` / `save_docs_config(...)` interact with the YAML manifest, `validate_docs_config(...)` enforces schema and path integrity, `upsert_doc_entry(...)` / `remove_doc_entry(...)` mutate single entries with validation, and `build_split_cli_args(doc_id, dry_run=True, output=...)` constructs validated argument lists for `split_text.py`.

## Repository Details Worth Knowing

- `scripts/split_text.py`
  - `build_document_chunks` returns a list of `DocumentChunk` dataclasses; `chunk_section_id` is omitted for non-markdown inputs.
  - `create_document_chunks_from_path` centralizes per-format branching; reuse it when adding new formats to keep `main()` slim.
  - `upsert_records` enforces `PINECONE_API_KEY` and host presence even in batch loops; adjust `batch_size` argument to tune throughput.
  - Errors are logged with full stack traces and cause an exit code of 1.
- `scripts/doc_dwnld.py`
  - Uses stdlib networking (`urllib`); no external dependencies besides config.
  - `_HTMLTextExtractor` strips scripts/styles and normalizes whitespace before saving text downloads.
  - Returns exit status 0 on success, 2 for selection errors, 3 when any download fails.
- `scripts/docs_config.py`
  - `DocsConfig` is a plain `dict[str, DocConfig]`; each `DocConfig` is a `TypedDict` with hyphenated keys to mirror CLI flags.
  - Validation checks ensure local files exist and that suffixes match formats (text entries must point to `.txt` files); during local development you may need to create placeholder docs before validation passes.
  - Extend `ALLOWED_INPUT_FORMATS` and adjust downstream logic if you add new ingestion modes.
- `scripts/db_create.py`
  - Only side effect is index creation via Pinecone control plane; no logging, simply exits once index exists (or is created).

## Common Agent Tasks & Tips

- **Dry-run new document ingestion:**
  1. Add/adjust entry in `scripts/docs_config_data.yaml` (or call `upsert_doc_entry` from `scripts.docs_config`).
  2. `python -m scripts.doc_dwnld --id <doc_id>` to fetch the source.
  3. `python -m scripts.split_text --dry-run ... --output logs/<doc_id>.json` to inspect chunk JSON.
  4. Review logs under `logs/` and optionally validate JSON contents.
- **Production upsert:** repeat the dry-run command without `--dry-run`; ensure `PINECONE_API_KEY` + host are set.
- **Namespace reset:** run `python -m scripts.ns_delete --namespace <ns>` before re-importing large structural changes.
- **Config validation:** invoke `validate_docs_config(load_docs_config())` in a Python shell or orchestrator to catch mistakes early. Remember relative `document-path` entries resolve against `scripts/` by default.
- **Logging hygiene:** `split_text.py` appends to a date-based log; clear or rotate logs manually if necessary. Failures bubble up via exit code 1.
- **Missing directories:** `docs/` and `logs/` are created at runtime. Agents should create them (`Path(...).mkdir(parents=True, exist_ok=True)`) before writing new assets.

## Coding & Contribution Guidelines

- Follow Ruff's configuration (line length 128, ignore `D203`, `D213`).
- Use type hints consistently (project already leverages `TypedDict`, dataclasses, and explicit type annotations).
- Prefer dependency-free utilities where practical (e.g., stdlib downloads). Keep CLI ergonomics aligned with existing scripts.
- When introducing new formats or pipelines, update `README.md`, this `AGENTS.md`, `docs_config` validation, and any relevant helper functions together.
- Keep the pytest suite current—add or update tests in `tests/` whenever behavior changes, and run `pytest` to confirm coverage before shipping.
- Before considering work complete, run `pyright` and `ruff check .` from the repository root and resolve every reported issue.

## Operational Checklist for Agents

- [ ] Ensure virtual environment + dependencies installed.
- [ ] Export `PINECONE_API_KEY` and `PINECONE_HOST` (or pass `--host`).
- [ ] If editing configs, run `python - <<'PY' ... PY` snippet to call `validate_docs_config` before invoking scripts.
- [ ] Maintain `document_id` + namespace consistency to benefit from chunk ID idempotency (`<document_id>:chunk<idx>`).
- [ ] Capture relevant logs/artifacts under `logs/` for debugging; summarize in PRs or change notes.
- [ ] Run `pytest` when modifying ingestion helpers to ensure regressions are caught early.
- [ ] Run `pyright` and `ruff check .` from the repository root and resolve any reported errors before shipping.

With this playbook an agentic coding tool should be able to answer questions about the project, modify ingestion behavior, and run the existing workflows without manual hand-holding.
