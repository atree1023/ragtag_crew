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
  - `scripts/doc_dwnld.py` – downloads sources declared in `scripts/docs_config.py` into `docs/`.
  - `scripts/docs_config.py` – typed config + validators + helpers for orchestration.
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
  - No unit test suite is present; rely on dry runs, logging, and targeted scripts for validation.

## Core Workflows

1. **Create the Pinecone index** (`scripts/db_create.py`)
   - Idempotent; safe to rerun. Uses `pinecone.Pinecone(...).create_index_for_model` with the `llama-text-embed-v2` preset (`field_map={"text": "chunk_content"}`) targeting AWS `us-east-1`.
2. **Download source documents** (`scripts/doc_dwnld.py`)
   - Configuration lives in `scripts/docs_config.py` (`docs_config` mapping).
   - `python -m scripts.doc_dwnld --list` prints `<id>\t<input-format>\t<document-url>`.
   - `--id <doc_id>` downloads a single entry; `--all` iterates the full config.
   - Text downloads auto-convert HTML to plain text unless the URL already ends with `.txt`.
   - Files land under `docs/` (created on demand) or at the configured path relative to the repo root.
3. **Split & ingest documents** (`scripts/split_text.py`)
   - Required flags: `--document-id`, `--document-url`, `--document-path`, `--pinecone-namespace` (`--namespace` alias), and a Pinecone host (`--host` or `PINECONE_HOST`).
   - Formats: `markdown` (header-aware via `MarkdownHeaderTextSplitter`), `text`, `pdf` (pypdf text extraction), `json` (`RecursiveJsonSplitter`), `yaml` (safe-load to Python, then treat as JSON).
   - Chunk tuning constants live at module top (`max_chunk_size=1792`, `chunk_overlap=128`).
   - Dry run (`--dry-run`) writes JSON records to `logs/document_chunks.json` (override with `--output`).
   - Upserts batch in groups of 64 via `index.upsert_records(...)`; responses are summarized in the log.
   - Logging: `logs/split_text.<YYYY-MM-DD>.log` (auto-created) + console output.
4. **Clean a namespace** (`scripts/ns_delete.py`)
   - CLI: `python -m scripts.ns_delete --namespace <name> [--host ...]`.
   - Programmatic helper: `delete_namespace_records(namespace, host, api_key=None)`; raises `NamespaceDeleteError` on missing inputs.
5. **Orchestrate from config** (`scripts/docs_config.py` helpers)
   - `validate_docs_config(...)` ensures entries have required keys, valid URLs, matching file suffixes, and non-empty namespaces.
   - `build_split_cli_args(doc_id, dry_run=True, output=...)` constructs a safe argument list for invoking `split_text.py` (performs validation first).

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
  - Validation checks ensure local files exist; during local development you may need to create placeholder docs before validation passes.
  - Extend `ALLOWED_INPUT_FORMATS` and adjust downstream logic if you add new ingestion modes.
- `scripts/db_create.py`
  - Only side effect is index creation via Pinecone control plane; no logging, simply exits once index exists (or is created).

## Common Agent Tasks & Tips

- **Dry-run new document ingestion:**
  1. Add/adjust entry in `scripts/docs_config.py`.
  2. `python -m scripts.doc_dwnld --id <doc_id>` to fetch the source.
  3. `python -m scripts.split_text --dry-run ... --output logs/<doc_id>.json` to inspect chunk JSON.
  4. Review logs under `logs/` and optionally validate JSON contents.
- **Production upsert:** repeat the dry-run command without `--dry-run`; ensure `PINECONE_API_KEY` + host are set.
- **Namespace reset:** run `python -m scripts.ns_delete --namespace <ns>` before re-importing large structural changes.
- **Config validation:** invoke `validate_docs_config(docs_config)` in a Python shell or orchestrator to catch mistakes early. Remember relative `document-path` entries resolve against `scripts/` by default.
- **Logging hygiene:** `split_text.py` appends to a date-based log; clear or rotate logs manually if necessary. Failures bubble up via exit code 1.
- **Missing directories:** `docs/` and `logs/` are created at runtime. Agents should create them (`Path(...).mkdir(parents=True, exist_ok=True)`) before writing new assets.

## Coding & Contribution Guidelines

- Follow Ruff's configuration (line length 128, ignore `D203`, `D213`).
- Use type hints consistently (project already leverages `TypedDict`, dataclasses, and explicit type annotations).
- Prefer dependency-free utilities where practical (e.g., stdlib downloads). Keep CLI ergonomics aligned with existing scripts.
- When introducing new formats or pipelines, update `README.md`, this `AGENTS.md`, `docs_config` validation, and any relevant helper functions together.
- No automated test harness exists; add smoke scripts or leverage dry-run outputs to demonstrate correctness.

## Operational Checklist for Agents

- [ ] Ensure virtual environment + dependencies installed.
- [ ] Export `PINECONE_API_KEY` and `PINECONE_HOST` (or pass `--host`).
- [ ] If editing configs, run `python - <<'PY' ... PY` snippet to call `validate_docs_config` before invoking scripts.
- [ ] Maintain `document_id` + namespace consistency to benefit from chunk ID idempotency (`<document_id>:chunk<idx>`).
- [ ] Capture relevant logs/artifacts under `logs/` for debugging; summarize in PRs or change notes.

With this playbook an agentic coding tool should be able to answer questions about the project, modify ingestion behavior, and run the existing workflows without manual hand-holding.
