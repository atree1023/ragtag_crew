"""Download documents defined in ``scripts.docs_config`` into the local ``docs/`` directory.

This helper reads the project-level ``docs_config`` mapping and downloads each
document's content from its ``document-url`` into the repository's ``docs/``
folder using consistent file names.

CLI usage
---------

- List available document IDs:
    python -m scripts.doc_dwnld --list

- Download a single document by id:
    python -m scripts.doc_dwnld --id fastmcp-docs

- Download all configured documents:
    python -m scripts.doc_dwnld --all

Special behavior for text inputs
--------------------------------

If an entry has ``input-format == "text"`` and its ``document-url`` does not
end with ".txt", the URL is treated as an HTML page. It is fetched, converted
to plain text by stripping HTML tags, and written to a file named
``<document-id>.txt`` in ``docs/`` (e.g., ``cribl-api.txt``).

Otherwise, content is downloaded as-is and saved to the configured
``document-path`` relative to the repository root (typically inside ``docs/``).

Notes
-----
- This tool does not require any external HTTP libraries; it uses the Python
  standard library (``urllib``) to download content.
- Existing files are overwritten by default.

"""

from __future__ import annotations

import argparse
import html
import logging
import re
import sys
from pathlib import Path
from html.parser import HTMLParser
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .docs_config import DocConfig, DocsConfig, docs_config, resolve_document_path

if TYPE_CHECKING:
    from collections.abc import Iterable


def project_root() -> Path:
    """Return the repository root directory.

    This assumes the script lives at ``<repo>/scripts/doc_dwnld.py``.
    """
    return Path(__file__).resolve().parent.parent


def docs_dir() -> Path:
    """Return the absolute path to the repository ``docs/`` directory."""
    return project_root() / "docs"


def _read_url_bytes(url: str, *, user_agent: str = "ragtag-crew/0.1") -> tuple[bytes, dict[str, str]]:
    """Read bytes from a URL and return (content, headers).

    Args:
        url: HTTP(S) URL to fetch.
        user_agent: Optional User-Agent header value.

    Raises:
        URLError, HTTPError: On network or HTTP issues.

    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        msg = f"URL must start with http(s): {url!r}"
        raise ValueError(msg)
    req = Request(url, headers={"User-Agent": user_agent})  # noqa: S310 - scheme validated above
    with urlopen(req) as resp:  # noqa: S310 - standard library urlopen allowed in this script
        data = resp.read()
        headers: dict[str, str] = {k.lower(): v for k, v in resp.headers.items()}
    return data, headers


def _infer_encoding(headers: dict[str, str]) -> str:
    ct = headers.get("content-type", "")
    m = re.search(r"charset=([\w.-]+)", ct, flags=re.IGNORECASE)
    return (m.group(1) if m else "utf-8").strip()


class _HTMLTextExtractor(HTMLParser):
    """HTML parser that extracts visible text, skipping script and style blocks.

    This class is used to convert HTML content to plain text by ignoring
    the contents of <script> and <style> tags.
    """
    def __init__(self):
        """Initialize the parser and internal state."""
        super().__init__()
        self.result = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        """Set skip flag when entering <script> or <style> tags."""
        if tag.lower() in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        """Clear skip flag when exiting <script> or <style> tags."""
        if tag.lower() in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        """Collect text data unless inside a skipped tag."""
        if self._skip_depth == 0:
            self.result.append(data)

    def get_text(self):
        """Return the concatenated visible text extracted from HTML."""
        return "".join(self.result)


def _strip_html_to_text(html_bytes: bytes, *, encoding: str | None = None) -> str:
    """Convert HTML bytes to a readable plain text string.

    This is a lightweight conversion:
    - Remove <script> and <style> blocks
    - Replace remaining tags with spaces
    - Unescape HTML entities
    - Normalize whitespace

    Returns:
        A simplified plain text representation.

    """
    text = html_bytes.decode(encoding or "utf-8", errors="ignore")
    parser = _HTMLTextExtractor()
    parser.feed(text)
    text = parser.get_text()
    # Unescape entities and collapse whitespace
    text = html.unescape(text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r" \n+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _url_path_basename(url: str) -> str:
    path = urlparse(url).path
    # Strip trailing slash to get a basename if the URL ends with '/'
    path = path.removesuffix("/")
    return Path(path).name


def _is_txt_url(url: str) -> bool:
    name = _url_path_basename(url).lower()
    return bool(name) and name.endswith(".txt")


def iter_selected_docs(
    cfg: DocsConfig,
    *,
    all_docs: bool,
    single_id: str | None,
) -> Iterable[tuple[str, DocConfig]]:
    """Yield selected (doc_id, entry) pairs from the config based on flags."""
    if all_docs:
        yield from cfg.items()
        return
    if single_id:
        if single_id not in cfg:
            msg = f"unknown document id: {single_id!r}"
            raise KeyError(msg)
        yield single_id, cfg[single_id]
        return
    msg = "either --all or --id must be provided (or use --list)"
    raise ValueError(msg)


def download_one(doc_id: str, entry: DocConfig) -> Path:
    """Download one document according to its config.

    Behavior:
    - For input-format == 'text' and URL not ending with .txt: treat as HTML -> text
      and save to docs/<doc_id>.txt
    - Otherwise: download bytes as-is and save to the configured document-path
      relative to the project root.

    Returns:
        The absolute path to the written file.

    """
    fmt = str(entry.get("input-format", "")).strip().lower()
    url = str(entry.get("document-url", "")).strip()

    root = project_root()
    target: Path

    # Destination resolution
    if fmt == "text" and not _is_txt_url(url):
        # Force document-id based .txt filename in docs/
        target = docs_dir() / f"{doc_id}.txt"
    else:
        # Use configured document-path (relative to project root)
        target = resolve_document_path(entry, base_dir=root)

    target.parent.mkdir(parents=True, exist_ok=True)

    # Fetch content
    data, headers = _read_url_bytes(url)

    if fmt == "text" and not _is_txt_url(url):
        # Convert HTML-ish content to plain text and save as UTF-8
        encoding = _infer_encoding(headers)
        text = _strip_html_to_text(data, encoding=encoding)
        target.write_text(text, encoding="utf-8")
    else:
        # Save as-is (binary)
        target.write_bytes(data)

    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for selecting which documents to download."""
    parser = argparse.ArgumentParser(description="Download configured documents into docs/")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--list", action="store_true", help="List available document ids and exit")
    group.add_argument("--all", action="store_true", help="Download all documents")
    group.add_argument("--id", dest="doc_id", help="Download a single document by id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint to list or download configured documents."""
    # Basic logging to stdout to mirror other scripts
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(__name__)

    args = parse_args(argv)

    if args.list:
        # Print one per line: id TAB input-format TAB url
        for doc_id, entry in docs_config.items():
            fmt = entry.get("input-format", "")
            url = entry.get("document-url", "")
            sys.stdout.write(f"{doc_id}\t{fmt}\t{url}\n")
        return 0

    try:
        selected = list(iter_selected_docs(docs_config, all_docs=args.all, single_id=args.doc_id))
    except (KeyError, ValueError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2

    exit_code = 0
    for doc_id, entry in selected:
        try:
            dest = download_one(doc_id, entry)
            logger.info("downloaded: %s -> %s", doc_id, dest)
        except (HTTPError, URLError):
            logger.exception("download failed: id=%s", doc_id)
            exit_code = 3
        except Exception:
            logger.exception("processing failed: id=%s", doc_id)
            exit_code = 3

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
