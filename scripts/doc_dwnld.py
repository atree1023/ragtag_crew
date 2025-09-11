r"""Download configured documents into the local ``docs/`` directory.

This module reads the project-level mapping from ``scripts.docs_config`` and
downloads each document's content from its ``document-url`` into this
repository, writing files under ``docs/`` with consistent names.

CLI usage
---------

- List available document IDs and their sources (tab-separated output):
        python -m scripts.doc_dwnld --list
    Output columns: ``<id>\t<input-format>\t<document-url>``

- Download a single document by id:
        python -m scripts.doc_dwnld --id fastmcp-docs

- Download all configured documents:
        python -m scripts.doc_dwnld --all

Special behavior for text inputs
--------------------------------

If an entry has ``input-format == "text"`` and its ``document-url`` does not
end with ``.txt``, the URL is treated as an HTML page. The page is fetched,
converted to plain text by stripping tags (``<script>`` and ``<style>``
contents are ignored), and written to ``docs/<document-id>.txt`` (e.g.,
``docs/cribl-api.txt``).

Otherwise, content is downloaded as-is and saved to the configured
``document-path`` relative to the repository root (typically inside ``docs/``).

Notes
-----
- Uses only the Python standard library (``urllib``) for HTTP(S) downloads and
    sets a small, fixed ``User-Agent``.
- Existing files are overwritten by default.

"""

from __future__ import annotations

import argparse
import html
import logging
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .docs_config import DocConfig, DocsConfig, docs_config, resolve_document_path

if TYPE_CHECKING:
    from collections.abc import Iterable


def project_root() -> Path:
    """Return the repository root directory.

    Assumes the script lives at ``<repo>/scripts/doc_dwnld.py`` and returns
    ``<repo>``.

    Returns:
        Absolute path to the repository root.

    """
    return Path(__file__).resolve().parent.parent


def docs_dir() -> Path:
    """Return the absolute path to the repository ``docs/`` directory."""
    return project_root() / "docs"


def _read_url_bytes(url: str, *, user_agent: str = "ragtag-crew/0.1") -> tuple[bytes, dict[str, str]]:
    """Read bytes from a URL and return ``(content, headers)``.

    Args:
        url: Absolute HTTP(S) URL to fetch.
        user_agent: Optional ``User-Agent`` header value.

    Returns:
        A tuple ``(data, headers)`` where ``data`` is the raw response body and
        ``headers`` is a case-insensitive dict of response headers (all keys lowered).

    Raises:
        ValueError: If the URL doesn't begin with ``http`` or ``https``.
        URLError, HTTPError: On network or HTTP failures.

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
    """Infer response text encoding from headers, defaulting to UTF-8.

    Args:
        headers: Response headers with lower-cased keys.

    Returns:
        A best-effort character set name, defaulting to ``"utf-8"``.

    """
    ct = headers.get("content-type", "")
    m = re.search(r"charset=([\w.-]+)", ct, flags=re.IGNORECASE)
    return (m.group(1) if m else "utf-8").strip()


class _HTMLTextExtractor(HTMLParser):
    """HTML parser that extracts visible text, skipping script/style blocks.

    Use this helper to convert HTML content to plain text. It ignores the
    contents of ``<script>`` and ``<style>`` tags and concatenates visible text.
    """

    def __init__(self) -> None:
        """Initialize the parser and internal state."""
        super().__init__()
        self.result = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Set skip flag when entering ``<script>``/``<style>`` tags."""
        # attributes are not used in extraction; explicitly discard to avoid linters
        del attrs
        if tag.lower() in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        """Clear skip flag when exiting ``<script>``/``<style>`` tags."""
        if tag.lower() in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        """Collect text data unless inside a skipped tag."""
        if self._skip_depth == 0:
            self.result.append(data)

    def get_text(self) -> str:
        """Return the concatenated visible text extracted from HTML."""
        return " ".join(s for s in (frag.strip() for frag in self.result) if s)


def _strip_html_to_text(html_bytes: bytes, *, encoding: str | None = None) -> str:
    """Convert HTML bytes to a readable plain-text string.

    This is a lightweight conversion that:
    - Removes ``<script>`` and ``<style>`` blocks
    - Drops all other tags and concatenates their text content
    - Unescapes HTML entities
    - Normalizes whitespace (collapses runs and trims)

    Args:
        html_bytes: Raw HTML response body.
        encoding: Optional character encoding. Defaults to ``"utf-8"`` if not provided.

    Returns:
        A simplified plain-text representation of the page.

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
    """Return the basename portion of the URL path (sans trailing slash)."""
    path = urlparse(url).path
    # Strip trailing slash to get a basename if the URL ends with '/'
    path = path.removesuffix("/")
    return Path(path).name


def _is_txt_url(url: str) -> bool:
    """Return True if the URL path ends with ``.txt`` (case-insensitive)."""
    name = _url_path_basename(url).lower()
    return bool(name) and name.endswith(".txt")


def iter_selected_docs(
    cfg: DocsConfig,
    *,
    all_docs: bool,
    single_id: str | None,
) -> Iterable[tuple[str, DocConfig]]:
    """Yield selected ``(doc_id, entry)`` pairs from the config based on flags.

    Args:
        cfg: The configuration mapping.
        all_docs: If True, iterate all configured documents.
        single_id: If provided, iterate only the requested document id.

    Yields:
        Pairs of ``(document-id, DocConfig)`` values.

    Raises:
        KeyError: If ``single_id`` is provided but not present in ``cfg``.
        ValueError: If neither ``all_docs`` nor ``single_id`` is provided.

    """
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
    """Download one document according to its configuration.

    Behavior:
    - If ``input-format == 'text'`` and the URL does not end with ``.txt``:
      treat the content as HTML, convert to plain text, and save to
      ``docs/<doc_id>.txt``.
    - Otherwise: download bytes as-is and save to the configured
      ``document-path`` relative to the project root.

    Args:
        doc_id: The document id key in the config.
        entry: The configuration mapping for this document.

    Returns:
        Absolute path to the written file.

    Raises:
        ValueError: If the ``document-url`` has a non-HTTP(S) scheme.
        URLError, HTTPError: For network/HTTP failures.

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
    """Parse CLI arguments for selecting which documents to download.

    Args:
        argv: Optional list of arguments (excluding program name). If omitted,
            arguments are read from ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with ``list``, ``all``, and ``doc_id`` fields.

    """
    parser = argparse.ArgumentParser(description="Download configured documents into docs/")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--list", action="store_true", help="List available document ids and exit")
    group.add_argument("--all", action="store_true", help="Download all documents")
    group.add_argument("--id", dest="doc_id", help="Download a single document by id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint to list or download configured documents.

    Returns:
        Process exit code: 0 for success, 2 for selection/argument issues,
        3 if any download failed.

    """
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
