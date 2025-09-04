"""Delete all records in a Pinecone namespace.

This module exposes an importable function and a small CLI wrapper to delete
every record within a specific namespace of the configured Pinecone index.

Why this exists:
- We often want to fully refresh a namespace by deleting then re-upserting
  records. This utility performs the delete step.

Requirements:
- Environment variable ``PINECONE_API_KEY`` must be set unless an API key is
  passed explicitly to the function.

Defaults:
- The default index host mirrors the constant in ``scripts/split_text.py``.
  You can override via the ``--host`` CLI argument or by calling the function
  with a different ``host`` value.

Examples:
Programmatic usage
    from scripts.ns_delete import delete_namespace_records

    delete_namespace_records(
        namespace="cribl",
        host="https://ragtag-db-f059e7z.svc.aped-4627-b74a.pinecone.io",
    )

CLI usage
    python -m scripts.ns_delete --namespace cribl

"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys

# Keep this in sync with scripts/split_text.py
PINECONE_HOST = os.getenv("PINECONE_HOST")


class NamespaceDeleteError(RuntimeError):
    """Raised when namespace deletion cannot proceed (bad input or env)."""


def delete_namespace_records(
    namespace: str,
    *,
    host: str | None = PINECONE_HOST,
    api_key: str | None = None,
) -> object:
    """Delete all records within a single Pinecone namespace.

    Args:
        namespace: The Pinecone namespace whose records should be removed.
        host: The Pinecone index host URL (not the control-plane URL).
        api_key: Optional API key override. If omitted, reads ``PINECONE_API_KEY``
            from the environment.

    Returns:
        The response object returned by the Pinecone client. Today this is
        typically an empty mapping (e.g., ``{}``) but callers should treat it
        as opaque and log/inspect for troubleshooting.

    Raises:
        RuntimeError: If the API key is missing or ``namespace`` is empty.

    """
    ns = (namespace or "").strip()
    if not ns:
        msg = "namespace must be a non-empty string"
        raise NamespaceDeleteError(msg)

    effective_api_key = api_key or os.getenv("PINECONE_API_KEY")
    if not effective_api_key:
        msg = "PINECONE_API_KEY is not set in the environment"
        raise NamespaceDeleteError(msg)

    pinecone_mod = importlib.import_module("pinecone")
    pc = pinecone_mod.Pinecone(api_key=effective_api_key)
    index = pc.Index(host=host)

    # Per Pinecone docs: delete all records by specifying delete_all=True and a namespace
    return index.delete(delete_all=True, namespace=ns)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for namespace deletion.

    Args:
        argv: Optional list of arguments (primarily for testing). Defaults to
            ``sys.argv[1:]`` when omitted.

    Returns:
        Parsed arguments namespace with fields: ``namespace`` and ``host``.

    """
    parser = argparse.ArgumentParser(description="Delete all records in a Pinecone namespace")
    parser.add_argument(
        "--namespace",
        required=True,
        help="Target Pinecone namespace to delete (use __default__ for the default namespace)",
    )
    parser.add_argument(
        "--host",
        default=PINECONE_HOST,
        help=(
            "Pinecone index host URL. Defaults to the project host used by scripts/split_text.py. "
            "Override if you created a new index in a different project/region."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint to delete all records for a namespace.

    Configures basic logging, parses CLI args, invokes deletion, and prints a
    compact JSON representation of the response for convenience.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(__name__)

    args = parse_args(argv)
    logger.info("deleting namespace: namespace=%s host=%s", args.namespace, args.host)
    try:
        resp = delete_namespace_records(args.namespace, host=args.host)
        # Pretty-print a JSON-ish response for easy inspection. Prefer JSON; fallback to repr.
        output: str
        try:
            output = json.dumps(resp, ensure_ascii=False, indent=2)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            output = repr(resp)
        sys.stdout.write(output + "\n")
        logger.info("delete request submitted successfully")
    except Exception as exc:  # surface a clear CLI error and exit non-zero
        logger.exception("delete failed")
        sys.stderr.write(f"ERROR: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
