"""Tests for the ``scripts.ns_delete`` helpers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from scripts import ns_delete


def fail_unless(*, condition: bool, message: str) -> None:
    """Fail the current test if the condition is not satisfied."""
    if not condition:
        pytest.fail(message)


def test_delete_namespace_records_invokes_pinecone_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deletion helper should call the Pinecone client with expected arguments."""
    fake_state: dict[str, object] = {}

    class FakeIndex:
        def delete(self, *, delete_all: bool, namespace: str) -> dict[str, bool]:
            fake_state["delete_args"] = (delete_all, namespace)
            return {"deleted": True}

    class FakePinecone:
        def __init__(self, api_key: str) -> None:
            fake_state["api_key"] = api_key
            self.index = FakeIndex()

        def Index(self, host: str) -> FakeIndex:  # noqa: N802 - match real client API
            fake_state["host"] = host
            return self.index

    fake_module = SimpleNamespace(Pinecone=FakePinecone)

    def fake_import(name: str) -> SimpleNamespace:
        fail_unless(condition=name == "pinecone", message=name)
        return fake_module

    monkeypatch.setenv("PINECONE_API_KEY", "key")
    monkeypatch.setattr(importlib, "import_module", fake_import)

    response = ns_delete.delete_namespace_records("docs", host="https://host")
    fail_unless(condition=response == {"deleted": True}, message=str(response))
    fail_unless(condition=fake_state["api_key"] == "key", message=str(fake_state))
    fail_unless(condition=fake_state["host"] == "https://host", message=str(fake_state))
    fail_unless(condition=fake_state["delete_args"] == (True, "docs"), message=str(fake_state))


def test_delete_namespace_records_rejects_empty_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty namespace values should raise NamespaceDeleteError."""
    monkeypatch.setenv("PINECONE_API_KEY", "key")
    with pytest.raises(ns_delete.NamespaceDeleteError, match="namespace"):
        ns_delete.delete_namespace_records("", host="https://host")


def test_delete_namespace_records_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing API key should raise NamespaceDeleteError."""
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(ns_delete.NamespaceDeleteError, match="PINECONE_API_KEY"):
        ns_delete.delete_namespace_records("docs", host="https://host")


def test_delete_namespace_records_requires_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing host should raise NamespaceDeleteError."""
    monkeypatch.setenv("PINECONE_API_KEY", "key")
    with pytest.raises(ns_delete.NamespaceDeleteError, match="host"):
        ns_delete.delete_namespace_records("docs", host="")
