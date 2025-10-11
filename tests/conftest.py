"""Shared pytest fixtures for the ragtag_crew test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts import docs_config as docs_config_module

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Temporarily override CONFIG_DATA_PATH with a test-local file."""
    original_path = docs_config_module.CONFIG_DATA_PATH
    temp_path = tmp_path / "docs_config_test.yaml"
    monkeypatch.setattr(docs_config_module, "CONFIG_DATA_PATH", temp_path)
    try:
        yield temp_path
    finally:
        monkeypatch.setattr(docs_config_module, "CONFIG_DATA_PATH", original_path)
