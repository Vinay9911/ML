"""Shared fixtures for the M21 tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from src.model import Model

MODEL_ID = "M21"
FOLDER = Path(__file__).resolve().parent.parent


def _load_model_class() -> type:
    """Import ``src.model.Model`` from this folder.

    A model folder is a standalone service, not a package of the repository, so ``src`` only
    resolves once the folder itself is on ``sys.path``. The import is done here rather than
    at module level because an import-sorter will hoist a top-level ``from src.model import
    Model`` above the ``sys.path`` line and break it.
    """
    if str(FOLDER) not in sys.path:
        sys.path.insert(0, str(FOLDER))
    return importlib.import_module("src.model").Model


@pytest.fixture(scope="session")
def model_class() -> type:
    return _load_model_class()


@pytest.fixture(scope="session")
def model(model_class: type) -> Model:
    """The model, loaded once from this folder's config.yaml."""
    return model_class.from_folder(FOLDER)


@pytest.fixture(scope="session")
def app(model_class: type):
    from twin_common.api import create_app

    return create_app(model_class, root=FOLDER)
