"""Shared fixtures for the M21 tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

MODEL_ID = "M21"
FOLDER = Path(__file__).resolve().parent.parent


def _load_model_class() -> type:
    """Import ``src.model.Model`` from THIS folder.

    Two things make this fiddly, and both bite once more than one model folder exists:

    1. A model folder is a standalone service, not a package of the repository, so ``src``
       only resolves once the folder itself is on ``sys.path``.
    2. Every model folder has its own ``src.model``. Python caches the first one it imports
       under the name ``src``, so a later folder would silently receive the earlier folder's
       ``Model`` class - which surfaces as
       "config.yaml: model_id is 'M21' but Model.model_id is 'M01'".
       The cached ``src`` modules are therefore purged before the import. Class objects
       already handed out stay valid; only the module name is reused.

    The import is done here rather than at module level because an import sorter would hoist
    a top-level ``from src.model import Model`` above the ``sys.path`` line and break it.
    """
    for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
        del sys.modules[name]
    folder = str(FOLDER)
    if folder in sys.path:
        sys.path.remove(folder)
    sys.path.insert(0, folder)
    try:
        return importlib.import_module("src.model").Model
    finally:
        if folder in sys.path:
            sys.path.remove(folder)


@pytest.fixture(scope="session")
def model_class() -> type:
    return _load_model_class()


@pytest.fixture(scope="session")
def model(model_class: type):
    """The model, loaded once from this folder's config.yaml."""
    return model_class.from_folder(FOLDER)


@pytest.fixture(scope="session")
def app(model_class: type):
    from twin_common.api import create_app

    return create_app(model_class, root=FOLDER)
