"""Logging for twin_common. Library code never uses print (enforced by ruff T20).

Level comes from the TWIN_LOG_LEVEL environment variable (default INFO). A single stderr
handler is installed once on the `twin` root logger; child loggers propagate to it.
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT_NAME = "twin"
_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def _configure_root() -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger(_ROOT_NAME)
    if _CONFIGURED:
        return root
    level_name = os.environ.get("TWIN_LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True
    return root


_PACKAGE_NAME = "twin_common"


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the `twin` namespace.

    `get_logger(__name__)` inside the package yields e.g. `twin.io.tables`: the
    `twin_common.` prefix is dropped so log lines stay short.
    """
    _configure_root()
    suffix = name.removeprefix(_PACKAGE_NAME).removeprefix(_ROOT_NAME).lstrip("._")
    return logging.getLogger(f"{_ROOT_NAME}.{suffix}" if suffix else _ROOT_NAME)
