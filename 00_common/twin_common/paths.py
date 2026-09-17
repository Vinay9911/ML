"""Filesystem locations for config, data and schemas.

One resolution rule keeps twin_common working both in-repo and vendored into a model
folder for delivery (docs/02 section 10):

    common_root = Path(twin_common.__file__).parent.parent

- in repo:   00_common/twin_common/..      -> 00_common/           (config/, data/, schemas/)
- vendored:  Mxx/vendor/twin_common/..     -> Mxx/vendor/          (packaging copies config/ there)

Every location can be overridden by an environment variable so a model service can point at
a different config set without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import ConfigError

_ENV_COMMON_ROOT = "TWIN_COMMON_ROOT"
_ENV_CONFIG_DIR = "TWIN_COMMON_CONFIG_DIR"
_ENV_DATA_DIR = "TWIN_COMMON_DATA_DIR"
_ENV_OFFLINE = "TWIN_OFFLINE"


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else None


def common_root() -> Path:
    """Directory that holds config/, data/ and schemas/."""
    override = _env_path(_ENV_COMMON_ROOT)
    if override is not None:
        return override
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    """Directory of the shared yaml config files."""
    override = _env_path(_ENV_CONFIG_DIR)
    if override is not None:
        return override
    return common_root() / "config"


def data_dir() -> Path:
    override = _env_path(_ENV_DATA_DIR)
    if override is not None:
        return override
    return common_root() / "data"


def schemas_dir() -> Path:
    return common_root() / "schemas"


def world_dir(scenario_id: str) -> Path:
    """Generated synthetic world for one scenario (gitignored, regenerable)."""
    return data_dir() / "world" / scenario_id


def real_cache_dir() -> Path:
    """Cached downloads: weather, air quality, OSM graph, AI4I, model weights."""
    return data_dir() / "real_cache"


def bundled_dir() -> Path:
    """Small offline fallbacks committed to git so a fresh clone works offline."""
    return data_dir() / "bundled"


def reports_dir() -> Path:
    return common_root() / "reports"


def is_offline() -> bool:
    """True when TWIN_OFFLINE is set to a truthy value. No network calls are allowed then."""
    return os.environ.get(_ENV_OFFLINE, "").strip().lower() in {"1", "true", "yes", "on"}


def require_config_file(name: str) -> Path:
    """Return the path of a shared config file, raising ConfigError when it is absent."""
    path = config_dir() / name
    if not path.is_file():
        raise ConfigError(f"shared config file not found: {path}")
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
