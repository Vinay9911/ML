"""Config loading and merging.

Three layers, lowest precedence first (docs/07 Phase 1):

1. shared yaml in ``00_common/config/`` - assumptions, world, scenarios, bands, registries
2. the model config.yaml - standard keys in docs/02 section 8 plus ``params``
3. environment overrides - ``TWIN_CFG_<dotted__path>=<yaml value>`` and named shortcuts

Shared files are cached, so repeated access inside a request costs nothing. Call
:func:`clear_cache` in tests that write temporary config files.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from ..errors import ConfigError
from ..logging import get_logger
from ..paths import config_dir, require_config_file

log = get_logger(__name__)

_ENV_PREFIX = "TWIN_CFG_"
_ENV_PATH_SEP = "__"

#: Environment shortcuts for the keys that are overridden most often.
_ENV_SHORTCUTS: dict[str, str] = {
    "TWIN_SEED": "seed",
    "TWIN_DEMO_NOW": "demo_now",
    "TWIN_DATA_SOURCE": "data_source",
    "TWIN_SCENARIO": "scenario_id",
    "TWIN_UPSTREAM_TIMEOUT_S": "upstream_timeout_s",
}

_MISSING = object()


# --------------------------------------------------------------------------- primitives
def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one yaml file into a dict. Raises ConfigError on anything unusable."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse yaml {p}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"top level of {p} must be a mapping, got {type(raw).__name__}")
    return raw


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``. Mappings merge; other types replace.

    Lists replace rather than concatenate, so a model can shorten a shared list.
    """
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(current, value)
        else:
            out[key] = value
    return out


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        return
    node = target
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def env_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Collect config overrides from the environment.

    ``TWIN_CFG_params__horizon_steps=12`` sets ``params.horizon_steps`` to the int 12.
    Values are parsed as yaml scalars, so ``true``, ``1.5``, ``[1,2]`` and ``null`` all work.

    Windows note: the OS stores environment variable names case-insensitively and
    ``os.environ`` reports them upper-cased, so ``TWIN_CFG_params__alpha`` arrives as
    ``TWIN_CFG_PARAMS__ALPHA``. Both the prefix match and the derived key are therefore
    case-normalised: the prefix is matched case-insensitively and the dotted path is
    lower-cased. Every config key in this project is lower-case snake_case, so this is
    lossless and makes the same variable behave identically on Windows and POSIX.
    """
    env = os.environ if environ is None else environ
    out: dict[str, Any] = {}
    for raw_key, raw_value in env.items():
        upper_key = raw_key.upper()
        if upper_key in _ENV_SHORTCUTS:
            dotted = _ENV_SHORTCUTS[upper_key]
        elif upper_key.startswith(_ENV_PREFIX):
            dotted = raw_key[len(_ENV_PREFIX) :].replace(_ENV_PATH_SEP, ".").lower()
        else:
            continue
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        _set_dotted(out, dotted, value)
        log.debug("config override from env: %s = %r", dotted, value)
    return out


# ------------------------------------------------------------------ shared config files
@cache
def _load_shared(name: str) -> dict[str, Any]:
    return load_yaml(require_config_file(name))


def assumptions() -> dict[str, Any]:
    """docs/04 section 6 - all values are expert-review placeholders."""
    return _load_shared("assumptions.yaml")


def world() -> dict[str, Any]:
    """docs/04 sections 2-3 - venue layout, entities and the time grid."""
    return _load_shared("world.yaml")


def scenarios_config() -> dict[str, Any]:
    """docs/05 section 1.1 - the 15 scenarios."""
    return _load_shared("scenarios.yaml")


def risk_bands() -> dict[str, Any]:
    """docs/05 section 3 - the four band tables."""
    return _load_shared("risk_bands.yaml")


def kpi_registry() -> dict[str, Any]:
    """docs/05 section 2 - 93 study KPIs plus 16 extensions."""
    return _load_shared("kpi_registry.yaml")


def model_registry() -> dict[str, Any]:
    """docs/02 section 13 - folders, ports, engines, phases and upstream lists."""
    return _load_shared("model_registry.yaml")


def reason_codes_config() -> dict[str, Any]:
    """docs/05 section 4 - the allowed reason codes, grouped by domain."""
    return _load_shared("reason_codes.yaml")


def resources_config() -> dict[str, Any]:
    """docs/05 section 5 - the 15 resource types."""
    return _load_shared("resources.yaml")


def zones_geojson_path() -> Path | None:
    """User-supplied real zone polygons, if any (docs/04 section 2)."""
    path = config_dir() / "zones.geojson"
    return path if path.is_file() else None


def clear_cache() -> None:
    """Drop cached shared config. Tests that write temporary config files must call this."""
    _load_shared.cache_clear()


# ------------------------------------------------------------------------ Config object
class Config(Mapping[str, Any]):
    """Read-only mapping over a merged config tree, with dotted access.

    ``cfg["params.alpha"]``, ``cfg.get("params.alpha", 0.15)`` and ``cfg.require("model_id")``
    all work. Plain ``cfg["params"]`` still returns the sub-dict, so mapping code is fine.
    """

    __slots__ = ("_data", "_source")

    def __init__(self, data: Mapping[str, Any], source: str = "<memory>") -> None:
        self._data = dict(data)
        self._source = source

    # -- Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._lookup(key)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Config(source={self._source!r}, keys={sorted(self._data)})"

    # -- access helpers ---------------------------------------------------
    def _lookup(self, dotted: str) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return _MISSING
        return node

    def get(self, key: str, default: Any = None) -> Any:
        value = self._lookup(key)
        return default if value is _MISSING else value

    def require(self, key: str) -> Any:
        """Return the value at ``key`` or raise ConfigError naming the source file."""
        value = self._lookup(key)
        if value is _MISSING:
            raise ConfigError(f"required config key {key!r} missing from {self._source}")
        return value

    def require_float(self, key: str) -> float:
        return float(self.require(key))

    def require_int(self, key: str) -> int:
        return int(self.require(key))

    def require_str(self, key: str) -> str:
        return str(self.require(key))

    def require_list(self, key: str) -> list[Any]:
        value = self.require(key)
        if not isinstance(value, list):
            raise ConfigError(f"config key {key!r} in {self._source} must be a list")
        return value

    def as_dict(self) -> dict[str, Any]:
        """A shallow copy for callers that want to mutate."""
        return dict(self._data)

    @property
    def source(self) -> str:
        return self._source


#: Keys every model config.yaml must define (docs/02 section 8).
REQUIRED_MODEL_KEYS: tuple[str, ...] = (
    "model_id",
    "model_name",
    "model_version",
    "port",
    "engine",
    "data_source",
    "seed",
    "demo_now",
    "horizons_min",
    "default_horizon_min",
    "upstream",
    "inputs",
    "kpis",
    "scenarios_supported",
    "upstream_timeout_s",
)


def load_model_config(
    path: str | Path,
    *,
    apply_env: bool = True,
    validate: bool = True,
) -> Config:
    """Load a model config.yaml, apply environment overrides and check the standard keys.

    The shared config files are NOT merged into the result: they are separate namespaces
    reached through :func:`assumptions`, :func:`world` and friends. Merging them would let a
    model silently redefine a shared assumption, which docs/02 section 8 forbids.

    ``upstream`` may legitimately be an empty list, so presence is checked with a sentinel
    rather than truthiness.
    """
    p = Path(path)
    data = load_yaml(p)
    if apply_env:
        data = deep_merge(data, env_overrides())
    if "params" not in data:
        data = deep_merge(data, {"params": {}})
    cfg = Config(data, source=str(p))
    if validate:
        missing = [k for k in REQUIRED_MODEL_KEYS if cfg.get(k, _MISSING) is _MISSING]
        if missing:
            raise ConfigError(f"{p}: missing required config keys {missing} (docs/02 section 8)")
    return cfg
