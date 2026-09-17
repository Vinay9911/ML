"""Cached registry lookups used by the contract validators.

Keeping these behind small typed functions means the pydantic models never read yaml
directly, and a test can point ``TWIN_COMMON_CONFIG_DIR`` at a fixture directory.

Everything here is cached per process. :func:`clear_cache` resets both this module and the
underlying config cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .. import config as cfg
from ..errors import RegistryError


@dataclass(frozen=True, slots=True)
class KpiInfo:
    """One row of ``kpi_registry.yaml``."""

    kpi: str
    domain: str
    definition: str
    unit: str
    threshold: str
    priority: str
    owner: str
    direction: str
    band: str | None
    band_input: str
    bounds: tuple[float | None, float | None]
    extension: bool

    @property
    def higher_is_worse(self) -> bool:
        return self.direction == "higher_is_worse"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One row of ``model_registry.yaml``."""

    model_id: str
    folder: str
    name: str
    port: int
    engine: str
    phase: int
    upstream: tuple[str, ...]


# ------------------------------------------------------------------------------ KPIs
@lru_cache(maxsize=1)
def _kpi_table() -> dict[str, KpiInfo]:
    raw = cfg.kpi_registry()
    kpis = raw.get("kpis") or {}
    unit_bounds = raw.get("unit_bounds") or {}
    allowed_units = set(raw.get("allowed_units") or [])
    out: dict[str, KpiInfo] = {}
    for name, row in kpis.items():
        unit = row["unit"]
        if unit not in allowed_units:
            raise RegistryError(
                f"kpi_registry.yaml: KPI {name!r} uses unit {unit!r}, which is not in "
                f"allowed_units (docs/02 section 5)"
            )
        bounds_raw = row.get("bounds", unit_bounds.get(unit, [None, None]))
        if isinstance(bounds_raw, list):
            lo, hi = ([*bounds_raw, None, None])[:2]
        else:
            lo, hi = None, None
        out[name] = KpiInfo(
            kpi=name,
            domain=row["domain"],
            definition=row.get("definition", ""),
            unit=unit,
            threshold=str(row.get("threshold", "")),
            priority=row.get("priority", ""),
            owner=row["owner"],
            direction=row.get("direction", "higher_is_worse"),
            band=row.get("band"),
            band_input=row.get("band_input", "value"),
            bounds=(None if lo is None else float(lo), None if hi is None else float(hi)),
            extension=bool(row.get("extension", False)),
        )
    return out


def all_kpis() -> dict[str, KpiInfo]:
    """Every registered KPI, study and extension, keyed by name."""
    return _kpi_table()


def kpi_info(kpi: str) -> KpiInfo:
    """Look up one KPI or raise RegistryError."""
    try:
        return _kpi_table()[kpi]
    except KeyError as exc:
        raise RegistryError(
            f"unknown KPI {kpi!r}: add it to 00_common/config/kpi_registry.yaml "
            f"(and to docs/05 section 2 first)"
        ) from exc


def kpi_exists(kpi: str) -> bool:
    return kpi in _kpi_table()


def kpi_unit(kpi: str) -> str:
    return kpi_info(kpi).unit


def kpis_owned_by(model_id: str) -> list[str]:
    """KPI names whose ``owner`` is ``model_id``, in registry order."""
    return [name for name, info in _kpi_table().items() if info.owner == model_id]


def allowed_units() -> frozenset[str]:
    return frozenset(cfg.kpi_registry().get("allowed_units") or [])


# ---------------------------------------------------------------------------- models
@lru_cache(maxsize=1)
def _model_table() -> dict[str, ModelInfo]:
    raw = cfg.model_registry().get("models") or {}
    return {
        mid: ModelInfo(
            model_id=mid,
            folder=row["folder"],
            name=row.get("name", mid),
            port=int(row["port"]),
            engine=row["engine"],
            phase=int(row["phase"]),
            upstream=tuple(row.get("upstream") or []),
        )
        for mid, row in raw.items()
    }


def all_models() -> dict[str, ModelInfo]:
    return _model_table()


def model_info(model_id: str) -> ModelInfo:
    try:
        return _model_table()[model_id]
    except KeyError as exc:
        raise RegistryError(f"unknown model_id {model_id!r} (not in model_registry.yaml)") from exc


def model_exists(model_id: str) -> bool:
    return model_id in _model_table()


def upstream_of(model_id: str) -> tuple[str, ...]:
    return model_info(model_id).upstream


def topological_order(model_ids: list[str] | None = None) -> list[str]:
    """Dependency order: every model appears after all of its upstreams.

    Raises RegistryError when the graph has a cycle, which is what makes
    ``scripts/refresh_samples.py`` safe to run over the whole registry.
    """
    table = _model_table()
    wanted = list(table) if model_ids is None else list(model_ids)
    ordered: list[str] = []
    seen: set[str] = set()
    in_progress: set[str] = set()

    def visit(mid: str, chain: tuple[str, ...]) -> None:
        if mid in seen:
            return
        if mid in in_progress:
            raise RegistryError(f"cycle in the upstream graph: {' -> '.join((*chain, mid))}")
        in_progress.add(mid)
        for dep in table[mid].upstream if mid in table else ():
            if dep not in table:
                raise RegistryError(f"model {mid} declares unknown upstream {dep!r}")
            visit(dep, (*chain, mid))
        in_progress.discard(mid)
        seen.add(mid)
        ordered.append(mid)

    for mid in wanted:
        if mid not in table:
            raise RegistryError(f"unknown model_id {mid!r}")
        visit(mid, ())
    return ordered


def downstream_of(model_id: str) -> list[str]:
    """Models that list ``model_id`` among their upstreams."""
    return [mid for mid, info in _model_table().items() if model_id in info.upstream]


# --------------------------------------------------------------------- reason codes
@lru_cache(maxsize=1)
def _reason_codes() -> frozenset[str]:
    groups = cfg.reason_codes_config().get("groups") or {}
    return frozenset(code for codes in groups.values() for code in codes)


def all_reason_codes() -> frozenset[str]:
    return _reason_codes()


def reason_code_exists(code: str) -> bool:
    return code in _reason_codes()


def validate_reason_codes(codes: list[str]) -> list[str]:
    """Return the codes unchanged, or raise RegistryError listing the unknown ones."""
    unknown = [c for c in codes if c not in _reason_codes()]
    if unknown:
        raise RegistryError(
            f"unknown reason codes {unknown}: allowed codes are defined in docs/05 section 4 "
            f"and 00_common/config/reason_codes.yaml"
        )
    return codes


# ------------------------------------------------------------------- resource types
@lru_cache(maxsize=1)
def _resource_types() -> dict[str, dict[str, Any]]:
    return dict(cfg.resources_config().get("resource_types") or {})


def all_resource_types() -> dict[str, dict[str, Any]]:
    return _resource_types()


def resource_type_exists(resource_type: str) -> bool:
    return resource_type in _resource_types()


def resource_available(resource_type: str) -> float:
    """Baseline availability from resources.yaml, before scenario multipliers."""
    table = _resource_types()
    if resource_type not in table:
        raise RegistryError(f"unknown resource_type {resource_type!r} (docs/05 section 5)")
    return float(table[resource_type]["available"])


# ----------------------------------------------------------------------- band tables
@lru_cache(maxsize=1)
def _bands() -> dict[str, dict[str, list[float]]]:
    return dict(cfg.risk_bands().get("bands") or {})


def all_bands() -> dict[str, dict[str, list[float]]]:
    return _bands()


def band_table(name: str) -> dict[str, list[float]]:
    try:
        return _bands()[name]
    except KeyError as exc:
        raise RegistryError(
            f"unknown risk band table {name!r}: defined tables are {sorted(_bands())}"
        ) from exc


def band_exists(name: str) -> bool:
    return name in _bands()


def clear_cache() -> None:
    """Reset every cache in this module and in :mod:`twin_common.config`."""
    for fn in (_kpi_table, _model_table, _reason_codes, _resource_types, _bands):
        fn.cache_clear()
    cfg.clear_cache()
