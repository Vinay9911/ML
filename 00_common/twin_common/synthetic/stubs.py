"""Contract-valid stub outputs for models that are not built yet (docs/02 section 7 step 5).

The dependency graph means M25 needs eleven upstreams and M24 thirteen. Waiting for all of
them before anything can run would make the build order rigid, so any model can be stood in
for by a stub: a real :class:`ModelOutput` carrying values read from the generated world,
marked ``source: stub`` and ``status: degraded`` so nobody mistakes it for a model result.

A stub is derived from the world rather than invented, so a downstream model sees numbers
that move consistently with the scenario it asked for. What it does not get is the model's
actual method - a stub for M03 reports density-derived risk, not the M03 rules.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

import pandas as pd

from ..config import assumptions, world
from ..contracts import registry as reg
from ..contracts.enums import DataSource, EntityType, State, Status
from ..contracts.models import ModelOutput, to_ist
from ..errors import TableError
from ..io.tables import load_table
from ..logging import get_logger
from ..output.builder import OutputBuilder

log = get_logger(__name__)

#: Version reported by every stub, so a consumer can tell them apart from real output.
STUB_VERSION = "0.0.0"
#: The warning every stub carries.
STUB_WARNING = "stub output generated from the synthetic world; not a model result"


@dataclass(frozen=True, slots=True)
class WorldSlice:
    """The few world tables a stub needs, loaded once per scenario."""

    scenario_id: str
    footfall: pd.DataFrame
    zones: pd.DataFrame
    weather: pd.DataFrame

    def at(self, as_of: datetime) -> pd.DataFrame:
        """Footfall rows at the step nearest ``as_of``."""
        if self.footfall.empty:
            return self.footfall
        stamps = pd.DatetimeIndex(self.footfall["timestamp"].unique())
        nearest = min(stamps, key=lambda value: abs(value - pd.Timestamp(as_of)))
        return self.footfall.loc[self.footfall["timestamp"] == nearest]

    def weather_at(self, as_of: datetime) -> pd.Series | None:
        if self.weather.empty:
            return None
        stamps = pd.DatetimeIndex(self.weather["timestamp"])
        nearest = min(stamps, key=lambda value: abs(value - pd.Timestamp(as_of)))
        return self.weather.loc[self.weather["timestamp"] == nearest].iloc[0]


@lru_cache(maxsize=8)
def _load_slice(scenario_id: str, base_dir: str | None) -> WorldSlice:
    """Load and cache the world tables stubs read. Falls back to empty frames."""
    kwargs: dict[str, Any] = {"scenario_id": scenario_id}
    if base_dir:
        kwargs["base_dir"] = base_dir

    def safe(name: str) -> pd.DataFrame:
        try:
            return load_table(name, **kwargs)
        except (TableError, FileNotFoundError) as exc:
            log.debug("stub source table %s unavailable for %s: %s", name, scenario_id, exc)
            return pd.DataFrame()

    return WorldSlice(
        scenario_id=scenario_id,
        footfall=safe("footfall_15min"),
        zones=safe("zones"),
        weather=safe("weather_hourly"),
    )


def clear_cache() -> None:
    """Drop the cached world slices. Tests that regenerate a world call this."""
    _load_slice.cache_clear()


# --------------------------------------------------------------------------- helpers
def _zone_ids(world_slice: WorldSlice, entity_ids: list[str] | None) -> list[str]:
    if not world_slice.zones.empty:
        available = world_slice.zones["zone_id"].tolist()
    else:
        available = list(world().get("zones", {}))
    if entity_ids:
        wanted = set(entity_ids)
        selected = [zone for zone in available if zone in wanted]
        if selected:
            return selected
    return available


def _population(world_slice: WorldSlice, as_of: datetime) -> dict[str, float]:
    rows = world_slice.at(as_of)
    if rows.empty:
        return {}
    return dict(zip(rows["zone_id"], rows["population"].astype(float), strict=True))


def _density(world_slice: WorldSlice, as_of: datetime) -> dict[str, float]:
    rows = world_slice.at(as_of)
    if rows.empty or "density_p_m2" not in rows.columns:
        return {}
    return dict(zip(rows["zone_id"], rows["density_p_m2"].astype(float), strict=True))


def _safe_capacity(world_slice: WorldSlice) -> dict[str, float]:
    if world_slice.zones.empty:
        return {}
    return dict(
        zip(
            world_slice.zones["zone_id"],
            world_slice.zones["safe_capacity"].astype(float),
            strict=True,
        )
    )


#: Fallback population when no world has been generated, so a stub still answers.
FALLBACK_POPULATION = 1000.0


def _reasons(kpi: str) -> list[str]:
    """Reason codes for a stub record.

    Any KPI with a band may land in amber or worse, and docs/02 section 4 makes reason codes
    mandatory there. SYNTHETIC_DATA is the honest one for a stub: the value came from the
    generated world, not from the model that owns the KPI.
    """
    return ["SYNTHETIC_DATA"] if reg.kpi_info(kpi).band is not None else []


def _clip(value: float, kpi: str) -> float:
    """Keep a stub value inside the KPI registry bounds."""
    low, high = reg.kpi_info(kpi).bounds
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


# ------------------------------------------------------------------ per-model builders
def _crowd_like(
    builder: OutputBuilder,
    world_slice: WorldSlice,
    as_of: datetime,
    zones: list[str],
    kpis: list[str],
    state: State,
    horizon_min: int,
) -> None:
    """KPIs that follow directly from population and density."""
    population = _population(world_slice, as_of)
    density = _density(world_slice, as_of)
    capacity = _safe_capacity(world_slice)
    for zone in zones:
        people = population.get(zone, FALLBACK_POPULATION)
        dense = density.get(zone, 0.5)
        for kpi in kpis:
            if kpi == "zone_population":
                value = people
            elif kpi == "crowd_density":
                value = dense
            elif kpi == "zone_capacity_utilization":
                value = 100.0 * people / max(capacity.get(zone, 1.0), 1e-9)
            elif kpi == "expected_footfall":
                # persons/hr from a 15-min population snapshot.
                value = people * 4.0
            elif kpi == "peak_footfall":
                value = people * 4.4
            else:
                continue
            builder.add(
                EntityType.ZONE,
                zone,
                kpi,
                as_of,
                _clip(value, kpi),
                zone_id=zone,
                horizon_min=horizon_min,
                state=state,
                confidence=0.3,
                reason_codes=_reasons(kpi),
                details={"stub": True, "derived_from": "footfall_15min"},
            )


def _weather_like(
    builder: OutputBuilder,
    world_slice: WorldSlice,
    as_of: datetime,
    kpis: list[str],
    state: State,
    horizon_min: int,
) -> None:
    row = world_slice.weather_at(as_of)
    defaults = {
        "temperature": 31.0,
        "heat_index": 34.0,
        "rainfall_intensity": 0.0,
        "waterlogging_probability": 5.0,
        "weather_arrival_multiplier": 1.0,
        "weather_medical_multiplier": 1.0,
        "weather_traffic_speed_multiplier": 1.0,
    }
    if row is not None:
        defaults["temperature"] = float(row["temperature_c"])
        defaults["heat_index"] = float(row["heat_index_c"])
        defaults["rainfall_intensity"] = float(row["rain_mm"])
        defaults["waterlogging_probability"] = min(100.0, float(row["rain_mm"]) * 2.0)
    for kpi in kpis:
        if kpi not in defaults:
            continue
        builder.add(
            EntityType.EVENT,
            "EVENT",
            kpi,
            as_of,
            _clip(defaults[kpi], kpi),
            horizon_min=horizon_min,
            state=state,
            confidence=0.3,
            reason_codes=_reasons(kpi),
            details={"stub": True, "derived_from": "weather_hourly"},
        )


def _score_like(
    builder: OutputBuilder,
    world_slice: WorldSlice,
    as_of: datetime,
    zones: list[str],
    kpis: list[str],
    state: State,
    horizon_min: int,
) -> None:
    """Risk-style KPIs, scaled off density so they move with the scenario."""
    density = _density(world_slice, as_of)
    bands = assumptions()["crowd"]["density_bands_p_m2"]
    critical = float(bands["critical"])
    for zone in zones:
        dense = density.get(zone, 0.5)
        fraction = min(1.0, dense / max(critical, 1e-9))
        for kpi in kpis:
            info = reg.kpi_info(kpi)
            if info.unit == "score_0_100":
                value = 100.0 * fraction
            elif info.unit == "%":
                value = (
                    100.0 * fraction
                    if info.direction == "higher_is_worse"
                    else 100.0 * (1.0 - fraction * 0.3)
                )
            else:
                continue
            value = _clip(value, kpi)
            builder.add(
                EntityType.ZONE,
                zone,
                kpi,
                as_of,
                value,
                zone_id=zone,
                horizon_min=horizon_min,
                state=state,
                confidence=0.3,
                reason_codes=_reasons(kpi) or ["SYNTHETIC_DATA"],
                details={"stub": True, "derived_from": "crowd_density"},
            )


def _generic(
    builder: OutputBuilder,
    world_slice: WorldSlice,
    as_of: datetime,
    zones: list[str],
    kpis: list[str],
    state: State,
    horizon_min: int,
) -> None:
    """Anything else: a population-scaled value inside the KPI bounds.

    Deliberately crude. A stub exists so a downstream model can run, not so it can be
    evaluated; every record says ``stub: true`` in its details.
    """
    population = _population(world_slice, as_of)
    for kpi in kpis:
        info = reg.kpi_info(kpi)
        entity_type = EntityType.ZONE
        targets = zones
        if info.domain in ("overall", "weather", "environment"):
            entity_type, targets = EntityType.EVENT, ["EVENT"]
        elif info.domain == "resources" and info.owner == "M24":
            entity_type, targets = EntityType.RESOURCE_POOL, ["police", "ambulance"]
        for entity in targets:
            people = population.get(entity, FALLBACK_POPULATION)
            scale = people / 1000.0
            if info.unit in ("%", "score_0_100"):
                value = min(60.0, 10.0 + scale)
            elif info.unit == "ratio":
                value = 1.0
            elif info.unit in ("min", "hours"):
                value = 10.0
            else:
                value = max(1.0, math.ceil(scale))
            builder.add(
                entity_type,
                entity,
                kpi,
                as_of,
                _clip(value, kpi),
                zone_id=entity if entity_type is EntityType.ZONE else None,
                horizon_min=horizon_min,
                state=state,
                confidence=0.2,
                reason_codes=_reasons(kpi),
                details={"stub": True},
            )


#: Which builder handles which KPI family. Checked in order.
_BUILDERS: tuple[tuple[Callable[..., None], frozenset[str]], ...] = (
    (
        _crowd_like,
        frozenset(
            {
                "zone_population",
                "crowd_density",
                "zone_capacity_utilization",
                "expected_footfall",
                "peak_footfall",
            }
        ),
    ),
    (
        _weather_like,
        frozenset(
            {
                "temperature",
                "heat_index",
                "rainfall_intensity",
                "waterlogging_probability",
                "weather_arrival_multiplier",
                "weather_medical_multiplier",
                "weather_traffic_speed_multiplier",
            }
        ),
    ),
    (
        _score_like,
        frozenset(
            {
                "crush_risk_score",
                "crowd_spillover_risk",
                "bottleneck_probability",
                "security_risk_score",
                "fire_risk_score",
                "overall_event_risk",
                "anomaly_score",
                "counterflow_risk",
            }
        ),
    ),
)


def build_stub(
    model_id: str,
    *,
    as_of: datetime,
    scenario_id: str = "S01",
    entity_ids: list[str] | None = None,
    kpis: list[str] | None = None,
    horizon_min: int = 0,
    base_dir: str | None = None,
) -> ModelOutput:
    """A contract-valid placeholder output for ``model_id``.

    The result always validates as a :class:`ModelOutput`, always reports
    ``status: degraded`` with an explanatory warning, and always marks each record
    ``details.stub = true``.
    """
    info = reg.model_info(model_id)
    owned = kpis or reg.kpis_owned_by(model_id)
    world_slice = _load_slice(scenario_id, base_dir)
    as_of = to_ist(as_of)
    state = State.SCENARIO if scenario_id != "S01" else State.CURRENT
    horizon = 0 if state is State.CURRENT else horizon_min

    builder = OutputBuilder(
        model_id=model_id,
        model_name=f"{info.name} (stub)",
        model_version=STUB_VERSION,
        as_of=as_of,
        data_source=DataSource.SYNTHETIC,
        scenario_id=scenario_id,
    )
    builder.warn(STUB_WARNING)

    zones = _zone_ids(world_slice, entity_ids)
    remaining = list(owned)
    for handler, handled in _BUILDERS:
        matching = [kpi for kpi in remaining if kpi in handled]
        if not matching:
            continue
        if handler is _weather_like:
            handler(builder, world_slice, as_of, matching, state, horizon)
        else:
            handler(builder, world_slice, as_of, zones, matching, state, horizon)
        remaining = [kpi for kpi in remaining if kpi not in handled]
    if remaining:
        _generic(builder, world_slice, as_of, zones, remaining, state, horizon)

    output = builder.build(status=Status.DEGRADED)
    log.debug(
        "stub %s for %s: %d records over %d KPIs",
        model_id,
        scenario_id,
        len(output.results),
        len(owned),
    )
    return output


def stub_provider(
    model_id: str,
    *,
    as_of: datetime,
    scenario_id: str,
    entity_ids: list[str] | None = None,
) -> ModelOutput:
    """Adapter matching :class:`twin_common.upstream.StubProvider`."""
    return build_stub(model_id, as_of=as_of, scenario_id=scenario_id, entity_ids=entity_ids)


def install() -> None:
    """Register :func:`stub_provider` as the upstream fallback.

    Called by ``twin_common.api.create_app`` so every model service can resolve an upstream
    that has not been built yet (docs/02 section 7 step 5).
    """
    from ..upstream import set_stub_provider

    set_stub_provider(stub_provider)


def build_all_stubs(
    *,
    as_of: datetime,
    scenario_id: str = "S01",
    model_ids: list[str] | None = None,
) -> dict[str, ModelOutput]:
    """A stub for every registered model, in topological order."""
    wanted = model_ids or list(reg.all_models())
    return {
        model_id: build_stub(model_id, as_of=as_of, scenario_id=scenario_id)
        for model_id in reg.topological_order(wanted)
    }


def summarise(outputs: Mapping[str, ModelOutput]) -> str:
    """One line per stub, for the refresh_samples CLI."""
    return "\n".join(
        f"  {model_id}: {len(output.results):4d} records, {len(output.kpis_present()):2d} KPIs"
        for model_id, output in sorted(outputs.items())
    )
