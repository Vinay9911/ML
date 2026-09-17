"""Scenario loading, override merging and generic adjustments (docs/05 section 1).

Two responsibilities:

- **Resolution**: turn a ``scenario_id`` plus request-level ``overrides`` into one merged
  override dict. Scenarios may be combined, e.g. S02 + S04 for the demo, and overrides
  merge left to right.
- **Generic adjustment**: :func:`apply_adjustments` applies the multiplier-type overrides to
  a series or scalar for models whose synthetic world was not regenerated for that
  scenario. It covers the mechanical cases only; anything structural (closed links,
  evacuation start) belongs to the model that understands it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

from ..config import deep_merge, scenarios_config
from ..errors import UnknownScenarioError
from ..logging import get_logger

log = get_logger(__name__)

BASELINE_SCENARIO = "S01"


@dataclass(frozen=True, slots=True)
class Scenario:
    """One row of ``scenarios.yaml``."""

    scenario_id: str
    name: str
    overrides: dict[str, Any]
    affects: tuple[str, ...]
    assertion: str

    @property
    def is_baseline(self) -> bool:
        return self.scenario_id == BASELINE_SCENARIO


def all_scenarios() -> dict[str, Scenario]:
    """Every scenario defined in ``scenarios.yaml``, keyed by ID."""
    raw = scenarios_config().get("scenarios") or {}
    return {
        sid: Scenario(
            scenario_id=sid,
            name=row.get("name", sid),
            overrides=dict(row.get("overrides") or {}),
            affects=tuple(row.get("affects") or []),
            assertion=str(row.get("assertion", "")),
        )
        for sid, row in raw.items()
    }


def get_scenario(scenario_id: str) -> Scenario:
    """Look up one scenario or raise UnknownScenarioError (mapped to HTTP 404)."""
    scenarios = all_scenarios()
    try:
        return scenarios[scenario_id]
    except KeyError as exc:
        raise UnknownScenarioError(
            f"unknown scenario_id {scenario_id!r}; defined scenarios are {sorted(scenarios)}"
        ) from exc


def scenario_exists(scenario_id: str) -> bool:
    return scenario_id in all_scenarios()


def adjustment_keys() -> dict[str, list[str]]:
    """The override keys grouped by how :func:`apply_adjustments` treats them."""
    return {k: list(v) for k, v in (scenarios_config().get("adjustment_keys") or {}).items()}


def resolve_overrides(
    scenario_ids: str | Iterable[str],
    request_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge the overrides of one or more scenarios, then the request overrides on top.

    Later entries win, so ``resolve_overrides(["S02", "S04"], {...})`` gives the demo
    combination with the request as the final say.
    """
    ids = [scenario_ids] if isinstance(scenario_ids, str) else list(scenario_ids)
    merged: dict[str, Any] = {}
    for sid in ids:
        merged = deep_merge(merged, get_scenario(sid).overrides)
    if request_overrides:
        merged = deep_merge(merged, dict(request_overrides))
    return merged


def affects_model(scenario_id: str, model_id: str) -> bool:
    """Whether docs/05 section 1.2 lists this model as one that must change."""
    return model_id in get_scenario(scenario_id).affects


def model_supports(scenario_id: str, supported: Iterable[str]) -> bool:
    """Whether the model config lists this scenario in ``scenarios_supported``."""
    return scenario_id in set(supported)


# --------------------------------------------------------------- generic adjustments
def _parse_clock(value: str) -> time:
    hour, _, minute = value.partition(":")
    return time(int(hour), int(minute or 0))


def in_window(ts: datetime, window: list[str] | tuple[str, str] | None) -> bool:
    """Whether a timestamp falls inside an ``["HH:MM", "HH:MM"]`` scenario window.

    A window that wraps past midnight (start > end) is handled as a union of the two ends.
    Returns True when no window is given, so an override with no window applies all day.
    """
    if not window:
        return True
    start, end = _parse_clock(str(window[0])), _parse_clock(str(window[1]))
    current = ts.timetz().replace(tzinfo=None)
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


@dataclass(frozen=True, slots=True)
class Adjustment:
    """The mechanical effect of an override set on one quantity."""

    multiplier: float = 1.0
    offset: float = 0.0
    absolute: float | None = None

    def apply(self, value: float) -> float:
        if self.absolute is not None:
            return self.absolute
        return value * self.multiplier + self.offset


#: Which override key adjusts which named quantity, and how.
#: `quantity` names are what models pass to `apply_adjustments`.
_QUANTITY_RULES: dict[str, dict[str, str]] = {
    "footfall": {"footfall_multiplier": "multiplicative"},
    "medical_rate": {"medical_rate_multiplier": "multiplicative"},
    "food_lead_time": {"food_lead_time_multiplier": "multiplicative"},
    "temperature_c": {"temperature_offset_c": "additive"},
    "rain_mm_hr": {"rain_mm_hr": "absolute"},
    "humidity_pct": {"humidity_pct_min": "floor"},
    "camera_outage_share": {"camera_outage_share": "absolute"},
}


def adjustment_for(quantity: str, overrides: Mapping[str, Any]) -> Adjustment:
    """Build the :class:`Adjustment` an override set implies for a named quantity.

    Unknown quantities return the identity adjustment, which is what makes a model
    "insensitive" to a scenario rather than wrong.
    """
    rules = _QUANTITY_RULES.get(quantity, {})
    multiplier, offset, absolute = 1.0, 0.0, None
    for key, kind in rules.items():
        if key not in overrides:
            continue
        raw = overrides[key]
        if kind == "multiplicative":
            multiplier *= float(raw)
        elif kind == "additive":
            offset += float(raw)
        elif kind in ("absolute", "floor"):
            absolute = float(raw)
    return Adjustment(multiplier=multiplier, offset=offset, absolute=absolute)


def apply_adjustments(
    value: float,
    quantity: str,
    overrides: Mapping[str, Any],
    *,
    timestamp: datetime | None = None,
    floor: bool = False,
) -> float:
    """Apply the mechanical part of a scenario to one number.

    Args:
        value: the baseline number.
        quantity: which quantity this is, e.g. ``footfall``, ``temperature_c``.
        overrides: merged overrides from :func:`resolve_overrides`.
        timestamp: when given, the scenario ``window`` is honoured and values outside it
            are returned unchanged.
        floor: for ``humidity_pct_min``-style overrides, take the max instead of replacing.

    Returns:
        The adjusted number. Identical to ``value`` when nothing applies.
    """
    if timestamp is not None and not in_window(timestamp, overrides.get("window")):
        return value
    adjustment = adjustment_for(quantity, overrides)
    if floor and adjustment.absolute is not None:
        return max(value, adjustment.absolute)
    return adjustment.apply(value)


def closed_entities(overrides: Mapping[str, Any]) -> dict[str, set[str]]:
    """The structural closures in an override set, grouped by kind.

    Returns a dict with the keys ``gates``, ``links``, ``exits`` and ``substations`` so a
    model can check membership without knowing the override spelling.
    """
    return {
        "gates": set(overrides.get("closed_gates") or []),
        "links": set(overrides.get("closed_links") or []),
        "exits": set(overrides.get("blocked_exits") or []),
        "substations": set(overrides.get("substations_down") or []),
    }


def resource_multiplier(overrides: Mapping[str, Any], resource_type: str) -> float:
    """Availability multiplier for one resource type (S10, S11)."""
    mapping = overrides.get("resource_availability_multiplier") or {}
    return float(mapping.get(resource_type, 1.0))
