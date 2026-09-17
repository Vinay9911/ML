"""Risk-level derivation from ``risk_bands.yaml`` (docs/05 section 3).

No model may decide a risk level with an inline comparison: the four band tables are the
single source of truth, and CLAUDE.md forbids numeric thresholds in ``src/``.

Bands are read as half-open intervals ``[lower, upper)`` in the order green, amber, red,
critical, with the top band closed at its upper bound so the maximum value still bands.

"Down bad" percentage KPIs (availability, coverage, effectiveness) carry
``band_input: complement_100`` in the KPI registry, so the band is applied to the shortfall
``100 - value``. See plans/PHASE_1.md decision D1.
"""

from __future__ import annotations

from ..contracts import registry as reg
from ..contracts.enums import RiskLevel
from ..errors import RegistryError

#: The order bands are evaluated in, lowest risk first.
BAND_ORDER: tuple[RiskLevel, ...] = (
    RiskLevel.GREEN,
    RiskLevel.AMBER,
    RiskLevel.RED,
    RiskLevel.CRITICAL,
)


def band_input_value(kpi: str, value: float) -> float:
    """Transform a KPI value into the number the band table should be applied to."""
    info = reg.kpi_info(kpi)
    if info.band_input == "complement_100":
        return 100.0 - value
    if info.band_input == "value":
        return value
    raise RegistryError(
        f"KPI {kpi!r} has unsupported band_input {info.band_input!r}; "
        f"allowed values are 'value' and 'complement_100'"
    )


def risk_level_from_table(table_name: str, value: float) -> RiskLevel:
    """Band a raw number against a named table in ``risk_bands.yaml``."""
    table = reg.band_table(table_name)
    highest = BAND_ORDER[-1]
    for level in BAND_ORDER:
        bounds = table.get(level.value)
        if bounds is None:
            continue
        lower, upper = float(bounds[0]), float(bounds[1])
        if value < lower:
            # Below the lowest defined band: treat as the lowest risk level.
            return BAND_ORDER[0]
        is_top = level is highest
        if value < upper or (is_top and value <= upper):
            return level
    # Above every band, including the top one: the top band is the answer.
    return highest


def risk_level_for_kpi(kpi: str, value: float) -> RiskLevel | None:
    """Risk level for a KPI value, or None when the registry assigns no band.

    A ``None`` result means the owning model decides (or that the KPI is informational,
    like ``weather_arrival_multiplier``).
    """
    info = reg.kpi_info(kpi)
    if info.band is None:
        return None
    return risk_level_from_table(info.band, band_input_value(kpi, value))


def worst_risk(levels: list[RiskLevel | None]) -> RiskLevel | None:
    """The most severe non-null level in a list, or None when all are null."""
    present = [level for level in levels if level is not None]
    if not present:
        return None
    return max(present, key=BAND_ORDER.index)


def risk_rank(level: RiskLevel | None) -> int:
    """0 for None, 1 green .. 4 critical. Useful for sorting and floor rules (M25)."""
    return 0 if level is None else BAND_ORDER.index(level) + 1


def band_thresholds(table_name: str) -> dict[str, tuple[float, float]]:
    """The band table as {level: (lower, upper)}, for model cards and READMEs."""
    table = reg.band_table(table_name)
    return {
        level.value: (float(table[level.value][0]), float(table[level.value][1]))
        for level in BAND_ORDER
        if level.value in table
    }
