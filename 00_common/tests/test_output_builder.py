"""Risk-band derivation and the OutputBuilder."""

from __future__ import annotations

from datetime import datetime

import pytest

from twin_common.contracts import IST
from twin_common.contracts.enums import DataSource, RiskLevel, State, Status, UpstreamSource
from twin_common.errors import ContractError
from twin_common.output import (
    OutputBuilder,
    band_input_value,
    band_thresholds,
    risk_level_for_kpi,
    risk_level_from_table,
    risk_rank,
    worst_risk,
)

AS_OF = datetime(2027, 8, 2, 6, 0, tzinfo=IST)
TS = datetime(2027, 8, 2, 6, 45, tzinfo=IST)


def _builder(**kwargs) -> OutputBuilder:
    defaults = {
        "model_id": "M03",
        "model_name": "Hotspot / Crush Risk",
        "model_version": "0.1.0",
        "as_of": AS_OF,
        "data_source": DataSource.SYNTHETIC,
    }
    defaults.update(kwargs)
    return OutputBuilder(**defaults)


# --------------------------------------------------------------------- band tables
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "green"),
        (39.9, "green"),
        (40, "amber"),
        (59.9, "amber"),
        (60, "red"),
        (79.9, "red"),
        (80, "critical"),
        (100, "critical"),
    ],
)
def test_score_band_boundaries(value: float, expected: str) -> None:
    """docs/05 section 3 intervals are [lower, upper), top band closed at its maximum."""
    assert risk_level_from_table("score_0_100", value).value == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "green"),
        (1.99, "green"),
        (2.0, "amber"),
        (3.9, "amber"),
        (4.0, "red"),
        (4.9, "red"),
        (5.0, "critical"),
        (8.0, "critical"),
    ],
)
def test_density_band_boundaries(value: float, expected: str) -> None:
    assert risk_level_from_table("crowd_density_p_m2", value).value == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "green"),
        (79.9, "green"),
        (80, "amber"),
        (89.9, "amber"),
        (90, "red"),
        (99.9, "red"),
        (100, "critical"),
        (250, "critical"),
    ],
)
def test_utilization_band_allows_overflow(value: float, expected: str) -> None:
    assert risk_level_from_table("utilization_pct", value).value == expected


def test_value_above_the_top_band_is_still_critical() -> None:
    """utilization_pct critical is [100, 9999]; beyond that must not fall through."""
    assert risk_level_from_table("utilization_pct", 100_000).value == "critical"


def test_band_thresholds_exposes_the_table() -> None:
    table = band_thresholds("probability_pct")
    assert table["green"] == (0.0, 20.0)
    assert table["critical"] == (70.0, 100.0)


# ----------------------------------------------------- the complement_100 decision
def test_complement_100_transform() -> None:
    """Decision D1: availability risk is carried by the shortfall, 100 - value."""
    assert band_input_value("emergency_route_availability", 100.0) == pytest.approx(0.0)
    assert band_input_value("emergency_route_availability", 70.0) == pytest.approx(30.0)
    assert band_input_value("crush_risk_score", 70.0) == pytest.approx(70.0)


@pytest.mark.parametrize(
    ("availability", "expected"),
    [(100.0, "green"), (85.0, "green"), (70.0, "amber"), (50.0, "red"), (10.0, "critical")],
)
def test_availability_kpi_risk_uses_the_shortfall(availability: float, expected: str) -> None:
    level = risk_level_for_kpi("emergency_route_availability", availability)
    assert level is not None and level.value == expected


def test_full_availability_is_green_and_zero_is_critical() -> None:
    """Sanity: the direction must not be inverted."""
    assert risk_level_for_kpi("camera_availability", 100.0) == RiskLevel.GREEN
    assert risk_level_for_kpi("camera_availability", 0.0) == RiskLevel.CRITICAL


def test_utilization_kpi_risk_uses_the_value_directly() -> None:
    assert risk_level_for_kpi("zone_capacity_utilization", 95.0) == RiskLevel.RED
    assert risk_level_for_kpi("zone_capacity_utilization", 134.0) == RiskLevel.CRITICAL


def test_unbanded_kpi_returns_none() -> None:
    """An informational KPI has no band; the owning model decides."""
    assert risk_level_for_kpi("weather_arrival_multiplier", 0.8) is None
    assert risk_level_for_kpi("expected_footfall", 50_000.0) is None


# ----------------------------------------------------------------- risk helpers
def test_worst_risk_and_rank() -> None:
    assert worst_risk([RiskLevel.GREEN, RiskLevel.RED, RiskLevel.AMBER]) == RiskLevel.RED
    assert worst_risk([None, RiskLevel.CRITICAL]) == RiskLevel.CRITICAL
    assert worst_risk([None, None]) is None
    assert risk_rank(None) == 0
    assert risk_rank(RiskLevel.GREEN) == 1
    assert risk_rank(RiskLevel.CRITICAL) == 4


# ----------------------------------------------------------------- OutputBuilder
def test_unit_is_filled_from_the_registry() -> None:
    builder = _builder()
    record = builder.add(
        "zone", "Z01", "crush_risk_score", TS, 30.0, state=State.FORECAST, horizon_min=45
    )
    assert record.unit == "score_0_100"


def test_risk_level_is_derived_from_the_band() -> None:
    builder = _builder()
    record = builder.add(
        "zone", "Z01", "crowd_density", TS, 4.3, state=State.CURRENT, reason_codes=["DENSITY_HIGH"]
    )
    assert record.risk_level == RiskLevel.RED


def test_explicit_risk_level_overrides_the_band() -> None:
    """The M03 sustained-exposure rule forces the score up; the builder must allow it."""
    builder = _builder()
    record = builder.add(
        "zone",
        "Z01",
        "crush_risk_score",
        TS,
        30.0,
        state=State.CURRENT,
        risk_level="critical",
        reason_codes=["SUSTAINED_EXPOSURE"],
    )
    assert record.risk_level == RiskLevel.CRITICAL


def test_derived_amber_without_reason_codes_fails_loudly() -> None:
    """The contract makes reason codes mandatory at amber+, including derived ones."""
    builder = _builder()
    with pytest.raises(Exception, match="reason_codes are mandatory"):
        builder.add("zone", "Z01", "crowd_density", TS, 3.0, state=State.CURRENT)


def test_delta_is_computed_from_baseline_value() -> None:
    builder = _builder(scenario_id="S02")
    record = builder.add(
        "zone",
        "Z01",
        "crush_risk_score",
        TS,
        55.0,
        state=State.SCENARIO,
        horizon_min=45,
        baseline_value=40.0,
        reason_codes=["DENSITY_HIGH"],
    )
    assert record.delta == pytest.approx(15.0)


def test_apply_baseline_fills_deltas_on_matching_records() -> None:
    baseline = _builder()
    baseline.add(
        "zone",
        "Z01",
        "crush_risk_score",
        TS,
        40.0,
        state=State.FORECAST,
        horizon_min=45,
        reason_codes=["DENSITY_HIGH"],
    )
    baseline.add("zone", "Z02", "crush_risk_score", TS, 20.0, state=State.FORECAST, horizon_min=45)
    baseline_output = baseline.build()

    scenario = _builder(scenario_id="S02")
    scenario.add(
        "zone",
        "Z01",
        "crush_risk_score",
        TS,
        62.0,
        state=State.SCENARIO,
        horizon_min=45,
        reason_codes=["DENSITY_HIGH"],
    )
    scenario.add("zone", "Z02", "crush_risk_score", TS, 25.0, state=State.SCENARIO, horizon_min=45)
    output = scenario.apply_baseline(baseline_output).build()

    by_entity = {r.entity_id: r for r in output.results}
    assert by_entity["Z01"].baseline_value == pytest.approx(40.0)
    assert by_entity["Z01"].delta == pytest.approx(22.0)
    assert by_entity["Z02"].delta == pytest.approx(5.0)


def test_apply_baseline_warns_when_nothing_matches() -> None:
    baseline = _builder()
    baseline.add(
        "zone",
        "Z03",
        "crush_risk_score",
        TS,
        40.0,
        state=State.FORECAST,
        horizon_min=45,
        reason_codes=["DENSITY_HIGH"],
    )
    scenario = _builder(scenario_id="S02")
    scenario.add(
        "zone",
        "Z01",
        "crush_risk_score",
        TS,
        62.0,
        state=State.SCENARIO,
        horizon_min=45,
        reason_codes=["DENSITY_HIGH"],
    )
    output = scenario.apply_baseline(baseline.build()).build()
    assert output.status is Status.DEGRADED
    assert any("no baseline records matched" in w for w in output.warnings)


def test_warn_degrades_the_status() -> None:
    builder = _builder()
    builder.add("zone", "Z01", "crush_risk_score", TS, 10.0, state=State.CURRENT)
    builder.warn("cached weather used")
    output = builder.build()
    assert output.status is Status.DEGRADED
    assert output.warnings == ["cached weather used"]


def test_warn_without_degrading() -> None:
    builder = _builder()
    builder.add("zone", "Z01", "crush_risk_score", TS, 10.0, state=State.CURRENT)
    builder.warn("informational only", degrade=False)
    output = builder.build()
    assert output.status is Status.OK


def test_duplicate_warnings_are_collapsed() -> None:
    builder = _builder()
    builder.add("zone", "Z01", "crush_risk_score", TS, 10.0, state=State.CURRENT)
    builder.warn("same thing")
    builder.warn("same thing")
    assert builder.build().warnings == ["same thing"]


def test_inline_upstream_does_not_degrade() -> None:
    builder = _builder()
    builder.add("zone", "Z01", "crush_risk_score", TS, 10.0, state=State.CURRENT)
    builder.add_upstream("M01", UpstreamSource.INLINE, run_id="r1")
    output = builder.build()
    assert output.status is Status.OK
    assert output.upstream_sources() == {"M01": UpstreamSource.INLINE}


@pytest.mark.parametrize("source", ["sample", "stub"])
def test_fallback_upstream_degrades_with_a_warning(source: str) -> None:
    """docs/02 section 4: a fallback means degraded plus an explanation."""
    builder = _builder()
    builder.add("zone", "Z01", "crush_risk_score", TS, 10.0, state=State.CURRENT)
    builder.add_upstream("M01", source)
    output = builder.build()
    assert output.status is Status.DEGRADED
    assert any(source in w for w in output.warnings)


def test_mark_insensitive_wording_matches_the_docs() -> None:
    builder = _builder(scenario_id="S06")
    builder.add("zone", "Z01", "crush_risk_score", TS, 10.0, state=State.SCENARIO)
    builder.mark_insensitive("S06")
    output = builder.build()
    assert "insensitive to S06" in output.warnings


def test_is_synthetic_defaults_from_data_source() -> None:
    assert _builder().is_synthetic is True
    assert _builder(data_source=DataSource.REPLAY).is_synthetic is False


def test_build_wraps_validation_failures_in_contract_error() -> None:
    """A model must fail at its own boundary, not emit a broken payload."""
    builder = _builder()
    builder.add("zone", "Z01", "crush_risk_score", TS, 10.0, state=State.CURRENT)
    builder.is_synthetic = False  # inconsistent with data_source synthetic
    with pytest.raises(ContractError, match="docs/02 section 4"):
        builder.build()


def test_from_config_reads_the_standard_keys(dummy_model) -> None:
    builder = OutputBuilder.from_config(dummy_model.config, as_of=AS_OF)
    assert builder.model_id == "M21"
    assert builder.model_version == "0.1.0"
    assert builder.data_source is DataSource.SYNTHETIC


def test_scenario_overrides_are_echoed_in_the_output() -> None:
    builder = _builder(scenario_id="S02", scenario_overrides={"footfall_multiplier": 1.3})
    builder.add("zone", "Z01", "crush_risk_score", TS, 10.0, state=State.SCENARIO)
    output = builder.build()
    assert output.scenario_overrides == {"footfall_multiplier": 1.3}


def test_recommendation_round_trips() -> None:
    from twin_common.contracts import Recommendation

    builder = _builder()
    builder.add(
        "zone",
        "Z01",
        "crush_risk_score",
        TS,
        65.0,
        state=State.CURRENT,
        reason_codes=["DENSITY_HIGH"],
        recommendation=Recommendation(
            action="open holding area",
            resource_type="police",
            quantity=40,
            target_entity_id="Z01",
            rationale="density rising above the amber band",
        ),
    )
    output = builder.build()
    rec = output.results[0].recommendation
    assert rec is not None
    assert rec.requires_approval is True
    assert rec.resource_type == "police"
