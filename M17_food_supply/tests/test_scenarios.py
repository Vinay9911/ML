"""M17 scenario directions (docs/05 section 1.2, docs/03 M17).

S02  more crowd, so more meals.
S13  food supply disruption - the lead time stretches by 1.5, so replenishment lands later
     and coverage falls. This is the card's named test.
S06  road closure - "up lead time optional" on the card, but it needs M04 to say how much
     longer a closure makes a delivery take. M04 is Phase 7, so M17 reports itself
     insensitive rather than inventing a number.
"""

from __future__ import annotations

import pytest

from twin_common.contracts import PredictRequest, ScenarioRequest
from twin_common.scenarios import get_scenario
from twin_common.testing import (
    assert_insensitive,
    assert_scenario_direction,
    assert_valid_output,
)

MODEL_ID = "M17"
OUTLETS = ("FO01", "FO02")


@pytest.fixture(scope="module")
def baseline(model):
    return model.predict(PredictRequest())


def _total(output, kpi: str) -> float:
    return sum(r.value for r in output.results if r.kpi == kpi)


def _flags(output, code: str) -> int:
    return sum(1 for r in output.results if code in r.reason_codes)


def test_every_supported_scenario_answers(model) -> None:
    for scenario_id in model.scenarios_supported:
        output = model.scenario(ScenarioRequest(scenario_id=scenario_id))
        assert_valid_output(output, model_id=MODEL_ID)
        assert output.scenario_id == scenario_id


def test_baseline_scenario_matches_predict(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S01"))
    assert scenario.kpis_present() == baseline.kpis_present()


def test_scenario_echoes_its_overrides(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S13"))
    assert output.scenario_overrides == get_scenario("S13").overrides


# ------------------------------------------------------------------------- S02
def test_s02_raises_food_demand(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert_scenario_direction(
        baseline,
        scenario,
        "food_demand",
        direction="up",
        entity_ids=OUTLETS,
        ratio_range=(1.2, 1.4),
    )


def test_s02_needs_at_least_as_many_deliveries(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert _total(scenario, "delivery_requirement") >= _total(baseline, "delivery_requirement")


# ------------------------------------------------------------------------- S13
def test_s13_lowers_coverage(model, baseline) -> None:
    """docs/05 section 1.2 S13: "coverage down". The card's named test for this model.

    A longer lead time changes neither the stock on hand nor the rate it is eaten, so this
    only moves because stock is SIMULATED forward: the replenishment lands later, or beyond
    the horizon entirely, and the shelf runs lower in the meantime.
    """
    scenario = model.scenario(ScenarioRequest(scenario_id="S13"))
    assert _total(scenario, "food_stock_coverage") < _total(baseline, "food_stock_coverage")


def test_s13_raises_the_stockout_risk(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S13"))
    assert _flags(scenario, "STOCKOUT_RISK") > _flags(baseline, "STOCKOUT_RISK")


def test_s13_reports_the_stretched_lead_time(model) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S13"))
    lead_times = {r.details["lead_time_hours"] for r in scenario.results}
    assert len(lead_times) == 1
    stretched = lead_times.pop()
    assert stretched > model._lead_time_hours
    assert any("lead time" in w for w in scenario.warnings)


def test_s13_does_not_change_demand(model, baseline) -> None:
    """A supply disruption does not make people hungrier. Only the supply side moves."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S13"))
    assert _total(scenario, "food_demand") == pytest.approx(
        _total(baseline, "food_demand"), rel=1e-6
    )


def test_no_delivery_can_land_inside_a_horizon_shorter_than_the_lead_time(model) -> None:
    """Documents why the default horizon is 12 h rather than 6.

    With a 6 h lead time, nothing ordered inside a 6 h window can arrive inside it, so the
    model's own inventory simulation cannot produce any replenishment there - only the
    opening stock differs between scenarios. That is what makes a short horizon a poor place
    to read a lead-time scenario, and why the default is longer than the lead time.
    """
    short = 360
    scenario = model.scenario(ScenarioRequest(scenario_id="S13", horizon_min=short))
    arrived = [
        r.details["meals_arrived"]
        for r in scenario.results
        if r.kpi == "food_demand" and "meals_arrived" in r.details
    ]
    assert arrived, "no records to inspect"
    assert max(arrived) == 0.0, (
        "a delivery landed inside a window shorter than the lead time, which should be "
        "impossible - the lead time and the simulation have drifted apart"
    )
    assert model.default_horizon_min > model._lead_time_hours * 60


def test_the_s13_world_holds_less_stock_overall(model) -> None:
    """The structural effect of the disruption, measured on the world rather than one instant.

    A delivery ordered under a stretched lead time lands later, so the shelf drains further
    while it waits: mean stock across the month falls even though each order is larger.

    This is asserted on the MEAN and not at ``as_of``, because at any single instant the sign
    depends on where each outlet happens to sit in its delivery cycle. At the demo instant
    S13's FO01 has just taken a large delivery (4,560 meals against S01's 3,157) while FO02 is
    empty. Averaged over the month the disruption is unambiguous; at one timestamp it is not,
    and a test that pretended otherwise would be testing the phase of a sawtooth.
    """
    baseline = sum(float(series.mean()) for series in model._stock["S01"].values())
    disrupted = sum(float(series.mean()) for series in model._stock["S13"].values())
    assert disrupted < baseline, (
        f"the S13 world holds {disrupted:,.0f} mean meals against {baseline:,.0f} in S01; a "
        f"longer lead time should drain the shelf, not fill it"
    )


def test_deltas_are_filled_against_the_baseline(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S13", compare_to_baseline=True))
    matched = [r for r in output.results if r.baseline_value is not None]
    assert matched
    assert all(r.delta == pytest.approx(r.value - r.baseline_value) for r in matched)


# ---------------------------------------------------------------- unsupported
def test_s06_is_declared_insensitive(model) -> None:
    """The card makes the S06 lead-time increase optional and dependent on M04 (Phase 7)."""
    output = model.handle_scenario(ScenarioRequest(scenario_id="S06"))
    assert_insensitive(output, "S06")


def test_unsupported_scenario_values_equal_the_baseline(model, baseline) -> None:
    output = model.handle_scenario(ScenarioRequest(scenario_id="S10"))
    assert_scenario_direction(baseline, output, "food_demand", direction="same", entity_ids=OUTLETS)
