"""M01 scenario directions (docs/05 section 1.2, docs/03 M01).

S02  total expected_footfall over the horizon is 1.2-1.4x S01
S03  arrivals fall during the rain window
S05  G02 goes to zero and its share moves to the other gates
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

MODEL_ID = "M01"
ZONES = ("Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07", "Z08")
GATES = ("G01", "G02", "G03", "G04")


@pytest.fixture(scope="module")
def baseline(model):
    return model.predict(PredictRequest())


def _total(output, kpi: str, entities=None) -> float:
    wanted = set(entities) if entities else None
    return sum(
        r.value
        for r in output.results
        if r.kpi == kpi and (wanted is None or r.entity_id in wanted)
    )


def test_every_supported_scenario_answers(model) -> None:
    for scenario_id in model.scenarios_supported:
        output = model.scenario(ScenarioRequest(scenario_id=scenario_id))
        assert_valid_output(output, model_id=MODEL_ID)
        assert output.scenario_id == scenario_id


def test_scenario_records_use_the_scenario_state(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert all(record.state.value in ("scenario", "current") for record in output.results)


def test_scenario_echoes_its_overrides(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert output.scenario_overrides == get_scenario("S02").overrides


# ------------------------------------------------------------------------- S02
def test_s02_raises_total_footfall_into_the_documented_band(model, baseline) -> None:
    """docs/03 M01: S02 total expected_footfall over the horizon is 1.2-1.4x S01."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    base_total, scenario_total = assert_scenario_direction(
        baseline,
        scenario,
        "expected_footfall",
        direction="up",
        entity_ids=ZONES,
        ratio_range=(1.2, 1.4),
    )
    assert base_total > 0 and scenario_total > base_total


def test_s02_raises_the_peak_too(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert_scenario_direction(baseline, scenario, "peak_footfall", direction="up", entity_ids=ZONES)


# ------------------------------------------------------------------------- S03
def test_s03_suppresses_arrivals(model, baseline) -> None:
    """docs/05 section 1.2: S03 arrivals down. demo_now 06:00 is inside the rain window."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S03"))
    assert_scenario_direction(
        baseline, scenario, "expected_footfall", direction="down", entity_ids=ZONES
    )


# ------------------------------------------------------------------------- S05
def test_s05_closes_g02(model) -> None:
    """docs/03 M01 and docs/04 section 8: S05 G02 forecast is zero."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S05"))
    g02 = [r for r in scenario.results if r.entity_id == "G02"]
    assert g02, "G02 must still be reported, at zero"
    assert all(r.value == pytest.approx(0.0, abs=1e-6) for r in g02)


def test_s05_keeps_the_other_gates_open(model) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S05"))
    others = _total(scenario, "expected_footfall", entities=("G01", "G03", "G04"))
    assert others > 0, "closing one gate must not close the rest"


def test_s05_total_arrivals_are_broadly_preserved(model, baseline) -> None:
    """The crowd does not vanish: a closed gate redistributes (docs/04 section 6)."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S05"))
    base = _total(baseline, "expected_footfall", entities=GATES)
    after = _total(scenario, "expected_footfall", entities=GATES)
    assert after > base * 0.5, f"arrivals collapsed from {base:,.0f} to {after:,.0f}"


# ---------------------------------------------------------------- unsupported
def test_s04_is_declared_insensitive(model) -> None:
    """The docs/03 M01 card lists S04, but nothing M01 reads responds to heat.

    The arrival multiplier in the docs/03 M21 card is a function of rain only, so claiming
    S04 support would mean returning an unchanged baseline while implying otherwise. The
    honest answer is the documented degraded response. See the model card.
    """
    output = model.handle_scenario(ScenarioRequest(scenario_id="S04"))
    assert_insensitive(output, "S04")


def test_unsupported_scenario_values_equal_the_baseline(model, baseline) -> None:
    output = model.handle_scenario(ScenarioRequest(scenario_id="S10"))
    assert_scenario_direction(
        baseline, output, "expected_footfall", direction="same", entity_ids=ZONES
    )


def test_baseline_scenario_matches_predict(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S01"))
    assert scenario.kpis_present() == baseline.kpis_present()


def test_deltas_are_filled_against_the_baseline(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S02", compare_to_baseline=True))
    matched = [r for r in output.results if r.baseline_value is not None]
    assert matched, "no record was matched to a baseline"
    assert all(r.delta == pytest.approx(r.value - r.baseline_value) for r in matched)
