"""M21 scenario directions (docs/05 section 1.2).

S03 heavy rain and S04 extreme heat are the two M21 must react to:

    S03  waterlogging up; arrivals down
    S04  heat_index up
"""

from __future__ import annotations

import pytest

from twin_common.contracts import PredictRequest, ScenarioRequest
from twin_common.scenarios import get_scenario
from twin_common.testing import assert_insensitive, assert_scenario_direction, assert_valid_output

MODEL_ID = "M21"


@pytest.fixture(scope="module")
def baseline(model):
    return model.predict(PredictRequest())


def test_every_supported_scenario_answers(model) -> None:
    for scenario_id in model.scenarios_supported:
        output = model.scenario(ScenarioRequest(scenario_id=scenario_id))
        assert_valid_output(output, model_id=MODEL_ID)
        assert output.scenario_id == scenario_id


def test_scenario_records_use_the_scenario_state(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S04"))
    assert all(record.state.value in ("scenario", "current") for record in output.results)


def test_scenario_echoes_its_overrides(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S03"))
    assert output.scenario_overrides == get_scenario("S03").overrides


# ------------------------------------------------------------------------- S03
def test_s03_raises_waterlogging(model, baseline) -> None:
    """docs/05 section 1.2: S03 waterlogging up."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S03"))
    assert_scenario_direction(baseline, scenario, "waterlogging_probability", direction="up")


def test_s03_raises_rainfall_to_the_override(model) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S03"))
    expected = float(get_scenario("S03").overrides["rain_mm_hr"])
    rain = [r for r in scenario.results if r.kpi == "rainfall_intensity"]
    assert rain, "no rainfall records"
    # The scenario window covers 06:00-09:00 and demo_now is 06:00, so every record is inside.
    assert max(r.value for r in rain) == pytest.approx(expected)


def test_s03_suppresses_arrivals(model, baseline) -> None:
    """docs/05 section 1.2: S03 arrivals down. M01 reads this multiplier."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S03"))
    assert_scenario_direction(baseline, scenario, "weather_arrival_multiplier", direction="down")


def test_s03_slows_traffic(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S03"))
    assert_scenario_direction(
        baseline, scenario, "weather_traffic_speed_multiplier", direction="down"
    )


def test_s03_floods_low_lying_zones_more(model) -> None:
    """The point of a per-zone drainage rate: the same rain does not flood every zone."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S03"))
    by_zone = {
        r.entity_id: r.value
        for r in scenario.results
        if r.kpi == "waterlogging_probability" and r.horizon_min == 0
    }
    # Z01 and Z06 are low-lying with poor drainage; Z08 is the well-drained hub.
    assert by_zone["Z01"] > by_zone["Z08"]
    assert by_zone["Z06"] > by_zone["Z08"]


def test_s03_raises_the_waterlogging_reason_code(model) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S03"))
    flagged = [
        r
        for r in scenario.results
        if r.kpi == "waterlogging_probability" and "WATERLOGGING" in r.reason_codes
    ]
    assert flagged, "heavy rain must raise WATERLOGGING somewhere"


# ------------------------------------------------------------------------- S04
def test_s04_raises_the_heat_index(model, baseline) -> None:
    """docs/05 section 1.2: S04 heat_index up."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S04"))
    assert_scenario_direction(baseline, scenario, "heat_index", direction="up")


def test_s04_raises_the_temperature_by_the_override(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S04"))
    offset = float(get_scenario("S04").overrides["temperature_offset_c"])
    base = {(r.entity_id, r.timestamp): r.value for r in baseline.results if r.kpi == "temperature"}
    for record in scenario.results:
        if record.kpi != "temperature":
            continue
        expected = base.get((record.entity_id, record.timestamp))
        if expected is not None:
            assert record.value == pytest.approx(expected + offset, abs=0.01)


def test_s04_raises_the_medical_multiplier(model, baseline) -> None:
    """M07 reads this: hotter weather means more presentations."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S04"))
    assert_scenario_direction(baseline, scenario, "weather_medical_multiplier", direction="up")


def test_s04_raises_the_heat_reason_code(model) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S04"))
    flagged = [r for r in scenario.results if "HEAT_STRESS" in r.reason_codes]
    assert flagged, "S04 must raise HEAT_STRESS"


def test_s04_does_not_change_rainfall(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S04"))
    assert_scenario_direction(baseline, scenario, "rainfall_intensity", direction="same")


# ---------------------------------------------------------------- unsupported
def test_unsupported_scenario_returns_a_degraded_baseline(model) -> None:
    """docs/05 section 1.2: a model not listed for a scenario returns baseline values.

    `handle_scenario` is the entry point `/scenario` uses; calling `scenario` directly
    skips the support check on purpose, because by then the check has already happened.
    """
    output = model.handle_scenario(ScenarioRequest(scenario_id="S08"))
    assert_insensitive(output, "S08")
    assert output.scenario_id == "S08"


def test_unsupported_scenario_values_equal_the_baseline(model, baseline) -> None:
    output = model.handle_scenario(ScenarioRequest(scenario_id="S08"))
    assert_scenario_direction(baseline, output, "heat_index", direction="same")


def test_baseline_scenario_matches_predict(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S01"))
    assert scenario.kpis_present() == baseline.kpis_present()
    assert_scenario_direction(baseline, scenario, "temperature", direction="same")


def test_deltas_are_filled_against_the_baseline(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S04", compare_to_baseline=True))
    heat = [r for r in output.results if r.kpi == "heat_index"]
    assert all(r.baseline_value is not None for r in heat)
    assert all(r.delta == pytest.approx(r.value - r.baseline_value) for r in heat)
    assert max(r.delta for r in heat) > 0
