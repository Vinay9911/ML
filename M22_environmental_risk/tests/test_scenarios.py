"""M22 scenario directions (docs/05 section 1.2, docs/03 M22).

S02  more crowd, so more noise and more electricity, hence more CO2.
S08  the grid fails and generators run, so diesel CO2 rises.
S15  a fire: the card says PM2.5 rises near Z03. D13 is venue-level, so only the venue
     average can move - documented rather than faked.
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

MODEL_ID = "M22"
ZONES = ("Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07", "Z08")


@pytest.fixture(scope="module")
def baseline(model):
    return model.predict(PredictRequest())


def _total(output, kpi: str) -> float:
    return sum(r.value for r in output.results if r.kpi == kpi)


def test_every_supported_scenario_answers(model) -> None:
    for scenario_id in model.scenarios_supported:
        output = model.scenario(ScenarioRequest(scenario_id=scenario_id))
        assert_valid_output(output, model_id=MODEL_ID)
        assert output.scenario_id == scenario_id


def test_baseline_scenario_matches_predict(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S01"))
    assert scenario.kpis_present() == baseline.kpis_present()


def test_scenario_echoes_its_overrides(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S15"))
    assert output.scenario_overrides == get_scenario("S15").overrides


# ------------------------------------------------------------------------- S02
def test_s02_raises_noise(model, baseline) -> None:
    """docs/05 section 1.2: S02 up noise. A denser crowd is a louder one."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert _total(scenario, "noise_level") > _total(baseline, "noise_level")


def test_s02_raises_co2(model, baseline) -> None:
    """docs/05 section 1.2: S02 up CO2, through the electricity term."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert _total(scenario, "co2_emissions") > _total(baseline, "co2_emissions")


def test_s02_noise_rises_less_than_the_crowd(model, baseline) -> None:
    """Decibels are logarithmic: 30 percent more crowd is nowhere near 30 percent more dB."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    ratio = _total(scenario, "noise_level") / _total(baseline, "noise_level")
    assert 1.0 < ratio < 1.1


# ------------------------------------------------------------------------- S08
def test_s08_raises_co2(model, baseline) -> None:
    """docs/03 M22: S08 up CO2, because the generators run on diesel."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S08"))
    assert _total(scenario, "co2_emissions") > _total(baseline, "co2_emissions")


def test_s08_reports_generator_litres(model) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S08"))
    litres = [
        r.details["generator_litres_per_hour"] for r in scenario.results if r.kpi == "co2_emissions"
    ]
    assert litres and max(litres) > 0, "S08 burns diesel; that must be visible"


def test_s08_does_not_change_the_noise(model, baseline) -> None:
    """A power cut does not change how many people are standing where."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S08"))
    assert _total(scenario, "noise_level") == pytest.approx(
        _total(baseline, "noise_level"), rel=1e-6
    )


# ------------------------------------------------------------------------- S15
def test_s15_answers_and_flags_the_fire(model) -> None:
    """The card wants PM2.5 up near Z03, which this data cannot resolve - so say so.

    D13 is a single venue-level series, so there is no "near Z03" to move. The scenario is
    answered, the fire is flagged in the reason codes and a warning, and the model card
    records the limitation. Inventing a per-zone plume would be fabricating spatial detail
    the input does not contain.
    """
    scenario = model.scenario(ScenarioRequest(scenario_id="S15"))
    assert_valid_output(scenario, model_id=MODEL_ID)
    assert any("FIRE_RISK" in r.reason_codes for r in scenario.results)
    assert any("fire in" in w for w in scenario.warnings)


def test_deltas_are_filled_against_the_baseline(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S02", compare_to_baseline=True))
    matched = [r for r in output.results if r.baseline_value is not None]
    assert matched
    assert all(r.delta == pytest.approx(r.value - r.baseline_value) for r in matched)


# ---------------------------------------------------------------- unsupported
def test_an_unsupported_scenario_is_declared_insensitive(model) -> None:
    output = model.handle_scenario(ScenarioRequest(scenario_id="S05"))
    assert_insensitive(output, "S05")


def test_unsupported_scenario_values_equal_the_baseline(model, baseline) -> None:
    output = model.handle_scenario(ScenarioRequest(scenario_id="S10"))
    assert_scenario_direction(baseline, output, "noise_level", direction="same", entity_ids=ZONES)
