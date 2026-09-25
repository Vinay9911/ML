"""M18 scenario directions (docs/05 section 1.2, docs/03 M18).

S02  more crowd, so more waste.
S06/S12  the card wants collection delayed, which needs M04 to say by how much. M04 is
     Phase 7, so M18 reports itself insensitive. The delay override is wired and tested.
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

MODEL_ID = "M18"
BINS = ("WB01", "WB02", "WB03", "WB04", "WB05", "WB06", "WB07", "WB08")


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
    output = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert output.scenario_overrides == get_scenario("S02").overrides


# ------------------------------------------------------------------------- S02
def test_s02_raises_waste_generation(model, baseline) -> None:
    """docs/03 M18 test list: S02 up."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert_scenario_direction(
        baseline,
        scenario,
        "waste_generation",
        direction="up",
        entity_ids=BINS,
        ratio_range=(1.2, 1.4),
    )


def test_s02_needs_at_least_as_many_trips(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert _total(scenario, "waste_collection_trips") >= _total(baseline, "waste_collection_trips")


def test_s02_fills_bins_at_least_as_fast(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert _total(scenario, "waste_bin_fill_level") >= _total(baseline, "waste_bin_fill_level")


def test_deltas_are_filled_against_the_baseline(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S02", compare_to_baseline=True))
    matched = [r for r in output.results if r.baseline_value is not None]
    assert matched
    assert all(r.delta == pytest.approx(r.value - r.baseline_value) for r in matched)


# ------------------------------------- the collection-delay mechanism
def test_a_delayed_round_stretches_the_window(model) -> None:
    """The mechanism the docs/03 M18 card wants from S06/S12, driven directly."""
    normal = model.collection_window_min({})
    delayed = model.collection_window_min({"waste_collection_delay_multiplier": 2.0})
    assert delayed == pytest.approx(normal * 2.0)


def test_a_delayed_round_leaves_bins_fuller(model) -> None:
    """A bin emptied half as often holds more, which is the whole point of the scenario."""
    request = ScenarioRequest(scenario_id="S02")
    overrides = {"footfall_multiplier": 1.0, "waste_collection_delay_multiplier": 3.0}
    builder = model.new_builder(request, scenario_id="S02", overrides=overrides)
    model._compute(request, builder, state="scenario", overrides=overrides)
    delayed = builder.build()

    baseline = model.predict(PredictRequest())
    assert _total(delayed, "waste_bin_fill_level") >= _total(baseline, "waste_bin_fill_level")
    assert any("collection round" in w for w in delayed.warnings)


# ---------------------------------------------------------------- unsupported
@pytest.mark.parametrize("scenario_id", ["S06", "S12"])
def test_closures_are_declared_insensitive(model, scenario_id: str) -> None:
    """The card wants collection delayed under these, but only M04 can say by how much."""
    output = model.handle_scenario(ScenarioRequest(scenario_id=scenario_id))
    assert_insensitive(output, scenario_id)


def test_unsupported_scenario_values_equal_the_baseline(model, baseline) -> None:
    output = model.handle_scenario(ScenarioRequest(scenario_id="S10"))
    assert_scenario_direction(
        baseline, output, "waste_generation", direction="same", entity_ids=BINS
    )
