"""M19 scenario directions (docs/05 section 1.2, docs/03 M19).

S02  more crowd, so more load.
S04  extreme heat raises the cooling term - the card's "S04 up load".
S08  substation SS01 down: affected assets on backup, a duration countdown, GRID_OUTAGE.
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

MODEL_ID = "M19"
ASSETS = ("SS01", "SS02")


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
    output = model.scenario(ScenarioRequest(scenario_id="S08"))
    assert output.scenario_overrides == get_scenario("S08").overrides


# ------------------------------------------------------------------------- S02
def test_s02_raises_electricity_demand(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert_scenario_direction(
        baseline, scenario, "electricity_demand", direction="up", entity_ids=ASSETS
    )


# ------------------------------------------------------------------------- S04
def test_s04_raises_electricity_demand(model, baseline) -> None:
    """docs/03 M19: "S04 up load". Heat does not add people, it adds cooling."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S04"))
    assert_scenario_direction(
        baseline, scenario, "electricity_demand", direction="up", entity_ids=ASSETS
    )


def test_s04_needs_at_least_as_many_generators(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S04"))
    assert _total(scenario, "backup_generator_requirement") >= _total(
        baseline, "backup_generator_requirement"
    )


# ------------------------------------------------------------------------- S08
def test_s08_raises_the_grid_outage_reason(model, baseline) -> None:
    """docs/03 M19: S08 puts the affected assets on backup with reason GRID_OUTAGE."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S08"))
    assert _flags(scenario, "GRID_OUTAGE") > _flags(baseline, "GRID_OUTAGE")


def test_s08_marks_the_affected_asset_on_backup(model) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S08"))
    on_backup = {r.entity_id for r in scenario.results if r.details.get("on_backup")}
    assert "SS01" in on_backup, "the downed substation must be shown on backup"


def test_s08_leaves_the_healthy_substation_alone(model) -> None:
    """S08 downs SS01 only. Flagging SS02 too would be a false alarm."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S08"))
    ss02 = [r for r in scenario.results if r.entity_id == "SS02"]
    assert ss02
    assert not any(r.details.get("on_backup") for r in ss02)


def test_s08_backup_duration_is_finite(model) -> None:
    """docs/03 M19 test list: "S08 -> backup duration finite"."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S08"))
    durations = [r.value for r in scenario.results if r.kpi == "generator_backup_duration"]
    assert durations
    ceiling = float(model.require_param("max_backup_hours"))
    assert all(0.0 < d <= ceiling for d in durations)


def test_s08_recommends_confirming_backup(model) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S08"))
    recommended = [r for r in scenario.results if r.recommendation is not None]
    assert recommended
    for record in recommended:
        assert record.recommendation.requires_approval is True
        assert record.recommendation.resource_type == "backup_generator"


def test_the_outage_path_without_a_world(model) -> None:
    """S08 has a generated world, so the hand-applied override path needs its own test."""
    assert model._on_backup("SS01", "S99", {"substations_down": ["SS01"]}) is True
    assert model._on_backup("SS02", "S99", {"substations_down": ["SS01"]}) is False
    assert model._on_backup("SS01", "S99", {}) is False


def test_deltas_are_filled_against_the_baseline(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S08", compare_to_baseline=True))
    matched = [r for r in output.results if r.baseline_value is not None]
    assert matched
    assert all(r.delta == pytest.approx(r.value - r.baseline_value) for r in matched)


# ---------------------------------------------------------------- unsupported
def test_an_unsupported_scenario_is_declared_insensitive(model) -> None:
    output = model.handle_scenario(ScenarioRequest(scenario_id="S05"))
    assert_insensitive(output, "S05")


def test_unsupported_scenario_values_equal_the_baseline(model, baseline) -> None:
    output = model.handle_scenario(ScenarioRequest(scenario_id="S10"))
    assert_scenario_direction(
        baseline, output, "electricity_demand", direction="same", entity_ids=ASSETS
    )
