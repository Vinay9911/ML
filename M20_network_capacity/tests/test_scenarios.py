"""M20 scenario directions (docs/05 section 1.2, docs/03 M20).

S02  more crowd, so more user traffic.
S08  the towers on SS01 lose power, so availability falls - the card's named test.
S09  cameras offline: their backhaul load goes with them, and the coverage loss is flagged.
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

MODEL_ID = "M20"
TOWERS = ("NT01", "NT02", "NT03")


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
    output = model.scenario(ScenarioRequest(scenario_id="S09"))
    assert output.scenario_overrides == get_scenario("S09").overrides


# ------------------------------------------------------------------------- S02
def test_s02_raises_utilisation(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert_scenario_direction(
        baseline, scenario, "bandwidth_utilization", direction="up", entity_ids=TOWERS
    )


def test_s02_does_not_touch_availability(model, baseline) -> None:
    """More traffic is not an outage. Conflating the two would be a false alarm."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert _total(scenario, "network_availability") == pytest.approx(
        _total(baseline, "network_availability")
    )


# ------------------------------------------------------------------------- S08
def test_s08_lowers_availability(model, baseline) -> None:
    """docs/03 M20 test list: the S08 direction. NT02 hangs off SS01."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S08"))
    assert _total(scenario, "network_availability") < _total(baseline, "network_availability")


def test_s08_takes_down_only_the_towers_on_that_substation(model) -> None:
    """SS01 feeds NT02 alone; NT01 and NT03 are on SS02 and must stay up."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S08"))
    down = {r.entity_id for r in scenario.results if r.details.get("powered_down")}
    assert down == {"NT02"}, f"expected only NT02 powered down, got {down}"


def test_s08_raises_the_grid_outage_reason(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S08"))
    assert _flags(scenario, "GRID_OUTAGE") > _flags(baseline, "GRID_OUTAGE")


def test_the_power_link_is_read_from_the_world(model) -> None:
    """Which tower dies with which substation is world.yaml's business, not a literal here."""
    assert model._tower_down("NT02", {"substations_down": ["SS01"]}) is True
    assert model._tower_down("NT01", {"substations_down": ["SS01"]}) is False
    assert model._tower_down("NT01", {"substations_down": ["SS02"]}) is True
    assert model._tower_down("NT02", {}) is False


# ------------------------------------------------------------------------- S09
def test_s09_lowers_the_camera_backhaul(model, baseline) -> None:
    """docs/03 M20: S09 cuts camera load. Fewer streams is less traffic."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S09"))
    assert _total(scenario, "bandwidth_utilization") < _total(baseline, "bandwidth_utilization")


def test_s09_flags_the_loss_of_coverage(model) -> None:
    """Less traffic is not good news here; the blind spot is the point."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S09"))
    assert _flags(scenario, "CCTV_BLIND_SPOT") > 0
    assert any("cameras are offline" in w for w in scenario.warnings)


def test_s09_reports_how_much_load_it_removed(model) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S09"))
    removed = {r.details["camera_load_removed_mbps"] for r in scenario.results}
    assert max(removed) > 0


def test_s09_leaves_availability_alone(model, baseline) -> None:
    """A dead camera is not a dead tower. M02 and M10 own camera coverage."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S09"))
    assert _total(scenario, "network_availability") == pytest.approx(
        _total(baseline, "network_availability")
    )


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
        baseline, output, "bandwidth_utilization", direction="same", entity_ids=TOWERS
    )
