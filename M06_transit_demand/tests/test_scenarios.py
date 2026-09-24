"""M06 scenario directions (docs/05 section 1.2, docs/03 M06).

S02  more crowd, so more riders: demand, vehicles and wait all rise.
S06  road closure - no shuttle route is closed and M04 does not exist yet, so insensitive.
S12  bridge closure - likewise.

The round-trip override the docs/03 M06 card asks for ("S06/S12 up round trip time via M04
details if present") is exercised directly, since M04 is a Phase 7 model.
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

MODEL_ID = "M06"
ROUTES = ("SH1", "SH2")


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


def test_scenario_records_use_the_scenario_state(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert all(record.state.value in ("scenario", "current") for record in output.results)


def test_scenario_echoes_its_overrides(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert output.scenario_overrides == get_scenario("S02").overrides


# ------------------------------------------------------------------------- S02
def test_s02_raises_shuttle_demand(model, baseline) -> None:
    """docs/05 section 1.2 lists M06 under S02: 30 percent more crowd is 30 percent more riders."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert_scenario_direction(
        baseline,
        scenario,
        "shuttle_demand",
        direction="up",
        entity_ids=ROUTES,
        ratio_range=(1.2, 1.4),
    )


def test_s02_raises_the_vehicle_requirement(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert_scenario_direction(
        baseline, scenario, "shuttle_requirement", direction="up", entity_ids=ROUTES
    )


def test_s02_raises_the_wait(model, baseline) -> None:
    """The fleet is already short, so extra demand lands entirely on the passenger."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert _total(scenario, "passenger_wait_time") > _total(baseline, "passenger_wait_time")


def test_deltas_are_filled_against_the_baseline(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S02", compare_to_baseline=True))
    matched = [r for r in output.results if r.baseline_value is not None]
    assert matched, "no record was matched to a baseline"
    assert all(r.delta == pytest.approx(r.value - r.baseline_value) for r in matched)


# ------------------------------------------------- the round-trip override
def test_a_longer_round_trip_needs_more_vehicles(model) -> None:
    """The mechanism the docs/03 M06 card wants from M04, driven directly.

    A diversion that makes the loop take twice as long needs about twice the fleet to hold
    the same headway, because each vehicle completes half as many trips.
    """
    demand = 5000.0
    base = model.buses_required(demand, round_trip_min=40.0)
    slower = model.buses_required(demand, round_trip_min=80.0)
    assert slower == pytest.approx(base * 2, rel=0.02)


def test_a_longer_round_trip_lengthens_the_wait(model) -> None:
    buses = 200.0
    assert model.wait_minutes(100.0, buses, round_trip_min=80.0) > model.wait_minutes(
        100.0, buses, round_trip_min=40.0
    )


def test_the_override_reaches_the_output(model) -> None:
    """A scenario can lengthen the round trip until M04 exists to supply it."""
    request = ScenarioRequest(scenario_id="S02")
    builder = model.new_builder(request, scenario_id="S02", overrides={"round_trip_min": 90.0})
    model._compute(request, builder, state="scenario", overrides={"round_trip_min": 90.0})
    output = builder.build()
    assert all(r.details["round_trip_min"] == 90.0 for r in output.results)
    assert any("round trip" in w for w in output.warnings)


# ---------------------------------------------------------------- unsupported
@pytest.mark.parametrize("scenario_id", ["S06", "S12"])
def test_link_closures_are_declared_insensitive(model, scenario_id: str) -> None:
    """The card hedges "via M04 details if present"; M04 is Phase 7 and is not present.

    Neither scenario closes a shuttle route, so with no M04 to lengthen the round trip there
    is nothing for M06 to respond to. Saying so beats returning an unchanged baseline while
    implying a response.
    """
    output = model.handle_scenario(ScenarioRequest(scenario_id=scenario_id))
    assert_insensitive(output, scenario_id)


def test_unsupported_scenario_values_equal_the_baseline(model, baseline) -> None:
    output = model.handle_scenario(ScenarioRequest(scenario_id="S10"))
    assert_scenario_direction(
        baseline, output, "shuttle_demand", direction="same", entity_ids=ROUTES
    )
