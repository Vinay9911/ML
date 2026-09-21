"""M05 scenario directions (docs/05 section 1.2, docs/03 M05).

S02  more crowd, so more cars: demand, occupancy and search time all rise.
S06  road closure - R07, not a parking access link, so M05 is insensitive and says so.
S12  bridge closure - B01, likewise.

The site-reallocation mechanism the docs/03 M05 card asks for is exercised directly, since
no scripted scenario closes a parking link.
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

MODEL_ID = "M05"
SITES = ("P1", "P2", "P3")


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
def test_s02_raises_parking_demand(model, baseline) -> None:
    """docs/05 section 1.2 lists M05 under S02: 30 percent more crowd is 30 percent more cars.

    The band is the same 1.2-1.4x the scenario table sets for the primary KPI, because the
    generator derives car arrivals straight from total arrivals.
    """
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert_scenario_direction(
        baseline,
        scenario,
        "parking_demand",
        direction="up",
        entity_ids=SITES,
        ratio_range=(1.2, 1.4),
    )


def test_s02_raises_occupancy(model, baseline) -> None:
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert_scenario_direction(
        baseline, scenario, "parking_occupancy", direction="up", entity_ids=SITES
    )


def test_s02_raises_search_time(model, baseline) -> None:
    """Search time is monotonic in occupancy, so it has to follow it up."""
    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert _total(scenario, "parking_search_time") > _total(baseline, "parking_search_time")


def test_s02_overflows_at_least_as_much_as_the_baseline(model, baseline) -> None:
    def overflowing(output) -> int:
        return sum(
            1
            for r in output.results
            if r.kpi == "parking_occupancy" and "PARKING_OVERFLOW" in r.reason_codes
        )

    scenario = model.scenario(ScenarioRequest(scenario_id="S02"))
    assert overflowing(scenario) >= overflowing(baseline)


def test_deltas_are_filled_against_the_baseline(model) -> None:
    output = model.scenario(ScenarioRequest(scenario_id="S02", compare_to_baseline=True))
    matched = [r for r in output.results if r.baseline_value is not None]
    assert matched, "no record was matched to a baseline"
    assert all(r.delta == pytest.approx(r.value - r.baseline_value) for r in matched)


# ------------------------------------------------- the reallocation mechanism
def test_closing_a_parking_link_empties_that_site(model) -> None:
    """The docs/03 M05 "shift demand between sites" behaviour, driven directly.

    No scripted scenario closes R04/R05/R06, so the mechanism is exercised with the override
    itself rather than through a scenario id.
    """
    demand, notes = model._scenario_demand("S01", {"closed_links": ["R05"]})
    assert demand["P2"].max() == pytest.approx(0.0), "P2's access link is closed"
    assert any("reallocated" in note for note in notes)


def test_a_closed_site_moves_its_demand_rather_than_losing_it(model) -> None:
    """Drivers do not evaporate when a lot is unreachable."""
    before, _ = model._scenario_demand("S01", {})
    after, _ = model._scenario_demand("S01", {"closed_links": ["R05"]})
    total_before = sum(series.max() for series in before.values())
    total_after = sum(series.max() for series in after.values())
    assert total_after == pytest.approx(total_before, rel=0.01)


def test_the_open_sites_absorb_in_proportion_to_capacity(model) -> None:
    after, _ = model._scenario_demand("S01", {"closed_links": ["R05"]})
    ratio = after["P1"].max() / after["P3"].max()
    expected = model.capacity("P1") / model.capacity("P3")
    assert ratio == pytest.approx(expected, rel=0.05)


def test_closing_a_road_that_serves_no_lot_changes_nothing(model) -> None:
    """R07 is the link S06 closes, and no parking site sits on it."""
    before, _ = model._scenario_demand("S01", {})
    after, notes = model._scenario_demand("S01", {"closed_links": ["R07"]})
    assert notes == []
    for site, series in after.items():
        assert series.max() == pytest.approx(before[site].max())


# ---------------------------------------------------------------- unsupported
@pytest.mark.parametrize("scenario_id", ["S06", "S12"])
def test_link_closures_are_declared_insensitive(model, scenario_id: str) -> None:
    """The docs/03 M05 card hedges "S06/S12 may shift demand between sites"; they do not.

    S06 closes R07 and S12 closes B01, while the parking access links are R04/R05/R06.
    Claiming support would mean returning an unchanged baseline while implying a response.
    """
    output = model.handle_scenario(ScenarioRequest(scenario_id=scenario_id))
    assert_insensitive(output, scenario_id)


def test_unsupported_scenario_values_equal_the_baseline(model, baseline) -> None:
    output = model.handle_scenario(ScenarioRequest(scenario_id="S10"))
    assert_scenario_direction(
        baseline, output, "parking_demand", direction="same", entity_ids=SITES
    )
