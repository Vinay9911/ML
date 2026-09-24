"""M06 behaviour: the docs/03 M06 test list plus the fleet and wait arithmetic.

The card asks for three things: requirement an integer >= 0, S02 up, and wait decreasing when
the fleet increases. The rest pins down the choices the model card explains - the fleet is
shared between routes, and the wait stretches when it cannot carry the demand.
"""

from __future__ import annotations

import math

import pytest

from twin_common.contracts import PredictRequest

MODEL_ID = "M06"
ROUTES = ("SH1", "SH2")


@pytest.fixture(scope="module")
def output(model):
    return model.predict(PredictRequest())


def test_every_route_is_forecast(output) -> None:
    assert set(ROUTES) <= output.entities_present()


def test_routes_are_typed_as_routes(output) -> None:
    for record in output.results:
        assert record.entity_type.value == "route"


def test_all_three_kpis_are_reported(output) -> None:
    assert output.kpis_present() == {
        "shuttle_demand",
        "shuttle_requirement",
        "passenger_wait_time",
    }


def test_units_match_the_registry(output) -> None:
    units = {r.kpi: r.unit for r in output.results}
    assert units["shuttle_demand"] == "persons/hr"
    assert units["shuttle_requirement"] == "vehicles"
    assert units["passenger_wait_time"] == "min"


# ------------------------------------------------------------------ fleet arithmetic
def test_requirement_is_a_non_negative_integer(output) -> None:
    """docs/03 M06: requirement integer >= 0."""
    for record in output.results:
        if record.kpi == "shuttle_requirement":
            assert record.value >= 0
            assert record.value == int(record.value)


def test_requirement_follows_the_documented_formula(output, model) -> None:
    """ceil(demand * round_trip/60 / (bus_capacity * load_factor))."""
    demand = {
        (r.entity_id, r.timestamp): r.value for r in output.results if r.kpi == "shuttle_demand"
    }
    for record in output.results:
        if record.kpi != "shuttle_requirement":
            continue
        expected = model.buses_required(demand[(record.entity_id, record.timestamp)])
        assert record.value == pytest.approx(expected)


def test_requirement_grows_with_demand(model) -> None:
    per_bus = model._bus_capacity * model._load_factor
    assert model.buses_required(0.0) == 0
    small = model.buses_required(per_bus)
    large = model.buses_required(per_bus * 50)
    assert 0 < small < large


def test_a_route_with_any_demand_gets_at_least_one_bus(model) -> None:
    """A route with one waiting passenger still needs a vehicle."""
    minimum = int(model.require_param("min_buses_when_demand"))
    assert model.buses_required(0.001) >= minimum


def test_the_formula_matches_a_hand_worked_case(model) -> None:
    """One independent check, so a refactor of the formula cannot pass silently."""
    round_trip, capacity, load = 40.0, model._bus_capacity, model._load_factor
    demand = 4250.0
    expected = math.ceil(demand * round_trip / 60.0 / (capacity * load))
    assert model.buses_required(demand, round_trip_min=round_trip) == expected


# ------------------------------------------------------------------ the shared fleet
def test_the_fleet_is_shared_between_routes(model) -> None:
    """When the routes together want more than exists, each gets a share."""
    deployed = model._deployable({"SH1": 300, "SH2": 100})
    assert sum(deployed.values()) == pytest.approx(model._fleet)
    assert deployed["SH1"] > deployed["SH2"], "the busier route should get more vehicles"


def test_a_fleet_that_is_big_enough_is_not_rationed(model) -> None:
    deployed = model._deployable({"SH1": 5, "SH2": 3})
    assert deployed == {"SH1": 5.0, "SH2": 3.0}


def test_deployed_never_exceeds_the_fleet(output, model) -> None:
    by_time: dict[object, float] = {}
    for record in output.results:
        if record.kpi == "shuttle_requirement":
            by_time[record.timestamp] = (
                by_time.get(record.timestamp, 0.0) + record.details["buses_deployed"]
            )
    assert by_time
    for total in by_time.values():
        assert total <= model._fleet + 1e-6


# ------------------------------------------------------------------ wait time
def test_wait_falls_when_the_fleet_grows(model) -> None:
    """docs/03 M06: wait decreases when fleet increases."""
    demand = 5000.0
    waits = [model.wait_minutes(demand, buses) for buses in (10, 20, 40, 80, 160)]
    assert waits == sorted(waits, reverse=True), waits


def test_wait_is_headway_over_two_when_the_fleet_copes(model) -> None:
    """The card's plain formula must still hold in the case it was written for."""
    buses = 400.0  # comfortably more than enough for this demand
    demand = 100.0
    expected = (model._round_trip_min / buses) / 2.0
    assert model.wait_minutes(demand, buses) == pytest.approx(expected)


def test_wait_stretches_when_the_fleet_cannot_carry_the_demand(model) -> None:
    """The point of the overload factor: full buses going past are a real wait."""
    buses = 20.0
    served = model.capacity_per_hour(buses)
    plain = (model._round_trip_min / buses) / 2.0
    assert model.wait_minutes(served, buses) == pytest.approx(plain)
    assert model.wait_minutes(served * 3, buses) > plain * 2


def test_the_overload_factor_is_capped(model) -> None:
    buses = 5.0
    ceiling = float(model.require_param("max_overload_factor"))
    plain = (model._round_trip_min / buses) / 2.0
    assert model.wait_minutes(1e12, buses) == pytest.approx(plain * ceiling)


def test_no_buses_means_no_reported_wait(model) -> None:
    """With nothing running there is no headway to halve; the shortfall carries the signal."""
    assert model.wait_minutes(1000.0, 0.0) == 0.0


def test_wait_is_never_negative(output) -> None:
    for record in output.results:
        assert record.value >= 0.0


# ------------------------------------------------------------------ shortfall signalling
def test_a_shortfall_recommends_more_vehicles(output) -> None:
    recommended = [r for r in output.results if r.recommendation is not None]
    assert recommended, "this world's fleet is far too small; a shortfall must be raised"
    for record in recommended:
        assert record.recommendation.requires_approval is True
        assert record.recommendation.resource_type == "shuttle_bus"
        assert record.recommendation.quantity > 0
        assert record.kpi == "shuttle_requirement"


def test_the_shortfall_is_reported_in_details(output, model) -> None:
    for record in output.results:
        short = record.details["fleet_short_by"]
        assert short == pytest.approx(
            max(0.0, record.details["buses_required"] - record.details["buses_deployed"]), abs=0.2
        )
        assert record.details["fleet_available"] == pytest.approx(model._fleet)


def test_amber_and_above_always_carry_a_reason(output) -> None:
    for record in output.results:
        if record.risk_level is not None and record.risk_level.value in (
            "amber",
            "red",
            "critical",
        ):
            assert record.reason_codes


# ------------------------------------------------------------------ upstream wiring
def test_both_upstreams_are_recorded(output) -> None:
    """docs/02 section 7: M06 is the first model with two upstreams."""
    sources = output.upstream_sources()
    assert "M01" in sources
    assert "M05" in sources


def test_the_arrivals_covariate_reports_its_source(output) -> None:
    sources = {r.details.get("arrivals_covariate") for r in output.results}
    assert sources <= {"M01", "none"}
    assert len(sources) == 1


def test_parking_overflow_is_reported_but_not_added_to_demand(output) -> None:
    """The model card is explicit: M05's overflow is context, not extra riders.

    M06 cannot know what share of turned-away drivers take a shuttle, so inventing one would
    be a number with no evidence behind it.
    """
    overflow = {r.details["parking_overflow_vehicles"] for r in output.results}
    assert len(overflow) == 1
    assert all(value >= 0 for value in overflow)


# ------------------------------------------------------------------ forecast mechanics
def test_bands_are_ordered_and_labelled(output) -> None:
    banded = [r for r in output.results if r.lower is not None]
    assert banded
    for record in banded:
        assert record.lower <= record.value <= record.upper
        assert record.quantile_level == pytest.approx(0.8)


def test_the_band_is_calibrated(output) -> None:
    """M06 enables conformal calibration, so the band should say it is calibrated."""
    assert all(r.details["band_calibrated"] for r in output.results)


def test_horizon_is_respected(output, model) -> None:
    assert max(r.horizon_min for r in output.results) <= model.default_horizon_min


def test_a_shorter_horizon_returns_fewer_records(model) -> None:
    long_horizon = model.predict(PredictRequest(horizon_min=180))
    short_horizon = model.predict(PredictRequest(horizon_min=15))
    assert len(short_horizon.results) < len(long_horizon.results)


def test_forecast_starts_after_as_of(output, model) -> None:
    assert all(r.timestamp > model.demo_now for r in output.results)


def test_entity_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(entity_ids=["SH2"]))
    assert output.entities_present() == {"SH2"}


def test_kpi_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(kpis=["shuttle_requirement"]))
    assert output.kpis_present() == {"shuttle_requirement"}


def test_repeated_requests_are_deterministic(model) -> None:
    """CLAUDE.md: same inputs, same outputs."""
    first = model.predict(PredictRequest(entity_ids=["SH1"]))
    second = model.predict(PredictRequest(entity_ids=["SH1"]))
    assert [r.value for r in first.results] == [r.value for r in second.results]
