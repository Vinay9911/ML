"""M17 behaviour: the docs/03 M17 test list plus the inventory simulation.

The card asks for two things: coverage non-negative and the S13 direction. The rest pins
down the simulation, which is what makes the lead time matter at all.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from twin_common.contracts import PredictRequest

MODEL_ID = "M17"
OUTLETS = ("FO01", "FO02")


@pytest.fixture(scope="module")
def output(model):
    return model.predict(PredictRequest())


def test_every_outlet_is_reported(output) -> None:
    assert set(OUTLETS) <= output.entities_present()


def test_outlets_are_typed_as_facilities(output) -> None:
    """FO extends the docs/02 section 3 facility pattern; see the model card."""
    for record in output.results:
        assert record.entity_type.value == "facility"
        assert record.zone_id is not None


def test_all_three_kpis_are_reported(output) -> None:
    assert output.kpis_present() == {
        "food_demand",
        "food_stock_coverage",
        "delivery_requirement",
    }


def test_units_match_the_registry(output) -> None:
    units = {r.kpi: r.unit for r in output.results}
    assert units["food_demand"] == "meals/hr"
    assert units["food_stock_coverage"] == "hours"
    assert units["delivery_requirement"] == "trips/day"


def test_the_grid_is_hourly(output) -> None:
    """D17 is hourly, unlike every other forecasting model in this project."""
    horizons = sorted({r.horizon_min for r in output.results})
    assert horizons[0] == 60
    assert all(h % 60 == 0 for h in horizons)


# ------------------------------------------------------------------ coverage
def test_coverage_is_non_negative(output) -> None:
    """docs/03 M17 test list: coverage non-negative."""
    for record in output.results:
        if record.kpi == "food_stock_coverage":
            assert record.value >= 0.0


def test_coverage_is_stock_over_rate(model) -> None:
    assert model.coverage_hours(1000.0, 100.0) == pytest.approx(10.0)
    assert model.coverage_hours(0.0, 100.0) == 0.0


def test_coverage_is_capped_not_infinite(model) -> None:
    """An outlet with stock and no demand has unbounded cover; that is useless on a dashboard."""
    ceiling = float(model.require_param("max_coverage_hours"))
    assert model.coverage_hours(1000.0, 0.0) == ceiling
    assert model.coverage_hours(1e9, 1.0) == ceiling


def test_no_stock_is_no_cover_even_with_no_demand(model) -> None:
    """Empty is empty. The cap applies to slow demand, not to an empty shelf."""
    assert model.coverage_hours(0.0, 0.0) == 0.0


def test_reported_coverage_never_exceeds_the_cap(output, model) -> None:
    ceiling = float(model.require_param("max_coverage_hours"))
    for record in output.results:
        if record.kpi == "food_stock_coverage":
            assert record.value <= ceiling


# ------------------------------------------------------------------ deliveries
def test_trips_are_non_negative_integers(output) -> None:
    for record in output.results:
        if record.kpi == "delivery_requirement":
            assert record.value >= 0
            assert record.value == int(record.value)


def test_trips_follow_the_documented_formula(model) -> None:
    """ceil(demand x hours_per_day / vehicle_payload_meals)."""
    hours = float(model.require_param("hours_per_day"))
    payload = model._payload_meals
    assert model.trips_required(0.0) == 0
    assert model.trips_required(payload / hours) == 1
    assert model.trips_required(payload / hours * 2.5) == 3


def test_trips_grow_with_demand(model) -> None:
    assert model.trips_required(5000.0) > model.trips_required(500.0)


# ------------------------------------------------------------------ the simulation
def test_the_simulation_draws_stock_down(model) -> None:
    """Without a delivery, stock falls by what is eaten."""
    rows = model.simulate_inventory(1000.0, [100.0, 100.0, 100.0], lead_time_hours=99.0)
    assert [row["stock"] for row in rows] == [900.0, 800.0, 700.0]


def test_stock_never_goes_negative(model) -> None:
    rows = model.simulate_inventory(150.0, [100.0, 100.0, 100.0], lead_time_hours=99.0)
    assert all(row["stock"] >= 0.0 for row in rows)


def test_an_order_arrives_after_the_lead_time(model) -> None:
    """The whole point of the simulation: WHEN the lorry lands is what a lead time changes."""
    demand = [500.0] * 8
    rows = model.simulate_inventory(600.0, demand, lead_time_hours=3.0)
    arrivals = [step for step, row in enumerate(rows) if row["arrived"] > 0]
    assert arrivals, "an outlet this short must have ordered and received"
    ordered_at = next(step for step, row in enumerate(rows) if row["ordered"] > 0)
    assert arrivals[0] == ordered_at + 3


def test_a_longer_lead_time_delays_the_delivery(model) -> None:
    demand = [500.0] * 12
    quick = model.simulate_inventory(600.0, demand, lead_time_hours=3.0)
    slow = model.simulate_inventory(600.0, demand, lead_time_hours=9.0)
    first_quick = next(s for s, r in enumerate(quick) if r["arrived"] > 0)
    first_slow = next((s for s, r in enumerate(slow) if r["arrived"] > 0), None)
    assert first_slow is None or first_slow > first_quick


def test_a_longer_lead_time_leaves_less_stock(model) -> None:
    """The mechanism behind the S13 assertion, isolated from the forecast."""
    demand = [500.0] * 12
    quick = model.simulate_inventory(600.0, demand, lead_time_hours=3.0)
    slow = model.simulate_inventory(600.0, demand, lead_time_hours=9.0)
    assert sum(r["stock"] for r in slow) < sum(r["stock"] for r in quick)


def test_orders_are_whole_vehicle_loads(model) -> None:
    rows = model.simulate_inventory(0.0, [500.0] * 6, lead_time_hours=2.0)
    for row in rows:
        if row["ordered"] > 0:
            assert row["ordered"] % model._payload_meals == pytest.approx(0.0)


def test_no_duplicate_orders_while_one_is_in_flight(model) -> None:
    """A real buyer does not reorder every hour while a lorry is already on the road."""
    rows = model.simulate_inventory(100.0, [500.0] * 10, lead_time_hours=5.0)
    ordering_steps = [s for s, r in enumerate(rows) if r["ordered"] > 0]
    assert len(ordering_steps) == len(set(ordering_steps))
    for first, second in pairwise(ordering_steps):
        assert second - first >= 5


# ------------------------------------------------------------------ stockout signalling
def test_stockout_risk_uses_the_minimum_cover(model) -> None:
    minimum = model._min_cover_hours
    assert model.is_stockout_risk(minimum - 0.1, lead_time_hours=1.0) is True
    assert model.is_stockout_risk(minimum + 100.0, lead_time_hours=1.0) is False


def test_stockout_risk_also_uses_the_lead_time(model) -> None:
    """Stock that outlasts the minimum but not the lorry is still a stockout in waiting."""
    comfortable = model._min_cover_hours + 1.0
    assert model.is_stockout_risk(comfortable, lead_time_hours=1.0) is False
    assert model.is_stockout_risk(comfortable, lead_time_hours=comfortable + 5.0) is True


def test_a_stockout_recommends_a_delivery(output, model) -> None:
    at_risk = [r for r in output.results if "STOCKOUT_RISK" in r.reason_codes]
    if not at_risk:
        pytest.skip("no outlet is short in the baseline horizon")
    recommended = [r for r in output.results if r.recommendation is not None]
    assert recommended
    for record in recommended:
        assert record.recommendation.requires_approval is True
        assert record.recommendation.resource_type == "food_vehicle"


# ------------------------------------------------------------------ mechanics
def test_demand_is_never_negative(output) -> None:
    for record in output.results:
        assert record.value >= 0.0


def test_bands_are_ordered_and_calibrated(output) -> None:
    banded = [r for r in output.results if r.lower is not None]
    assert banded
    for record in banded:
        assert record.lower <= record.value <= record.upper
        assert record.details["band_calibrated"] is True


def test_upstream_m01_is_recorded(output) -> None:
    assert "M01" in output.upstream_sources()


def test_entity_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(entity_ids=["FO01"]))
    assert output.entities_present() == {"FO01"}


def test_kpi_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(kpis=["food_stock_coverage"]))
    assert output.kpis_present() == {"food_stock_coverage"}


def test_repeated_requests_are_deterministic(model) -> None:
    first = model.predict(PredictRequest(entity_ids=["FO01"]))
    second = model.predict(PredictRequest(entity_ids=["FO01"]))
    assert [r.value for r in first.results] == [r.value for r in second.results]


def test_the_trip_formula_matches_a_hand_worked_case(model) -> None:
    meals_per_hour, hours, payload = 900.0, 24.0, model._payload_meals
    assert model.trips_required(meals_per_hour) == math.ceil(meals_per_hour * hours / payload)
