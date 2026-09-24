"""M15 behaviour: the docs/03 M15 test list plus the gap, point and tanker arithmetic.

The card asks for three things: the gap sign, S04 demand up, and integer requirements. The
rest pins down the choices the model card explains - supply is carried forward, a surplus is
reported as a negative gap rather than clipped, and population is recovered from demand.
"""

from __future__ import annotations

import math

import pytest

from twin_common.contracts import PredictRequest

MODEL_ID = "M15"
ZONES = ("Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07", "Z08")


@pytest.fixture(scope="module")
def output(model):
    return model.predict(PredictRequest())


def test_every_zone_is_reported(output) -> None:
    assert set(ZONES) <= output.entities_present()


def test_records_are_zone_typed(output) -> None:
    for record in output.results:
        assert record.entity_type.value == "zone"
        assert record.zone_id == record.entity_id


def test_all_four_kpis_are_reported(output) -> None:
    assert output.kpis_present() == {
        "expected_water_demand",
        "water_supply_demand_gap",
        "drinking_water_point_requirement",
        "water_tanker_requirement",
    }


def test_units_match_the_registry(output) -> None:
    units = {r.kpi: r.unit for r in output.results}
    assert units["expected_water_demand"] == "L/hr"
    assert units["water_supply_demand_gap"] == "L/hr"
    assert units["drinking_water_point_requirement"] == "points"
    assert units["water_tanker_requirement"] == "vehicles"


# ------------------------------------------------------------------ the gap
def test_gap_is_demand_minus_supply(output) -> None:
    """docs/03 M15: gap = demand - supply. Positive means a shortage."""
    demand = {
        (r.entity_id, r.timestamp): r.value
        for r in output.results
        if r.kpi == "expected_water_demand"
    }
    for record in output.results:
        if record.kpi != "water_supply_demand_gap":
            continue
        expected = demand[(record.entity_id, record.timestamp)] - record.details["supply_l_per_hr"]
        assert record.value == pytest.approx(expected)


def test_a_surplus_is_reported_as_a_negative_gap(output) -> None:
    """Not clipped at zero: "how much spare capacity is there" is a real question."""
    gaps = [r.value for r in output.results if r.kpi == "water_supply_demand_gap"]
    assert min(gaps) < 0, "no zone has spare capacity anywhere in the horizon"


def test_some_zone_is_actually_short(output) -> None:
    """If nothing is ever short the KPI cannot be exercised and the demo says nothing."""
    gaps = [r.value for r in output.results if r.kpi == "water_supply_demand_gap"]
    assert max(gaps) > 0


def test_a_shortage_raises_the_documented_reason(output) -> None:
    for record in output.results:
        if record.kpi == "water_supply_demand_gap" and record.value > 0:
            assert "WATER_GAP" in record.reason_codes


def test_supply_is_constant_across_the_horizon(output) -> None:
    """Supply is a planned quantity carried forward, not a forecast."""
    by_zone: dict[str, set[float]] = {}
    for record in output.results:
        if record.kpi == "expected_water_demand":
            by_zone.setdefault(record.entity_id, set()).add(record.details["supply_l_per_hr"])
    assert by_zone
    for zone, values in by_zone.items():
        assert len(values) == 1, f"{zone} supply varied across the horizon: {values}"


# ------------------------------------------------------------------ requirements
def test_requirements_are_integers(output) -> None:
    """docs/03 M15: integer requirements."""
    for record in output.results:
        if record.kpi in ("drinking_water_point_requirement", "water_tanker_requirement"):
            assert record.value >= 0
            assert record.value == int(record.value)


def test_points_follow_the_documented_formula(model) -> None:
    """ceil(peak population / persons_per_water_point)."""
    per_point = model._persons_per_point
    assert model.points_required(0.0) == 0
    assert model.points_required(per_point) == 1
    assert model.points_required(per_point + 1) == 2
    assert model.points_required(per_point * 10) == 10


def test_tankers_follow_the_documented_formula(model) -> None:
    """ceil(max(gap, 0) * hours / tanker_payload_l)."""
    payload = model._tanker_payload_l
    hours = float(model.require_param("tanker_cover_hours"))
    assert model.tankers_required(0.0) == 0
    assert model.tankers_required(-5000.0) == 0, "a surplus needs no tankers"
    assert model.tankers_required(payload / hours) == 1
    assert model.tankers_required(payload / hours * 3) == 3


def test_tankers_are_only_sent_where_there_is_a_gap(output) -> None:
    gap = {
        (r.entity_id, r.timestamp): r.value
        for r in output.results
        if r.kpi == "water_supply_demand_gap"
    }
    for record in output.results:
        if record.kpi != "water_tanker_requirement":
            continue
        if gap[(record.entity_id, record.timestamp)] <= 0:
            assert record.value == 0


def test_tanker_count_matches_the_gap(output, model) -> None:
    gap = {
        (r.entity_id, r.timestamp): r.value
        for r in output.results
        if r.kpi == "water_supply_demand_gap"
    }
    for record in output.results:
        if record.kpi == "water_tanker_requirement":
            expected = model.tankers_required(gap[(record.entity_id, record.timestamp)])
            assert record.value == pytest.approx(expected)


def test_points_are_sized_once_per_zone(output) -> None:
    """A water point is placed for the busiest moment, not re-sited every 15 minutes."""
    points = [r for r in output.results if r.kpi == "drinking_water_point_requirement"]
    assert len(points) == len({r.entity_id for r in points})
    for record in points:
        assert record.details["sized_against"] == "peak of the horizon"


def test_a_tanker_run_is_recommended_when_short(output) -> None:
    recommended = [r for r in output.results if r.recommendation is not None]
    assert recommended
    for record in recommended:
        assert record.recommendation.requires_approval is True
        assert record.recommendation.resource_type == "water_tanker"
        assert record.recommendation.quantity > 0


# ------------------------------------------------------------------ heat and population
def test_heat_factor_is_one_below_thirty(model) -> None:
    assert model.heat_factor(25.0) == pytest.approx(1.0)
    assert model.heat_factor(30.0) == pytest.approx(1.0)


def test_heat_factor_rises_above_thirty(model) -> None:
    """docs/04 section 6: demand rises with every degree above 30."""
    assert model.heat_factor(36.0) == pytest.approx(1.0 + model._heat_bonus * 6.0)
    assert model.heat_factor(40.0) > model.heat_factor(35.0) > model.heat_factor(31.0)


def test_population_inverts_the_consumption_relation(model) -> None:
    """Recovering population from demand keeps the two KPIs consistent by construction."""
    people = 4000.0
    temperature = 34.0
    litres = people * model._litres_pph * model.heat_factor(temperature)
    assert model.population_from_demand(litres, temperature) == pytest.approx(people)


def test_hotter_weather_means_fewer_people_for_the_same_litres(model) -> None:
    """The same consumption in hotter weather implies a smaller crowd, not a larger one."""
    litres = 5000.0
    assert model.population_from_demand(litres, 38.0) < model.population_from_demand(litres, 30.0)


def test_points_match_the_reported_peak_population(output, model) -> None:
    for record in output.results:
        if record.kpi == "drinking_water_point_requirement":
            expected = model.points_required(record.details["peak_population"])
            assert record.value == pytest.approx(expected, abs=1)


# ------------------------------------------------------------------ mechanics
def test_demand_is_never_negative(output) -> None:
    for record in output.results:
        if record.kpi != "water_supply_demand_gap":
            assert record.value >= 0.0


def test_bands_are_ordered_and_calibrated(output) -> None:
    banded = [r for r in output.results if r.lower is not None]
    assert banded
    for record in banded:
        assert record.lower <= record.value <= record.upper
        assert record.quantile_level == pytest.approx(0.8)
        assert record.details["band_calibrated"] is True


def test_both_upstreams_are_recorded(output) -> None:
    sources = output.upstream_sources()
    assert "M01" in sources
    assert "M21" in sources


def test_horizon_is_respected(output, model) -> None:
    assert max(r.horizon_min for r in output.results) <= model.default_horizon_min


def test_entity_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(entity_ids=["Z01"]))
    assert output.entities_present() == {"Z01"}


def test_kpi_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(kpis=["water_supply_demand_gap"]))
    assert output.kpis_present() == {"water_supply_demand_gap"}


def test_repeated_requests_are_deterministic(model) -> None:
    first = model.predict(PredictRequest(entity_ids=["Z01"]))
    second = model.predict(PredictRequest(entity_ids=["Z01"]))
    assert [r.value for r in first.results] == [r.value for r in second.results]


def test_the_formula_matches_a_hand_worked_case(model) -> None:
    """One independent check so a refactor cannot pass silently."""
    gap, hours, payload = 25_000.0, 4.0, model._tanker_payload_l
    assert model.tankers_required(gap, hours=hours) == math.ceil(gap * hours / payload)
