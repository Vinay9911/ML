"""M05 behaviour: the docs/03 M05 test list plus the demand/occupancy relationship.

The card asks for three things: occupancy within [0, 100 + overflow], S02 up, and search
time monotonic in occupancy. The rest of this file pins down the choices the model card
explains - demand is uncapped, occupancy may exceed 100, and the overflow reason code means
overflow rather than "busy".
"""

from __future__ import annotations

import pytest

from twin_common.contracts import PredictRequest

MODEL_ID = "M05"
SITES = ("P1", "P2", "P3")


@pytest.fixture(scope="module")
def output(model):
    return model.predict(PredictRequest())


def test_every_site_is_forecast(output) -> None:
    assert set(SITES) <= output.entities_present()


def test_sites_are_typed_as_parking_sites(output) -> None:
    for record in output.results:
        assert record.entity_type.value == "parking_site"
        assert record.zone_id is not None, "a lot sits in a zone; M04 and M06 join on it"


def test_all_three_kpis_are_reported(output) -> None:
    assert output.kpis_present() == {
        "parking_demand",
        "parking_occupancy",
        "parking_search_time",
    }


def test_units_match_the_registry(output) -> None:
    units = {r.kpi: r.unit for r in output.results}
    assert units["parking_demand"] == "vehicles"
    assert units["parking_occupancy"] == "%"
    assert units["parking_search_time"] == "min"


# ------------------------------------------------------------------ occupancy
def test_occupancy_stays_within_the_documented_bound(output, model) -> None:
    """docs/03 M05: occupancy within [0, 100 + overflow]."""
    ceiling = float(model.require_param("max_occupancy_pct"))
    for record in output.results:
        if record.kpi == "parking_occupancy":
            assert 0.0 <= record.value <= ceiling


def test_occupancy_is_demand_over_capacity(output, model) -> None:
    """The relationship the model card claims, checked rather than asserted in prose."""
    demand = {
        (r.entity_id, r.timestamp): r.value for r in output.results if r.kpi == "parking_demand"
    }
    for record in output.results:
        if record.kpi != "parking_occupancy":
            continue
        capacity = model.capacity(record.entity_id)
        expected = demand[(record.entity_id, record.timestamp)] / capacity * 100.0
        assert record.value == pytest.approx(min(expected, 400.0), rel=1e-6)


def test_occupancy_may_exceed_one_hundred(output) -> None:
    """The whole point of forecasting demand rather than the capped `occupied` column.

    If this ever fails the model has silently lost its ability to warn about an overflow,
    which is the decision it exists to support.
    """
    peak = max(r.value for r in output.results if r.kpi == "parking_occupancy")
    assert peak > 100.0, f"no overflow anywhere in the horizon (peak {peak:.1f}%)"


def test_occupied_spaces_never_exceed_capacity(output, model) -> None:
    """Occupancy may pass 100, but the number of PARKED cars cannot."""
    for record in output.results:
        capacity = model.capacity(record.entity_id)
        assert record.details["occupied_spaces"] <= capacity + 1e-6
        assert record.details["capacity_spaces"] == pytest.approx(capacity)


def test_overflow_vehicles_are_the_shortfall(output, model) -> None:
    for record in output.results:
        if record.kpi != "parking_demand":
            continue
        capacity = model.capacity(record.entity_id)
        assert record.details["overflow_vehicles"] == pytest.approx(
            max(0.0, record.value - capacity), abs=0.2
        )


# ------------------------------------------------------------------ search time
def test_search_time_is_monotonic_in_occupancy(model) -> None:
    """docs/03 M05: search time monotonic in occupancy."""
    times = [model.search_time_min(occ) for occ in range(0, 200, 5)]
    assert times == sorted(times)


def test_search_time_is_flat_below_the_knee(model) -> None:
    """Below the knee a driver parks straight away, so the curve is the base value."""
    settings = model.require_param("search_time")
    knee_pct = float(settings["occupancy_knee"]) * 100.0
    base = float(settings["base_min"])
    assert model.search_time_min(0.0) == pytest.approx(base)
    assert model.search_time_min(knee_pct) == pytest.approx(base)


def test_search_time_hits_the_documented_threshold_when_full(model) -> None:
    """The curve is calibrated so a full lot sits exactly on the docs/05 15-minute mark."""
    settings = model.require_param("search_time")
    expected = float(settings["base_min"]) + float(settings["k_min"])
    assert model.search_time_min(100.0) == pytest.approx(expected)


def test_search_time_keeps_rising_past_full(model) -> None:
    """An oversubscribed lot must not saturate, or the KPI stops discriminating."""
    assert model.search_time_min(150.0) > model.search_time_min(100.0)


def test_reported_search_time_matches_the_curve(output, model) -> None:
    occupancy = {
        (r.entity_id, r.timestamp): r.value for r in output.results if r.kpi == "parking_occupancy"
    }
    for record in output.results:
        if record.kpi != "parking_search_time":
            continue
        occ = occupancy[(record.entity_id, record.timestamp)]
        assert record.value == pytest.approx(model.search_time_min(occ))


# ------------------------------------------------------------------ overflow signalling
def test_overflow_raises_the_documented_reason_code(output, model) -> None:
    overflow_pct = float(model.require_param("overflow_pct"))
    for record in output.results:
        if record.kpi != "parking_occupancy":
            continue
        if record.value >= overflow_pct:
            assert "PARKING_OVERFLOW" in record.reason_codes


def test_a_busy_lot_is_not_called_an_overflow(output, model) -> None:
    """PARKING_OVERFLOW must mean overflow; a filling lot reports CONGESTION instead."""
    overflow_pct = float(model.require_param("overflow_pct"))
    for record in output.results:
        if record.value < overflow_pct and record.kpi == "parking_occupancy":
            assert "PARKING_OVERFLOW" not in record.reason_codes


def test_amber_and_above_always_carry_a_reason(output) -> None:
    """docs/02 section 4, checked here because the band turns amber well before overflow."""
    for record in output.results:
        if record.risk_level is not None and record.risk_level.value in (
            "amber",
            "red",
            "critical",
        ):
            assert record.reason_codes, f"{record.kpi} on {record.entity_id} has no reason"


def test_an_overflow_recommends_a_diversion(output) -> None:
    recommended = [r for r in output.results if r.recommendation is not None]
    assert recommended, "an overflowing lot must propose an action"
    for record in recommended:
        assert record.recommendation.requires_approval is True
        assert record.recommendation.resource_type == "shuttle_bus"
        assert record.recommendation.quantity > 0
        assert record.recommendation.target_entity_id == record.entity_id


# ------------------------------------------------------------------ forecast mechanics
def test_no_negative_vehicle_counts(output) -> None:
    for record in output.results:
        assert record.value >= 0.0
        if record.lower is not None:
            assert record.lower >= 0.0


def test_bands_are_ordered_and_labelled(output) -> None:
    banded = [r for r in output.results if r.lower is not None]
    assert banded, "the default backend should produce uncertainty bands"
    for record in banded:
        assert record.lower <= record.value <= record.upper
        assert record.quantile_level == pytest.approx(0.8)


def test_horizon_is_respected(output, model) -> None:
    assert max(r.horizon_min for r in output.results) <= model.default_horizon_min


def test_a_shorter_horizon_returns_fewer_records(model) -> None:
    long_horizon = model.predict(PredictRequest(horizon_min=180))
    short_horizon = model.predict(PredictRequest(horizon_min=15))
    assert len(short_horizon.results) < len(long_horizon.results)
    assert max(r.horizon_min for r in short_horizon.results) <= 15


def test_forecast_starts_after_as_of(output, model) -> None:
    assert all(r.timestamp > model.demo_now for r in output.results)


def test_entity_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(entity_ids=["P2"]))
    assert output.entities_present() == {"P2"}


def test_kpi_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(kpis=["parking_occupancy"]))
    assert output.kpis_present() == {"parking_occupancy"}


def test_upstream_m01_is_recorded(output) -> None:
    """docs/02 section 7: the resolution source must be reported."""
    assert "M01" in output.upstream_sources()


def test_the_arrivals_covariate_reports_its_source(output) -> None:
    """Whether M01 actually reached the forecast is visible to the caller, not implied."""
    sources = {r.details.get("arrivals_covariate") for r in output.results}
    assert sources <= {"M01", "none"}
    assert len(sources) == 1


def test_backend_is_reported_in_details(output) -> None:
    backends = {r.details.get("backend") for r in output.results}
    assert backends <= {"chronos2", "lightgbm", "naive"}
    assert len(backends) == 1, f"one request should use one backend, got {backends}"


def test_repeated_requests_are_deterministic(model) -> None:
    """CLAUDE.md: same inputs, same outputs."""
    first = model.predict(PredictRequest(entity_ids=["P1"]))
    second = model.predict(PredictRequest(entity_ids=["P1"]))
    assert [r.value for r in first.results] == [r.value for r in second.results]
