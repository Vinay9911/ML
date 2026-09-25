"""M18 behaviour: the docs/03 M18 test list plus the fill projection.

The card asks for two things: fill within [0, 100] after a collection reset, and S02 up.
"""

from __future__ import annotations

import math

import pytest

from twin_common.contracts import PredictRequest

MODEL_ID = "M18"
BINS = ("WB01", "WB02", "WB03", "WB04", "WB05", "WB06", "WB07", "WB08")


@pytest.fixture(scope="module")
def output(model):
    return model.predict(PredictRequest())


def test_every_bin_group_is_reported(output) -> None:
    assert set(BINS) <= output.entities_present()


def test_bins_are_typed_as_service_points(output) -> None:
    for record in output.results:
        assert record.entity_type.value == "service_point"
        assert record.zone_id is not None


def test_all_four_kpis_are_reported(output) -> None:
    assert output.kpis_present() == {
        "waste_generation",
        "waste_bin_fill_level",
        "waste_collection_trips",
        "bin_overflow_time",
    }


def test_units_match_the_registry(output) -> None:
    units = {r.kpi: r.unit for r in output.results}
    assert units["waste_generation"] == "kg/day"
    assert units["waste_bin_fill_level"] == "%"
    assert units["waste_collection_trips"] == "trips/day"
    assert units["bin_overflow_time"] == "min"


# ------------------------------------------------------------------ fill
def test_fill_is_within_zero_and_one_hundred(output) -> None:
    """docs/03 M18 test list: fill within [0, 100]."""
    for record in output.results:
        if record.kpi == "waste_bin_fill_level":
            assert 0.0 <= record.value <= 100.0


def test_fill_uses_the_group_capacity_not_one_bin(model) -> None:
    """A bin GROUP holds many bins; dividing a zone by one bin pinned every KPI at 100 %."""
    single = 120.0
    assert model._bin_capacity_kg > single
    assert model._bin_capacity_kg == pytest.approx(single * model._bins_per_group)


def test_fill_rises_with_what_is_added(model) -> None:
    assert model.fill_after(0.0, 0.0) == 0.0
    low = model.fill_after(0.0, model._bin_capacity_kg * 0.25)
    high = model.fill_after(0.0, model._bin_capacity_kg * 0.5)
    assert low == pytest.approx(25.0)
    assert high == pytest.approx(50.0)


def test_fill_is_clipped_at_full(model) -> None:
    """A bin cannot be more than full; bin_overflow_time carries what happens past that."""
    assert model.fill_after(90.0, model._bin_capacity_kg) == 100.0
    assert model.fill_after(0.0, model._bin_capacity_kg * 10) == 100.0


def test_fill_is_never_negative(model) -> None:
    assert model.fill_after(0.0, -500.0) == 0.0


def test_the_bin_is_emptied_on_the_round(model) -> None:
    """The projection must restart at a collection, not grow without bound.

    The round is 4 hours and the default horizon is 3, so no collection falls inside a
    default request and this has to be asked over a longer window. Without the reset a busy
    bin saturates at 100 percent and stays there - which is exactly what every bin looked
    like before the group capacity was fixed.
    """
    window = model.collection_window_min()
    horizon = int(window * 2)
    output = model.predict(PredictRequest(horizon_min=horizon, entity_ids=["WB01"]))
    fills = [(r.horizon_min, r.value) for r in output.results if r.kpi == "waste_bin_fill_level"]
    fills.sort()
    after_collection = [value for offset, value in fills if offset > window]
    assert after_collection, f"no step falls past the {window:.0f} min round"
    assert min(after_collection) < max(value for _, value in fills), (
        "fill never drops after a collection; the reset is not working"
    )


# ------------------------------------------------------------------ overflow time
def test_overflow_time_is_reported_once_per_bin(output) -> None:
    records = [r for r in output.results if r.kpi == "bin_overflow_time"]
    assert len(records) == len({r.entity_id for r in records})


def test_overflow_time_is_non_negative(output) -> None:
    for record in output.results:
        if record.kpi == "bin_overflow_time":
            assert record.value >= 0.0


def test_a_full_bin_overflows_now(model) -> None:
    assert model.minutes_to_overflow(100.0, 10.0) == 0.0


def test_a_bin_that_is_not_filling_reports_the_sentinel(model) -> None:
    """Not 0, which a reader could take for "overflowing now", and not null."""
    sentinel = float(model.require_param("no_overflow_minutes"))
    assert model.minutes_to_overflow(50.0, 0.0) == sentinel


def test_faster_filling_overflows_sooner(model) -> None:
    quick = model.minutes_to_overflow(0.0, model._bin_capacity_kg * 0.5)
    slow = model.minutes_to_overflow(0.0, model._bin_capacity_kg * 0.1)
    assert quick < slow


def test_a_fuller_bin_overflows_sooner(model) -> None:
    rate = model._bin_capacity_kg * 0.1
    assert model.minutes_to_overflow(80.0, rate) < model.minutes_to_overflow(10.0, rate)


def test_an_overflow_recommends_an_early_collection(output) -> None:
    recommended = [r for r in output.results if r.recommendation is not None]
    if not recommended:
        pytest.skip("no bin overflows inside the baseline horizon")
    for record in recommended:
        assert record.recommendation.requires_approval is True
        assert record.recommendation.resource_type == "waste_vehicle"
        assert record.kpi == "bin_overflow_time"


# ------------------------------------------------------------------ trips
def test_trips_are_non_negative_integers(output) -> None:
    for record in output.results:
        if record.kpi == "waste_collection_trips":
            assert record.value >= 0
            assert record.value == int(record.value)


def test_trips_follow_the_documented_formula(model) -> None:
    """ceil(kg per day / vehicle_payload_kg)."""
    payload = model._payload_kg
    assert model.trips_required(0.0) == 0
    assert model.trips_required(payload) == 1
    assert model.trips_required(payload * 2.1) == 3
    assert model.trips_required(9000.0) == math.ceil(9000.0 / payload)


# ------------------------------------------------------------------ mechanics
def test_amber_and_above_always_carry_a_reason(output) -> None:
    for record in output.results:
        if record.risk_level is not None and record.risk_level.value in (
            "amber",
            "red",
            "critical",
        ):
            assert record.reason_codes


def test_bands_are_ordered_and_calibrated(output) -> None:
    banded = [r for r in output.results if r.lower is not None]
    assert banded
    for record in banded:
        assert record.lower <= record.value <= record.upper
        assert record.details["band_calibrated"] is True


def test_upstream_m01_is_recorded(output) -> None:
    assert "M01" in output.upstream_sources()


def test_entity_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(entity_ids=["WB01"]))
    assert output.entities_present() == {"WB01"}


def test_kpi_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(kpis=["waste_generation"]))
    assert output.kpis_present() == {"waste_generation"}


def test_repeated_requests_are_deterministic(model) -> None:
    first = model.predict(PredictRequest(entity_ids=["WB01"]))
    second = model.predict(PredictRequest(entity_ids=["WB01"]))
    assert [r.value for r in first.results] == [r.value for r in second.results]
