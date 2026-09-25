"""M20 behaviour: the docs/03 M20 test list plus the risk curve.

The card asks for two things: the utilisation formula and the S08 direction.
"""

from __future__ import annotations

import pytest

from twin_common.contracts import PredictRequest

MODEL_ID = "M20"
TOWERS = ("NT01", "NT02", "NT03")


@pytest.fixture(scope="module")
def output(model):
    return model.predict(PredictRequest())


def test_every_tower_is_reported(output) -> None:
    assert set(TOWERS) <= output.entities_present()


def test_records_are_asset_typed(output) -> None:
    for record in output.results:
        assert record.entity_type.value == "asset"


def test_all_three_kpis_are_reported(output) -> None:
    assert output.kpis_present() == {
        "bandwidth_utilization",
        "network_availability",
        "network_capacity_risk",
    }


def test_units_are_all_percentages(output) -> None:
    for record in output.results:
        assert record.unit == "%"


def test_the_grid_is_five_minutes(model) -> None:
    """D19 is 5-minute, unlike the 15-minute grid the rest of the project uses."""
    assert model.horizon_steps(60) == 12
    assert model.horizon_steps(180) == 36
    assert model.horizon_steps(5) == 1


# ------------------------------------------------------------------ utilisation
def test_utilisation_formula(model) -> None:
    """docs/05 section 2: used / capacity * 100."""
    assert model.utilization_pct(500.0, 2000.0) == pytest.approx(25.0)
    assert model.utilization_pct(0.0, 2000.0) == 0.0
    assert model.utilization_pct(2000.0, 2000.0) == pytest.approx(100.0)


def test_utilisation_is_clipped_at_full(model) -> None:
    """Demand past capacity is dropped packets, not throughput."""
    assert model.utilization_pct(5000.0, 2000.0) == 100.0


def test_utilisation_is_never_negative(model) -> None:
    assert model.utilization_pct(-50.0, 2000.0) == 0.0


def test_zero_capacity_reads_as_saturated(model) -> None:
    """A tower with no capacity cannot carry anything; that is full, not empty."""
    assert model.utilization_pct(10.0, 0.0) == 100.0


def test_reported_utilisation_matches_the_details(output, model) -> None:
    for record in output.results:
        if record.kpi != "bandwidth_utilization":
            continue
        expected = model.utilization_pct(
            record.details["used_mbps"], record.details["capacity_mbps"]
        )
        assert record.value == pytest.approx(expected, abs=0.1)


def test_utilisation_is_within_zero_and_one_hundred(output) -> None:
    for record in output.results:
        if record.kpi == "bandwidth_utilization":
            assert 0.0 <= record.value <= 100.0


# ------------------------------------------------------------------ the risk curve
def test_risk_is_a_percentage(output) -> None:
    for record in output.results:
        if record.kpi == "network_capacity_risk":
            assert 0.0 <= record.value <= 100.0


def test_risk_is_half_at_the_midpoint(model) -> None:
    """The defining property of the sigmoid in the docs/03 M20 card."""
    midpoint = float(model.require_param("risk_midpoint_pct"))
    assert model.capacity_risk_pct(midpoint) == pytest.approx(50.0)


def test_risk_rises_with_utilisation(model) -> None:
    values = [model.capacity_risk_pct(u) for u in range(0, 101, 10)]
    assert values == sorted(values)


def test_risk_is_low_when_the_link_is_quiet(model) -> None:
    assert model.capacity_risk_pct(10.0) < 5.0


def test_risk_is_high_when_the_link_is_saturated(model) -> None:
    assert model.capacity_risk_pct(100.0) > 90.0


def test_the_risk_curve_cannot_overflow(model) -> None:
    """A wild utilisation must not raise; the exponential is guarded."""
    assert model.capacity_risk_pct(-1e6) == 0.0
    assert model.capacity_risk_pct(1e6) == 100.0


def test_this_world_does_not_run_out_of_bandwidth(output) -> None:
    """Documents the finding in the model card rather than hiding it.

    Peak utilisation here is under 30 percent, so the capacity risk is near zero throughout.
    That is the honest answer for these tower capacities. If this ever starts failing, either
    the capacities or the demand assumptions changed and the model card is stale.
    """
    risks = [r.value for r in output.results if r.kpi == "network_capacity_risk"]
    assert max(risks) < 20.0, "capacity risk is no longer negligible; revisit the M20 model card"


# ------------------------------------------------------------------ availability
def test_availability_is_a_percentage(output) -> None:
    for record in output.results:
        if record.kpi == "network_availability":
            assert 0.0 <= record.value <= 100.0


def test_a_healthy_world_is_fully_available(output) -> None:
    for record in output.results:
        if record.kpi == "network_availability":
            assert record.value == pytest.approx(100.0)


# ------------------------------------------------------------------ mechanics
def test_amber_and_above_always_carry_a_reason(output) -> None:
    for record in output.results:
        if record.risk_level is not None and record.risk_level.value in (
            "amber",
            "red",
            "critical",
        ):
            assert record.reason_codes


def test_camera_counts_are_attached(output, model) -> None:
    """S09 removes camera backhaul, so the model has to know what hangs off each tower."""
    total = sum(model._cameras_on_tower.values())
    assert total > 0, "no cameras were found on any tower"
    for record in output.results:
        assert record.details["cameras_on_tower"] >= 0


def test_bands_are_ordered_and_calibrated(output) -> None:
    banded = [r for r in output.results if r.lower is not None]
    assert banded
    for record in banded:
        assert record.lower <= record.value <= record.upper
        assert record.details["band_calibrated"] is True


def test_upstream_m01_is_recorded(output) -> None:
    assert "M01" in output.upstream_sources()


def test_entity_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(entity_ids=["NT01"]))
    assert output.entities_present() == {"NT01"}


def test_kpi_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(kpis=["bandwidth_utilization"]))
    assert output.kpis_present() == {"bandwidth_utilization"}


def test_repeated_requests_are_deterministic(model) -> None:
    first = model.predict(PredictRequest(entity_ids=["NT01"]))
    second = model.predict(PredictRequest(entity_ids=["NT01"]))
    assert [r.value for r in first.results] == [r.value for r in second.results]
