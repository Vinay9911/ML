"""M19 behaviour: the docs/03 M19 test list plus the backup arithmetic.

The card asks for two things: the MW conversion, and that S08 gives a finite backup duration
that falls as load rises.
"""

from __future__ import annotations

import math

import pytest

from twin_common.contracts import PredictRequest

MODEL_ID = "M19"
ASSETS = ("SS01", "SS02")


@pytest.fixture(scope="module")
def output(model):
    return model.predict(PredictRequest())


def test_every_substation_is_reported(output) -> None:
    assert set(ASSETS) <= output.entities_present()


def test_records_are_asset_typed(output) -> None:
    for record in output.results:
        assert record.entity_type.value == "asset"


def test_all_three_kpis_are_reported(output) -> None:
    assert output.kpis_present() == {
        "electricity_demand",
        "generator_backup_duration",
        "backup_generator_requirement",
    }


def test_units_match_the_registry(output) -> None:
    units = {r.kpi: r.unit for r in output.results}
    assert units["electricity_demand"] == "MW"
    assert units["generator_backup_duration"] == "hours"
    assert units["backup_generator_requirement"] == "units"


# ------------------------------------------------------------------ the MW conversion
def test_megawatt_conversion(model) -> None:
    """docs/03 M19 test list: MW conversion. D18 is kW."""
    assert model.to_megawatts(1000.0) == pytest.approx(1.0)
    assert model.to_megawatts(2500.0) == pytest.approx(2.5)
    assert model.to_megawatts(0.0) == 0.0


def test_reported_demand_matches_the_kw_in_details(output, model) -> None:
    """details.kw is rounded to 0.1 kW for readability, so the tolerance is 0.0001 MW."""
    for record in output.results:
        if record.kpi == "electricity_demand":
            assert record.value == pytest.approx(model.to_megawatts(record.details["kw"]), abs=1e-4)


def test_demand_is_plausible_for_a_substation(output) -> None:
    """A sanity bound: a venue substation is megawatts, not gigawatts or milliwatts."""
    for record in output.results:
        if record.kpi == "electricity_demand":
            assert 0.0 <= record.value < 100.0


def test_the_load_sums_the_zones_a_substation_feeds(model) -> None:
    """D18 has one row per (substation, zone); keeping one zone would report a quarter.

    SS01 feeds four zones, so the series must be materially larger than any single zone's
    baseline load.
    """
    from twin_common.config import assumptions

    base_per_zone = float(assumptions()["power"]["base_kw_per_zone"])
    typical = float(model._kw["S01"]["SS01"].median())
    assert typical > base_per_zone * 2, (
        f"SS01 median load {typical:.0f} kW looks like one zone, not four"
    )


# ------------------------------------------------------------------ backup duration
def test_backup_duration_is_finite_and_positive(output) -> None:
    for record in output.results:
        if record.kpi == "generator_backup_duration":
            assert 0.0 <= record.value < math.inf


def test_backup_duration_follows_the_documented_formula(model) -> None:
    """fuel / (l_per_hr_at_full_load x load_fraction)."""
    fuel, fraction = 400.0, 0.5
    expected = fuel / (model._l_per_hr_full * fraction)
    assert model.backup_hours(fuel, load_fraction=fraction) == pytest.approx(expected)


def test_backup_duration_falls_as_load_rises(model) -> None:
    """docs/03 M19 test list: duration decreasing with load."""
    fuel = 400.0
    light = model.backup_hours(fuel, load_fraction=0.25)
    heavy = model.backup_hours(fuel, load_fraction=1.0)
    assert light > heavy > 0


def test_no_fuel_is_no_runtime(model) -> None:
    assert model.backup_hours(0.0) == 0.0


def test_backup_duration_is_capped(model) -> None:
    """A full tank against a trivial load is arithmetically enormous and operationally wrong."""
    ceiling = float(model.require_param("max_backup_hours"))
    assert model.backup_hours(1e9) == ceiling
    assert model.backup_hours(400.0, load_fraction=1e-9) == ceiling


# ------------------------------------------------------------------ generators
def test_generator_count_is_a_non_negative_integer(output) -> None:
    for record in output.results:
        if record.kpi == "backup_generator_requirement":
            assert record.value >= 0
            assert record.value == int(record.value)


def test_generators_follow_the_documented_formula(model) -> None:
    """ceil(critical_kw x redundancy / generator_unit_kw)."""
    unit, redundancy = model._generator_unit_kw, model._redundancy
    assert model.generators_required(0.0) == 0
    critical = unit * 2 / redundancy
    assert model.generators_required(critical) == 2


def test_generators_include_the_redundancy_factor(model) -> None:
    """Without the factor a load exactly equal to one unit would ask for one unit."""
    unit = model._generator_unit_kw
    assert model._redundancy > 1.0
    assert model.generators_required(unit) == math.ceil(unit * model._redundancy / unit)


def test_critical_load_is_a_share_of_the_whole(model) -> None:
    fraction = float(model.require_param("critical_load_fraction"))
    assert model.critical_load_kw(1000.0) == pytest.approx(1000.0 * fraction)
    assert model.critical_load_kw(-5.0) == 0.0


# ------------------------------------------------------------------ the cooling term
def test_cooling_uplift_is_zero_without_extra_heat(model) -> None:
    assert model.cooling_uplift_kw(30.0, 30.0) == 0.0
    assert model.cooling_uplift_kw(25.0, 30.0) == 0.0


def test_cooling_uplift_rises_with_temperature(model) -> None:
    """The S04 mechanism: heat does not add people, it makes the cooling work harder."""
    assert model.cooling_uplift_kw(36.0, 30.0) == pytest.approx(model._cooling_kw_per_c * 6.0)
    assert model.cooling_uplift_kw(40.0, 30.0) > model.cooling_uplift_kw(35.0, 30.0)


# ------------------------------------------------------------------ M12 handoff
def test_load_utilization_is_published_for_m12(output) -> None:
    """The docs/03 M12 card reads electrical utilisation as a fire driver."""
    for record in output.results:
        assert "load_utilization" in record.details
        assert 0.0 <= record.details["load_utilization"] <= 100.0


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


def test_both_upstreams_are_recorded(output) -> None:
    sources = output.upstream_sources()
    assert "M01" in sources
    assert "M21" in sources


def test_entity_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(entity_ids=["SS01"]))
    assert output.entities_present() == {"SS01"}


def test_kpi_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(kpis=["electricity_demand"]))
    assert output.kpis_present() == {"electricity_demand"}


def test_repeated_requests_are_deterministic(model) -> None:
    first = model.predict(PredictRequest(entity_ids=["SS01"]))
    second = model.predict(PredictRequest(entity_ids=["SS01"]))
    assert [r.value for r in first.results] == [r.value for r in second.results]
