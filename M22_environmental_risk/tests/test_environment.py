"""M22 behaviour: the docs/03 M22 test list plus the noise and CO2 arithmetic.

The card's named test is "AQI sub-index piecewise-linear correctness on breakpoints", which
is what most of this file checks. The breakpoints are placeholders transcribed from docs/05
and must be verified against CPCB before real use - the tests assert the *method*, not that
the numbers are authoritative.
"""

from __future__ import annotations

import math

import pytest

from twin_common.contracts import PredictRequest

MODEL_ID = "M22"
ZONES = ("Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07", "Z08")


@pytest.fixture(scope="module")
def output(model):
    return model.predict(PredictRequest())


def test_all_four_kpis_are_reported(output) -> None:
    assert output.kpis_present() == {
        "air_quality_index",
        "pm25",
        "noise_level",
        "co2_emissions",
    }


def test_units_match_the_registry(output) -> None:
    units = {r.kpi: r.unit for r in output.results}
    assert units["air_quality_index"] == "index"
    assert units["pm25"] == "ug/m3"
    assert units["noise_level"] == "dB"
    assert units["co2_emissions"] == "tCO2e/day"


def test_air_quality_is_venue_level_and_noise_is_per_zone(output) -> None:
    """D13 is venue-level and hourly; crowd density is per zone."""
    for record in output.results:
        if record.kpi in ("pm25", "air_quality_index", "co2_emissions"):
            assert record.entity_type.value == "event"
            assert record.entity_id == "EVENT"
        elif record.kpi == "noise_level":
            assert record.entity_type.value == "zone"


def test_the_grid_is_hourly(output) -> None:
    horizons = sorted({r.horizon_min for r in output.results})
    assert all(h % 60 == 0 for h in horizons)


# ------------------------------------------------- the NAQI sub-index
def test_sub_index_is_zero_at_zero(model) -> None:
    assert model.sub_index(0.0, "pm2_5") == pytest.approx(0.0)


def test_sub_index_hits_every_breakpoint_exactly(model) -> None:
    """docs/03 M22 test list: piecewise-linear correctness ON the breakpoints.

    At each concentration breakpoint the sub-index must equal the matching index breakpoint.
    An off-by-one in the interpolation shows up here immediately.
    """
    naqi = model.require_param("naqi")
    for pollutant in ("pm2_5", "pm10"):
        concentrations = naqi["concentration_breakpoints"][pollutant]
        indices = naqi["index_breakpoints"]
        for concentration, index in zip(concentrations, indices, strict=True):
            assert model.sub_index(float(concentration), pollutant) == pytest.approx(
                float(index)
            ), f"{pollutant} at {concentration} should be index {index}"


def test_sub_index_is_linear_between_breakpoints(model) -> None:
    """Halfway in concentration must be halfway in index, within a band."""
    naqi = model.require_param("naqi")
    concentrations = naqi["concentration_breakpoints"]["pm2_5"]
    indices = naqi["index_breakpoints"]
    low_c, high_c = float(concentrations[0]), float(concentrations[1])
    low_i, high_i = float(indices[0]), float(indices[1])
    midpoint_c = (low_c + high_c) / 2.0
    expected = (low_i + high_i) / 2.0
    assert model.sub_index(midpoint_c, "pm2_5") == pytest.approx(expected, rel=0.02)


def test_sub_index_rises_monotonically(model) -> None:
    values = [model.sub_index(c, "pm2_5") for c in range(0, 400, 10)]
    assert values == sorted(values)


def test_sub_index_is_capped_at_the_top_of_the_scale(model) -> None:
    """The CPCB scale stops at 500; past it a reading is off the scale, not a bigger number."""
    ceiling = float(model.require_param("naqi")["max_index"])
    assert model.sub_index(1e6, "pm2_5") == ceiling


def test_a_negative_concentration_reads_as_zero(model) -> None:
    assert model.sub_index(-10.0, "pm2_5") == pytest.approx(0.0)


def test_aqi_is_the_maximum_sub_index_not_the_average(model) -> None:
    """The rule that stops a clean pollutant masking a dangerous one."""
    dirty_pm25, clean_pm10 = 250.0, 10.0
    combined = model.aqi(dirty_pm25, clean_pm10)
    assert combined == pytest.approx(model.sub_index(dirty_pm25, "pm2_5"))
    assert combined > model.sub_index(clean_pm10, "pm10")

    average = (model.sub_index(dirty_pm25, "pm2_5") + model.sub_index(clean_pm10, "pm10")) / 2
    assert combined > average, "an averaged AQI would understate the dangerous pollutant"


def test_aqi_works_with_pm25_alone(model) -> None:
    assert model.aqi(45.0) == pytest.approx(model.sub_index(45.0, "pm2_5"))


def test_aqi_categories_follow_the_breakpoints(model) -> None:
    naqi = model.require_param("naqi")
    for edge, name in zip(naqi["index_breakpoints"], naqi["category_names"], strict=True):
        assert model.aqi_category(float(edge)) == name


def test_reported_aqi_matches_the_sub_index_method(output, model) -> None:
    pm25 = {r.timestamp: r.value for r in output.results if r.kpi == "pm25"}
    for record in output.results:
        if record.kpi != "air_quality_index":
            continue
        expected = model.aqi(pm25[record.timestamp], record.details.get("pm10_ug_m3"))
        assert record.value == pytest.approx(expected, rel=1e-6)


# ------------------------------------------------------------------ noise
def test_noise_is_the_base_with_no_crowd(model) -> None:
    """base + 10 log10(1 + 0) = base."""
    base = float(model.require_param("noise")["base_db"])
    assert model.noise_db(0.0) == pytest.approx(base)


def test_noise_follows_the_documented_formula(model) -> None:
    settings = model.require_param("noise")
    base = float(settings["base_db"])
    reference = float(settings["density_ref_p_m2"])
    density = 3.0
    expected = base + 10.0 * math.log10(1.0 + density / reference)
    assert model.noise_db(density) == pytest.approx(expected)


def test_noise_rises_with_density(model) -> None:
    assert model.noise_db(5.0) > model.noise_db(2.0) > model.noise_db(0.5)


def test_noise_is_logarithmic_not_linear(model) -> None:
    """Doubling the crowd must not double the decibels; that is the point of the log."""
    base = float(model.require_param("noise")["base_db"])
    quiet = model.noise_db(1.0) - base
    loud = model.noise_db(2.0) - base
    assert loud < quiet * 2


def test_noise_is_capped(model) -> None:
    ceiling = float(model.require_param("noise")["max_db"])
    assert model.noise_db(1e9) == ceiling


def test_reported_noise_matches_the_density_in_details(output, model) -> None:
    for record in output.results:
        if record.kpi == "noise_level":
            assert record.value == pytest.approx(
                model.noise_db(record.details["density_p_m2"]), abs=0.05
            )


# ------------------------------------------------------------------ CO2
def test_co2_is_never_negative(output) -> None:
    for record in output.results:
        if record.kpi == "co2_emissions":
            assert record.value >= 0.0


def test_co2_sums_its_three_terms(model) -> None:
    factors = model.require_param("co2")
    hours = float(factors["hours_per_day"])
    tonne = float(factors["kg_per_tonne"])
    expected = (
        (
            100.0 * float(factors["kg_per_vehicle_km"])
            + 50.0 * float(factors["kg_per_kwh"])
            + 10.0 * float(factors["kg_per_diesel_litre"])
        )
        * hours
        / tonne
    )
    assert model.co2_tonnes_per_day(
        vehicle_km=100.0, kwh=50.0, generator_litres=10.0
    ) == pytest.approx(expected)


def test_each_co2_term_adds(model) -> None:
    nothing = model.co2_tonnes_per_day(vehicle_km=0.0, kwh=0.0, generator_litres=0.0)
    assert nothing == 0.0
    assert model.co2_tonnes_per_day(vehicle_km=10.0, kwh=0.0, generator_litres=0.0) > 0
    assert model.co2_tonnes_per_day(vehicle_km=0.0, kwh=10.0, generator_litres=0.0) > 0
    assert model.co2_tonnes_per_day(vehicle_km=0.0, kwh=0.0, generator_litres=10.0) > 0


def test_the_missing_traffic_term_is_declared(output) -> None:
    """M04 does not exist yet, so the traffic term is zero and the output must say so."""
    for record in output.results:
        assert record.details["traffic_term"] in ("M04", "unavailable (zero)")
    assert any("no M04 output" in w for w in output.warnings)


# ------------------------------------------------------------------ mechanics
def test_upstream_m21_is_recorded(output) -> None:
    assert "M21" in output.upstream_sources()


def test_bands_are_ordered(output) -> None:
    banded = [r for r in output.results if r.lower is not None]
    assert banded
    for record in banded:
        assert record.lower <= record.value <= record.upper


def test_kpi_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(kpis=["noise_level"]))
    assert output.kpis_present() == {"noise_level"}


def test_repeated_requests_are_deterministic(model) -> None:
    first = model.predict(PredictRequest())
    second = model.predict(PredictRequest())
    assert [r.value for r in first.results] == [r.value for r in second.results]
