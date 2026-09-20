"""M21 formula checks: the NOAA heat index and the waterlogging curve.

docs/03 M21 requires the heat index to match NOAA reference values within tolerance. The
reference table lives in the engine so this test and the engine test check the same cells.
"""

from __future__ import annotations

import pytest

from twin_common.contracts import PredictRequest
from twin_common.engines.formula import (
    NOAA_CHART_REFERENCE,
    NOAA_CHART_TOLERANCE_F,
    heat_index_c,
    heat_index_f,
)


@pytest.mark.parametrize(("temp_f", "humidity", "expected"), NOAA_CHART_REFERENCE)
def test_heat_index_matches_the_noaa_chart(temp_f: float, humidity: float, expected: float) -> None:
    """docs/03 M21: heat index matches NOAA reference values within tolerance."""
    got = float(heat_index_f(temp_f, humidity))
    assert abs(got - expected) <= NOAA_CHART_TOLERANCE_F, (
        f"{temp_f} degF / {humidity}% RH gives {got:.1f}, the NOAA chart says {expected}"
    )


def test_reported_heat_index_is_consistent_with_its_inputs(model) -> None:
    """The reported heat index must be recomputable from the temperature and humidity.

    Interpolating the heat index onto the 15-minute grid instead of recomputing it would
    break this, because the NOAA regression is non-linear.
    """
    output = model.predict(PredictRequest())
    by_time: dict = {}
    for record in output.results:
        if record.kpi in ("temperature", "heat_index"):
            by_time.setdefault(record.timestamp, {})[record.kpi] = record
    checked = 0
    for values in by_time.values():
        if "temperature" not in values or "heat_index" not in values:
            continue
        humidity = float(values["heat_index"].details["humidity_pct"])
        expected = float(heat_index_c(values["temperature"].value, humidity))
        assert values["heat_index"].value == pytest.approx(expected, abs=0.15)
        checked += 1
    assert checked > 0, "no paired temperature and heat index records to check"


def test_waterlogging_rises_with_rainfall(model) -> None:
    """The curve must be monotonic in rainfall for a fixed zone."""
    zone = model._zones.iloc[0]
    values = [model._waterlogging(rain, zone) for rain in (0, 5, 10, 20, 50, 100)]
    assert values == sorted(values)
    assert all(0.0 <= value <= 100.0 for value in values)


def test_waterlogging_is_higher_for_a_low_lying_zone(model) -> None:
    zones = model._zones
    low = zones.loc[zones["low_lying"]].iloc[0]
    high = zones.loc[~zones["low_lying"]].iloc[0]
    rain = 30.0
    # Compare at equal drainage so the low_lying flag is the only difference.
    high_same_drainage = high.copy()
    high_same_drainage["drainage_capacity_mm_hr"] = low["drainage_capacity_mm_hr"]
    assert model._waterlogging(rain, low) > model._waterlogging(rain, high_same_drainage)


def test_waterlogging_is_near_zero_in_dry_weather(model) -> None:
    """A bare sigmoid would read 50 percent at z=0; the offset is what prevents that."""
    for _, zone in model._zones.iterrows():
        assert model._waterlogging(0.0, zone) < 5.0


def test_better_drainage_lowers_waterlogging(model) -> None:
    zone = model._zones.iloc[0].copy()
    zone["drainage_capacity_mm_hr"] = 10.0
    poor = model._waterlogging(40.0, zone)
    zone["drainage_capacity_mm_hr"] = 60.0
    good = model._waterlogging(40.0, zone)
    assert good < poor


def test_multiplier_coefficients_match_the_shared_assumptions(model) -> None:
    """M21 and the synthetic generator must not drift apart.

    Both compute the arrival and medical multipliers; if the coefficients diverged, a model
    reading M21 would see a different multiplier from the one that shaped the world.
    """
    from twin_common.synthetic.weather import default_weather_params

    shared = default_weather_params()
    for key in (
        "rain_cap_mm_hr",
        "k_rain_arrival",
        "k_rain_speed",
        "k_heat_medical",
        "hi_threshold_c",
    ):
        assert model.params_dict[key] == pytest.approx(shared[key]), (
            f"{key} differs between M21 config.yaml and assumptions.yaml"
        )


def test_multipliers_are_one_in_benign_weather(model) -> None:
    """No rain and no heat means no effect on anything downstream."""
    output = model.predict(PredictRequest())
    rain = {r.timestamp: r.value for r in output.results if r.kpi == "rainfall_intensity"}
    heat = {r.timestamp: r.value for r in output.results if r.kpi == "heat_index"}
    threshold = float(model.require_param("hi_threshold_c"))
    for record in output.results:
        if record.kpi != "weather_arrival_multiplier":
            continue
        if rain.get(record.timestamp, 0.0) == 0.0:
            assert record.value == pytest.approx(1.0)
    for record in output.results:
        if record.kpi != "weather_medical_multiplier":
            continue
        if heat.get(record.timestamp, 0.0) <= threshold:
            assert record.value == pytest.approx(1.0)
