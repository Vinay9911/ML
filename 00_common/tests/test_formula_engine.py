"""The shared formula engine (docs/05 section 6).

The heat index gets the most attention because M21 depends on it and docs/03 requires it to
be checked against NOAA reference values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from twin_common.engines.formula import (
    NOAA_CHART_REFERENCE,
    NOAA_CHART_TOLERANCE_F,
    bpr_travel_time,
    c_to_f,
    clip_unit,
    evacuation_time_min,
    f_to_c,
    heat_index_c,
    heat_index_f,
    inflow_outflow_ratio,
    logistic,
    logistic_pct,
    naqi,
    naqi_sub_index,
    normalize_range,
    poisson_at_least_one_pct,
    weighted_score,
)


# --------------------------------------------------------------------- unit conversion
def test_temperature_round_trip() -> None:
    for celsius in (-10.0, 0.0, 25.0, 37.0, 50.0):
        assert f_to_c(c_to_f(celsius)) == pytest.approx(celsius)


def test_known_conversions() -> None:
    assert float(c_to_f(0)) == pytest.approx(32.0)
    assert float(c_to_f(100)) == pytest.approx(212.0)
    assert float(f_to_c(98.6)) == pytest.approx(37.0)


# ------------------------------------------------------------------------- heat index
@pytest.mark.parametrize(("temp_f", "humidity", "expected"), NOAA_CHART_REFERENCE)
def test_heat_index_matches_the_noaa_chart(temp_f: float, humidity: float, expected: float) -> None:
    """docs/03 M21: the heat index must match NOAA reference values within tolerance."""
    got = float(heat_index_f(temp_f, humidity))
    assert abs(got - expected) <= NOAA_CHART_TOLERANCE_F, (
        f"heat index at {temp_f} degF / {humidity}% RH is {got:.1f}, chart says {expected}"
    )


def test_chart_mode_reproduces_the_published_80f_row_exactly() -> None:
    """The weather.gov chart is the un-adjusted regression; ``apply_adjustments=False`` matches.

    This row is the one place the chart and the operational algorithm disagree, so it is
    pinned in both modes to document which is which.
    """
    published = {85: 85, 90: 86, 95: 86, 100: 87}
    for humidity, expected in published.items():
        chart = float(heat_index_f(80, humidity, apply_adjustments=False))
        assert abs(chart - expected) <= NOAA_CHART_TOLERANCE_F, (
            f"chart mode at 80 degF / {humidity}% gives {chart:.1f}, expected {expected}"
        )


def test_humid_adjustment_raises_the_value_in_its_band() -> None:
    """The adjustment is additive and confined to RH > 85 with 80 <= T <= 87."""
    inside_adjusted = float(heat_index_f(82, 95))
    inside_raw = float(heat_index_f(82, 95, apply_adjustments=False))
    assert inside_adjusted > inside_raw
    # Outside the band the two modes agree.
    assert float(heat_index_f(95, 60)) == pytest.approx(
        float(heat_index_f(95, 60, apply_adjustments=False))
    )
    assert float(heat_index_f(82, 70)) == pytest.approx(
        float(heat_index_f(82, 70, apply_adjustments=False))
    )


def test_dry_adjustment_lowers_the_value_in_its_band() -> None:
    assert float(heat_index_f(95, 10)) < float(heat_index_f(95, 10, apply_adjustments=False))


def test_humid_air_feels_hotter_than_the_thermometer() -> None:
    """Above roughly 55 percent RH the heat index exceeds the air temperature."""
    for temp_f in (85.0, 90.0, 95.0, 100.0, 105.0):
        for humidity in (60.0, 70.0, 80.0, 90.0):
            assert float(heat_index_f(temp_f, humidity)) > temp_f


def test_dry_air_feels_cooler_than_the_thermometer() -> None:
    """Evaporative cooling is real, and the published chart shows it.

    weather.gov reads 84 at 85 degF / 40% RH - below the air temperature. The crossover sits
    near 45 percent RH, which is why the dry-air NWS adjustment subtracts rather than adds.
    A test that demanded heat index >= temperature everywhere would be asserting bad physics.
    """
    for temp_f in (85.0, 90.0, 95.0):
        assert float(heat_index_f(temp_f, 20.0)) < temp_f
        assert float(heat_index_f(temp_f, 30.0)) < temp_f
    assert float(heat_index_f(85.0, 40.0)) == pytest.approx(84.0, abs=1.5)


def test_heat_index_rises_with_humidity() -> None:
    values = [float(heat_index_f(95, rh)) for rh in (20, 30, 40, 50, 60, 70)]
    assert values == sorted(values)


def test_heat_index_rises_with_temperature() -> None:
    values = [float(heat_index_f(t, 60)) for t in (80, 85, 90, 95, 100)]
    assert values == sorted(values)


def test_cool_conditions_use_the_simple_formula() -> None:
    """Below the regression range the value must stay close to the air temperature."""
    assert float(heat_index_f(60, 50)) == pytest.approx(60.0, abs=4.0)
    assert float(heat_index_c(20, 50)) == pytest.approx(20.0, abs=3.0)


def test_heat_index_is_vectorised() -> None:
    temps = np.array([80.0, 90.0, 100.0])
    humidity = np.array([40.0, 55.0, 40.0])
    got = heat_index_f(temps, humidity)
    assert got.shape == (3,)
    assert got == pytest.approx([79.6, 97.0, 109.3], abs=0.5)


def test_heat_index_broadcasts_a_scalar_humidity() -> None:
    got = heat_index_c(np.array([30.0, 35.0, 40.0]), 60.0)
    assert got.shape == (3,)
    assert (np.diff(got) > 0).all()


def test_humidity_is_clamped_to_a_valid_range() -> None:
    assert float(heat_index_f(90, 150)) == pytest.approx(float(heat_index_f(90, 100)))
    assert float(heat_index_f(90, -5)) == pytest.approx(float(heat_index_f(90, 0)))


def test_celsius_entry_point_agrees_with_fahrenheit() -> None:
    celsius, humidity = 35.0, 65.0
    assert float(heat_index_c(celsius, humidity)) == pytest.approx(
        float(f_to_c(heat_index_f(c_to_f(celsius), humidity)))
    )


# ------------------------------------------------------------------------ probabilities
def test_logistic_is_centred_and_bounded() -> None:
    assert float(logistic(0.0)) == pytest.approx(0.5)
    assert 0.0 < float(logistic(-50.0)) < 1e-10
    assert 1.0 - 1e-10 < float(logistic(50.0)) <= 1.0


def test_logistic_is_stable_at_large_magnitudes() -> None:
    """The naive form overflows at -800; the stable branch must not."""
    for z in (-800.0, -100.0, 100.0, 800.0):
        value = float(logistic(z))
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


def test_logistic_pct_is_a_percentage() -> None:
    assert float(logistic_pct(0.0)) == pytest.approx(50.0)
    assert 0.0 <= float(logistic_pct(-20.0)) <= 100.0


def test_poisson_at_least_one() -> None:
    assert float(poisson_at_least_one_pct(0.0, 1.0)) == pytest.approx(0.0)
    assert float(poisson_at_least_one_pct(1.0, 1.0)) == pytest.approx(63.21, abs=0.01)
    # Monotonic in both the rate and the window.
    assert poisson_at_least_one_pct(2.0, 1.0) > poisson_at_least_one_pct(1.0, 1.0)
    assert poisson_at_least_one_pct(1.0, 2.0) > poisson_at_least_one_pct(1.0, 1.0)


def test_negative_rate_is_treated_as_zero() -> None:
    assert float(poisson_at_least_one_pct(-3.0, 1.0)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- sub-scores
def test_clip_unit() -> None:
    assert clip_unit([-1.0, 0.0, 0.5, 1.0, 2.0]) == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


def test_normalize_range_is_the_m03_sub_score_shape() -> None:
    """s_density = (d - d_amber) / (d_critical - d_amber), clipped to [0,1]."""
    assert float(normalize_range(2.0, 2.0, 5.0)) == pytest.approx(0.0)
    assert float(normalize_range(3.5, 2.0, 5.0)) == pytest.approx(0.5)
    assert float(normalize_range(5.0, 2.0, 5.0)) == pytest.approx(1.0)
    assert float(normalize_range(9.0, 2.0, 5.0)) == pytest.approx(1.0)
    assert float(normalize_range(0.5, 2.0, 5.0)) == pytest.approx(0.0)


def test_normalize_range_is_monotonic() -> None:
    values = [float(normalize_range(d, 2.0, 5.0)) for d in (1, 2, 3, 4, 5, 6)]
    assert values == sorted(values)


def test_normalize_range_with_equal_bounds_is_a_step() -> None:
    assert float(normalize_range(1.9, 2.0, 2.0)) == pytest.approx(0.0)
    assert float(normalize_range(2.0, 2.0, 2.0)) == pytest.approx(1.0)


def test_weighted_score() -> None:
    score = weighted_score(
        {"density": 1.0, "trend": 0.0, "heat": 0.5},
        {"density": 0.5, "trend": 0.3, "heat": 0.2},
    )
    assert score == pytest.approx(60.0)


def test_weighted_score_rejects_weights_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        weighted_score({"a": 1.0, "b": 1.0}, {"a": 0.5, "b": 0.6})


def test_weighted_score_reports_a_missing_sub_score() -> None:
    with pytest.raises(ValueError, match="no sub-score supplied"):
        weighted_score({"a": 1.0}, {"a": 0.5, "b": 0.5})


def test_weighted_score_clips_out_of_range_sub_scores() -> None:
    assert weighted_score({"a": 5.0}, {"a": 1.0}) == pytest.approx(100.0)
    assert weighted_score({"a": -2.0}, {"a": 1.0}) == pytest.approx(0.0)


# ------------------------------------------------------------------------------- NAQI
PM25_BREAKPOINTS = [[0, 30], [31, 60], [61, 90], [91, 120], [121, 250], [251, 380]]


@pytest.mark.parametrize(
    ("concentration", "expected"),
    [(0, 0), (30, 50), (60, 100), (90, 200), (120, 300), (250, 400), (380, 500)],
)
def test_naqi_sub_index_hits_the_breakpoints(concentration: float, expected: float) -> None:
    """docs/05 section 6: piecewise-linear interpolation must be exact at the breakpoints."""
    assert naqi_sub_index(concentration, PM25_BREAKPOINTS) == pytest.approx(expected, abs=2.0)


def test_naqi_sub_index_interpolates_inside_a_band() -> None:
    midpoint = naqi_sub_index(15, PM25_BREAKPOINTS)
    assert 20 < midpoint < 30


def test_naqi_sub_index_is_monotonic() -> None:
    values = [naqi_sub_index(c, PM25_BREAKPOINTS) for c in range(0, 400, 20)]
    assert values == sorted(values)


def test_naqi_sub_index_caps_above_the_top_breakpoint() -> None:
    assert naqi_sub_index(2000, PM25_BREAKPOINTS) == pytest.approx(500.0)


def test_naqi_is_the_worst_sub_index() -> None:
    assert naqi({"pm2_5": 120.0, "pm10": 80.0, "no2": 40.0}) == pytest.approx(120.0)


def test_naqi_needs_at_least_one_sub_index() -> None:
    with pytest.raises(ValueError, match="at least one"):
        naqi({})


# ---------------------------------------------------------------------------- traffic
def test_bpr_at_free_flow() -> None:
    assert float(bpr_travel_time(10.0, 0.0, 1800.0, alpha=0.15, beta=4)) == pytest.approx(10.0)


def test_bpr_at_capacity_adds_alpha() -> None:
    """At V/C == 1 the travel time is t0 * (1 + alpha)."""
    assert float(bpr_travel_time(10.0, 1800.0, 1800.0, alpha=0.15, beta=4)) == pytest.approx(11.5)


def test_bpr_grows_with_volume() -> None:
    times = [
        float(bpr_travel_time(10.0, v, 1800.0, alpha=0.15, beta=4)) for v in (0, 900, 1800, 2700)
    ]
    assert times == sorted(times)
    assert times[-1] > times[0]


def test_bpr_closed_link_is_infinite() -> None:
    """A closed link (capacity 0) must be unusable, not a divide-by-zero."""
    assert math.isinf(float(bpr_travel_time(10.0, 100.0, 0.0, alpha=0.15, beta=4)))


def test_bpr_is_vectorised() -> None:
    got = bpr_travel_time([10.0, 20.0], [1800.0, 900.0], [1800.0, 1800.0], alpha=0.15, beta=4)
    assert got.shape == (2,)


# -------------------------------------------------------------------------- evacuation
def test_evacuation_flow_model() -> None:
    """T = premovement + max_e N_e / (specific_flow * width_e)."""
    time_min = evacuation_time_min(
        {"E01": 1000.0},
        {"E01": 8.0},
        specific_flow_p_per_m_s=1.3,
        premovement_min=2.0,
    )
    # 1000 / (1.3 * 8) = 96.15 s = 1.6 min, plus 2 min premovement.
    assert time_min == pytest.approx(2.0 + 1000.0 / (1.3 * 8.0) / 60.0)


def test_evacuation_uses_the_worst_exit() -> None:
    time_min = evacuation_time_min(
        {"E01": 1000.0, "E02": 5000.0},
        {"E01": 8.0, "E02": 4.0},
        specific_flow_p_per_m_s=1.3,
        premovement_min=2.0,
    )
    assert time_min == pytest.approx(2.0 + 5000.0 / (1.3 * 4.0) / 60.0)


def test_blocking_an_exit_increases_evacuation_time() -> None:
    """docs/07 Phase 8 acceptance, checked at the formula level."""
    people = {"E01": 2000.0, "E02": 2000.0}
    both_open = evacuation_time_min(
        people,
        {"E01": 8.0, "E02": 8.0},
        specific_flow_p_per_m_s=1.3,
        premovement_min=2.0,
    )
    one_narrow = evacuation_time_min(
        people,
        {"E01": 8.0, "E02": 2.0},
        specific_flow_p_per_m_s=1.3,
        premovement_min=2.0,
    )
    assert one_narrow > both_open


def test_fully_blocked_exit_with_people_is_infinite() -> None:
    assert math.isinf(
        evacuation_time_min(
            {"E03": 500.0},
            {"E03": 0.0},
            specific_flow_p_per_m_s=1.3,
            premovement_min=2.0,
        )
    )


def test_blocked_exit_with_nobody_assigned_is_fine() -> None:
    time_min = evacuation_time_min(
        {"E01": 100.0, "E03": 0.0},
        {"E01": 8.0, "E03": 0.0},
        specific_flow_p_per_m_s=1.3,
        premovement_min=2.0,
    )
    assert math.isfinite(time_min)


def test_doubling_agents_increases_time() -> None:
    small = evacuation_time_min(
        {"E01": 1000.0}, {"E01": 8.0}, specific_flow_p_per_m_s=1.3, premovement_min=2.0
    )
    large = evacuation_time_min(
        {"E01": 2000.0}, {"E01": 8.0}, specific_flow_p_per_m_s=1.3, premovement_min=2.0
    )
    assert large > small


def test_empty_zone_is_just_premovement() -> None:
    assert evacuation_time_min(
        {}, {}, specific_flow_p_per_m_s=1.3, premovement_min=2.0
    ) == pytest.approx(2.0)


# ------------------------------------------------------------------- inflow / outflow
def test_inflow_outflow_ratio() -> None:
    assert float(inflow_outflow_ratio(100.0, 50.0, ratio_max=10.0)) == pytest.approx(2.0)
    assert float(inflow_outflow_ratio(50.0, 100.0, ratio_max=10.0)) == pytest.approx(0.5)


def test_zero_outflow_is_capped_not_infinite() -> None:
    """docs/05 section 6: guard outflow = 0 by capping at ratio_max."""
    assert float(inflow_outflow_ratio(100.0, 0.0, ratio_max=10.0)) == pytest.approx(10.0)


def test_ratio_is_capped_at_ratio_max() -> None:
    assert float(inflow_outflow_ratio(10_000.0, 1.0, ratio_max=10.0)) == pytest.approx(10.0)
