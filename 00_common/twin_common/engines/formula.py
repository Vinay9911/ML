"""Physics and statistical formulas shared across models (docs/05 section 6).

This module exists so a formula is written once and tested once. The heat index in
particular is needed by the synthetic generator (D12 ``heat_index_c``) in Phase 2 and by
M21 in Phase 3; duplicating the NOAA regression in both would guarantee they drift.

Everything here is a pure function of its arguments. Rates, weights and breakpoints are
never hard-coded: the caller passes them in from config (CLAUDE.md hard rule). The only
constants are published algorithm coefficients, which are math, not tunable thresholds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------- temperature units
def c_to_f(celsius: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(celsius, dtype=float) * 9.0 / 5.0 + 32.0


def f_to_c(fahrenheit: ArrayLike) -> NDArray[np.float64]:
    return (np.asarray(fahrenheit, dtype=float) - 32.0) * 5.0 / 9.0


# ------------------------------------------------------------------- NOAA heat index
# Rothfusz regression coefficients, NWS Technical Attachment SR 90-23. Inputs in degF.
_R = (
    -42.379,
    2.04901523,
    10.14333127,
    -0.22475541,
    -0.00683783,
    -0.05481717,
    0.00122874,
    0.00085282,
    -0.00000199,
)


def heat_index_f(
    temp_f: ArrayLike,
    humidity_pct: ArrayLike,
    *,
    apply_adjustments: bool = True,
) -> NDArray[np.float64]:
    """NOAA heat index in degF, following the NWS procedure.

    1. Compute the simple (Steadman-derived) heat index.
    2. If the average of that and the temperature is below 80 degF, keep the simple value -
       the Rothfusz regression is not valid in the cool range.
    3. Otherwise use the Rothfusz regression, plus the NWS low-humidity and high-humidity
       adjustments when ``apply_adjustments`` is set.

    **On the adjustments and the published chart.** The NWS operational algorithm documents
    two corrections, and docs/03 (M21) asks for them. The heat index *chart* on weather.gov
    is, however, the un-adjusted regression: at 80 degF the chart reads 85 / 86 / 86 / 87 for
    RH 85 / 90 / 95 / 100, which matches the raw regression (84.9 / 85.6 / 86.4 / 87.2) and
    not the adjusted values (84.9 / 86.3 / 87.8 / 89.3). The two therefore disagree by up to
    about 2 degF inside one narrow band: RH above 85 percent with the temperature between 80
    and 87 degF. Outside that band, and in the dry band (RH below 13 percent), the
    implementation reproduces every published chart cell to within 0.6 degF.

    The default keeps the adjustments on, because the operational algorithm is what NWS
    issues warnings from and it is the more conservative reading for a crowd-safety system.
    ``apply_adjustments=False`` reproduces the chart exactly, which is what the chart-based
    reference test uses.

    Vectorised, so the generator can convert a whole month at once.
    """
    t = np.asarray(temp_f, dtype=float)
    rh = np.clip(np.asarray(humidity_pct, dtype=float), 0.0, 100.0)
    t, rh = np.broadcast_arrays(t, rh)

    simple = 0.5 * (t + 61.0 + ((t - 68.0) * 1.2) + (rh * 0.094))

    rothfusz = (
        _R[0]
        + _R[1] * t
        + _R[2] * rh
        + _R[3] * t * rh
        + _R[4] * t**2
        + _R[5] * rh**2
        + _R[6] * t**2 * rh
        + _R[7] * t * rh**2
        + _R[8] * t**2 * rh**2
    )

    if apply_adjustments:
        # NWS adjustment 1: very dry air, 80-112 degF.
        dry = (rh < 13.0) & (t >= 80.0) & (t <= 112.0)
        dry_adjust = ((13.0 - rh) / 4.0) * np.sqrt(
            np.clip(17.0 - np.abs(t - 95.0), 0.0, None) / 17.0
        )
        rothfusz = np.where(dry, rothfusz - dry_adjust, rothfusz)

        # NWS adjustment 2: very humid air, 80-87 degF.
        humid = (rh > 85.0) & (t >= 80.0) & (t <= 87.0)
        humid_adjust = ((rh - 85.0) / 10.0) * ((87.0 - t) / 5.0)
        rothfusz = np.where(humid, rothfusz + humid_adjust, rothfusz)

    use_simple = ((simple + t) / 2.0) < 80.0
    return np.where(use_simple, simple, rothfusz)


def heat_index_c(
    temp_c: ArrayLike,
    humidity_pct: ArrayLike,
    *,
    apply_adjustments: bool = True,
) -> NDArray[np.float64]:
    """NOAA heat index in degC. The project-facing entry point (KPI ``heat_index``)."""
    return f_to_c(heat_index_f(c_to_f(temp_c), humidity_pct, apply_adjustments=apply_adjustments))


#: Reference cells read from the published NWS heat index chart, as (temp_F, RH_pct, HI_F).
#: Every cell here is one the chart actually prints, and all lie outside the adjustment
#: bands, so the chart and the operational algorithm agree on them. docs/03 (M21) requires a
#: reference table of known points; this is it, and it is shared so the Phase 3 M21 test and
#: the engine test check the same numbers.
NOAA_CHART_REFERENCE: tuple[tuple[float, float, float], ...] = (
    (80.0, 40.0, 80.0),
    (80.0, 65.0, 82.0),
    (84.0, 60.0, 88.0),
    (86.0, 70.0, 95.0),
    (90.0, 40.0, 91.0),
    (90.0, 55.0, 97.0),
    (90.0, 70.0, 106.0),
    (90.0, 85.0, 117.0),
    (94.0, 50.0, 103.0),
    (100.0, 25.0, 99.0),
    (100.0, 40.0, 109.0),
    (100.0, 55.0, 124.0),
    (104.0, 40.0, 119.0),
    (110.0, 40.0, 136.0),
    (110.0, 45.0, 143.0),
)
#: The chart is printed in whole degrees, so cells are matched to +/- 1.5 degF.
NOAA_CHART_TOLERANCE_F = 1.5


# --------------------------------------------------------------------- probabilities
def logistic(z: ArrayLike) -> NDArray[np.float64]:
    """Numerically stable logistic function on (0, 1)."""
    x = np.asarray(z, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def logistic_pct(z: ArrayLike) -> NDArray[np.float64]:
    """``100 / (1 + exp(-z))`` - the docs/05 section 6 "logistic probability" row.

    Probabilities are expressed in percent (0-100) across this project (docs/02 section 4).
    """
    return 100.0 * logistic(z)


def poisson_at_least_one_pct(rate_per_hr: ArrayLike, hours: float) -> NDArray[np.float64]:
    """``100 * (1 - exp(-lambda * t))``: chance of at least one event in a window.

    Used by M09 for ``predicted_incident_probability``.
    """
    lam = np.clip(np.asarray(rate_per_hr, dtype=float), 0.0, None)
    return 100.0 * (1.0 - np.exp(-lam * float(hours)))


def clip_unit(value: ArrayLike) -> NDArray[np.float64]:
    """Clip a sub-score to [0, 1], as every docs/03 rules card requires."""
    return np.clip(np.asarray(value, dtype=float), 0.0, 1.0)


def normalize_range(
    value: ArrayLike,
    start: float,
    end: float,
) -> NDArray[np.float64]:
    """Map ``[start, end]`` onto ``[0, 1]`` and clip outside it.

    This is the shape every M03-style sub-score uses, e.g.
    ``s_density = (d - d_amber) / (d_critical - d_amber)``. ``start == end`` would divide by
    zero, so it degenerates to a step at ``start``.
    """
    x = np.asarray(value, dtype=float)
    if end == start:
        return (x >= start).astype(float)
    return clip_unit((x - start) / (end - start))


def weighted_score(sub_scores: dict[str, float], weights: dict[str, float]) -> float:
    """``100 * sum(w_i * s_i)`` with the weights checked to sum to 1.

    Used by every composite score KPI (crush_risk_score, security_risk_score,
    fire_risk_score, overall_event_risk). The weight check is what stops a config edit from
    silently rescaling a risk score.
    """
    missing = sorted(set(weights) - set(sub_scores))
    if missing:
        raise ValueError(f"weighted_score: no sub-score supplied for {missing}")
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {total_weight:.6f} ({weights})")
    return 100.0 * sum(weights[key] * float(clip_unit(sub_scores[key])) for key in weights)


# ----------------------------------------------------------------- Indian NAQI (M22)
def naqi_sub_index(
    concentration: float,
    breakpoints: list[list[float]],
    *,
    index_bands: list[list[float]] | None = None,
) -> float:
    """CPCB sub-index by linear interpolation between concentration breakpoints.

    ``breakpoints`` is the concentration range per category, e.g. for PM2.5 24h
    ``[[0,30],[31,60],[61,90],[91,120],[121,250],[251,380]]``; ``index_bands`` are the
    matching index ranges and default to the standard NAQI bands. Above the top breakpoint
    the sub-index is capped at the top of the index scale.

    The breakpoints live in model config; docs/05 section 6 flags them as placeholders to
    verify against CPCB before real use.
    """
    bands = index_bands or [[0, 50], [51, 100], [101, 200], [201, 300], [301, 400], [401, 500]]
    if len(bands) < len(breakpoints):
        raise ValueError("index_bands must cover every concentration breakpoint")
    value = max(0.0, float(concentration))
    for (c_lo, c_hi), (i_lo, i_hi) in zip(breakpoints, bands, strict=False):
        if value <= c_hi:
            span = c_hi - c_lo
            if span <= 0:
                return float(i_lo)
            return float(i_lo) + (float(i_hi) - float(i_lo)) * (value - c_lo) / span
    return float(bands[len(breakpoints) - 1][1])


def naqi(sub_indices: dict[str, float]) -> float:
    """Overall NAQI is the worst sub-index (docs/05 section 6)."""
    if not sub_indices:
        raise ValueError("naqi needs at least one sub-index")
    return max(sub_indices.values())


# ------------------------------------------------------------------ traffic and flow
def bpr_travel_time(
    free_flow_min: ArrayLike,
    volume: ArrayLike,
    capacity: ArrayLike,
    *,
    alpha: float,
    beta: float,
) -> NDArray[np.float64]:
    """BPR volume-delay function ``t = t0 * (1 + alpha * (V/C)^beta)`` (docs/05 section 6).

    ``alpha`` and ``beta`` come from ``assumptions.transport``. Zero capacity means a closed
    link, which yields infinite time; callers treat that as unusable rather than dividing.
    """
    t0 = np.asarray(free_flow_min, dtype=float)
    v = np.clip(np.asarray(volume, dtype=float), 0.0, None)
    c = np.asarray(capacity, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(c > 0, v / c, np.inf)
    return np.where(c > 0, t0 * (1.0 + alpha * ratio**beta), np.inf)


def evacuation_time_min(
    persons_per_exit: dict[str, float],
    exit_widths_m: dict[str, float],
    *,
    specific_flow_p_per_m_s: float,
    premovement_min: float,
) -> float:
    """Flow-model evacuation time (docs/05 section 6).

    ``T = premovement + max_e N_e / (specific_flow * width_e)``, in minutes. An exit with
    zero width is blocked; if every exit is blocked the time is infinite, which is the
    honest answer and what M11 reports as a blocked route.
    """
    if not persons_per_exit:
        return float(premovement_min)
    worst = 0.0
    for exit_id, people in persons_per_exit.items():
        width = float(exit_widths_m.get(exit_id, 0.0))
        if width <= 0:
            if people > 0:
                return float("inf")
            continue
        seconds = float(people) / (specific_flow_p_per_m_s * width)
        worst = max(worst, seconds / 60.0)
    return float(premovement_min) + worst


def inflow_outflow_ratio(
    inflow: ArrayLike,
    outflow: ArrayLike,
    *,
    ratio_max: float,
) -> NDArray[np.float64]:
    """``inflow / outflow``, capped at ``ratio_max`` when outflow is zero (docs/05 section 6)."""
    numerator = np.clip(np.asarray(inflow, dtype=float), 0.0, None)
    denominator = np.asarray(outflow, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), ratio_max)
    return np.clip(ratio, 0.0, ratio_max)
