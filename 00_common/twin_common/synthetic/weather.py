"""Scenario layer over the fetched weather (docs/04 section 5.3).

:mod:`twin_common.real.weather` returns the observed (or modelled) series. This module
applies the S03 and S04 overrides on top and recomputes everything that depends on them, so
the rest of the pipeline only ever sees one consistent weather table.

It also produces the arrival and medical multipliers the crowd model and M21 share, from the
same formulas as the M21 card (docs/03) - defined once here so the generated world and the
model agree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import assumptions
from ..engines.formula import heat_index_c
from ..logging import get_logger
from .grid import in_window_series

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WeatherScenario:
    """The weather table after scenario overrides, plus the derived multipliers."""

    frame: pd.DataFrame
    #: hour timestamp -> arrival multiplier, consumed by the crowd model.
    arrival_multiplier: dict[pd.Timestamp, float]
    #: hour timestamp -> medical rate multiplier, for traceability.
    medical_multiplier: dict[pd.Timestamp, float]
    #: True when an override actually changed the series.
    modified: bool


def apply_weather_overrides(
    weather: pd.DataFrame,
    *,
    overrides: Mapping[str, Any],
    params: Mapping[str, float],
) -> WeatherScenario:
    """Apply S03 / S04 to a weather table and recompute the derived columns.

    Args:
        weather: the D12 table from :mod:`twin_common.real.weather`.
        overrides: merged scenario overrides. Understands ``temperature_offset_c``,
            ``humidity_pct_min``, ``rain_mm_hr`` and the ``window`` they apply in.
        params: the weather-response coefficients (``k_rain_arrival``, ``rain_cap_mm_hr``,
            ``k_heat_medical``, ``hi_threshold_c``, ``k_rain_speed``), from config.

    Returns:
        A :class:`WeatherScenario`. The heat index is always recomputed from the final
        temperature and humidity, so it can never disagree with them.
    """
    out = weather.copy()
    index = pd.DatetimeIndex(out["timestamp"])
    inside = in_window_series(index, overrides.get("window"))
    modified = False

    offset = float(overrides.get("temperature_offset_c", 0.0))
    if offset:
        # S04 is a heatwave, not a three-hour spike: a temperature offset applies all day
        # even when the scenario also carries a window for another effect.
        out["temperature_c"] = out["temperature_c"] + offset
        modified = True
        log.info("weather: temperature offset %+.1f C applied to the whole series", offset)

    humidity_floor = overrides.get("humidity_pct_min")
    if humidity_floor is not None:
        out["humidity_pct"] = out["humidity_pct"].clip(lower=float(humidity_floor))
        modified = True
        log.info("weather: humidity floored at %.0f%%", float(humidity_floor))

    rain_override = overrides.get("rain_mm_hr")
    if rain_override is not None:
        # Rain is a windowed event: S03 is "heavy rain 06:00-09:00".
        out["rain_mm"] = np.where(inside, float(rain_override), out["rain_mm"])
        modified = True
        log.info(
            "weather: rain forced to %.0f mm/hr in %s (%d of %d hours)",
            float(rain_override),
            overrides.get("window") or "the whole series",
            int(inside.sum()),
            len(out),
        )

    # Always recomputed, so temperature, humidity and heat index stay consistent.
    out["heat_index_c"] = heat_index_c(out["temperature_c"], out["humidity_pct"])

    if modified:
        out["source"] = out["source"].astype(str) + "+scenario"

    arrival, medical = weather_multipliers(out, params=params)
    return WeatherScenario(
        frame=out,
        arrival_multiplier=arrival,
        medical_multiplier=medical,
        modified=modified,
    )


def weather_multipliers(
    weather: pd.DataFrame,
    *,
    params: Mapping[str, float],
) -> tuple[dict[pd.Timestamp, float], dict[pd.Timestamp, float]]:
    """The arrival and medical multipliers of the docs/03 M21 card.

    ``arrival = 1 - k_rain_arrival * min(rain, rain_cap) / rain_cap``
    ``medical = 1 + k_heat * max(0, heat_index - hi_threshold)``

    Both are returned as {hour timestamp: value} so callers can align them to any grid.
    """
    rain_cap = float(params["rain_cap_mm_hr"])
    k_rain_arrival = float(params["k_rain_arrival"])
    k_heat = float(params["k_heat_medical"])
    hi_threshold = float(params["hi_threshold_c"])

    rain = weather["rain_mm"].to_numpy()
    heat = weather["heat_index_c"].to_numpy()
    arrival = 1.0 - k_rain_arrival * np.minimum(rain, rain_cap) / max(rain_cap, 1e-9)
    medical = 1.0 + k_heat * np.clip(heat - hi_threshold, 0.0, None)
    stamps = list(weather["timestamp"])
    return (
        dict(zip(stamps, arrival.tolist(), strict=True)),
        dict(zip(stamps, medical.tolist(), strict=True)),
    )


def default_weather_params() -> dict[str, float]:
    """Weather-response coefficients from ``assumptions.yaml``.

    They live under ``assumptions.weather_response`` so the generator and M21 read the same
    values rather than each keeping its own copy.
    """
    response = assumptions().get("weather_response") or {}
    medical = assumptions()["medical"]
    return {
        "rain_cap_mm_hr": float(response["rain_cap_mm_hr"]),
        "k_rain_arrival": float(response["k_rain_arrival"]),
        "k_rain_speed": float(response["k_rain_speed"]),
        "k_heat_medical": float(medical["heat_rate_increase_per_c"]),
        "hi_threshold_c": float(medical["heat_index_threshold_c"]),
        "waterlogging_k": float(response["waterlogging_k"]),
        "waterlogging_low_lying_bonus": float(response["waterlogging_low_lying_bonus"]),
    }
