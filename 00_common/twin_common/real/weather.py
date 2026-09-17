"""Open-Meteo weather and air quality, with a cache and a fallback chain (docs/04 section 7).

Resolution order, so a fresh clone works with no network:

1. ``data/real_cache/weather_<year>.parquet`` - a previous download
2. ``data/bundled/weather_<year>.parquet`` - a small sample committed to git
3. the Open-Meteo archive API (skipped entirely when ``TWIN_OFFLINE=1``), then cached
4. a synthetic diurnal model with monsoon rain events

Rows that came from the API carry ``is_synthetic=False`` and ``source="open-meteo"``; the
synthetic fallback carries ``True`` and ``source="synthetic_diurnal"``. Nothing downstream
has to know which one it got, but the flag travels with every row (CLAUDE.md hard rule).

Licence: the Open-Meteo free tier is non-commercial and asks for CC BY 4.0 attribution
(docs/06 section 3). The attribution string is recorded in the world manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts.models import IST
from ..engines.formula import heat_index_c
from ..logging import get_logger
from ..paths import bundled_dir, ensure_dir, is_offline, real_cache_dir

log = get_logger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

ATTRIBUTION = "Weather and air quality data by Open-Meteo.com, CC BY 4.0 (non-commercial tier)"

#: Hourly variables requested from the archive API, mapped to our D12 column names.
WEATHER_VARIABLES: dict[str, str] = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "precipitation": "rain_mm",
    "wind_speed_10m": "wind_kmh",
    "visibility": "visibility_m",
}
#: Hourly variables requested from the air-quality API, mapped to D13 column names.
AIR_QUALITY_VARIABLES: dict[str, str] = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "carbon_monoxide": "co",
}

REQUEST_TIMEOUT_S = 20.0


@dataclass(frozen=True, slots=True)
class FetchResult:
    """A fetched table plus where it actually came from."""

    frame: pd.DataFrame
    source: str
    is_real: bool

    @property
    def degraded(self) -> bool:
        """True when a fallback was used rather than the real source."""
        return not self.is_real


# ------------------------------------------------------------------------ synthetic model
def _synthetic_weather(
    days: list[date],
    rng: np.random.Generator,
    *,
    base_temp_c: float = 28.0,
    diurnal_amplitude_c: float = 6.0,
    base_humidity_pct: float = 62.0,
    monsoon_event_probability: float = 0.28,
) -> pd.DataFrame:
    """A plausible monsoon-season diurnal model for the venue latitude.

    The shape matters more than the exact numbers: a daily temperature cycle peaking in the
    afternoon, humidity moving opposite to temperature, and occasional multi-hour rain
    events. Defaults are only used when no real data is available, and every row produced
    here is flagged synthetic.
    """
    rows: list[dict[str, float | pd.Timestamp]] = []
    for day in days:
        # A slow seasonal drift plus day-to-day variation.
        day_offset = rng.normal(0.0, 1.4)
        rain_today = rng.random() < monsoon_event_probability
        rain_start = int(rng.integers(11, 20)) if rain_today else -1
        rain_hours = int(rng.integers(1, 5)) if rain_today else 0
        rain_peak = float(rng.gamma(shape=2.0, scale=3.5)) if rain_today else 0.0
        for hour in range(24):
            # Peak at 15:00, trough at 05:00.
            cycle = -np.cos(2.0 * np.pi * (hour - 5) / 24.0)
            temperature = base_temp_c + day_offset + diurnal_amplitude_c * cycle
            humidity = float(
                np.clip(base_humidity_pct - 14.0 * cycle + rng.normal(0.0, 3.0), 25.0, 99.0)
            )
            if rain_today and rain_start <= hour < rain_start + rain_hours:
                # Triangular intensity over the event.
                position = (hour - rain_start + 0.5) / max(1, rain_hours)
                rain = rain_peak * (1.0 - abs(2.0 * position - 1.0))
                humidity = float(np.clip(humidity + 18.0, 25.0, 99.0))
                temperature -= 2.5
            else:
                rain = 0.0
            wind = float(np.clip(rng.gamma(shape=2.2, scale=3.0), 0.0, 45.0))
            visibility = 10000.0 - min(8000.0, rain * 350.0) - max(0.0, humidity - 80.0) * 40.0
            rows.append(
                {
                    "timestamp": pd.Timestamp(
                        year=day.year, month=day.month, day=day.day, hour=hour, tz=IST
                    ),
                    "temperature_c": float(temperature),
                    "humidity_pct": humidity,
                    "rain_mm": float(rain),
                    "wind_kmh": wind,
                    "visibility_m": float(max(500.0, visibility)),
                }
            )
    return pd.DataFrame(rows)


def _synthetic_air_quality(
    days: list[date],
    rng: np.random.Generator,
    *,
    base_pm25: float = 38.0,
    base_pm10: float = 78.0,
) -> pd.DataFrame:
    """Background air quality with a morning and evening peak (cooking and traffic)."""
    rows: list[dict[str, float | pd.Timestamp]] = []
    for day in days:
        day_factor = float(np.clip(rng.normal(1.0, 0.18), 0.5, 1.9))
        for hour in range(24):
            # Twin peaks around 08:00 and 20:00, cleanest mid-afternoon.
            peak = (
                1.0
                + 0.45 * np.exp(-(((hour - 8) / 2.5) ** 2))
                + 0.55 * np.exp(-(((hour - 20) / 3.0) ** 2))
            )
            pm25 = base_pm25 * day_factor * peak * float(np.clip(rng.normal(1.0, 0.1), 0.6, 1.5))
            pm10 = base_pm10 * day_factor * peak * float(np.clip(rng.normal(1.0, 0.1), 0.6, 1.5))
            rows.append(
                {
                    "timestamp": pd.Timestamp(
                        year=day.year, month=day.month, day=day.day, hour=hour, tz=IST
                    ),
                    "pm2_5": float(max(2.0, pm25)),
                    "pm10": float(max(4.0, pm10)),
                    "no2": float(max(1.0, 18.0 * day_factor * peak + rng.normal(0, 3))),
                    "co": float(max(50.0, 420.0 * day_factor * peak + rng.normal(0, 40))),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ fetching
def _fetch_open_meteo(
    url: str,
    *,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    variables: dict[str, str],
) -> pd.DataFrame:
    """One archive API call. Raises on any problem so the caller can fall back."""
    import httpx

    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(variables),
        "timezone": "Asia/Kolkata",
    }
    log.info("fetching %s for %s..%s", url, start, end)
    response = httpx.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError(f"unexpected Open-Meteo payload keys: {sorted(payload)}")
    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(IST)
    for api_name, column in variables.items():
        series = hourly.get(api_name)
        frame[column] = pd.Series(series, dtype="float64") if series is not None else np.nan
    return frame


def _shift_year(frame: pd.DataFrame, source_year: int, target_year: int) -> pd.DataFrame:
    """Move a fetched series from the source year to the event year (docs/04 section 3).

    Shifting by whole days keeps the hour-of-day alignment, which is what the diurnal
    pattern depends on. A leap-day mismatch is absorbed by the day offset.
    """
    if source_year == target_year:
        return frame
    out = frame.copy()
    offset = pd.Timestamp(year=target_year, month=1, day=1, tz=IST) - pd.Timestamp(
        year=source_year, month=1, day=1, tz=IST
    )
    out["timestamp"] = out["timestamp"] + offset
    return out


def _load_cached(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # a truncated cache file should not be fatal
        log.warning("ignoring unreadable cache %s: %s", path, exc)
        return None
    log.info("using cached %s (%d rows)", path.name, len(frame))
    return frame


def _finalize_weather(frame: pd.DataFrame, *, source: str, is_real: bool) -> pd.DataFrame:
    """Add the derived heat index and the provenance columns (D12 schema)."""
    out = frame.copy()
    for column in ("temperature_c", "humidity_pct", "rain_mm", "wind_kmh"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    # A gap in the API series must not propagate as NaN into every downstream KPI.
    out[["temperature_c", "humidity_pct", "rain_mm", "wind_kmh"]] = (
        out[["temperature_c", "humidity_pct", "rain_mm", "wind_kmh"]].ffill().bfill()
    )
    if "visibility_m" not in out.columns:
        out["visibility_m"] = np.nan
    out["heat_index_c"] = heat_index_c(out["temperature_c"], out["humidity_pct"])
    out["is_synthetic"] = not is_real
    out["source"] = source
    return out.sort_values("timestamp").reset_index(drop=True)


def _finalize_air_quality(
    frame: pd.DataFrame,
    *,
    source: str,
    is_real: bool,
    noise_base_db: float,
) -> pd.DataFrame:
    """Add the noise column and provenance (D13 schema).

    Open-Meteo has no noise variable, so ``noise_db`` is always modelled: the configured
    base level plus a traffic-correlated term derived from NO2. M22 refines it with crowd
    density; this is the ambient background.
    """
    out = frame.copy()
    for column in ("pm2_5", "pm10", "no2", "co"):
        out[column] = pd.to_numeric(out[column], errors="coerce").ffill().bfill()
    out["noise_db"] = noise_base_db + 6.0 * np.log10(1.0 + out["no2"] / 20.0)
    out["is_synthetic"] = not is_real
    out["source"] = source
    return out.sort_values("timestamp").reset_index(drop=True)


def get_weather_hourly(
    days: list[date],
    *,
    latitude: float,
    longitude: float,
    source_year: int,
    seed: int = 42,
    use_cache: bool = True,
) -> FetchResult:
    """Hourly weather for ``days``, from cache, the API, or the synthetic model.

    ``days`` are event-year dates. The API is asked for the same calendar days of
    ``source_year`` and the result is shifted forward, per docs/04 section 3.
    """
    target_year = days[0].year
    cache_path = real_cache_dir() / f"weather_{source_year}.parquet"
    bundled_path = bundled_dir() / f"weather_{source_year}.parquet"
    rng = np.random.default_rng(seed)

    cached = _load_cached(cache_path) if use_cache else None
    if cached is not None:
        shifted = _shift_year(cached, source_year, target_year)
        return FetchResult(
            _finalize_weather(shifted, source="open-meteo (cached)", is_real=True),
            source="open-meteo (cached)",
            is_real=True,
        )

    bundled = _load_cached(bundled_path) if use_cache else None
    if bundled is not None:
        shifted = _shift_year(bundled, source_year, target_year)
        return FetchResult(
            _finalize_weather(shifted, source="open-meteo (bundled sample)", is_real=True),
            source="open-meteo (bundled sample)",
            is_real=True,
        )

    if not is_offline():
        source_start = date(source_year, days[0].month, days[0].day)
        source_end = source_start + timedelta(days=len(days) - 1)
        try:
            fetched = _fetch_open_meteo(
                ARCHIVE_URL,
                latitude=latitude,
                longitude=longitude,
                start=source_start,
                end=source_end,
                variables=WEATHER_VARIABLES,
            )
            ensure_dir(cache_path.parent)
            fetched.to_parquet(cache_path, index=False)
            log.info("cached %d weather rows to %s", len(fetched), cache_path)
            shifted = _shift_year(fetched, source_year, target_year)
            return FetchResult(
                _finalize_weather(shifted, source="open-meteo", is_real=True),
                source="open-meteo",
                is_real=True,
            )
        except Exception as exc:  # any network or payload problem falls back
            log.warning("Open-Meteo weather fetch failed (%s); using the synthetic model", exc)
    else:
        log.info("TWIN_OFFLINE is set; skipping the Open-Meteo weather request")

    synthetic = _synthetic_weather(days, rng)
    return FetchResult(
        _finalize_weather(synthetic, source="synthetic_diurnal", is_real=False),
        source="synthetic_diurnal",
        is_real=False,
    )


def get_air_quality_hourly(
    days: list[date],
    *,
    latitude: float,
    longitude: float,
    source_year: int,
    noise_base_db: float,
    seed: int = 42,
    use_cache: bool = True,
) -> FetchResult:
    """Hourly air quality for ``days``, same resolution order as the weather."""
    target_year = days[0].year
    cache_path = real_cache_dir() / f"aq_{source_year}.parquet"
    bundled_path = bundled_dir() / f"aq_{source_year}.parquet"
    rng = np.random.default_rng(seed + 1)

    for path, label in (
        (cache_path, "open-meteo-aq (cached)"),
        (bundled_path, "open-meteo-aq (bundled sample)"),
    ):
        if not use_cache:
            break
        cached = _load_cached(path)
        if cached is not None:
            shifted = _shift_year(cached, source_year, target_year)
            return FetchResult(
                _finalize_air_quality(
                    shifted, source=label, is_real=True, noise_base_db=noise_base_db
                ),
                source=label,
                is_real=True,
            )

    if not is_offline():
        source_start = date(source_year, days[0].month, days[0].day)
        source_end = source_start + timedelta(days=len(days) - 1)
        try:
            fetched = _fetch_open_meteo(
                AIR_QUALITY_URL,
                latitude=latitude,
                longitude=longitude,
                start=source_start,
                end=source_end,
                variables=AIR_QUALITY_VARIABLES,
            )
            ensure_dir(cache_path.parent)
            fetched.to_parquet(cache_path, index=False)
            log.info("cached %d air-quality rows to %s", len(fetched), cache_path)
            shifted = _shift_year(fetched, source_year, target_year)
            return FetchResult(
                _finalize_air_quality(
                    shifted, source="open-meteo-aq", is_real=True, noise_base_db=noise_base_db
                ),
                source="open-meteo-aq",
                is_real=True,
            )
        except Exception as exc:
            log.warning("Open-Meteo air-quality fetch failed (%s); using the synthetic model", exc)
    else:
        log.info("TWIN_OFFLINE is set; skipping the Open-Meteo air-quality request")

    synthetic = _synthetic_air_quality(days, rng)
    return FetchResult(
        _finalize_air_quality(
            synthetic, source="synthetic_aq", is_real=False, noise_base_db=noise_base_db
        ),
        source="synthetic_aq",
        is_real=False,
    )


def write_bundled_samples(
    days: list[date], *, latitude: float, longitude: float, source_year: int
) -> list[Path]:
    """Fetch once and write the committed offline fallbacks in ``data/bundled/``.

    Run this deliberately, with a network, to refresh the samples a fresh clone relies on.
    """
    if is_offline():
        raise RuntimeError("cannot refresh bundled samples while TWIN_OFFLINE is set")
    written: list[Path] = []
    source_start = date(source_year, days[0].month, days[0].day)
    source_end = source_start + timedelta(days=len(days) - 1)
    ensure_dir(bundled_dir())
    for url, variables, name in (
        (ARCHIVE_URL, WEATHER_VARIABLES, f"weather_{source_year}.parquet"),
        (AIR_QUALITY_URL, AIR_QUALITY_VARIABLES, f"aq_{source_year}.parquet"),
    ):
        frame = _fetch_open_meteo(
            url,
            latitude=latitude,
            longitude=longitude,
            start=source_start,
            end=source_end,
            variables=variables,
        )
        path = bundled_dir() / name
        frame.to_parquet(path, index=False)
        written.append(path)
        log.info("wrote bundled sample %s (%d rows)", path, len(frame))
    return written
