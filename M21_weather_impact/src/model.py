"""M21 - Weather Impact.

What is the weather now and over the next hours, and how does it change risk elsewhere?

M21 is the root of the dependency graph: it has no upstream, and M01, M03, M07, M12, M15,
M19, M22 and M25 all read it. Its job is to turn a weather series into the handful of
numbers the rest of the twin reasons about.

**Method** (docs/03 M21 card):

``temperature``, ``rainfall_intensity``
    Read from D12 ``weather_hourly`` - cached Open-Meteo history for the venue, interpolated
    from the hourly series onto the 15-minute grid.

``heat_index``
    The NOAA algorithm, from :func:`twin_common.engines.formula.heat_index_c`. Shared with
    the synthetic generator so the world and the model can never disagree; validated against
    15 published NWS chart cells.

``waterlogging_probability``
    Per zone: ``100 * sigmoid(k * (rain - drainage_capacity) + bonus * low_lying)``. A zone
    drains at its own configured rate, so the same rainfall floods a low-lying ghat and not
    the hub.

``weather_arrival_multiplier``, ``weather_medical_multiplier``, ``weather_traffic_speed_multiplier``
    The three couplings the rest of the twin consumes: rain suppresses arrivals and speeds,
    heat raises medical demand. Computed by
    :func:`twin_common.synthetic.weather.weather_multipliers`, which is the same function
    the generator used, so a downstream model reading M21 sees exactly the multiplier that
    shaped the world it is reading.

"Forecast" here is a replay of the stored series rather than a live prediction - the demo
world already contains the whole event window (docs/03 M21).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from twin_common.contracts import Metadata, ModelOutput, PredictRequest, ScenarioRequest
from twin_common.contracts import registry as reg
from twin_common.contracts.enums import EntityType, State
from twin_common.engines.formula import heat_index_c, logistic_pct
from twin_common.io.tables import load_table
from twin_common.logging import get_logger
from twin_common.model import TwinModel
from twin_common.output import OutputBuilder
from twin_common.synthetic.weather import apply_weather_overrides, weather_multipliers

log = get_logger(__name__)

#: Reason codes M21 can raise (docs/05 section 4).
REASON_HEAT = "HEAT_STRESS"
REASON_RAIN = "HEAVY_RAIN"
REASON_WATERLOGGING = "WATERLOGGING"
REASON_DRY_HEAT = "HEAT_DRY"


class Model(TwinModel):
    """Weather Impact (M21)."""

    model_id = "M21"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Read D12 once per scenario and cache it on the instance.

        The table is 744 rows, so every supported scenario is loaded up front; that keeps
        ``/scenario`` from touching the disk mid-request.
        """
        self._weather: dict[str, pd.DataFrame] = {}
        self._zones = load_table(
            "zones", self.data_source, scenario_id="S01", base_dir=self.synthetic_dir
        )
        for scenario_id in self.scenarios_supported:
            try:
                self._weather[scenario_id] = load_table(
                    "weather_hourly",
                    self.data_source,
                    scenario_id=scenario_id,
                    base_dir=self.synthetic_dir,
                )
            except Exception as exc:  # a missing slice is not fatal; S01 is the fallback
                log.warning("no weather slice for %s (%s)", scenario_id, exc)
        if "S01" not in self._weather:
            raise RuntimeError(
                "M21 needs at least the S01 weather slice. Run `python scripts/slice_world.py M21`."
            )
        log.info(
            "M21 loaded weather for %s (%d rows each)",
            sorted(self._weather),
            len(self._weather["S01"]),
        )

    # ------------------------------------------------------------------ helpers
    @property
    def params_dict(self) -> dict[str, float]:
        """The response coefficients, all from config.yaml."""
        return {
            "rain_cap_mm_hr": self.require_param("rain_cap_mm_hr"),
            "k_rain_arrival": self.require_param("k_rain_arrival"),
            "k_rain_speed": self.require_param("k_rain_speed"),
            "k_heat_medical": self.require_param("k_heat_medical"),
            "hi_threshold_c": self.require_param("hi_threshold_c"),
        }

    def entity_ids(self) -> list[str]:
        """Zones, for the per-zone waterlogging KPI."""
        return self._zones["zone_id"].tolist()

    def _weather_for(
        self, scenario_id: str, overrides: dict[str, Any] | None
    ) -> tuple[pd.DataFrame, bool]:
        """The weather series for a scenario, and whether a fallback was used.

        A scenario with its own generated world is read directly. Otherwise the S01 series is
        read and the overrides are applied on top with the same function the generator uses,
        so the answer is identical either way.
        """
        if scenario_id in self._weather and not overrides:
            return self._weather[scenario_id], False
        if scenario_id in self._weather:
            return self._weather[scenario_id], False
        adjusted = apply_weather_overrides(
            self._weather["S01"], overrides=overrides or {}, params=self.params_dict
        )
        return adjusted.frame, True

    def _series_at(
        self, weather: pd.DataFrame, as_of: pd.Timestamp, horizon_min: int
    ) -> pd.DataFrame:
        """Rows from ``as_of`` to ``as_of + horizon``, on the 15-minute grid.

        D12 is hourly; the twin runs on 15 minutes, so the series is interpolated rather
        than forward-filled - a step change in temperature every hour would show up as a
        spurious ``density_trend``-style artefact downstream.
        """
        indexed = weather.set_index("timestamp").sort_index()
        end = as_of + timedelta(minutes=max(horizon_min, 0))
        grid = pd.date_range(start=as_of, end=end, freq="15min", tz=as_of.tz)
        numeric = indexed[["temperature_c", "humidity_pct", "rain_mm", "wind_kmh"]].astype(float)
        combined = numeric.reindex(numeric.index.union(grid)).interpolate(
            method="time", limit_direction="both"
        )
        out = combined.reindex(grid)
        # The heat index is recomputed from the interpolated inputs rather than interpolated
        # itself: it is non-linear, so interpolating it would not match its own inputs.
        out["heat_index_c"] = heat_index_c(out["temperature_c"], out["humidity_pct"])
        return out

    def _waterlogging(self, rain_mm_hr: float, zone: pd.Series) -> float:
        """``100 * sigmoid(k * (rain - drainage) + bonus * low_lying)`` (docs/03 M21)."""
        k = float(self.require_param("waterlogging_k"))
        bonus = float(self.require_param("waterlogging_low_lying_bonus"))
        offset = float(self.require_param("waterlogging_offset"))
        drainage = float(zone["drainage_capacity_mm_hr"])
        low_lying = 1.0 if bool(zone["low_lying"]) else 0.0
        z = k * (rain_mm_hr - drainage) + bonus * low_lying + offset
        return float(logistic_pct(z))

    # ------------------------------------------------------------------ compute
    def _compute(
        self,
        request: PredictRequest,
        builder: OutputBuilder,
        *,
        state: State,
        overrides: dict[str, Any] | None = None,
    ) -> OutputBuilder:
        as_of = pd.Timestamp(self.resolve_as_of(request))
        horizon = self.resolve_horizon(request)
        kpis = set(self.resolve_kpis(request))
        zones = self.resolve_entities(request, self.entity_ids())

        weather, used_fallback = self._weather_for(builder.scenario_id, overrides)
        if used_fallback:
            builder.warn(
                f"no generated world for {builder.scenario_id}; the overrides were applied "
                f"to the S01 weather series instead"
            )
        series = self._series_at(weather, as_of, horizon)
        arrival, medical = weather_multipliers(
            series.reset_index(names="timestamp"), params=self.params_dict
        )

        heat_threshold = float(self.require_param("hi_threshold_c"))
        heavy_rain = float(self.require_param("heavy_rain_mm_hr"))
        dry_humidity = float(self.require_param("dry_humidity_pct"))
        waterlogging_alert = float(self.require_param("waterlogging_alert_pct"))
        rain_cap = float(self.require_param("rain_cap_mm_hr"))
        k_rain_speed = float(self.require_param("k_rain_speed"))

        for timestamp, row in series.iterrows():
            offset_min = int((timestamp - as_of).total_seconds() // 60)
            record_state = State.CURRENT if offset_min == 0 else state
            horizon_min = 0 if record_state is State.CURRENT else offset_min

            temperature = float(row["temperature_c"])
            humidity = float(row["humidity_pct"])
            rain = float(row["rain_mm"])
            heat = float(row["heat_index_c"])

            reasons: list[str] = []
            if heat >= heat_threshold:
                reasons.append(REASON_HEAT)
                if humidity <= dry_humidity:
                    reasons.append(REASON_DRY_HEAT)
            if rain >= heavy_rain:
                reasons.append(REASON_RAIN)

            details = {
                "humidity_pct": round(humidity, 1),
                "wind_kmh": round(float(row["wind_kmh"]), 1),
                "source": "replay of the stored weather series",
            }

            if "temperature" in kpis:
                builder.add(
                    EntityType.EVENT,
                    "EVENT",
                    "temperature",
                    timestamp,
                    temperature,
                    horizon_min=horizon_min,
                    state=record_state,
                    confidence=self.require_param("confidence"),
                    reason_codes=[r for r in reasons if r in (REASON_HEAT, REASON_DRY_HEAT)],
                    details=details,
                )
            if "heat_index" in kpis:
                builder.add(
                    EntityType.EVENT,
                    "EVENT",
                    "heat_index",
                    timestamp,
                    heat,
                    horizon_min=horizon_min,
                    state=record_state,
                    confidence=self.require_param("confidence"),
                    reason_codes=[r for r in reasons if r in (REASON_HEAT, REASON_DRY_HEAT)],
                    details={**details, "threshold_c": heat_threshold},
                )
            if "rainfall_intensity" in kpis:
                builder.add(
                    EntityType.EVENT,
                    "EVENT",
                    "rainfall_intensity",
                    timestamp,
                    rain,
                    horizon_min=horizon_min,
                    state=record_state,
                    confidence=self.require_param("confidence"),
                    reason_codes=[REASON_RAIN] if REASON_RAIN in reasons else [],
                    details=details,
                )

            if "waterlogging_probability" in kpis:
                for _, zone in self._zones.iterrows():
                    zone_id = str(zone["zone_id"])
                    if zone_id not in zones:
                        continue
                    probability = self._waterlogging(rain, zone)
                    zone_reasons: list[str] = []
                    if probability >= waterlogging_alert:
                        zone_reasons.append(REASON_WATERLOGGING)
                    if REASON_RAIN in reasons:
                        zone_reasons.append(REASON_RAIN)
                    builder.add(
                        EntityType.ZONE,
                        zone_id,
                        "waterlogging_probability",
                        timestamp,
                        probability,
                        zone_id=zone_id,
                        horizon_min=horizon_min,
                        state=record_state,
                        confidence=self.require_param("confidence"),
                        reason_codes=zone_reasons,
                        details={
                            "rain_mm_hr": round(rain, 2),
                            "drainage_capacity_mm_hr": float(zone["drainage_capacity_mm_hr"]),
                            "low_lying": bool(zone["low_lying"]),
                        },
                    )

            # The three couplings the rest of the twin reads.
            multipliers = {
                "weather_arrival_multiplier": arrival.get(timestamp, 1.0),
                "weather_medical_multiplier": medical.get(timestamp, 1.0),
                "weather_traffic_speed_multiplier": 1.0
                - k_rain_speed * min(rain, rain_cap) / max(rain_cap, 1e-9),
            }
            for kpi, value in multipliers.items():
                if kpi not in kpis:
                    continue
                builder.add(
                    EntityType.EVENT,
                    "EVENT",
                    kpi,
                    timestamp,
                    float(value),
                    horizon_min=horizon_min,
                    state=record_state,
                    confidence=self.require_param("confidence"),
                    details={
                        "formula": "docs/03 M21 card",
                        "rain_mm_hr": round(rain, 2),
                        "heat_index_c": round(heat, 2),
                    },
                )
        return builder

    # ------------------------------------------------------------------ endpoints
    def predict(self, request: PredictRequest) -> ModelOutput:
        builder = self.new_builder(request)
        self._compute(request, builder, state=State.FORECAST)
        return builder.build()

    def scenario(self, request: ScenarioRequest) -> ModelOutput:
        overrides = self.scenario_overrides(request)
        builder = self.new_builder(request, scenario_id=request.scenario_id, overrides=overrides)
        self._compute(request, builder, state=State.SCENARIO, overrides=overrides)
        if request.compare_to_baseline:
            builder.apply_baseline(self.predict(self.as_predict(request)))
        return builder.build()

    # ------------------------------------------------------------------ metadata
    def metadata(self) -> Metadata:
        return Metadata(
            model_id=self.model_id,
            model_name=self.model_name,
            model_version=self.model_version,
            question=(
                "What is the weather now and over the next hours, and how does it change "
                "risk elsewhere?"
            ),
            engine=self.engine,
            method_summary=(
                "NOAA heat index from the Rothfusz regression; per-zone waterlogging as a "
                "logistic function of rainfall against each zone's drainage capacity; "
                "arrival, medical and traffic-speed multipliers from the docs/03 M21 "
                "formulas. Weather is a replay of the cached Open-Meteo series."
            ),
            kpis=[
                {
                    "kpi": kpi,
                    "unit": reg.kpi_unit(kpi),
                    "description": reg.kpi_info(kpi).definition,
                }
                for kpi in self.owned_kpis
            ],
            upstream=self.upstream_ids,
            inputs=self.input_tables,
            horizons_min=self.horizons_min,
            scenarios_supported=self.scenarios_supported,
            limitations=[
                "Weather is a replay of a stored series, not a live forecast; there is no "
                "forecast uncertainty and the bands are absent by design.",
                "Waterlogging is a logistic function of rainfall against a per-zone drainage "
                "rate, not a hydraulic model. Ponding, runoff and river level are not "
                "represented.",
                "Every coefficient is a placeholder; the arrival, medical and speed "
                "multipliers have not been fitted to observed behaviour.",
                "The underlying observations are real Open-Meteo history for a past year "
                "shifted onto the event dates, so they are plausible but not a prediction.",
            ],
            license_notes=[
                "Weather and air quality: Open-Meteo, CC BY 4.0, free tier is "
                "non-commercial (docs/06 section 3).",
                "Heat index: NOAA/NWS Rothfusz regression, public domain.",
            ],
        )
