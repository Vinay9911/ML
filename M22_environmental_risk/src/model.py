"""M22 - Environmental Risk.

Where will air quality or noise stress increase, and what are the emissions?

**Method** (docs/03 M22 card)

PM2.5
    Forecast from D13 on the shared engine. **D13 is the one table in this project built from
    genuinely real data** - cached Open-Meteo air-quality history - so its rows carry
    ``is_synthetic: false``. D13 is hourly and venue-level, so PM2.5 and AQI are reported for
    the event as a whole rather than per zone.

AQI
    The Indian National AQI: a piecewise-linear sub-index per pollutant between CPCB
    breakpoints, with the AQI being the **maximum** sub-index, not the average. Every
    breakpoint is in config and every one is a placeholder - see the model card.

Noise
    ``base_db + 10 * log10(1 + density / density_ref)``, per zone, because crowd density is
    per zone. Decibels are logarithmic, which is why the crowd term is too.

CO2
    ``vehicle-km x factor + kWh x grid factor + generator litres x diesel factor``. The
    traffic term needs M04, which is a Phase 7 model; until it exists that term is **zero and
    the response says so** rather than being invented.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from twin_common.config import assumptions
from twin_common.contracts import Metadata, ModelOutput, PredictRequest, ScenarioRequest
from twin_common.contracts import registry as reg
from twin_common.contracts.enums import EntityType, State
from twin_common.engines.forecast import (
    ForecastEngine,
    ForecastResult,
    calendar_covariates,
)
from twin_common.io.tables import load_table
from twin_common.logging import get_logger
from twin_common.model import TwinModel
from twin_common.output import OutputBuilder

log = get_logger(__name__)

#: Reason codes M22 can raise (docs/05 section 4).
REASON_HEAVY_RAIN = "HEAVY_RAIN"
REASON_GENERATOR_OPERATION = "GENERATOR_OPERATION"
REASON_FIRE_RISK = "FIRE_RISK"

#: The venue-level entity PM2.5 and AQI are reported against (docs/02 section 3).
EVENT_ID = "EVENT"


class Model(TwinModel):
    """Environmental Risk (M22)."""

    model_id = "M22"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        power = assumptions()["power"]
        self._generator_lph = float(power["generator_l_per_hr_at_full_load"])

        self._pm25: dict[str, pd.Series] = {}
        self._noise_db: dict[str, pd.Series] = {}
        self._density: dict[str, dict[str, pd.Series]] = {}
        self._kwh: dict[str, pd.Series] = {}
        self._generator_litres: dict[str, pd.Series] = {}
        self._snan_days: dict[str, list[pd.Timestamp]] = {}
        for scenario_id in self.scenarios_supported:
            try:
                self._load_scenario(scenario_id)
            except Exception as exc:
                log.warning("no world slice for %s (%s)", scenario_id, exc)
        if "S01" not in self._pm25:
            raise RuntimeError(
                "M22 needs at least the S01 world slice. Run `python scripts/slice_world.py M22`."
            )

        self.forecast_engine = ForecastEngine.from_config(self.param("forecast"), seed=self.seed)
        try:
            result = self.forecast_engine.warmup(
                {EVENT_ID: self._pm25["S01"]},
                self.horizon_steps(self.default_horizon_min),
                future_covariates=self._covariate_frames("S01"),
            )
            log.info("M22 warm: backend %s, %.1fs", result.backend_used, result.elapsed_s)
        except Exception as exc:
            log.warning("M22 warmup failed (%s); the first request will be slower", exc)

    def _load_scenario(self, scenario_id: str) -> None:
        air = load_table(
            "air_quality_hourly",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        ).set_index("timestamp")
        self._pm25[scenario_id] = air["pm2_5"].astype("float64").sort_index()
        self._noise_db[scenario_id] = air["noise_db"].astype("float64").sort_index()
        self._pm10 = getattr(self, "_pm10", {})
        self._pm10[scenario_id] = air["pm10"].astype("float64").sort_index()

        footfall = load_table(
            "footfall_15min",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        if "density_p_m2" in footfall.columns:
            density = (
                footfall.pivot_table(
                    index="timestamp", columns="zone_id", values="density_p_m2", aggfunc="max"
                )
                .resample("1h")
                .max()
            )
            self._density[scenario_id] = {
                str(zone): density[zone].astype("float64") for zone in density.columns
            }
        else:
            self._density[scenario_id] = {}

        # D18 drives the electricity and diesel terms of co2_emissions. It is loaded last and
        # defensively: without it the air-quality and noise KPIs are still answerable, and a
        # model that half-loaded would be worse than one that says the term is missing.
        empty = pd.Series(dtype="float64")
        self._kwh[scenario_id] = empty
        self._generator_litres[scenario_id] = empty
        try:
            power = load_table(
                "power_15min",
                self.data_source,
                scenario_id=scenario_id,
                base_dir=self.synthetic_dir,
            )
        except Exception as exc:
            log.warning(
                "%s: no power table (%s); the CO2 energy terms will be zero", scenario_id, exc
            )
        else:
            hourly = power.groupby("timestamp")[["kwh"]].sum().resample("1h").sum()
            self._kwh[scenario_id] = hourly["kwh"].astype("float64")
            # Litres burnt is the fall in the tank level, summed across assets and clipped at
            # zero so a refuel does not read as negative emissions.
            fuel = power.pivot_table(
                index="timestamp", columns="asset_id", values="fuel_l", aggfunc="max"
            ).sort_index()
            burnt = (-fuel.diff()).clip(lower=0.0).sum(axis=1)
            self._generator_litres[scenario_id] = burnt.resample("1h").sum().astype("float64")

        calendar = load_table(
            "event_calendar",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        self._snan_days[scenario_id] = [
            pd.Timestamp(day) for day in calendar.loc[calendar["is_snan_day"], "date"].unique()
        ]

    # ------------------------------------------------------------------ covariates
    def _covariate_frames(
        self, scenario_id: str, *, wind_kmh: float | None = None
    ) -> dict[str, pd.DataFrame]:
        series = self._pm25[scenario_id]
        index = pd.DatetimeIndex(series.index)
        horizon = self.horizon_steps(max(self.horizons_min))
        extended = pd.date_range(
            start=index[0], periods=len(index) + horizon, freq="1h", tz=index.tz
        )
        weather = load_table(
            "weather_hourly",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        ).set_index("timestamp")
        extra = {"wind_kmh": weather["wind_kmh"].astype("float64").sort_index()}
        frame = calendar_covariates(
            extended, snan_days=self._snan_days.get(scenario_id, []), extra=extra
        )
        return {EVENT_ID: frame}

    @staticmethod
    def _m21_wind(upstream: dict[str, Any]) -> float | None:
        """Wind from M21, which disperses whatever is in the air."""
        resolution = upstream.get("M21")
        if resolution is None:
            return None
        for record in resolution.output.results:
            if record.kpi in ("wind_speed", "wind_kmh"):
                return float(record.value)
        return None

    @staticmethod
    def _m04_vehicle_km(upstream: dict[str, Any]) -> float:
        """Vehicle-kilometres from M04, or zero while M04 does not exist."""
        resolution = upstream.get("M04")
        if resolution is None:
            return 0.0
        return sum(
            float(record.value)
            for record in resolution.output.results
            if record.kpi in ("vehicle_km", "traffic_volume")
        )

    # ------------------------------------------------------------------ formulas
    def horizon_steps(self, horizon_min: int) -> int:
        """Minutes to HOURLY steps: D13 is hourly."""
        return max(1, round(horizon_min / 60))

    def entity_ids(self) -> list[str]:
        return [EVENT_ID, *sorted(self._density.get("S01", {}))]

    def sub_index(self, concentration: float, pollutant: str) -> float:
        """One pollutant's NAQI sub-index by piecewise-linear interpolation.

        docs/05 section 6. Within a band the index moves linearly between the band edges::

            index = index_lo + (index_hi - index_lo) * (c - c_lo) / (c_hi - c_lo)

        Below the first breakpoint the band starts at zero; above the last the scale is
        exhausted and the maximum is reported, because the CPCB scale stops at 500.
        """
        naqi = self.require_param("naqi")
        index_points = [float(v) for v in naqi["index_breakpoints"]]
        concentrations = [float(v) for v in naqi["concentration_breakpoints"][pollutant]]
        ceiling = float(naqi["max_index"])
        value = max(0.0, float(concentration))

        low_c, low_i = 0.0, 0.0
        for high_c, high_i in zip(concentrations, index_points, strict=True):
            if value <= high_c:
                span = high_c - low_c
                if span <= 0:
                    return float(min(ceiling, high_i))
                return float(min(ceiling, low_i + (high_i - low_i) * (value - low_c) / span))
            low_c, low_i = high_c, high_i
        return ceiling

    def aqi(self, pm25: float, pm10: float | None = None) -> float:
        """docs/05 section 6: AQI is the MAXIMUM sub-index, not the average.

        Averaging would let a clean pollutant mask a dangerous one, which is exactly what the
        maximum rule exists to prevent.
        """
        indices = [self.sub_index(pm25, "pm2_5")]
        if pm10 is not None:
            indices.append(self.sub_index(pm10, "pm10"))
        return max(indices)

    def aqi_category(self, index: float) -> str:
        naqi = self.require_param("naqi")
        names = list(naqi["category_names"])
        for edge, name in zip(naqi["index_breakpoints"], names, strict=True):
            if index <= float(edge):
                return str(name)
        return str(names[-1])

    def noise_db(self, density_p_m2: float, *, base_db: float | None = None) -> float:
        """docs/03 M22: base_db + 10 * log10(1 + density / density_ref)."""
        settings = self.require_param("noise")
        base = float(base_db if base_db is not None else settings["base_db"])
        reference = float(settings["density_ref_p_m2"])
        ceiling = float(settings["max_db"])
        if reference <= 0:
            return base
        ratio = max(0.0, density_p_m2) / reference
        return float(min(ceiling, base + 10.0 * math.log10(1.0 + ratio)))

    def co2_tonnes_per_day(
        self, *, vehicle_km: float, kwh: float, generator_litres: float
    ) -> float:
        """docs/03 M22: traffic + electricity + generator diesel, in tCO2e/day.

        The inputs are per hour; the KPI is per day, so the hourly rate is scaled up.
        """
        factors = self.require_param("co2")
        kg = (
            vehicle_km * float(factors["kg_per_vehicle_km"])
            + kwh * float(factors["kg_per_kwh"])
            + generator_litres * float(factors["kg_per_diesel_litre"])
        )
        per_day = kg * float(factors["hours_per_day"])
        return max(0.0, per_day / float(factors["kg_per_tonne"]))

    def _floor(self, value: float) -> float:
        return max(float(self.require_param("floor_value")), value)

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
        steps = self.horizon_steps(self.resolve_horizon(request))
        kpis = set(self.resolve_kpis(request))
        scenario_id = builder.scenario_id
        overrides = overrides or {}

        source = scenario_id if scenario_id in self._pm25 else "S01"
        entities = set(self.resolve_entities(request, self.entity_ids()))

        upstream = self.resolve_upstream(request, scenario_id=scenario_id, entity_ids=None)
        self.record_upstream(builder, upstream)
        vehicle_km = self._m04_vehicle_km(upstream)
        if vehicle_km == 0.0:
            builder.warn(
                "no M04 output; the traffic term of co2_emissions is zero rather than "
                "estimated, so the figure is electricity and generator diesel only"
            )

        history = self._pm25[source].loc[self._pm25[source].index <= as_of]
        if len(history) <= steps + 2:
            builder.warn(f"no history at or before {as_of.isoformat()}")
            return builder

        result = self.forecast_engine.predict(
            {EVENT_ID: history},
            steps,
            future_covariates=self._covariate_frames(source),
        )
        if result.degraded:
            builder.warn(
                f"forecast ran on the {result.backend_used} backend rather than "
                f"{result.backend_requested}"
            )
        for message in result.warnings:
            builder.warn(message)

        fire_zone = overrides.get("fire_zone")
        if fire_zone:
            builder.warn(
                f"fire in {fire_zone}: PM2.5 near it rises, which the scenario world already "
                f"carries where one was generated"
            )

        self._emit(
            builder,
            result,
            kpis,
            as_of,
            state,
            source=source,
            entities=entities,
            vehicle_km=vehicle_km,
            fire_zone=fire_zone,
        )
        return builder

    def _emit(
        self,
        builder: OutputBuilder,
        result: ForecastResult,
        kpis: set[str],
        as_of: pd.Timestamp,
        state: State,
        *,
        source: str,
        entities: set[str],
        vehicle_km: float,
        fire_zone: str | None,
    ) -> None:
        confidence = float(self.require_param("confidence"))
        pm10_series = self._pm10[source]
        kwh_series = self._kwh.get(source, pd.Series(dtype="float64"))
        litres_series = self._generator_litres.get(source, pd.Series(dtype="float64"))

        for timestamp in result.values[EVENT_ID].index:
            value, low, high = result.at(EVENT_ID, timestamp)
            pm25 = self._floor(value)
            offset = max(0, int((timestamp - as_of).total_seconds() // 60))
            pm10 = self._nearest(pm10_series, timestamp)
            index = self.aqi(pm25, pm10)
            reasons = [REASON_FIRE_RISK] if fire_zone else []

            shared = {
                "backend": result.backend_used,
                "band_calibrated": result.calibrated,
                "pm10_ug_m3": None if pm10 is None else round(pm10, 2),
                "aqi_category": self.aqi_category(index),
                "traffic_term": "M04" if vehicle_km else "unavailable (zero)",
            }

            if EVENT_ID in entities and "pm25" in kpis:
                builder.add(
                    EntityType.EVENT,
                    EVENT_ID,
                    "pm25",
                    timestamp,
                    pm25,
                    horizon_min=offset,
                    state=state,
                    lower=None if low is None else self._floor(low),
                    upper=None if high is None else self._floor(high),
                    quantile_level=result.quantile_level if low is not None else None,
                    confidence=confidence,
                    reason_codes=reasons,
                    details=shared,
                )

            if EVENT_ID in entities and "air_quality_index" in kpis:
                builder.add(
                    EntityType.EVENT,
                    EVENT_ID,
                    "air_quality_index",
                    timestamp,
                    index,
                    horizon_min=offset,
                    state=state,
                    confidence=confidence,
                    reason_codes=reasons,
                    details={**shared, "pm25_sub_index": round(self.sub_index(pm25, "pm2_5"), 1)},
                )

            if EVENT_ID in entities and "co2_emissions" in kpis:
                kwh = self._nearest(kwh_series, timestamp) or 0.0
                litres = self._nearest(litres_series, timestamp) or 0.0
                tonnes = self.co2_tonnes_per_day(
                    vehicle_km=vehicle_km, kwh=kwh, generator_litres=litres
                )
                builder.add(
                    EntityType.EVENT,
                    EVENT_ID,
                    "co2_emissions",
                    timestamp,
                    tonnes,
                    horizon_min=offset,
                    state=state,
                    confidence=confidence,
                    reason_codes=[*reasons, REASON_GENERATOR_OPERATION] if litres > 0 else reasons,
                    details={
                        **shared,
                        "kwh_per_hour": round(kwh, 1),
                        "generator_litres_per_hour": round(litres, 2),
                        "vehicle_km": round(vehicle_km, 1),
                    },
                )

            if "noise_level" in kpis:
                for zone, density in self._density[source].items():
                    if zone not in entities:
                        continue
                    level = self._nearest(density, timestamp) or 0.0
                    builder.add(
                        EntityType.ZONE,
                        zone,
                        "noise_level",
                        timestamp,
                        self.noise_db(level),
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        confidence=confidence,
                        reason_codes=reasons,
                        details={**shared, "density_p_m2": round(level, 2)},
                    )

    @staticmethod
    def _nearest(series: pd.Series, timestamp: pd.Timestamp) -> float | None:
        """The value at or most recently before ``timestamp``."""
        if series is None or series.empty:
            return None
        earlier = series.loc[series.index <= timestamp]
        if len(earlier):
            return float(earlier.iloc[-1])
        return float(series.iloc[0])

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
        backend = getattr(getattr(self, "forecast_engine", None), "backend", "chronos2")
        return Metadata(
            model_id=self.model_id,
            model_name=self.model_name,
            model_version=self.model_version,
            question="Where will air quality or noise stress increase, and what are emissions?",
            engine=self.engine,
            method_summary=(
                f"Darts {backend} over the venue PM2.5 series from D13 - the one table built "
                f"from real cached Open-Meteo data - with wind and calendar covariates. AQI "
                f"is the Indian NAQI maximum sub-index; noise is a logarithmic crowd-density "
                f"term per zone; CO2 sums electricity, generator diesel and (once M04 exists) "
                f"traffic."
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
                "THE NAQI BREAKPOINTS ARE PLACEHOLDERS transcribed from docs/05 and must be "
                "verified against the published CPCB table before any real use. The official "
                "method also uses 24-hour averages for PM2.5 and PM10; this uses the "
                "instantaneous forecast, which will read differently during a sharp episode.",
                "The CO2 traffic term is ZERO because M04 does not exist yet. The reported "
                "figure is electricity and generator diesel only, and the response says so.",
                "Every emission factor is a placeholder. Do not publish a tCO2e figure from "
                "this model without replacing them with official factors.",
                "D13 is venue-level and hourly, so PM2.5 and AQI have no spatial detail. The "
                "S15 claim that PM2.5 rises 'near Z03' cannot be represented at this "
                "resolution - only the venue average moves.",
                "Noise is modelled from crowd density alone. Public address, generators, "
                "traffic and music are not represented, and those usually dominate.",
                "Only PM2.5 and PM10 contribute to the AQI here. The official index also "
                "covers NO2, SO2, CO, O3 and NH3, any of which could be the binding "
                "sub-index on a given day.",
            ],
            license_notes=[
                "Darts: Apache-2.0. Chronos-2 weights: see the autogluon model card.",
                "Open-Meteo air-quality data: CC BY 4.0, free tier is non-commercial.",
                "The Indian NAQI method is published by CPCB; the breakpoints here are a "
                "transcription placeholder and are not authoritative.",
            ],
        )
