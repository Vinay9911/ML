"""M19 - Power Load.

What electrical load is expected, and how long can backup last?

**Method** (docs/03 M19 card)

Forecast
    Kilowatts per power asset from D18, on the shared forecast engine, with calendar,
    temperature and M01 arrival covariates. Reported as **MW**, which is what the KPI unit
    asks for, with the kW figure kept in ``details``.

Backup duration
    ``fuel_l / (l_per_hr_at_full_load * load_fraction)``. The load fraction is what the
    generator is actually carrying, not its rating: a set carrying a light load burns less and
    runs longer. Capped for reporting, because a full tank against a tiny load gives an
    arithmetically enormous runtime and the binding constraint past a day is refuelling, which
    this model does not represent.

Generators
    ``ceil(critical_load_kw * redundancy_factor / generator_unit_kw)``, where the critical load
    is the share of the asset that must stay up - life safety, lighting, comms, medical.

``details.load_utilization`` is published for M12 (fire risk), which the docs/03 M12 card
reads as a driver.

**S08 is the scenario this model exists for.** A downed substation puts its zones on backup
and starts a countdown, which is the ``GRID_OUTAGE`` reason code and a finite, falling
``generator_backup_duration``.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from twin_common.config import assumptions, world
from twin_common.contracts import (
    REASON_REQUIRED_LEVELS,
    Metadata,
    ModelOutput,
    PredictRequest,
    ScenarioRequest,
)
from twin_common.contracts import registry as reg
from twin_common.contracts.enums import EntityType, State
from twin_common.contracts.models import Recommendation
from twin_common.engines.forecast import (
    ForecastEngine,
    ForecastResult,
    calendar_covariates,
    series_by_entity,
)
from twin_common.io.tables import load_table
from twin_common.logging import get_logger
from twin_common.model import TwinModel
from twin_common.output import OutputBuilder
from twin_common.output.risk import risk_level_for_kpi
from twin_common.scenarios import closed_entities
from twin_common.upstream import series_for_entity

log = get_logger(__name__)

#: Reason codes M19 can raise (docs/05 section 4, utilities group).
REASON_GRID_OUTAGE = "GRID_OUTAGE"
REASON_ELECTRICAL_OVERLOAD = "ELECTRICAL_OVERLOAD"
REASON_GENERATOR_OPERATION = "GENERATOR_OPERATION"

#: The resource a shortfall recommendation asks for (resources.yaml).
RESOURCE_GENERATOR = "backup_generator"


class Model(TwinModel):
    """Power Load (M19)."""

    model_id = "M19"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        power = assumptions()["power"]
        self._generator_unit_kw = float(power["generator_unit_kw"])
        self._generator_fuel_l = float(power["generator_fuel_l"])
        self._l_per_hr_full = float(power["generator_l_per_hr_at_full_load"])
        self._redundancy = float(power["redundancy_factor"])
        self._cooling_kw_per_c = float(power["cooling_kw_per_c_above_30_per_zone"])
        self._substations = dict(world().get("substations", {}))

        self._kw: dict[str, dict[str, pd.Series]] = {}
        self._zone_of: dict[str, str] = {}
        self._fuel: dict[str, dict[str, pd.Series]] = {}
        self._temperature: dict[str, pd.Series] = {}
        self._arrivals: dict[str, pd.Series] = {}
        self._snan_days: dict[str, list[pd.Timestamp]] = {}
        for scenario_id in self.scenarios_supported:
            try:
                self._load_scenario(scenario_id)
            except Exception as exc:
                log.warning("no world slice for %s (%s)", scenario_id, exc)
        if "S01" not in self._kw:
            raise RuntimeError(
                "M19 needs at least the S01 world slice. Run `python scripts/slice_world.py M19`."
            )

        self.forecast_engine = ForecastEngine.from_config(self.param("forecast"), seed=self.seed)
        try:
            result = self.forecast_engine.warmup(
                self._kw["S01"],
                self.horizon_steps(self.default_horizon_min),
                future_covariates=self._covariate_frames("S01", list(self._kw["S01"])),
            )
            log.info(
                "M19 warm: backend %s, %d assets, %.1fs",
                result.backend_used,
                len(result.entities),
                result.elapsed_s,
            )
        except Exception as exc:
            log.warning("M19 warmup failed (%s); the first request will be slower", exc)

    def _load_scenario(self, scenario_id: str) -> None:
        table = load_table(
            "power_15min",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        # D18 carries one row per (substation, zone) pair, so a substation appears four
        # times at each timestamp. Splitting on asset_id alone would keep only the last
        # zone and silently report a quarter of the load. Load SUMS across the zones a
        # substation feeds; fuel is a property of its generator and must not be summed.
        by_asset = table.groupby(["timestamp", "asset_id"], as_index=False).agg(
            kw=("kw", "sum"), fuel_l=("fuel_l", "max")
        )
        self._kw[scenario_id] = series_by_entity(
            by_asset, entity_column="asset_id", value_column="kw"
        )
        self._fuel[scenario_id] = series_by_entity(
            by_asset, entity_column="asset_id", value_column="fuel_l"
        )
        # A substation feeds several zones, so it is not "in" one. world.yaml lists them.
        for asset_id, spec in self._substations.items():
            zones = spec.get("feeds_zones") or []
            if zones:
                self._zone_of.setdefault(asset_id, zones[0])

        weather = load_table(
            "weather_hourly",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        ).set_index("timestamp")
        self._temperature[scenario_id] = weather["temperature_c"].astype("float64").sort_index()

        footfall = load_table(
            "footfall_15min",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        self._arrivals[scenario_id] = (
            footfall.groupby("timestamp")["entries"].sum().astype("float64").sort_index()
        )
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
        self,
        scenario_id: str,
        entities: list[str],
        *,
        future_arrivals: pd.Series | None = None,
    ) -> dict[str, pd.DataFrame]:
        series = next(iter(self._kw[scenario_id].values()))
        index = pd.DatetimeIndex(series.index)
        horizon = self.horizon_steps(max(self.horizons_min))
        extended = pd.date_range(
            start=index[0], periods=len(index) + horizon, freq="15min", tz=index.tz
        )
        extra: dict[str, pd.Series] = {"temperature_c": self._temperature[scenario_id]}
        history = self._arrivals.get(scenario_id)
        if history is not None and future_arrivals is not None and len(future_arrivals):
            combined = pd.concat([history, future_arrivals])
            extra["arrivals"] = combined[~combined.index.duplicated(keep="last")].sort_index()
        frame = calendar_covariates(
            extended, snan_days=self._snan_days.get(scenario_id, []), extra=extra
        )
        return dict.fromkeys(entities, frame)

    def _m01_arrivals(self, upstream: dict[str, Any], as_of: pd.Timestamp) -> pd.Series | None:
        resolution = upstream.get("M01")
        if resolution is None:
            return None
        zones = sorted(
            {
                record.entity_id
                for record in resolution.output.results
                if record.kpi == "expected_footfall" and record.entity_id.startswith("Z")
            }
        )
        totals: dict[pd.Timestamp, float] = {}
        for zone in zones:
            for timestamp, value in series_for_entity(resolution.output, "expected_footfall", zone):
                stamp = pd.Timestamp(timestamp)
                if stamp > as_of:
                    totals[stamp] = totals.get(stamp, 0.0) + float(value) / 4.0
        return pd.Series(totals).sort_index() if totals else None

    @staticmethod
    def _m21_temperature(upstream: dict[str, Any]) -> float | None:
        resolution = upstream.get("M21")
        if resolution is None:
            return None
        values = [float(r.value) for r in resolution.output.results if r.kpi == "temperature"]
        return max(values) if values else None

    # ------------------------------------------------------------------ formulas
    def horizon_steps(self, horizon_min: int) -> int:
        return max(1, round(horizon_min / 15))

    def entity_ids(self) -> list[str]:
        return list(self._kw["S01"])

    def to_megawatts(self, kw: float) -> float:
        """docs/03 M19 reports MW; D18 is in kW."""
        divisor = float(self.require_param("kw_per_mw"))
        return 0.0 if divisor <= 0 else kw / divisor

    def critical_load_kw(self, kw: float) -> float:
        """The share that must stay up when the grid goes."""
        return max(0.0, kw) * float(self.require_param("critical_load_fraction"))

    def generators_required(self, critical_kw: float) -> int:
        """docs/03 M19: ceil(critical_load_kw * redundancy_factor / generator_unit_kw)."""
        if critical_kw <= 0 or self._generator_unit_kw <= 0:
            return 0
        return math.ceil(critical_kw * self._redundancy / self._generator_unit_kw)

    def backup_hours(self, fuel_l: float, *, load_fraction: float | None = None) -> float:
        """docs/03 M19: fuel / (litres per hour at full load x load fraction).

        A set carrying a light load burns less and runs longer, which is why the fraction is
        in the denominator rather than assumed to be one.
        """
        fraction = float(
            load_fraction
            if load_fraction is not None
            else self.require_param("assumed_load_fraction")
        )
        ceiling = float(self.require_param("max_backup_hours"))
        if fuel_l <= 0:
            return 0.0
        burn = self._l_per_hr_full * max(0.0, fraction)
        if burn <= 0:
            return ceiling
        return float(min(ceiling, fuel_l / burn))

    def cooling_uplift_kw(self, temperature_c: float, baseline_c: float) -> float:
        """Extra cooling load per asset for each degree above the baseline (docs/04 section 6).

        This is the S04 mechanism: heat does not change how many people are present, it
        changes how hard the air conditioning works.
        """
        return self._cooling_kw_per_c * max(0.0, temperature_c - baseline_c)

    def _floor(self, value: float) -> float:
        return max(float(self.require_param("floor_value")), value)

    def _window_temperature(self, scenario_id: str, as_of: pd.Timestamp, steps: int) -> float:
        series = self._temperature[scenario_id]
        end = as_of + pd.Timedelta(minutes=15 * steps)
        window = series.loc[(series.index >= as_of) & (series.index <= end)]
        return float(window.max() if len(window) else series.max())

    def _on_backup(self, asset_id: str, scenario_id: str, overrides: dict[str, Any]) -> bool:
        """True when this asset is running on generators for the scenario.

        A generated world already carries the outage in ``grid_available``; the override path
        is for a scenario with no world of its own.
        """
        down = closed_entities(overrides)["substations"]
        if not down:
            return False
        if asset_id in down:
            return True
        substation = self._substations.get(asset_id, {}).get("substation")
        return substation in down

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

        entities = self.resolve_entities(request, self.entity_ids())
        if not entities:
            builder.warn("no known entities matched the request")
            return builder

        upstream = self.resolve_upstream(request, scenario_id=scenario_id, entity_ids=entities)
        self.record_upstream(builder, upstream)

        source = scenario_id if scenario_id in self._kw else "S01"
        baseline_temp = self._window_temperature(source, as_of, steps)
        temperature = baseline_temp
        uplift_kw = 0.0
        adjustment = 1.0

        if source != scenario_id:
            multiplier = float(overrides.get("footfall_multiplier", 1.0))
            if multiplier != 1.0:
                adjustment = multiplier
                builder.warn(
                    f"no generated world for {scenario_id}; the forecast was scaled by "
                    f"{multiplier} through the generic scenario layer"
                )
            reported = self._m21_temperature(upstream)
            offset = float(overrides.get("temperature_offset_c", 0.0))
            if reported is not None:
                temperature = reported
                builder.warn(f"temperature {reported:.1f} C taken from M21")
            elif offset:
                temperature = baseline_temp + offset
            if temperature != baseline_temp:
                uplift_kw = self.cooling_uplift_kw(temperature, baseline_temp)
                builder.warn(
                    f"cooling load raised by {uplift_kw:.0f} kW per asset for "
                    f"{temperature:.1f} C against a window baseline of {baseline_temp:.1f} C"
                )

        trimmed = {
            asset: series.loc[series.index <= as_of]
            for asset, series in self._kw[source].items()
            if asset in entities
        }
        trimmed = {a: s for a, s in trimmed.items() if len(s) > steps + 2}
        if not trimmed:
            builder.warn(f"no history at or before {as_of.isoformat()}")
            return builder

        future_arrivals = self._m01_arrivals(upstream, as_of)
        if future_arrivals is None:
            builder.warn(
                "no M01 forecast available; the arrivals covariate was dropped and the "
                "forecast used the calendar and temperature covariates only"
            )

        result = self.forecast_engine.predict(
            trimmed,
            steps,
            future_covariates=self._covariate_frames(
                source, list(trimmed), future_arrivals=future_arrivals
            ),
        )
        if result.degraded:
            builder.warn(
                f"forecast ran on the {result.backend_used} backend rather than "
                f"{result.backend_requested}"
            )
        for message in result.warnings:
            builder.warn(message)

        self._emit(
            builder,
            result,
            kpis,
            as_of,
            state,
            source=source,
            scenario_id=scenario_id,
            overrides=overrides,
            adjustment=adjustment,
            uplift_kw=uplift_kw,
            temperature=temperature,
            used_m01=future_arrivals is not None,
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
        scenario_id: str,
        overrides: dict[str, Any],
        adjustment: float,
        uplift_kw: float,
        temperature: float,
        used_m01: bool,
    ) -> None:
        confidence = float(self.require_param("confidence"))

        for asset in result.entities:
            zone = self._zone_of.get(asset)
            fuel_series = self._fuel[source].get(asset)
            observed = (
                fuel_series.loc[fuel_series.index <= as_of] if fuel_series is not None else None
            )
            fuel = (
                float(observed.iloc[-1])
                if observed is not None and len(observed)
                else self._generator_fuel_l
            )
            on_backup = self._on_backup(asset, scenario_id, overrides)

            for timestamp in result.values[asset].index:
                value, low, high = result.at(asset, timestamp)
                kw = self._floor(value * adjustment + uplift_kw)
                critical = self.critical_load_kw(kw)
                generators = self.generators_required(critical)
                offset = max(0, int((timestamp - as_of).total_seconds() // 60))

                # Published for M12, whose card reads electrical utilisation as a fire driver.
                capacity_kw = max(1.0, generators * self._generator_unit_kw)
                utilization = min(100.0, 100.0 * critical / capacity_kw) if generators else 0.0

                reasons: list[str] = []
                if on_backup:
                    reasons.extend([REASON_GRID_OUTAGE, REASON_GENERATOR_OPERATION])
                banded = risk_level_for_kpi("electricity_demand", self.to_megawatts(kw))
                if banded in REASON_REQUIRED_LEVELS and REASON_ELECTRICAL_OVERLOAD not in reasons:
                    reasons.append(REASON_ELECTRICAL_OVERLOAD)

                details = {
                    "backend": result.backend_used,
                    "band_calibrated": result.calibrated,
                    "kw": round(kw, 1),
                    "critical_load_kw": round(critical, 1),
                    "load_utilization": round(utilization, 1),
                    "cooling_uplift_kw": round(uplift_kw, 1),
                    "temperature_c": round(temperature, 1),
                    "on_backup": on_backup,
                    "fuel_l": round(fuel, 1),
                    "arrivals_covariate": "M01" if used_m01 else "none",
                }

                if "electricity_demand" in kpis:
                    builder.add(
                        EntityType.ASSET,
                        asset,
                        "electricity_demand",
                        timestamp,
                        self.to_megawatts(kw),
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        lower=None
                        if low is None
                        else self.to_megawatts(self._floor(low * adjustment + uplift_kw)),
                        upper=None
                        if high is None
                        else self.to_megawatts(self._floor(high * adjustment + uplift_kw)),
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=details,
                    )

                if "backup_generator_requirement" in kpis:
                    builder.add(
                        EntityType.ASSET,
                        asset,
                        "backup_generator_requirement",
                        timestamp,
                        float(generators),
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        confidence=confidence,
                        resource_type=RESOURCE_GENERATOR,
                        reason_codes=reasons,
                        recommendation=self._stage_generators(asset, generators, critical)
                        if on_backup and generators > 0
                        else None,
                        details=details,
                    )

                if "generator_backup_duration" in kpis:
                    builder.add(
                        EntityType.ASSET,
                        asset,
                        "generator_backup_duration",
                        timestamp,
                        self.backup_hours(fuel),
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=details,
                    )

    def _stage_generators(self, asset: str, generators: int, critical_kw: float) -> Recommendation:
        return Recommendation(
            action=f"confirm backup generation at {asset}",
            resource_type=RESOURCE_GENERATOR,
            quantity=float(generators),
            target_entity_id=asset,
            rationale=(
                f"{asset} is on backup with {critical_kw:,.0f} kW of critical load, needing "
                f"{generators} unit(s) at {self._generator_unit_kw:,.0f} kW with a "
                f"{self._redundancy:.2f} redundancy factor"
            ),
        )

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
            question="What electrical load is expected, and how long can backup last?",
            engine=self.engine,
            method_summary=(
                f"Darts {backend} over one kW series per power asset, with calendar, "
                f"temperature and M01 arrival covariates, reported in MW. Backup duration is "
                f"fuel over burn at the carried load; generators are the critical load with a "
                f"redundancy factor. details.load_utilization is published for M12."
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
                "The critical load is one fraction of the whole asset, a placeholder for a "
                "proper per-circuit schedule. Which circuits are actually life-safety is a "
                "design decision, not a ratio.",
                "Backup duration assumes a constant load fraction and a full burn rate. Real "
                "consumption varies with load, ambient temperature and set condition, and "
                "refuelling - not tank size - is usually the binding constraint past a day.",
                "Duration is capped at 24 h for reporting; a light load on a full tank is "
                "arithmetically much longer and operationally meaningless.",
                "Fuel is the last observed level and does not deplete across the horizon, so "
                "the countdown is a starting estimate rather than a live one.",
                "The cooling uplift is a flat kW per degree per asset, applied only when a "
                "scenario moves the temperature; it does not vary by what the asset feeds.",
                "Synthetic context; the backtest measures pipeline correctness, not accuracy.",
            ],
            license_notes=[
                "Darts: Apache-2.0. Chronos-2 weights: see the autogluon model card.",
                "LightGBM: MIT.",
            ],
        )
