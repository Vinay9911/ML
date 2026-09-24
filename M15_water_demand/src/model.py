"""M15 - Water Demand.

How much water is needed by zone, is there a gap, and how many points and tankers?

**Method** (docs/03 M15 card)

Forecast
    Consumption per zone from D14, on the shared forecast engine. Covariates are the calendar,
    temperature from D12 - genuinely known across the window, as for M01 - and arrivals whose
    future half comes from M01.

Supply
    Carried forward from the last observed value rather than forecast. A pump's design output
    does not vary with the crowd, so supply is a *planned* quantity; reading it across the
    forecast window is no more peeking than reading the calendar. A scenario that downs a
    substation scales the affected zones by ``water.pump_outage_supply_factor``.

Gap
    ``demand - supply``. **Positive means a shortage**, which is the sign convention the KPI
    name implies and the one the tanker requirement depends on. A surplus is reported as a
    negative number rather than clipped, because "how much spare capacity is there" is a real
    operational question.

Points and tankers
    ``points = ceil(peak population / persons_per_water_point)`` and
    ``tankers = ceil(max(gap, 0) * hours / tanker_payload_l)``. Population is recovered from
    the forecast demand by inverting the generator's own relation rather than forecast
    separately, so the two KPIs cannot disagree - see :meth:`population_from_demand`.

**S04 has no generated world.** Extreme heat reaches M15 through M21's temperature, which
scales demand by the documented heat multiplier. That is the same coupling the generator uses,
so the scenario answer is consistent with what a generated S04 world would have contained.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from twin_common.config import assumptions, world
from twin_common.contracts import Metadata, ModelOutput, PredictRequest, ScenarioRequest
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
from twin_common.scenarios import closed_entities
from twin_common.upstream import series_for_entity

log = get_logger(__name__)

#: Reason codes M15 can raise. All three are on the docs/05 section 4 allowed list:
#: WATER_GAP is the supply-group code for a shortfall, HEAT_STRESS the weather-group code for
#: the heat term, and GRID_OUTAGE the utilities code for a pump whose substation is down.
REASON_WATER_GAP = "WATER_GAP"
REASON_HEAT_STRESS = "HEAT_STRESS"
REASON_GRID_OUTAGE = "GRID_OUTAGE"

#: Resources a shortfall recommendation asks for (resources.yaml).
RESOURCE_TANKER = "water_tanker"
RESOURCE_POINT = "water_point"


class Model(TwinModel):
    """Water Demand (M15)."""

    model_id = "M15"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        water = assumptions()["water"]
        self._litres_pph = float(water["litres_per_person_per_hour_present"])
        self._heat_bonus = float(water["heat_multiplier_per_c_above_30"])
        self._persons_per_point = float(water["persons_per_water_point"])
        self._tanker_payload_l = float(water["tanker_payload_l"])
        self._outage_factor = float(water["pump_outage_supply_factor"])
        self._pumps: dict[str, dict[str, Any]] = dict(world()["pumps"])

        self._demand: dict[str, dict[str, pd.Series]] = {}
        self._supply: dict[str, dict[str, pd.Series]] = {}
        self._temperature: dict[str, pd.Series] = {}
        self._arrivals: dict[str, pd.Series] = {}
        self._snan_days: dict[str, list[pd.Timestamp]] = {}
        for scenario_id in self.scenarios_supported:
            try:
                self._load_scenario(scenario_id)
            except Exception as exc:
                log.warning("no world slice for %s (%s)", scenario_id, exc)
        if "S01" not in self._demand:
            raise RuntimeError(
                "M15 needs at least the S01 world slice. Run `python scripts/slice_world.py M15`."
            )

        self.forecast_engine = ForecastEngine.from_config(self.param("forecast"), seed=self.seed)
        try:
            result = self.forecast_engine.warmup(
                self._demand["S01"],
                self.horizon_steps(self.default_horizon_min),
                future_covariates=self._covariate_frames("S01", list(self._demand["S01"])),
            )
            log.info(
                "M15 warm: backend %s, %d zones, %.1fs",
                result.backend_used,
                len(result.entities),
                result.elapsed_s,
            )
        except Exception as exc:
            log.warning("M15 warmup failed (%s); the first request will be slower", exc)

    def _load_scenario(self, scenario_id: str) -> None:
        table = load_table(
            "water_15min",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        # D14 carries one row per zone per step; asset_id names the pump where a zone has one.
        by_zone = table.groupby(["timestamp", "zone_id"], as_index=False)[
            ["consumption_l", "supply_l"]
        ].sum()
        self._demand[scenario_id] = series_by_entity(
            by_zone, entity_column="zone_id", value_column="consumption_l"
        )
        self._supply[scenario_id] = series_by_entity(
            by_zone, entity_column="zone_id", value_column="supply_l"
        )

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
        series = next(iter(self._demand[scenario_id].values()))
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
        per_step = float(self.require_param("steps_per_hour"))
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
                    totals[stamp] = totals.get(stamp, 0.0) + float(value) / per_step
        return pd.Series(totals).sort_index() if totals else None

    @staticmethod
    def _m21_temperature(upstream: dict[str, Any]) -> float | None:
        """The peak temperature M21 reports across its horizon."""
        resolution = upstream.get("M21")
        if resolution is None:
            return None
        values = [
            float(record.value)
            for record in resolution.output.results
            if record.kpi == "temperature"
        ]
        return max(values) if values else None

    # ------------------------------------------------------------------ formulas
    def horizon_steps(self, horizon_min: int) -> int:
        return max(1, round(horizon_min / 15))

    def entity_ids(self) -> list[str]:
        return list(self._demand["S01"])

    def heat_factor(self, temperature_c: float) -> float:
        """docs/04 section 6: demand rises with every degree above 30.

        The same expression the generator uses, so a heat scenario applied here gives the
        answer a generated world would have contained.
        """
        return 1.0 + self._heat_bonus * max(0.0, temperature_c - 30.0)

    def population_from_demand(self, litres_per_hr: float, temperature_c: float) -> float:
        """Invert the generator's consumption relation to recover people present.

        ``consumption = population * litres_per_person_per_hour * heat_factor``. Recovering
        population this way rather than forecasting it separately guarantees the water-point
        requirement and the demand forecast cannot contradict each other.
        """
        divisor = self._litres_pph * self.heat_factor(temperature_c)
        return 0.0 if divisor <= 0 else max(0.0, litres_per_hr / divisor)

    def points_required(self, population: float) -> int:
        """docs/03 M15: ceil(peak population / persons_per_water_point)."""
        if population <= 0 or self._persons_per_point <= 0:
            return 0
        return math.ceil(population / self._persons_per_point)

    def tankers_required(self, gap_l_per_hr: float, *, hours: float | None = None) -> int:
        """docs/03 M15: ceil(max(gap, 0) * hours / tanker_payload_l)."""
        cover = float(hours if hours is not None else self.require_param("tanker_cover_hours"))
        if gap_l_per_hr <= 0 or self._tanker_payload_l <= 0:
            return 0
        return math.ceil(gap_l_per_hr * cover / self._tanker_payload_l)

    def _floor(self, value: float) -> float:
        return max(float(self.require_param("floor_value")), value)

    # ------------------------------------------------------------------ scenario state
    def _supply_at(
        self,
        scenario_id: str,
        zone: str,
        as_of: pd.Timestamp,
        overrides: dict[str, Any],
    ) -> tuple[float, bool]:
        """Planned supply per step for a zone, and whether an outage is suppressing it."""
        source = scenario_id if scenario_id in self._supply else "S01"
        series = self._supply[source].get(zone)
        if series is None or series.empty:
            return 0.0, False
        observed = series.loc[series.index <= as_of]
        supply = float(observed.iloc[-1] if len(observed) else series.iloc[0])

        # A generated world already contains the outage; only the generic layer needs it
        # applying by hand.
        if scenario_id in self._supply:
            return supply, False
        down = closed_entities(overrides)["substations"]
        if not down:
            return supply, False
        affected = any(
            spec.get("zone_id") == zone and spec.get("substation") in down
            for spec in self._pumps.values()
        )
        if not affected:
            return supply, False
        return supply * self._outage_factor, True

    def _window_temperature(self, scenario_id: str, as_of: pd.Timestamp, steps: int) -> float:
        """Peak temperature over the forecast window, not over all of history.

        Comparing a scenario window against the historical maximum compares different things:
        this world peaks at 28.9 C across 31 days while the 06:00 demo window sits at 24.6 C,
        so the historical maximum would silently absorb most of a heat offset.
        """
        series = self._temperature[scenario_id]
        end = as_of + pd.Timedelta(minutes=15 * steps)
        window = series.loc[(series.index >= as_of) & (series.index <= end)]
        return float(window.max() if len(window) else series.max())

    def _scenario_state(
        self,
        scenario_id: str,
        overrides: dict[str, Any] | None,
        upstream: dict[str, Any],
        *,
        as_of: pd.Timestamp,
        steps: int,
    ) -> tuple[dict[str, pd.Series], float, float, list[str]]:
        """Demand history, the temperature in force, the multiplier to apply, and notes.

        The adjustment is RETURNED rather than folded into the history, and applied to the
        forecast instead. Scaling the history and re-forecasting does not reliably move the
        answer: the S04 heat uplift is about 1.8 percent, and re-forecasting a history scaled
        by 1.018 while its covariates stayed put moved demand 3.4 percent the WRONG way. The
        adjustment was well inside the noise of the forecast itself. Applying a known
        multiplier to the prediction is exact and monotone, and is how the M21 multipliers
        already work.
        """
        notes: list[str] = []
        overrides = overrides or {}
        source = scenario_id if scenario_id in self._demand else "S01"
        demand = dict(self._demand[source])

        baseline_temp = self._window_temperature(source, as_of, steps)
        temperature = baseline_temp
        adjustment = 1.0

        if source != scenario_id:
            multiplier = float(overrides.get("footfall_multiplier", 1.0))
            if multiplier != 1.0:
                adjustment *= multiplier
                notes.append(
                    f"no generated world for {scenario_id}; the forecast was scaled by "
                    f"{multiplier} through the generic scenario layer"
                )
            # S04 is the case this exists for: heat reaches M15 only through temperature.
            offset = float(overrides.get("temperature_offset_c", 0.0))
            reported = self._m21_temperature(upstream)
            if reported is not None:
                temperature = reported
                notes.append(f"temperature {reported:.1f} C taken from M21")
            elif offset:
                temperature = baseline_temp + offset
                notes.append(
                    f"no M21 output; temperature raised by the scenario offset of {offset} C"
                )
            if temperature != baseline_temp:
                ratio = self.heat_factor(temperature) / self.heat_factor(baseline_temp)
                adjustment *= ratio
                notes.append(
                    f"demand scaled by {ratio:.3f} for {temperature:.1f} C against a window "
                    f"baseline of {baseline_temp:.1f} C"
                )
        return demand, temperature, adjustment, notes

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

        demand, temperature, adjustment, notes = self._scenario_state(
            scenario_id, overrides, upstream, as_of=as_of, steps=steps
        )
        for note in notes:
            builder.warn(note)

        trimmed = {
            zone: series.loc[series.index <= as_of]
            for zone, series in demand.items()
            if zone in entities
        }
        trimmed = {z: s for z, s in trimmed.items() if len(s) > steps + 2}
        if not trimmed:
            builder.warn(f"no history at or before {as_of.isoformat()}")
            return builder

        future_arrivals = self._m01_arrivals(upstream, as_of)
        if future_arrivals is None:
            builder.warn(
                "no M01 forecast available; the arrivals covariate was dropped and the "
                "forecast used the calendar and temperature covariates only"
            )
        source = scenario_id if scenario_id in self._demand else "S01"
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
            scenario_id=scenario_id,
            overrides=overrides,
            temperature=temperature,
            adjustment=adjustment,
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
        scenario_id: str,
        overrides: dict[str, Any],
        temperature: float,
        adjustment: float,
        used_m01: bool,
    ) -> None:
        confidence = float(self.require_param("confidence"))
        per_step = float(self.require_param("steps_per_hour"))
        heat = self.heat_factor(temperature)

        for zone in result.entities:
            supply_step, outage = self._supply_at(scenario_id, zone, as_of, overrides)
            supply_hr = supply_step * per_step
            peak_population = 0.0
            peak_gap = 0.0

            for timestamp in result.values[zone].index:
                value, low, high = result.at(zone, timestamp)
                demand_hr = self._floor(value * per_step * adjustment)
                gap_hr = demand_hr - supply_hr
                population = self.population_from_demand(demand_hr, temperature)
                peak_population = max(peak_population, population)
                peak_gap = max(peak_gap, gap_hr)
                offset = max(0, int((timestamp - as_of).total_seconds() // 60))

                reasons: list[str] = []
                if gap_hr > 0:
                    reasons.append(REASON_WATER_GAP)
                if heat > self.heat_factor(0.0):  # hotter than the no-heat baseline
                    reasons.append(REASON_HEAT_STRESS)
                if outage:
                    reasons.append(REASON_GRID_OUTAGE)

                details = {
                    "backend": result.backend_used,
                    "band_calibrated": result.calibrated,
                    "supply_l_per_hr": round(supply_hr, 1),
                    "heat_factor": round(heat, 4),
                    "scenario_adjustment": round(adjustment, 4),
                    "temperature_c": round(temperature, 1),
                    "population_present": round(population),
                    "pump_outage": outage,
                    "arrivals_covariate": "M01" if used_m01 else "none",
                }

                if "expected_water_demand" in kpis:
                    builder.add(
                        EntityType.ZONE,
                        zone,
                        "expected_water_demand",
                        timestamp,
                        demand_hr,
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        lower=None if low is None else self._floor(low * per_step * adjustment),
                        upper=None if high is None else self._floor(high * per_step * adjustment),
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=details,
                    )

                if "water_supply_demand_gap" in kpis:
                    # NOT floored: a surplus is a real answer to "have we got enough".
                    builder.add(
                        EntityType.ZONE,
                        zone,
                        "water_supply_demand_gap",
                        timestamp,
                        gap_hr,
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        lower=None
                        if low is None
                        else self._floor(low * per_step * adjustment) - supply_hr,
                        upper=None
                        if high is None
                        else self._floor(high * per_step * adjustment) - supply_hr,
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=details,
                    )

                if "water_tanker_requirement" in kpis:
                    tankers = self.tankers_required(gap_hr)
                    builder.add(
                        EntityType.ZONE,
                        zone,
                        "water_tanker_requirement",
                        timestamp,
                        float(tankers),
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        confidence=confidence,
                        resource_type=RESOURCE_TANKER,
                        reason_codes=reasons,
                        recommendation=self._tanker_run(zone, tankers, gap_hr)
                        if tankers > 0
                        else None,
                        details=details,
                    )

            # Points are sized once per zone, against the PEAK of the horizon: a water point
            # is placed for the busiest moment, not re-sited every 15 minutes.
            if "drinking_water_point_requirement" in kpis:
                points = self.points_required(peak_population)
                builder.add(
                    EntityType.ZONE,
                    zone,
                    "drinking_water_point_requirement",
                    as_of,
                    float(points),
                    zone_id=zone,
                    horizon_min=0,
                    state=state,
                    confidence=confidence,
                    resource_type=RESOURCE_POINT,
                    reason_codes=[REASON_WATER_GAP] if peak_gap > 0 else [],
                    details={
                        "backend": result.backend_used,
                        "peak_population": round(peak_population),
                        "persons_per_point": self._persons_per_point,
                        "sized_against": "peak of the horizon",
                        "temperature_c": round(temperature, 1),
                    },
                )

    def _tanker_run(self, zone: str, tankers: int, gap_l_per_hr: float) -> Recommendation:
        hours = float(self.require_param("tanker_cover_hours"))
        return Recommendation(
            action=f"dispatch water tankers to {zone}",
            resource_type=RESOURCE_TANKER,
            quantity=float(tankers),
            target_entity_id=zone,
            rationale=(
                f"{zone} is forecast {gap_l_per_hr:,.0f} L/hr short of supply; "
                f"{tankers} tanker(s) cover {hours:.0f} hours at "
                f"{self._tanker_payload_l:,.0f} L each"
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
            question=(
                "How much water is needed by zone, is there a gap, and how many points and tankers?"
            ),
            engine=self.engine,
            method_summary=(
                f"Darts {backend} over one consumption series per zone, with calendar, "
                f"temperature and M01 arrivals as covariates. Supply is carried forward as a "
                f"planned quantity; gap is demand minus supply, positive meaning a shortage. "
                f"Points come from peak population recovered by inverting the consumption "
                f"relation; tankers from the gap over a cover window."
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
                "Supply is the design output of the pumps, carried forward. Real pump curves "
                "vary with pressure and head, and leakage is not modelled at all - a real "
                "network loses a substantial share between pump and tap.",
                "Storage is not drawn down. A zone with a positive gap is reported as short "
                "even when its tanks could cover the shortfall for hours, which makes the "
                "tanker requirement conservative.",
                "Population is recovered by inverting the generator's consumption relation, "
                "so it is exactly consistent with the demand forecast but inherits any error "
                "in the litres-per-person assumption.",
                "S04 has no generated world; heat is applied through M21's temperature using "
                "the same multiplier the generator uses.",
                "Water points are sized against the peak of the horizon and are not placed "
                "geographically - that is M16's and M24's work.",
                "Synthetic context; the backtest measures pipeline correctness, not accuracy.",
            ],
            license_notes=[
                "Darts: Apache-2.0. Chronos-2 weights: see the autogluon model card.",
                "LightGBM: MIT.",
            ],
        )
