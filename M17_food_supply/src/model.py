"""M17 - Food & Supply.

How much food is needed, how long does stock last, how many deliveries?

**Method** (docs/03 M17 card)

Forecast
    Meals sold per outlet from D17, on the shared forecast engine. **D17 is hourly**, not on
    the 15-minute grid the other forecasting models use, so one step is one hour and the
    horizons are whole hours. The default is six, matching the food lead time: a shorter
    horizon could not support the decision this model exists for, which is whether to order
    now.

Coverage
    ``stock / consumption rate``, in hours, where stock comes from an **inventory simulation**
    walked forward across the horizon rather than a constant snapshot. Capped for reporting,
    because an outlet holding stock against near-zero overnight demand has mathematically
    unbounded cover - true, and useless on a dashboard.

    The simulation is the reason S13 does anything. A longer lead time changes neither the
    stock on hand nor the rate it is eaten, so a snapshot of ``stock / rate`` is identical
    under S13. What changes is when a replenishment lands.

Deliveries
    ``ceil(replenishment_meals / vehicle_payload_meals)``, quoted per day to match the KPI
    unit. Replenishment is what the outlet must take in over a day to hold its stock level,
    which is its forecast daily consumption.

Stockout
    Coverage below ``min_stock_cover_hours`` raises ``STOCKOUT_RISK`` and recommends a
    delivery. The comparison is against the **lead time** as well: stock that outlasts the
    minimum cover but not the time it takes a lorry to arrive is still a stockout in waiting,
    which is the mechanism S13 exercises by stretching the lead time by 1.5.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from twin_common.config import assumptions
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
from twin_common.upstream import series_for_entity

log = get_logger(__name__)

#: Reason codes M17 can raise (docs/05 section 4, supply group).
REASON_STOCKOUT_RISK = "STOCKOUT_RISK"

#: The resource a delivery recommendation asks for (resources.yaml).
RESOURCE_FOOD_VEHICLE = "food_vehicle"


class Model(TwinModel):
    """Food & Supply (M17)."""

    model_id = "M17"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        food = assumptions()["food"]
        self._payload_meals = float(food["vehicle_payload_meals"])
        self._lead_time_hours = float(food["lead_time_hours"])
        self._min_cover_hours = float(food["min_stock_cover_hours"])

        self._meals: dict[str, dict[str, pd.Series]] = {}
        self._stock: dict[str, dict[str, pd.Series]] = {}
        self._zone_of: dict[str, str] = {}
        self._arrivals: dict[str, pd.Series] = {}
        self._snan_days: dict[str, list[pd.Timestamp]] = {}
        for scenario_id in self.scenarios_supported:
            try:
                self._load_scenario(scenario_id)
            except Exception as exc:
                log.warning("no world slice for %s (%s)", scenario_id, exc)
        if "S01" not in self._meals:
            raise RuntimeError(
                "M17 needs at least the S01 world slice. Run `python scripts/slice_world.py M17`."
            )

        self.forecast_engine = ForecastEngine.from_config(self.param("forecast"), seed=self.seed)
        try:
            result = self.forecast_engine.warmup(
                self._meals["S01"],
                self.horizon_steps(self.default_horizon_min),
                future_covariates=self._covariate_frames("S01", list(self._meals["S01"])),
            )
            log.info(
                "M17 warm: backend %s, %d outlets, %.1fs",
                result.backend_used,
                len(result.entities),
                result.elapsed_s,
            )
        except Exception as exc:
            log.warning("M17 warmup failed (%s); the first request will be slower", exc)

    def _load_scenario(self, scenario_id: str) -> None:
        table = load_table(
            "food_inventory_hourly",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        self._meals[scenario_id] = series_by_entity(
            table, entity_column="outlet_id", value_column="meals_sold"
        )
        self._stock[scenario_id] = series_by_entity(
            table, entity_column="outlet_id", value_column="stock_meals"
        )
        self._zone_of.update(dict(zip(table["outlet_id"], table["zone_id"], strict=True)))

        footfall = load_table(
            "footfall_15min",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        # D01 is 15-minute; D17 is hourly. Resample so the covariate shares the target grid.
        self._arrivals[scenario_id] = (
            footfall.groupby("timestamp")["entries"]
            .sum()
            .astype("float64")
            .sort_index()
            .resample("1h")
            .sum()
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
        series = next(iter(self._meals[scenario_id].values()))
        index = pd.DatetimeIndex(series.index)
        horizon = self.horizon_steps(max(self.horizons_min))
        extended = pd.date_range(
            start=index[0], periods=len(index) + horizon, freq="1h", tz=index.tz
        )
        extra: dict[str, pd.Series] = {}
        history = self._arrivals.get(scenario_id)
        if history is not None and future_arrivals is not None and len(future_arrivals):
            combined = pd.concat([history, future_arrivals])
            extra["arrivals"] = combined[~combined.index.duplicated(keep="last")].sort_index()
        frame = calendar_covariates(
            extended, snan_days=self._snan_days.get(scenario_id, []), extra=extra
        )
        return dict.fromkeys(entities, frame)

    def _m01_arrivals(self, upstream: dict[str, Any], as_of: pd.Timestamp) -> pd.Series | None:
        """M01's forecast arrivals, resampled onto the hourly grid D17 uses."""
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
                    # M01 reports persons/hr on a 15-minute grid; a quarter of each falls in
                    # the hour it belongs to.
                    totals[stamp] = totals.get(stamp, 0.0) + float(value) / 4.0
        if not totals:
            return None
        return pd.Series(totals).sort_index().resample("1h").sum()

    # ------------------------------------------------------------------ formulas
    def horizon_steps(self, horizon_min: int) -> int:
        """Minutes to HOURLY steps: D17 is not on the 15-minute grid."""
        return max(1, round(horizon_min / 60))

    def entity_ids(self) -> list[str]:
        return list(self._meals["S01"])

    def coverage_hours(self, stock_meals: float, meals_per_hour: float) -> float:
        """docs/03 M17: stock divided by the consumption rate.

        Capped for reporting: an outlet with stock and near-zero overnight demand has
        unbounded cover, which is true and useless. Zero demand reports the cap rather than
        raising, because "nothing is being eaten" is not a stockout.
        """
        ceiling = float(self.require_param("max_coverage_hours"))
        if stock_meals <= 0:
            return 0.0
        if meals_per_hour <= 0:
            return ceiling
        return float(min(ceiling, stock_meals / meals_per_hour))

    def trips_required(self, meals_per_hour: float) -> int:
        """docs/03 M17: ceil(replenishment / vehicle_payload_meals), quoted per day."""
        hours = float(self.require_param("hours_per_day"))
        if meals_per_hour <= 0 or self._payload_meals <= 0:
            return 0
        return math.ceil(meals_per_hour * hours / self._payload_meals)

    def simulate_inventory(
        self,
        opening_stock: float,
        demand: list[float],
        lead_time_hours: float,
    ) -> list[dict[str, float]]:
        """Walk stock forward, ordering when cover runs short (docs/03 M17).

        This is what makes S13 bite. A longer lead time does not change how much stock an
        outlet holds today, nor how fast it is eaten - so a snapshot of ``stock / rate`` is
        identical under S13 and says the scenario does nothing. What actually changes is
        WHEN a replenishment lands: an order placed now arrives ``lead_time`` hours later,
        and if that is beyond the horizon the stock simply keeps falling.

        The policy is the simplest one that reflects the card: when projected cover drops
        below the minimum, order enough whole vehicle loads to carry the outlet through the
        lead time plus that minimum. Orders already in flight are not duplicated.
        """
        lead_steps = max(1, math.ceil(lead_time_hours))
        pending: dict[int, float] = {}
        stock = max(0.0, opening_stock)
        rows: list[dict[str, float]] = []

        for step, meals in enumerate(demand):
            arrived = pending.pop(step, 0.0)
            stock = max(0.0, stock + arrived - max(0.0, meals))
            coverage = self.coverage_hours(stock, meals)

            ordered = 0.0
            if coverage < max(self._min_cover_hours, lead_time_hours) and not pending:
                # Enough to cover the wait for the lorry plus the minimum buffer.
                target = meals * (lead_time_hours + self._min_cover_hours)
                loads = math.ceil(max(0.0, target - stock) / self._payload_meals)
                ordered = loads * self._payload_meals
                if ordered > 0:
                    pending[step + lead_steps] = ordered

            rows.append(
                {
                    "stock": stock,
                    "coverage": coverage,
                    "arrived": arrived,
                    "ordered": ordered,
                }
            )
        return rows

    def is_stockout_risk(self, coverage: float, lead_time_hours: float) -> bool:
        """Short of the minimum cover, OR short of the time a delivery takes to arrive.

        The second test is what makes S13 bite: stretching the lead time does not change how
        much stock an outlet holds, but it does change whether that stock lasts long enough
        for the lorry to get there.
        """
        return coverage < max(self._min_cover_hours, lead_time_hours)

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

        source = scenario_id if scenario_id in self._meals else "S01"
        meals = dict(self._meals[source])
        adjustment = 1.0
        if source != scenario_id:
            multiplier = float(overrides.get("footfall_multiplier", 1.0))
            if multiplier != 1.0:
                adjustment = multiplier
                builder.warn(
                    f"no generated world for {scenario_id}; the forecast was scaled by "
                    f"{multiplier} through the generic scenario layer"
                )

        # S13 stretches the lead time. It does not change stock, only whether the stock lasts
        # long enough for a delivery to arrive.
        lead_multiplier = float(overrides.get("food_lead_time_multiplier", 1.0))
        lead_time = self._lead_time_hours * lead_multiplier
        if lead_multiplier != 1.0:
            builder.warn(
                f"lead time stretched to {lead_time:.1f} h by a factor of {lead_multiplier}"
            )

        entities = self.resolve_entities(request, list(meals))
        if not entities:
            builder.warn("no known entities matched the request")
            return builder

        trimmed = {
            outlet: series.loc[series.index <= as_of]
            for outlet, series in meals.items()
            if outlet in entities
        }
        trimmed = {o: s for o, s in trimmed.items() if len(s) > steps + 2}
        if not trimmed:
            builder.warn(f"no history at or before {as_of.isoformat()}")
            return builder

        upstream = self.resolve_upstream(request, scenario_id=scenario_id, entity_ids=entities)
        self.record_upstream(builder, upstream)
        future_arrivals = self._m01_arrivals(upstream, as_of)
        if future_arrivals is None:
            builder.warn(
                "no M01 forecast available; the arrivals covariate was dropped and the "
                "forecast used the calendar covariates only"
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
            adjustment=adjustment,
            lead_time=lead_time,
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
        adjustment: float,
        lead_time: float,
        used_m01: bool,
    ) -> None:
        confidence = float(self.require_param("confidence"))

        for outlet in result.entities:
            zone = self._zone_of.get(outlet)
            stock_series = self._stock[source].get(outlet)
            observed = (
                stock_series.loc[stock_series.index <= as_of] if stock_series is not None else None
            )
            opening = float(observed.iloc[-1]) if observed is not None and len(observed) else 0.0

            timestamps = list(result.values[outlet].index)
            demands = [
                self._floor(result.at(outlet, stamp)[0] * adjustment) for stamp in timestamps
            ]
            simulated = self.simulate_inventory(opening, demands, lead_time)

            for position, timestamp in enumerate(timestamps):
                _, low, high = result.at(outlet, timestamp)
                demand = demands[position]
                stock = simulated[position]["stock"]
                coverage = simulated[position]["coverage"]
                at_risk = self.is_stockout_risk(coverage, lead_time)
                offset = max(0, int((timestamp - as_of).total_seconds() // 60))
                reasons = [REASON_STOCKOUT_RISK] if at_risk else []
                details = {
                    "backend": result.backend_used,
                    "band_calibrated": result.calibrated,
                    "zone_id": zone,
                    "stock_meals": round(stock, 1),
                    "opening_stock_meals": round(opening, 1),
                    "meals_arrived": round(simulated[position]["arrived"], 1),
                    "meals_ordered": round(simulated[position]["ordered"], 1),
                    "lead_time_hours": round(lead_time, 2),
                    "min_cover_hours": self._min_cover_hours,
                    "arrivals_covariate": "M01" if used_m01 else "none",
                }

                if "food_demand" in kpis:
                    builder.add(
                        EntityType.FACILITY,
                        outlet,
                        "food_demand",
                        timestamp,
                        demand,
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        lower=None if low is None else self._floor(low * adjustment),
                        upper=None if high is None else self._floor(high * adjustment),
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=details,
                    )

                if "food_stock_coverage" in kpis:
                    builder.add(
                        EntityType.FACILITY,
                        outlet,
                        "food_stock_coverage",
                        timestamp,
                        coverage,
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        confidence=confidence,
                        reason_codes=reasons,
                        recommendation=self._delivery(outlet, coverage, demand)
                        if at_risk
                        else None,
                        details=details,
                    )

                if "delivery_requirement" in kpis:
                    builder.add(
                        EntityType.FACILITY,
                        outlet,
                        "delivery_requirement",
                        timestamp,
                        float(self.trips_required(demand)),
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        confidence=confidence,
                        resource_type=RESOURCE_FOOD_VEHICLE,
                        reason_codes=reasons,
                        details=details,
                    )

    def _delivery(self, outlet: str, coverage: float, demand: float) -> Recommendation:
        return Recommendation(
            action=f"schedule a food delivery to {outlet}",
            resource_type=RESOURCE_FOOD_VEHICLE,
            quantity=float(self.trips_required(demand)),
            target_entity_id=outlet,
            rationale=(
                f"{outlet} has {coverage:.1f} h of cover against a {self._min_cover_hours:.0f} h "
                f"minimum and a lead time that is longer still, so an order placed later "
                f"would not arrive in time"
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
            question="How much food is needed, how long does stock last, how many deliveries?",
            engine=self.engine,
            method_summary=(
                f"Darts {backend} over one meals-sold series per outlet on the HOURLY D17 "
                f"grid, with calendar and M01 arrival covariates. Coverage is stock over the "
                f"forecast consumption rate; a stockout risk is raised when cover is shorter "
                f"than the minimum OR than the delivery lead time."
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
                "The inventory simulation uses one simple reorder policy: order enough whole "
                "vehicle loads to cover the lead time plus the minimum buffer, and never "
                "order while an order is in flight. A real operation batches, splits and "
                "expedites; this is a first-order model of when stock runs out, not a "
                "replenishment plan.",
                "Opening stock is the last observed level. Any drift between the recorded "
                "level and what is physically on the shelf propagates through the whole "
                "simulated horizon.",
                "Coverage is capped for reporting; an outlet with stock and no demand shows "
                "the cap rather than infinity.",
                "Only two outlets exist in this world, both in the market zone, so the "
                "per-outlet split carries very little information.",
                "Deliveries are quoted per day from an hourly rate, which assumes the "
                "forecast hour is representative of the day.",
                "Insensitive to S06: the docs/03 M17 card makes the lead-time increase "
                "optional and dependent on M04, which is a Phase 7 model. The override is "
                "wired and tested, so adding M04 upstream is all that is needed.",
                "Synthetic context; the backtest measures pipeline correctness, not accuracy.",
            ],
            license_notes=[
                "Darts: Apache-2.0. Chronos-2 weights: see the autogluon model card.",
                "LightGBM: MIT.",
            ],
        )
