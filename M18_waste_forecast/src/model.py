"""M18 - Waste Generation.

How much waste, when do bins overflow, how many collection trips?

**Method** (docs/03 M18 card)

Forecast
    Kilograms added per bin group from D16, on the shared forecast engine, with calendar and
    M01 arrival covariates.

Fill projection
    ``current fill + inflow x t`` from the last observed level until the next collection.
    The bin empties on the round, so the projection restarts there rather than growing without
    bound.

Overflow time
    The first moment the projection reaches 100 percent, in minutes from ``as_of``. A bin that
    does not fill inside the window reports ``no_overflow_minutes`` rather than null, so the
    series stays numeric for a dashboard, and rather than 0, which a reader could mistake for
    "overflowing right now".

Trips
    ``ceil(kg per day / vehicle_payload_kg)``.

**Fill is reported clipped at 100 percent.** A bin cannot be more than full; what happens past
that is waste on the ground, and the thing an operator needs is not "137 percent" but *when*
it crossed - which is exactly what ``bin_overflow_time`` says.
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
from twin_common.upstream import series_for_entity

log = get_logger(__name__)

#: Reason code M18 can raise (docs/05 section 4, supply group).
REASON_BIN_OVERFLOW = "BIN_OVERFLOW"

#: The resource a collection recommendation asks for (resources.yaml).
RESOURCE_WASTE_VEHICLE = "waste_vehicle"


class Model(TwinModel):
    """Waste Generation (M18)."""

    model_id = "M18"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        waste = assumptions()["waste"]
        # A bin GROUP holds many bins, so the group capacity is what fill is measured
        # against. See the note in assumptions.yaml.
        self._bins_per_group = float(waste["bins_per_group"])
        self._bin_capacity_kg = float(waste["bin_capacity_kg"]) * self._bins_per_group
        self._payload_kg = float(waste["vehicle_payload_kg"])
        self._zone_of = {
            bin_id: spec["zone_id"] for bin_id, spec in world()["waste_bin_groups"].items()
        }

        self._added: dict[str, dict[str, pd.Series]] = {}
        self._fill: dict[str, dict[str, pd.Series]] = {}
        self._arrivals: dict[str, pd.Series] = {}
        self._snan_days: dict[str, list[pd.Timestamp]] = {}
        for scenario_id in self.scenarios_supported:
            try:
                self._load_scenario(scenario_id)
            except Exception as exc:
                log.warning("no world slice for %s (%s)", scenario_id, exc)
        if "S01" not in self._added:
            raise RuntimeError(
                "M18 needs at least the S01 world slice. Run `python scripts/slice_world.py M18`."
            )

        self.forecast_engine = ForecastEngine.from_config(self.param("forecast"), seed=self.seed)
        try:
            result = self.forecast_engine.warmup(
                self._added["S01"],
                self.horizon_steps(self.default_horizon_min),
                future_covariates=self._covariate_frames("S01", list(self._added["S01"])),
            )
            log.info(
                "M18 warm: backend %s, %d bin groups, %.1fs",
                result.backend_used,
                len(result.entities),
                result.elapsed_s,
            )
        except Exception as exc:
            log.warning("M18 warmup failed (%s); the first request will be slower", exc)

    def _load_scenario(self, scenario_id: str) -> None:
        table = load_table(
            "waste", self.data_source, scenario_id=scenario_id, base_dir=self.synthetic_dir
        )
        self._added[scenario_id] = series_by_entity(
            table, entity_column="bin_group_id", value_column="kg_added"
        )
        self._fill[scenario_id] = series_by_entity(
            table, entity_column="bin_group_id", value_column="fill_pct"
        )
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
        series = next(iter(self._added[scenario_id].values()))
        index = pd.DatetimeIndex(series.index)
        horizon = self.horizon_steps(max(self.horizons_min))
        extended = pd.date_range(
            start=index[0], periods=len(index) + horizon, freq="15min", tz=index.tz
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

    # ------------------------------------------------------------------ formulas
    def horizon_steps(self, horizon_min: int) -> int:
        return max(1, round(horizon_min / 15))

    def entity_ids(self) -> list[str]:
        return list(self._added["S01"])

    def fill_after(self, opening_pct: float, kg_added: float) -> float:
        """Fill after adding ``kg_added`` to a bin group, clipped at full.

        A bin cannot be more than full. What happens past that is waste on the ground, and
        `bin_overflow_time` is what says when it started.
        """
        ceiling = float(self.require_param("max_fill_pct"))
        if self._bin_capacity_kg <= 0:
            return 0.0
        added_pct = kg_added / self._bin_capacity_kg * 100.0
        return float(min(ceiling, max(0.0, opening_pct + added_pct)))

    def minutes_to_overflow(
        self, opening_pct: float, kg_per_step: float, *, step_minutes: float = 15.0
    ) -> float:
        """When the projection first reaches full, in minutes.

        Returns ``no_overflow_minutes`` when the bin is not filling, so a dashboard series
        stays numeric and "not overflowing" cannot be read as "overflowing now".
        """
        ceiling = float(self.require_param("max_fill_pct"))
        none = float(self.require_param("no_overflow_minutes"))
        if opening_pct >= ceiling:
            return 0.0
        if kg_per_step <= 0 or self._bin_capacity_kg <= 0:
            return none
        per_step_pct = kg_per_step / self._bin_capacity_kg * 100.0
        if per_step_pct <= 0:
            return none
        steps = (ceiling - opening_pct) / per_step_pct
        return float(min(none, steps * step_minutes))

    def trips_required(self, kg_per_day: float) -> int:
        """docs/03 M18: ceil(kg per day / vehicle_payload_kg)."""
        if kg_per_day <= 0 or self._payload_kg <= 0:
            return 0
        return math.ceil(kg_per_day / self._payload_kg)

    def collection_window_min(self, overrides: dict[str, Any] | None = None) -> float:
        """How long until the next round, including any scenario delay."""
        base = float(self.require_param("collection_interval_min"))
        multiplier = float(self.require_param("collection_delay_multiplier"))
        if overrides:
            multiplier = float(overrides.get("waste_collection_delay_multiplier", multiplier))
        return base * multiplier

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

        source = scenario_id if scenario_id in self._added else "S01"
        added = dict(self._added[source])
        adjustment = 1.0
        if source != scenario_id:
            multiplier = float(overrides.get("footfall_multiplier", 1.0))
            if multiplier != 1.0:
                adjustment = multiplier
                builder.warn(
                    f"no generated world for {scenario_id}; the forecast was scaled by "
                    f"{multiplier} through the generic scenario layer"
                )

        window = self.collection_window_min(overrides)
        if window != float(self.require_param("collection_interval_min")):
            builder.warn(f"collection round stretched to {window:.0f} min")

        entities = self.resolve_entities(request, list(added))
        if not entities:
            builder.warn("no known entities matched the request")
            return builder

        trimmed = {
            bin_id: series.loc[series.index <= as_of]
            for bin_id, series in added.items()
            if bin_id in entities
        }
        trimmed = {b: s for b, s in trimmed.items() if len(s) > steps + 2}
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
            window_min=window,
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
        window_min: float,
        used_m01: bool,
    ) -> None:
        confidence = float(self.require_param("confidence"))
        per_step = float(self.require_param("steps_per_hour"))
        hours_per_day = float(self.require_param("hours_per_day"))

        for bin_id in result.entities:
            zone = self._zone_of.get(bin_id)
            fill_series = self._fill[source].get(bin_id)
            observed = (
                fill_series.loc[fill_series.index <= as_of] if fill_series is not None else None
            )
            opening = float(observed.iloc[-1]) if observed is not None and len(observed) else 0.0

            fill = opening
            steps_to_collection = max(1, round(window_min / 15.0))
            overflow_at = None

            for position, timestamp in enumerate(result.values[bin_id].index):
                value, low, high = result.at(bin_id, timestamp)
                kg_step = self._floor(value * adjustment)
                offset = max(0, int((timestamp - as_of).total_seconds() // 60))

                # The round empties the bin, so the projection restarts rather than growing
                # without bound.
                if position > 0 and position % steps_to_collection == 0:
                    fill = 0.0
                fill = self.fill_after(fill, kg_step)
                if overflow_at is None and fill >= float(self.require_param("max_fill_pct")):
                    overflow_at = offset

                kg_day = kg_step * per_step * hours_per_day
                overflowing = fill >= float(self.require_param("max_fill_pct"))
                # docs/02 section 4 makes a reason code mandatory from amber upwards, and the
                # fill band turns amber well before a bin is actually full. BIN_OVERFLOW is
                # the only code in the docs/05 supply group that fits, and at amber it reads
                # as what it is: overflow RISK. The band comes from risk_bands.yaml rather
                # than a number here, which keeps the threshold out of src/ (CLAUDE.md).
                banded = risk_level_for_kpi("waste_bin_fill_level", fill)
                reasons = (
                    [REASON_BIN_OVERFLOW] if overflowing or banded in REASON_REQUIRED_LEVELS else []
                )
                details = {
                    "backend": result.backend_used,
                    "band_calibrated": result.calibrated,
                    "opening_fill_pct": round(opening, 1),
                    "bin_capacity_kg": self._bin_capacity_kg,
                    "collection_window_min": window_min,
                    "kg_per_step": round(kg_step, 2),
                    "arrivals_covariate": "M01" if used_m01 else "none",
                }

                if "waste_generation" in kpis:
                    builder.add(
                        EntityType.SERVICE_POINT,
                        bin_id,
                        "waste_generation",
                        timestamp,
                        kg_day,
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        lower=None
                        if low is None
                        else self._floor(low * adjustment) * per_step * hours_per_day,
                        upper=None
                        if high is None
                        else self._floor(high * adjustment) * per_step * hours_per_day,
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=details,
                    )

                if "waste_bin_fill_level" in kpis:
                    builder.add(
                        EntityType.SERVICE_POINT,
                        bin_id,
                        "waste_bin_fill_level",
                        timestamp,
                        fill,
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=details,
                    )

                if "waste_collection_trips" in kpis:
                    builder.add(
                        EntityType.SERVICE_POINT,
                        bin_id,
                        "waste_collection_trips",
                        timestamp,
                        float(self.trips_required(kg_day)),
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        confidence=confidence,
                        resource_type=RESOURCE_WASTE_VEHICLE,
                        reason_codes=reasons,
                        details=details,
                    )

            if "bin_overflow_time" in kpis:
                minutes = (
                    float(overflow_at)
                    if overflow_at is not None
                    else self.minutes_to_overflow(opening, 0.0)
                )
                builder.add(
                    EntityType.SERVICE_POINT,
                    bin_id,
                    "bin_overflow_time",
                    as_of,
                    minutes,
                    zone_id=zone,
                    horizon_min=0,
                    state=state,
                    confidence=confidence,
                    reason_codes=[REASON_BIN_OVERFLOW] if overflow_at is not None else [],
                    recommendation=self._collection(bin_id, minutes)
                    if overflow_at is not None
                    else None,
                    details={
                        "backend": result.backend_used,
                        "opening_fill_pct": round(opening, 1),
                        "collection_window_min": window_min,
                        "overflows_inside_horizon": overflow_at is not None,
                    },
                )

    def _collection(self, bin_id: str, minutes: float) -> Recommendation:
        return Recommendation(
            action=f"bring forward the waste collection at {bin_id}",
            resource_type=RESOURCE_WASTE_VEHICLE,
            quantity=1.0,
            target_entity_id=bin_id,
            rationale=(
                f"{bin_id} is projected to reach full in {minutes:.0f} min, before the next "
                f"scheduled round"
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
            question="How much waste, when do bins overflow, how many collection trips?",
            engine=self.engine,
            method_summary=(
                f"Darts {backend} over one kg-added series per bin group, with calendar and "
                f"M01 arrival covariates. Fill is projected from the last observed level and "
                f"reset on each collection round; overflow time is when that projection first "
                f"reaches full."
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
                "Collection is a fixed round at a fixed interval, not a schedule. A real "
                "operation collects on a route with travel time between stops.",
                "Fill is clipped at 100 percent. Waste on the ground past that point is not "
                "quantified; bin_overflow_time says when it started, not how much.",
                "Opening fill is the last observed level, so any sensor drift propagates "
                "through the whole projected horizon.",
                "Insensitive to S06 and S12: the docs/03 M18 card wants collection delayed, "
                "which needs M04 to say by how much. The delay override is wired and tested.",
                "Trips are quoted per day from a 15-minute rate, which assumes the forecast "
                "step is representative of the day.",
                "Synthetic context; the backtest measures pipeline correctness, not accuracy.",
            ],
            license_notes=[
                "Darts: Apache-2.0. Chronos-2 weights: see the autogluon model card.",
                "LightGBM: MIT.",
            ],
        )
