"""M06 - Transit / Shuttle Demand.

How many passengers and shuttles are needed per route?

**Method** (docs/03 M06 card)

Forecast
    Boardings per shuttle route from D07, on the shared forecast engine. Covariates are the
    calendar plus arrivals, whose future half comes from M01 - the same arrangement M05 uses,
    and for the same reason: arrivals are not known ahead, so taking them from the generated
    world would be peeking.

Fleet
    ``buses = ceil(demand * round_trip_min / 60 / (bus_capacity * bus_load_factor))``, with
    every input from ``assumptions.transport`` and the fleet size from ``resources.yaml``, so
    M06 and M24 cannot disagree about how many vehicles exist.

Wait
    ``headway / 2`` with ``headway = round_trip / deployed``, exactly as the card says - but
    multiplied by an overload factor when the deployed fleet cannot carry the demand. Without
    that, capping a hopelessly oversubscribed route at the available fleet makes the wait
    *shorter* (fewer buses, but the formula only sees headway), which is the opposite of what
    happens to a passenger. See :meth:`wait_minutes`.

**The fleet is far too small for this world.** At the snan peak the two routes need about
1,447 buses against the 80 in ``resources.yaml`` - 18x - and 369 even on an ordinary day.
That number is left alone deliberately: ``available`` is M24's input, read by several models,
and changing it here would pre-empt a model that does not exist yet. M06 reports the true
requirement, raises the shortfall, and recommends more vehicles. The gap is the finding.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from twin_common.config import assumptions, resources_config, world
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

#: Reason codes M06 can raise (docs/05 section 4).
REASON_CONGESTION = "CONGESTION"
REASON_PARKING_OVERFLOW = "PARKING_OVERFLOW"

#: The resource a shortfall recommendation asks for (resources.yaml).
RESOURCE_SHUTTLE = "shuttle_bus"


class Model(TwinModel):
    """Transit / Shuttle Demand (M06)."""

    model_id = "M06"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Read D07 per scenario, note the routes and the fleet, and warm the engine."""
        self._routes: dict[str, dict[str, Any]] = dict(world()["shuttle_routes"])
        transport = assumptions()["transport"]
        self._round_trip_min = float(transport["round_trip_min"])
        self._bus_capacity = float(transport["bus_capacity"])
        self._load_factor = float(transport["bus_load_factor"])
        self._fleet = float(resources_config()["resource_types"][RESOURCE_SHUTTLE]["available"])

        self._boardings: dict[str, dict[str, pd.Series]] = {}
        self._arrivals: dict[str, pd.Series] = {}
        self._snan_days: dict[str, list[pd.Timestamp]] = {}
        for scenario_id in self.scenarios_supported:
            try:
                self._load_scenario(scenario_id)
            except Exception as exc:
                log.warning("no world slice for %s (%s)", scenario_id, exc)
        if "S01" not in self._boardings:
            raise RuntimeError(
                "M06 needs at least the S01 world slice. Run `python scripts/slice_world.py M06`."
            )

        self.forecast_engine = ForecastEngine.from_config(self.param("forecast"), seed=self.seed)
        try:
            result = self.forecast_engine.warmup(
                self._boardings["S01"],
                self.horizon_steps(self.default_horizon_min),
                future_covariates=self._covariate_frames("S01", list(self._boardings["S01"])),
            )
            log.info(
                "M06 warm: backend %s, %d routes, %.1fs",
                result.backend_used,
                len(result.entities),
                result.elapsed_s,
            )
        except Exception as exc:
            log.warning("M06 warmup failed (%s); the first request will be slower", exc)

    def _load_scenario(self, scenario_id: str) -> None:
        transit = load_table(
            "transit_15min",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        self._boardings[scenario_id] = series_by_entity(
            transit, entity_column="route_id", value_column="boardings"
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
        """Calendar for every route, plus arrivals when M01 supplies the future half."""
        series = next(iter(self._boardings[scenario_id].values()))
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
        """M01's forecast arrivals per step, summed over its zones."""
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
    def _m05_overflow(upstream: dict[str, Any]) -> float:
        """Peak vehicles that will not fit in a car park, from M05.

        Those drivers still have to reach the venue, so an overflow is extra shuttle demand.
        It is reported in ``details`` rather than added to the forecast: the split between
        "park elsewhere and walk" and "take a shuttle" is not something M06 can know, and
        inventing a share would be a number with no evidence behind it.
        """
        resolution = upstream.get("M05")
        if resolution is None:
            return 0.0
        return max(
            (
                float(record.details.get("overflow_vehicles", 0.0))
                for record in resolution.output.results
                if record.kpi == "parking_demand"
            ),
            default=0.0,
        )

    # ------------------------------------------------------------------ fleet maths
    def horizon_steps(self, horizon_min: int) -> int:
        return max(1, round(horizon_min / 15))

    def entity_ids(self) -> list[str]:
        return list(self._routes)

    def buses_required(self, demand_per_hr: float, *, round_trip_min: float | None = None) -> int:
        """docs/03 M06: ceil(demand * round_trip/60 / (capacity * load_factor))."""
        if demand_per_hr <= 0:
            return 0
        round_trip = float(round_trip_min if round_trip_min is not None else self._round_trip_min)
        per_bus = self._bus_capacity * self._load_factor
        if per_bus <= 0:
            raise ValueError("bus_capacity * bus_load_factor must be positive")
        required = math.ceil(demand_per_hr * round_trip / 60.0 / per_bus)
        return max(int(self.require_param("min_buses_when_demand")), required)

    def capacity_per_hour(self, buses: float, *, round_trip_min: float | None = None) -> float:
        """Passengers a given fleet can move in an hour on this route."""
        round_trip = float(round_trip_min if round_trip_min is not None else self._round_trip_min)
        if round_trip <= 0 or buses <= 0:
            return 0.0
        return buses * (60.0 / round_trip) * self._bus_capacity * self._load_factor

    def wait_minutes(
        self,
        demand_per_hr: float,
        deployed: float,
        *,
        round_trip_min: float | None = None,
    ) -> float:
        """docs/03 M06: headway/2, stretched when the deployed fleet cannot cope.

        The card's formula alone has a perverse property: cap a hopelessly oversubscribed
        route at the available fleet and the computed wait gets *shorter*, because fewer
        buses on the road only ever shows up as headway. What actually happens to a passenger
        is that full buses go past. The overload factor is how many bus-loads of demand exist
        per bus-load of capacity, so a route carrying twice what it can hold reports roughly
        twice the wait. It is a first-order approximation: when demand exceeds capacity for a
        sustained period the real queue grows without bound, and no steady-state wait exists.
        """
        round_trip = float(round_trip_min if round_trip_min is not None else self._round_trip_min)
        if deployed <= 0:
            return 0.0
        headway = round_trip / deployed
        wait = headway / 2.0
        if not bool(self.require_param("apply_overload_to_wait")):
            return wait
        served = self.capacity_per_hour(deployed, round_trip_min=round_trip)
        if served <= 0:
            return wait
        overload = min(
            float(self.require_param("max_overload_factor")), max(1.0, demand_per_hr / served)
        )
        return wait * overload

    def _deployable(self, required_by_route: dict[str, int]) -> dict[str, float]:
        """Split the fleet between routes in proportion to what each one needs.

        docs/03 M06 says the wait is "capped by available fleet", so when the routes together
        want more than exists, each gets a share rather than the first one served taking all.
        """
        total = sum(required_by_route.values())
        if total <= self._fleet or total <= 0:
            return {route: float(need) for route, need in required_by_route.items()}
        return {route: self._fleet * (need / total) for route, need in required_by_route.items()}

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

        source_scenario = scenario_id if scenario_id in self._boardings else "S01"
        boardings = dict(self._boardings[source_scenario])
        if source_scenario != scenario_id:
            multiplier = float(overrides.get("footfall_multiplier", 1.0))
            if multiplier != 1.0:
                boardings = {r: s * multiplier for r, s in boardings.items()}
                builder.warn(
                    f"no generated world for {scenario_id}; the S01 history was scaled by "
                    f"{multiplier} through the generic scenario layer"
                )

        # A scenario may lengthen the round trip (a closure, a diversion). M04 will supply
        # this once it exists; until then it can be set directly in the overrides.
        round_trip_min = float(overrides.get("round_trip_min", self._round_trip_min))
        if round_trip_min != self._round_trip_min:
            builder.warn(
                f"round trip time overridden to {round_trip_min} min (baseline "
                f"{self._round_trip_min})"
            )

        entities = self.resolve_entities(request, list(boardings))
        if not entities:
            builder.warn("no known entities matched the request")
            return builder

        trimmed = {
            route: series.loc[series.index <= as_of]
            for route, series in boardings.items()
            if route in entities
        }
        trimmed = {r: s for r, s in trimmed.items() if len(s) > steps + 2}
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
        overflow = self._m05_overflow(upstream)

        result = self.forecast_engine.predict(
            trimmed,
            steps,
            future_covariates=self._covariate_frames(
                source_scenario, list(trimmed), future_arrivals=future_arrivals
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
            round_trip_min=round_trip_min,
            used_m01=future_arrivals is not None,
            overflow_vehicles=overflow,
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
        round_trip_min: float,
        used_m01: bool,
        overflow_vehicles: float,
    ) -> None:
        confidence = float(self.require_param("confidence"))
        per_step = float(self.require_param("steps_per_hour"))
        routes = result.entities
        if not routes:
            return

        timestamps = list(result.values[routes[0]].index)
        for timestamp in timestamps:
            # The fleet is shared, so every route's requirement at this instant is needed
            # before any one route's deployment can be worked out.
            demand: dict[str, float] = {}
            band: dict[str, tuple[float | None, float | None]] = {}
            for route in routes:
                value, low, high = result.at(route, timestamp)
                demand[route] = self._floor(value * per_step)
                band[route] = (
                    None if low is None else self._floor(low * per_step),
                    None if high is None else self._floor(high * per_step),
                )
            required = {
                route: self.buses_required(value, round_trip_min=round_trip_min)
                for route, value in demand.items()
            }
            deployed = self._deployable(required)
            offset = max(0, int((timestamp - as_of).total_seconds() // 60))
            short = sum(required.values()) > self._fleet

            for route in routes:
                spec = self._routes.get(route, {})
                low, high = band[route]
                shortfall = max(0.0, required[route] - deployed[route])
                details = {
                    "backend": result.backend_used,
                    "band_calibrated": result.calibrated,
                    "route_name": spec.get("name", route),
                    "from": spec.get("from"),
                    "to": spec.get("to"),
                    "round_trip_min": round_trip_min,
                    "buses_required": required[route],
                    "buses_deployed": round(deployed[route], 1),
                    "fleet_available": self._fleet,
                    "fleet_short_by": round(shortfall, 1),
                    "arrivals_covariate": "M01" if used_m01 else "none",
                    "parking_overflow_vehicles": round(overflow_vehicles, 1),
                }
                reasons = [REASON_CONGESTION] if short else []
                if overflow_vehicles > 0:
                    reasons.append(REASON_PARKING_OVERFLOW)

                if "shuttle_demand" in kpis:
                    builder.add(
                        EntityType.ROUTE,
                        route,
                        "shuttle_demand",
                        timestamp,
                        demand[route],
                        horizon_min=offset,
                        state=state,
                        lower=low,
                        upper=high,
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=details,
                    )

                if "shuttle_requirement" in kpis:
                    builder.add(
                        EntityType.ROUTE,
                        route,
                        "shuttle_requirement",
                        timestamp,
                        float(required[route]),
                        horizon_min=offset,
                        state=state,
                        lower=None
                        if low is None
                        else float(self.buses_required(low, round_trip_min=round_trip_min)),
                        upper=None
                        if high is None
                        else float(self.buses_required(high, round_trip_min=round_trip_min)),
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        resource_type=RESOURCE_SHUTTLE,
                        reason_codes=reasons,
                        recommendation=self._more_buses(route, required[route], shortfall)
                        if shortfall > 0
                        else None,
                        details=details,
                    )

                if "passenger_wait_time" in kpis:
                    builder.add(
                        EntityType.ROUTE,
                        route,
                        "passenger_wait_time",
                        timestamp,
                        self._floor(
                            self.wait_minutes(
                                demand[route], deployed[route], round_trip_min=round_trip_min
                            )
                        ),
                        horizon_min=offset,
                        state=state,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=details,
                    )

    def _more_buses(self, route: str, required: int, shortfall: float) -> Recommendation:
        spec = self._routes.get(route, {})
        return Recommendation(
            action=f"add shuttle vehicles to {spec.get('name', route)}",
            resource_type=RESOURCE_SHUTTLE,
            quantity=round(shortfall, 1),
            target_entity_id=route,
            rationale=(
                f"{spec.get('name', route)} needs {required:,} vehicles to hold the forecast "
                f"demand but the fleet of {self._fleet:,.0f} cannot supply them; "
                f"{shortfall:,.0f} short on this route"
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
            question="How many passengers and shuttles are needed per route?",
            engine=self.engine,
            method_summary=(
                f"Darts {backend} over one boardings series per shuttle route, with calendar "
                f"covariates and arrivals whose future half comes from M01. Vehicles are "
                f"ceil(demand x round_trip/60 / (capacity x load factor)); wait is headway/2 "
                f"stretched by an overload factor when the fleet cannot carry the demand."
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
                "The shuttle fleet in resources.yaml (80) is far below what this world needs: "
                "about 1,447 vehicles at the snan peak and 369 on an ordinary day. The "
                "requirement is reported honestly and the shortfall recommended; the fleet "
                "size is M24's input and was deliberately not changed here.",
                "Wait time past capacity is a first-order approximation. When demand exceeds "
                "what the fleet can carry for a sustained period the real queue grows without "
                "bound and no steady-state wait exists.",
                "Round-trip time is a single constant for every route and hour. M04 will "
                "supply a congestion-dependent value; until then a scenario can override it.",
                "M05's parking overflow is reported in details but NOT added to demand: the "
                "split between drivers who park elsewhere and drivers who take a shuttle is "
                "not something M06 can know.",
                "The fleet is split between routes in proportion to need, which assumes "
                "vehicles are interchangeable and can be moved instantly.",
                "Synthetic context; the backtest measures pipeline correctness, not accuracy.",
            ],
            license_notes=[
                "Darts: Apache-2.0. Chronos-2 weights: see the autogluon model card.",
                "LightGBM: MIT.",
            ],
        )
