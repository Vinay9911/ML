"""M05 - Parking Demand.

How full will each parking site be, and when does it overflow?

**What is forecast** (docs/03 M05 card)

The card asks for "occupied spaces per site". M05 forecasts *demand* instead - the uncapped
running total of vehicles wanting a bay - and derives occupancy from it. The reason is in the
data: D06 writes ``occupied`` clipped to ``capacity``, so a forecast trained on it can never
predict an overflow, which is the one thing this model exists to warn about. The uncapped
series is recovered from the same table as ``cumsum(entries - exits)``, because D06 records
arrival demand in ``entries`` rather than admitted vehicles.

Occupancy is therefore demand over capacity and **may exceed 100 percent**. That is
deliberate and the rest of the contract already expects it: the ``utilization_pct`` band in
risk_bands.yaml has a ``critical`` range of 100-9999, and the docs/03 M05 test asks for
occupancy "within [0, 100 + overflow]". ``details.occupied_spaces`` carries the physically
parked count, ``min(demand, capacity)``, for anyone who wants the capped view.

**Covariates**

Calendar (hour sine and cosine, day of week, the snan flag) always. Arrivals are added as a
covariate whose past half comes from D01 and whose future half comes from **M01's forecast** -
arrivals are not known ahead, so taking the future half from the generated world would be
peeking. With no M01 available the covariate is dropped entirely and the response says so,
rather than quietly substituting data the caller could not have had.

**KPIs**

``parking_demand`` vehicles seeking a bay · ``parking_occupancy`` demand over capacity as a
percentage · ``parking_search_time`` from the docs/03 M05 curve, monotonic in occupancy.
At or above ``overflow_pct`` the record carries ``PARKING_OVERFLOW`` and a shuttle-diversion
recommendation.
"""

from __future__ import annotations

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

#: Reason codes M05 can raise (docs/05 section 4).
REASON_PARKING_OVERFLOW = "PARKING_OVERFLOW"
REASON_CONGESTION = "CONGESTION"
REASON_SYNTHETIC = "SYNTHETIC_DATA"

#: The resource a diversion recommendation asks for (resources.yaml).
RESOURCE_SHUTTLE = "shuttle_bus"


class Model(TwinModel):
    """Parking Demand (M05)."""

    model_id = "M05"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Read D06 once per scenario, build the demand series, and warm the engine."""
        sites = world()["parking_sites"]
        capacities = assumptions()["transport"]["parking_capacity"]
        self._sites: dict[str, dict[str, Any]] = {
            site_id: {
                "name": spec["name"],
                "zone_id": spec["zone_id"],
                "link": spec["link"],
                "capacity": float(capacities[site_id]),
            }
            for site_id, spec in sites.items()
        }

        self._demand: dict[str, dict[str, pd.Series]] = {}
        self._arrivals: dict[str, pd.Series] = {}
        self._snan_days: dict[str, list[pd.Timestamp]] = {}
        for scenario_id in self.scenarios_supported:
            try:
                self._load_scenario(scenario_id)
            except Exception as exc:
                log.warning("no world slice for %s (%s)", scenario_id, exc)
        if "S01" not in self._demand:
            raise RuntimeError(
                "M05 needs at least the S01 world slice. Run `python scripts/slice_world.py M05`."
            )

        self.forecast_engine = ForecastEngine.from_config(self.param("forecast"), seed=self.seed)
        try:
            result = self.forecast_engine.warmup(
                self._demand["S01"],
                self.horizon_steps(self.default_horizon_min),
                future_covariates=self._covariate_frames("S01", list(self._demand["S01"])),
            )
            log.info(
                "M05 warm: backend %s, %d sites, %.1fs",
                result.backend_used,
                len(result.entities),
                result.elapsed_s,
            )
        except Exception as exc:
            log.warning("M05 warmup failed (%s); the first request will be slower", exc)

    def _load_scenario(self, scenario_id: str) -> None:
        """The uncapped demand series per site, plus the covariate inputs."""
        parking = load_table(
            "parking_15min",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        entries = series_by_entity(parking, entity_column="parking_id", value_column="entries")
        exits = series_by_entity(parking, entity_column="parking_id", value_column="exits")

        # D06 clips `occupied` at capacity, so a model trained on it could never see an
        # overflow. `entries` is arrival demand rather than admitted vehicles, so the
        # uncapped running total is the demand the operator actually has to place.
        demand: dict[str, pd.Series] = {}
        for site_id, arriving in entries.items():
            leaving = exits[site_id].reindex(arriving.index).fillna(0.0)
            demand[site_id] = (arriving - leaving).cumsum().clip(lower=0.0)
        self._demand[scenario_id] = demand

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
    def _covariate_index(self, scenario_id: str) -> pd.DatetimeIndex:
        """History plus the longest horizon: a future covariate must span the window."""
        series = next(iter(self._demand[scenario_id].values()))
        index = pd.DatetimeIndex(series.index)
        horizon = self.horizon_steps(max(self.horizons_min))
        return pd.date_range(
            start=index[0], periods=len(index) + horizon, freq="15min", tz=index.tz
        )

    def _covariate_frames(
        self,
        scenario_id: str,
        entities: list[str],
        *,
        future_arrivals: pd.Series | None = None,
    ) -> dict[str, pd.DataFrame]:
        """One covariate frame per site.

        Every site sees the same frame: the venue's calendar and its total arrivals drive
        all three lots. ``future_arrivals`` is M01's forecast; without it the arrivals
        column is left out rather than filled from the generated world, which the caller
        could not have known at ``as_of``.
        """
        index = self._covariate_index(scenario_id)
        extra: dict[str, pd.Series] = {}
        history = self._arrivals.get(scenario_id)
        if history is not None and future_arrivals is not None and len(future_arrivals):
            combined = pd.concat([history, future_arrivals])
            extra["arrivals"] = combined[~combined.index.duplicated(keep="last")].sort_index()
        frame = calendar_covariates(
            index, snan_days=self._snan_days.get(scenario_id, []), extra=extra
        )
        return dict.fromkeys(entities, frame)

    def _m01_arrivals(self, upstream: dict[str, Any], as_of: pd.Timestamp) -> pd.Series | None:
        """M01's forecast arrivals per step, summed over its zones.

        M01 reports ``expected_footfall`` in persons/hr; the covariate is per 15-min step,
        so it is divided back down by ``steps_per_hour``.
        """
        resolution = upstream.get("M01")
        if resolution is None:
            return None
        per_step = float(self.require_param("steps_per_hour"))
        totals: dict[pd.Timestamp, float] = {}
        for zone in self._zone_ids(resolution):
            for timestamp, value in series_for_entity(resolution.output, "expected_footfall", zone):
                stamp = pd.Timestamp(timestamp)
                if stamp <= as_of:
                    continue
                totals[stamp] = totals.get(stamp, 0.0) + float(value) / per_step
        if not totals:
            return None
        return pd.Series(totals).sort_index()

    @staticmethod
    def _zone_ids(resolution: Any) -> list[str]:
        return sorted(
            {
                record.entity_id
                for record in resolution.output.results
                if record.kpi == "expected_footfall" and record.entity_id.startswith("Z")
            }
        )

    # ------------------------------------------------------------------ helpers
    def horizon_steps(self, horizon_min: int) -> int:
        """Minutes to 15-minute steps, at least one."""
        return max(1, round(horizon_min / 15))

    def entity_ids(self) -> list[str]:
        return list(self._sites)

    def capacity(self, site_id: str) -> float:
        return float(self._sites[site_id]["capacity"])

    def _occupancy_pct(self, vehicles: float, capacity: float) -> float:
        """Demand over capacity, floored at zero and capped at the reporting ceiling."""
        ceiling = float(self.require_param("max_occupancy_pct"))
        if capacity <= 0:
            return 0.0
        return float(min(ceiling, max(0.0, vehicles / capacity * 100.0)))

    def search_time_min(self, occupancy_pct: float) -> float:
        """docs/03 M05: base + k * max(0, occ - knee) / (1 - knee), occ as a fraction.

        Deliberately not capped at full: an oversubscribed lot keeps getting worse, which is
        what keeps the KPI monotonic in occupancy.
        """
        settings = self.require_param("search_time")
        base = float(settings["base_min"])
        k = float(settings["k_min"])
        knee = float(settings["occupancy_knee"])
        occupancy = max(0.0, occupancy_pct / 100.0)
        # A knee at or past "full" would make this zero or negative. That is a broken config,
        # not a threshold, and it should flatten the curve rather than raise mid-request.
        span = 1.0 - knee
        if span <= 0:
            return base
        return base + k * max(0.0, occupancy - knee) / span

    def _vehicles(self, value: float) -> float:
        floor = float(self.require_param("floor_vehicles"))
        return max(floor, value)

    # ------------------------------------------------------------------ scenario state
    def _scenario_demand(
        self, scenario_id: str, overrides: dict[str, Any] | None
    ) -> tuple[dict[str, pd.Series], list[str]]:
        """Demand series for a scenario, with any site reallocation applied.

        A scenario with its own generated world is read directly. Otherwise the S01 history
        is scaled by the generic adjustment layer, and demand is moved off any site whose
        access link is closed - that is the "S06/S12 may shift demand between sites" note on
        the docs/03 M05 card. Neither scenario actually closes a parking link, so in practice
        M05 declares itself insensitive to both; the mechanism is here for one that does.
        """
        notes: list[str] = []
        overrides = overrides or {}

        if scenario_id in self._demand:
            demand = dict(self._demand[scenario_id])
        else:
            demand = dict(self._demand["S01"])
            multiplier = float(overrides.get("footfall_multiplier", 1.0))
            if multiplier != 1.0:
                demand = {site: series * multiplier for site, series in demand.items()}
                notes.append(
                    f"no generated world for {scenario_id}; the S01 history was scaled by "
                    f"{multiplier} through the generic scenario layer"
                )

        closed_links = set(closed_entities(overrides)["links"])
        if closed_links:
            cut = [s for s, spec in self._sites.items() if spec["link"] in closed_links]
            open_sites = [s for s in demand if s not in cut]
            if cut and open_sites:
                spare = sum(self.capacity(s) for s in open_sites)
                moved = {site: demand[site] for site in cut}
                for site in cut:
                    demand[site] = demand[site] * 0.0
                for site in open_sites:
                    share = self.capacity(site) / spare if spare > 0 else 1.0 / len(open_sites)
                    demand[site] = demand[site] + sum(moved.values()) * share
                notes.append(
                    f"access link closed for {', '.join(sorted(cut))}; their demand was "
                    f"reallocated to {', '.join(sorted(open_sites))} in proportion to capacity"
                )
        return demand, notes

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
        horizon_min = self.resolve_horizon(request)
        steps = self.horizon_steps(horizon_min)
        kpis = set(self.resolve_kpis(request))
        scenario_id = builder.scenario_id

        demand, notes = self._scenario_demand(scenario_id, overrides)
        for note in notes:
            builder.warn(note)

        entities = self.resolve_entities(request, list(demand))
        if not entities:
            builder.warn("no known entities matched the request")
            return builder

        trimmed = {
            site: series.loc[series.index <= as_of]
            for site, series in demand.items()
            if site in entities
        }
        trimmed = {s: series for s, series in trimmed.items() if len(series) > steps + 2}
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
        covariate_scenario = scenario_id if scenario_id in self._demand else "S01"
        covariates = self._covariate_frames(
            covariate_scenario, list(trimmed), future_arrivals=future_arrivals
        )

        result = self.forecast_engine.predict(trimmed, steps, future_covariates=covariates)
        if result.degraded:
            builder.warn(
                f"forecast ran on the {result.backend_used} backend rather than "
                f"{result.backend_requested}"
            )
        for message in result.warnings:
            builder.warn(message)

        self._emit(builder, result, kpis, as_of, state, used_m01=future_arrivals is not None)
        return builder

    def _emit(
        self,
        builder: OutputBuilder,
        result: ForecastResult,
        kpis: set[str],
        as_of: pd.Timestamp,
        state: State,
        *,
        used_m01: bool,
    ) -> None:
        """Turn the demand forecast into the three contract KPIs."""
        confidence = float(self.require_param("confidence"))
        overflow_pct = float(self.require_param("overflow_pct"))

        for site_id in result.entities:
            series = result.values[site_id]
            capacity = self.capacity(site_id)
            spec = self._sites[site_id]

            for timestamp in series.index:
                value, low, high = result.at(site_id, timestamp)
                offset = max(0, int((timestamp - as_of).total_seconds() // 60))
                vehicles = self._vehicles(value)
                occupancy = self._occupancy_pct(vehicles, capacity)
                overflowing = occupancy >= overflow_pct
                reasons = self._reason_codes(occupancy, overflowing=overflowing)
                shared = {
                    "backend": result.backend_used,
                    "site_name": spec["name"],
                    "capacity_spaces": capacity,
                    "occupied_spaces": round(min(vehicles, capacity), 1),
                    "overflow_vehicles": round(max(0.0, vehicles - capacity), 1),
                    "arrivals_covariate": "M01" if used_m01 else "none",
                }

                if "parking_demand" in kpis:
                    builder.add(
                        EntityType.PARKING_SITE,
                        site_id,
                        "parking_demand",
                        timestamp,
                        vehicles,
                        zone_id=spec["zone_id"],
                        horizon_min=offset,
                        state=state,
                        lower=None if low is None else self._vehicles(low),
                        upper=None if high is None else self._vehicles(high),
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=shared,
                    )

                if "parking_occupancy" in kpis:
                    builder.add(
                        EntityType.PARKING_SITE,
                        site_id,
                        "parking_occupancy",
                        timestamp,
                        occupancy,
                        zone_id=spec["zone_id"],
                        horizon_min=offset,
                        state=state,
                        lower=None
                        if low is None
                        else self._occupancy_pct(self._vehicles(low), capacity),
                        upper=None
                        if high is None
                        else self._occupancy_pct(self._vehicles(high), capacity),
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        reason_codes=reasons,
                        recommendation=self._diversion(site_id, vehicles, capacity)
                        if overflowing
                        else None,
                        details=shared,
                    )

                if "parking_search_time" in kpis:
                    builder.add(
                        EntityType.PARKING_SITE,
                        site_id,
                        "parking_search_time",
                        timestamp,
                        self.search_time_min(occupancy),
                        zone_id=spec["zone_id"],
                        horizon_min=offset,
                        state=state,
                        lower=None
                        if low is None
                        else self.search_time_min(
                            self._occupancy_pct(self._vehicles(low), capacity)
                        ),
                        upper=None
                        if high is None
                        else self.search_time_min(
                            self._occupancy_pct(self._vehicles(high), capacity)
                        ),
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        reason_codes=reasons,
                        details={**shared, "occupancy_pct": round(occupancy, 1)},
                    )

    def _reason_codes(self, occupancy_pct: float, *, overflowing: bool) -> list[str]:
        """The codes a record must carry (docs/02 section 4).

        Amber and above make a reason code mandatory, and the ``utilization_pct`` band turns
        amber at 80 percent - well before the lot actually overflows. ``PARKING_OVERFLOW``
        would be the wrong word for a lot that is merely filling, so a busy-but-coping site
        reports ``CONGESTION`` and only a genuinely oversubscribed one reports an overflow.
        The band itself is read from risk_bands.yaml rather than compared against a number
        here, which is what keeps the threshold out of src/ (CLAUDE.md).
        """
        if overflowing:
            return [REASON_PARKING_OVERFLOW]
        level = risk_level_for_kpi("parking_occupancy", occupancy_pct)
        return [REASON_CONGESTION] if level in REASON_REQUIRED_LEVELS else []

    def _diversion(self, site_id: str, vehicles: float, capacity: float) -> Recommendation:
        """Shuttle diversion for the vehicles that will not fit (docs/03 M05)."""
        spec = self._sites[site_id]
        overflow = max(0.0, vehicles - capacity)
        persons_per_car = float(assumptions()["transport"]["persons_per_car"])
        bus_capacity = float(assumptions()["transport"]["bus_capacity"])
        load_factor = float(assumptions()["transport"]["bus_load_factor"])
        buses = overflow * persons_per_car / max(1.0, bus_capacity * load_factor)
        return Recommendation(
            action=f"divert arrivals from {spec['name']} and run a shuttle from the overflow area",
            resource_type=RESOURCE_SHUTTLE,
            quantity=round(buses, 1),
            target_entity_id=site_id,
            rationale=(
                f"{spec['name']} is forecast at {vehicles:,.0f} vehicles against "
                f"{capacity:,.0f} bays, so about {overflow:,.0f} will not fit"
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
            question="How full will each parking site be, and when does it overflow?",
            engine=self.engine,
            method_summary=(
                f"Darts {backend} over one demand series per parking site, where demand is "
                f"the uncapped running total of vehicles seeking a bay. Covariates: hour "
                f"sine and cosine, day of week, the snan flag, and arrivals whose future "
                f"half comes from M01. Occupancy is demand over capacity and may exceed 100 "
                f"percent; search time follows the docs/03 M05 curve."
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
                "Occupancy is demand over capacity, so it exceeds 100 percent when more "
                "vehicles want a bay than the site holds. The physically parked count is in "
                "details.occupied_spaces.",
                "Site capacities were raised from the docs/04 section 6 placeholder of 4,800 "
                "total to 23,200, which is what makes the KPI informative. See the model card.",
                "Demand is split between sites in proportion to capacity, as the generator "
                "does. Real drivers choose by distance, price and signage.",
                "Without M01 the arrivals covariate is dropped and only the calendar drives "
                "the forecast; the response says so and marks itself degraded.",
                "Trained context is synthetic, so the backtest measures pipeline correctness "
                "rather than real-world accuracy.",
                "Confidence is a fixed heuristic, not a calibrated probability.",
            ],
            license_notes=[
                "Darts: Apache-2.0. Chronos-2 weights: see the autogluon model card.",
                "LightGBM: MIT.",
            ],
        )
