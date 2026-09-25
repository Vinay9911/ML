"""M20 - Network Capacity.

Will communications - CCTV backhaul and command links - overload or fail?

**Method** (docs/03 M20 card)

Forecast
    Bandwidth used per tower from D19, on the shared forecast engine. **D19 is on a
    five-minute grid**, not the fifteen-minute grid the rest of the project uses, so a step
    here is five minutes.

Utilisation
    ``used / capacity * 100``, clipped at 100. Demand above capacity is a real state, but the
    excess is dropped packets rather than extra throughput, so the overflow is carried by
    ``network_capacity_risk`` instead of by a utilisation above 100.

Availability
    The share of recent steps the tower was up, carried forward. A scenario that downs a
    substation takes the towers fed by it offline for its window.

Risk
    ``100 * sigmoid(k * (utilisation - midpoint))`` with both constants in config.

**A finding, stated plainly: this world's network does not run out of bandwidth.** Peak
utilisation at the busiest tower is 27.5 percent, so ``network_capacity_risk`` sits near zero
throughout. That is the correct answer for the capacities in ``assumptions.network``, not a
broken KPI - the binding constraint here is power-driven *availability* (S08), not congestion.
The threshold was deliberately left where an operator would actually want it rather than
lowered to make the number move.
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

#: Reason codes M20 can raise (docs/05 section 4).
REASON_NETWORK_OVERLOAD = "NETWORK_OVERLOAD"
REASON_GRID_OUTAGE = "GRID_OUTAGE"
REASON_CCTV_BLIND_SPOT = "CCTV_BLIND_SPOT"


class Model(TwinModel):
    """Network Capacity (M20)."""

    model_id = "M20"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        self._towers = dict(world()["network_towers"])
        self._capacity_cfg = dict(assumptions()["network"]["tower_capacity_mbps"])
        self._camera_bitrate = float(assumptions()["network"]["camera_bitrate_mbps"])
        self._cameras_on_tower = self._count_cameras()

        self._used: dict[str, dict[str, pd.Series]] = {}
        self._capacity: dict[str, dict[str, pd.Series]] = {}
        self._up: dict[str, dict[str, pd.Series]] = {}
        self._arrivals: dict[str, pd.Series] = {}
        self._snan_days: dict[str, list[pd.Timestamp]] = {}
        for scenario_id in self.scenarios_supported:
            try:
                self._load_scenario(scenario_id)
            except Exception as exc:
                log.warning("no world slice for %s (%s)", scenario_id, exc)
        if "S01" not in self._used:
            raise RuntimeError(
                "M20 needs at least the S01 world slice. Run `python scripts/slice_world.py M20`."
            )

        self.forecast_engine = ForecastEngine.from_config(self.param("forecast"), seed=self.seed)
        try:
            result = self.forecast_engine.warmup(
                self._used["S01"],
                self.horizon_steps(self.default_horizon_min),
                future_covariates=self._covariate_frames("S01", list(self._used["S01"])),
            )
            log.info(
                "M20 warm: backend %s, %d towers, %.1fs",
                result.backend_used,
                len(result.entities),
                result.elapsed_s,
            )
        except Exception as exc:
            log.warning("M20 warmup failed (%s); the first request will be slower", exc)

    @staticmethod
    def _count_cameras() -> dict[str, int]:
        """Cameras hanging off each tower, from the world's camera placement table."""
        counts: dict[str, int] = {}
        placement = world()["cameras"].get("placement") or {}
        for entries in placement.values():
            for entry in entries:
                if len(entry) > 2:
                    counts[str(entry[2])] = counts.get(str(entry[2]), 0) + 1
        return counts

    def _load_scenario(self, scenario_id: str) -> None:
        table = load_table(
            "network_5min",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        self._used[scenario_id] = series_by_entity(
            table, entity_column="tower_id", value_column="bandwidth_used_mbps"
        )
        self._capacity[scenario_id] = series_by_entity(
            table, entity_column="tower_id", value_column="capacity_mbps"
        )
        up = table.copy()
        up["up"] = up["up"].astype("float64")
        self._up[scenario_id] = series_by_entity(up, entity_column="tower_id", value_column="up")

        footfall = load_table(
            "footfall_15min",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        # D01 is 15-minute and D19 is 5-minute, so the covariate is upsampled and held.
        arrivals = footfall.groupby("timestamp")["entries"].sum().astype("float64").sort_index()
        self._arrivals[scenario_id] = arrivals.resample("5min").ffill()
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
        series = next(iter(self._used[scenario_id].values()))
        index = pd.DatetimeIndex(series.index)
        horizon = self.horizon_steps(max(self.horizons_min))
        extended = pd.date_range(
            start=index[0], periods=len(index) + horizon, freq="5min", tz=index.tz
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
        if not totals:
            return None
        return pd.Series(totals).sort_index().resample("5min").ffill()

    # ------------------------------------------------------------------ formulas
    def horizon_steps(self, horizon_min: int) -> int:
        """Minutes to FIVE-minute steps: D19 is not on the 15-minute grid."""
        step = float(self.require_param("step_minutes"))
        return max(1, round(horizon_min / step))

    def entity_ids(self) -> list[str]:
        return list(self._used["S01"])

    def utilization_pct(self, used_mbps: float, capacity_mbps: float) -> float:
        """docs/05 section 2: used / capacity * 100, clipped at full.

        Demand past capacity is dropped packets, not throughput; `network_capacity_risk`
        carries how bad the overload is.
        """
        ceiling = float(self.require_param("max_utilization_pct"))
        if capacity_mbps <= 0:
            return ceiling
        return float(min(ceiling, max(0.0, used_mbps / capacity_mbps * 100.0)))

    def capacity_risk_pct(self, utilization_pct: float) -> float:
        """docs/03 M20: 100 * sigmoid(k * (utilisation - midpoint))."""
        midpoint = float(self.require_param("risk_midpoint_pct"))
        steepness = float(self.require_param("risk_steepness"))
        exponent = -steepness * (utilization_pct - midpoint)
        # Guard the exponential so a wild utilisation cannot overflow the float.
        if exponent > 60:
            return 0.0
        if exponent < -60:
            return 100.0
        return 100.0 / (1.0 + math.exp(exponent))

    def _floor(self, value: float) -> float:
        return max(float(self.require_param("floor_value")), value)

    def _tower_down(self, tower: str, overrides: dict[str, Any]) -> bool:
        """True when the tower loses power for this scenario (its substation is down)."""
        down = closed_entities(overrides)["substations"]
        if not down:
            return False
        return self._towers.get(tower, {}).get("substation") in down

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

        source = scenario_id if scenario_id in self._used else "S01"
        adjustment = 1.0
        if source != scenario_id:
            multiplier = float(overrides.get("footfall_multiplier", 1.0))
            if multiplier != 1.0:
                adjustment = multiplier
                builder.warn(
                    f"no generated world for {scenario_id}; the forecast was scaled by "
                    f"{multiplier} through the generic scenario layer"
                )

        camera_outage = float(overrides.get("camera_outage_share", 0.0))
        if camera_outage:
            builder.warn(
                f"{camera_outage:.0%} of cameras are offline; their backhaul load is removed "
                f"and the loss of coverage is flagged"
            )

        entities = self.resolve_entities(request, list(self._used[source]))
        if not entities:
            builder.warn("no known entities matched the request")
            return builder

        trimmed = {
            tower: series.loc[series.index <= as_of]
            for tower, series in self._used[source].items()
            if tower in entities
        }
        trimmed = {t: s for t, s in trimmed.items() if len(s) > steps + 2}
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
            overrides=overrides,
            adjustment=adjustment,
            camera_outage=camera_outage,
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
        overrides: dict[str, Any],
        adjustment: float,
        camera_outage: float,
        used_m01: bool,
    ) -> None:
        confidence = float(self.require_param("confidence"))

        for tower in result.entities:
            zone = self._towers.get(tower, {}).get("zone_id")
            capacity_series = self._capacity[source].get(tower)
            capacity = (
                float(capacity_series.iloc[-1])
                if capacity_series is not None and len(capacity_series)
                else float(self._capacity_cfg.get(tower, 0.0))
            )
            cameras = self._cameras_on_tower.get(tower, 0)
            # S09 takes cameras offline, so their backhaul stops too.
            removed_mbps = cameras * self._camera_bitrate * camera_outage

            up_series = self._up[source].get(tower)
            observed_up = up_series.loc[up_series.index <= as_of] if up_series is not None else None
            availability = (
                100.0 * float(observed_up.tail(288).mean())
                if observed_up is not None and len(observed_up)
                else 100.0
            )
            powered_down = self._tower_down(tower, overrides)
            if powered_down:
                availability = 0.0

            for timestamp in result.values[tower].index:
                value, low, high = result.at(tower, timestamp)
                used = self._floor(value * adjustment - removed_mbps)
                utilization = self.utilization_pct(used, capacity)
                risk = self.capacity_risk_pct(utilization)
                offset = max(0, int((timestamp - as_of).total_seconds() // 60))

                reasons: list[str] = []
                if powered_down:
                    reasons.append(REASON_GRID_OUTAGE)
                if camera_outage > 0:
                    reasons.append(REASON_CCTV_BLIND_SPOT)
                for kpi_name, kpi_value in (
                    ("bandwidth_utilization", utilization),
                    ("network_capacity_risk", risk),
                ):
                    if (
                        risk_level_for_kpi(kpi_name, kpi_value) in REASON_REQUIRED_LEVELS
                        and REASON_NETWORK_OVERLOAD not in reasons
                    ):
                        reasons.append(REASON_NETWORK_OVERLOAD)

                details = {
                    "backend": result.backend_used,
                    "band_calibrated": result.calibrated,
                    "used_mbps": round(used, 1),
                    "capacity_mbps": capacity,
                    "cameras_on_tower": cameras,
                    "camera_load_removed_mbps": round(removed_mbps, 1),
                    "powered_down": powered_down,
                    "arrivals_covariate": "M01" if used_m01 else "none",
                }

                if "bandwidth_utilization" in kpis:
                    builder.add(
                        EntityType.ASSET,
                        tower,
                        "bandwidth_utilization",
                        timestamp,
                        utilization,
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        lower=None
                        if low is None
                        else self.utilization_pct(
                            self._floor(low * adjustment - removed_mbps), capacity
                        ),
                        upper=None
                        if high is None
                        else self.utilization_pct(
                            self._floor(high * adjustment - removed_mbps), capacity
                        ),
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=details,
                    )

                if "network_capacity_risk" in kpis:
                    builder.add(
                        EntityType.ASSET,
                        tower,
                        "network_capacity_risk",
                        timestamp,
                        risk,
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        confidence=confidence,
                        reason_codes=reasons,
                        details={**details, "utilization_pct": round(utilization, 1)},
                    )

                if "network_availability" in kpis:
                    builder.add(
                        EntityType.ASSET,
                        tower,
                        "network_availability",
                        timestamp,
                        availability,
                        zone_id=zone,
                        horizon_min=offset,
                        state=state,
                        confidence=confidence,
                        reason_codes=reasons,
                        details=details,
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
            question="Will communications (CCTV backhaul, command links) overload or fail?",
            engine=self.engine,
            method_summary=(
                f"Darts {backend} over one bandwidth series per tower on the FIVE-minute D19 "
                f"grid. Utilisation is used over capacity; risk is a sigmoid of utilisation "
                f"about a configured midpoint; availability is the recent uptime share, "
                f"zeroed for a tower whose substation is down."
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
                "This world's network does not run out of bandwidth: peak utilisation at the "
                "busiest tower is 27.5 percent, so network_capacity_risk sits near zero "
                "throughout. That is the honest answer for the capacities in "
                "assumptions.network, and the risk threshold was left where an operator would "
                "want it rather than lowered to make the number move. Validate "
                "tower_capacity_mbps before reading anything into the risk KPI.",
                "Availability is the recent uptime share carried forward, not a forecast. An "
                "outage that has not started yet is invisible unless a scenario declares it.",
                "S09 removes camera backhaul load but does NOT model the loss of coverage "
                "itself, which is M02's and M10's work; the CCTV_BLIND_SPOT code is a pointer.",
                "Each tower is treated as an independent link. Shared backhaul, contention "
                "and failover between towers are not modelled.",
                "Synthetic context; the backtest measures pipeline correctness, not accuracy.",
            ],
            license_notes=[
                "Darts: Apache-2.0. Chronos-2 weights: see the autogluon model card.",
                "LightGBM: MIT.",
            ],
        )
