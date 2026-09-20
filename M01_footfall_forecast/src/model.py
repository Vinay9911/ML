"""M01 - Footfall Forecast.

How many people will arrive, by zone, gate and time?

M01 is the reference model for ``twin_common.engines.forecast`` and the busiest node in the
dependency graph: M03, M04, M05, M06, M07, M11, M15, M16, M17, M18, M19, M20 and M24 all
read it. If M01 is wrong, most of the twin is wrong.

**Method** (docs/03 M01 card)

Series
    One per zone (Z01-Z08) from D01 ``footfall_15min``, and one per gate (G01-G04) from the
    gate table. Zones and gates are forecast in the same call, because a Darts global model
    takes a list of series.

Backend
    Darts ``Chronos2Model`` zero-shot by default, with ``LightGBMModel`` and
    ``NaiveSeasonal`` behind it. docs/06 section 3 requires Chronos-2 rather than TimesFM,
    because only Chronos-2 accepts covariates.

Covariates
    Future: hour-of-day sine and cosine, day of week, the snan-day flag, and temperature and
    rain from D12. They are genuinely known across the window because the world is generated,
    which is what makes a zero-shot forecast meaningful here.

KPIs
    ``expected_footfall`` is the forecast entries per step converted to persons/hr (x4), with
    the 0.1 and 0.9 quantiles as the band. ``peak_footfall`` is the maximum of the median
    forecast across the horizon, with the time it occurs in ``details``.

Warm start
    The engine fits once in :meth:`load`. Without that the first request pays the fit -
    measured 8.9 s for Chronos-2 - against a 10 s budget (docs/02 section 6).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from twin_common.contracts import Metadata, ModelOutput, PredictRequest, ScenarioRequest
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
from twin_common.scenarios import closed_entities

log = get_logger(__name__)

#: Reason codes M01 can raise (docs/05 section 4).
REASON_SURGE = "SURGE"
REASON_HEAVY_RAIN = "HEAVY_RAIN"
REASON_SYNTHETIC = "SYNTHETIC_DATA"


class Model(TwinModel):
    """Footfall Forecast (M01)."""

    model_id = "M01"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Read the history once, build the covariates, and warm the engine."""
        self._history: dict[str, dict[str, pd.Series]] = {}
        self._covariates: dict[str, dict[str, pd.DataFrame]] = {}
        self._snan_days: list[pd.Timestamp] = []

        for scenario_id in self.scenarios_supported:
            try:
                self._load_scenario(scenario_id)
            except Exception as exc:
                log.warning("no world slice for %s (%s)", scenario_id, exc)
        if "S01" not in self._history:
            raise RuntimeError(
                "M01 needs at least the S01 world slice. Run `python scripts/slice_world.py M01`."
            )

        self.forecast_engine = ForecastEngine.from_config(self.param("forecast"), seed=self.seed)
        # Fit now so the first request does not pay for it (docs/02 section 6).
        try:
            result = self.forecast_engine.warmup(
                self._history["S01"],
                self.horizon_steps(self.default_horizon_min),
                future_covariates=self._covariates.get("S01"),
            )
            log.info(
                "M01 warm: backend %s, %d series, %.1fs",
                result.backend_used,
                len(result.entities),
                result.elapsed_s,
            )
        except Exception as exc:
            log.warning("M01 warmup failed (%s); the first request will be slower", exc)

    def _load_scenario(self, scenario_id: str) -> None:
        """History series and future covariates for one scenario."""
        footfall = load_table(
            "footfall_15min",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        history = series_by_entity(footfall, entity_column="zone_id", value_column="entries")
        try:
            gates = load_table(
                "gate_entries",
                self.data_source,
                scenario_id=scenario_id,
                base_dir=self.synthetic_dir,
            )
            history |= series_by_entity(gates, entity_column="gate_id", value_column="entries")
        except Exception as exc:
            log.warning("%s: no gate series (%s); forecasting zones only", scenario_id, exc)

        weather = load_table(
            "weather_hourly",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        ).set_index("timestamp")
        calendar = load_table(
            "event_calendar",
            self.data_source,
            scenario_id=scenario_id,
            base_dir=self.synthetic_dir,
        )
        snan = [pd.Timestamp(day) for day in calendar.loc[calendar["is_snan_day"], "date"].unique()]
        if scenario_id == "S01":
            self._snan_days = snan

        # The covariate index must span the history AND the forecast window, which is what
        # makes a future covariate "known in the future".
        index = pd.DatetimeIndex(next(iter(history.values())).index)
        horizon = self.horizon_steps(max(self.horizons_min))
        extended = pd.date_range(
            start=index[0], periods=len(index) + horizon, freq="15min", tz=index.tz
        )
        covariates = calendar_covariates(
            extended,
            snan_days=snan,
            extra={
                "temperature_c": weather["temperature_c"],
                "rain_mm": weather["rain_mm"],
            },
        )
        self._history[scenario_id] = history
        self._covariates[scenario_id] = dict.fromkeys(history, covariates)

    # ------------------------------------------------------------------ helpers
    def horizon_steps(self, horizon_min: int) -> int:
        """Minutes to 15-minute steps, at least one."""
        return max(1, round(horizon_min / 15))

    def entity_ids(self) -> list[str]:
        """Zones first, then gates, in the order the world declares them."""
        return list(self._history["S01"])

    @staticmethod
    def _entity_type(entity_id: str) -> EntityType:
        return EntityType.GATE if entity_id.startswith("G") else EntityType.ZONE

    def _to_persons_hr(self, value: float) -> float:
        """15-minute counts to persons/hr (docs/03 M01), floored at zero.

        A quantile band can dip below zero; a count of people cannot.
        """
        floor = float(self.require_param("floor_persons_hr"))
        return max(floor, value * float(self.require_param("steps_per_hour")))

    def _scenario_history(
        self, scenario_id: str, overrides: dict[str, Any] | None
    ) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame] | None, list[str]]:
        """History and covariates for a scenario, with any generic adjustment applied.

        A scenario with its own generated world is read directly, which is the accurate
        path. Otherwise the S01 history is scaled by the generic adjustment layer and the
        response says so (docs/03 M01: "apply via the synthetic world for that scenario when
        available, else the generic scenario layer").
        """
        notes: list[str] = []
        if scenario_id in self._history:
            return self._history[scenario_id], self._covariates.get(scenario_id), notes

        overrides = overrides or {}
        history = dict(self._history["S01"])
        multiplier = float(overrides.get("footfall_multiplier", 1.0))
        if multiplier != 1.0:
            history = {entity: series * multiplier for entity, series in history.items()}
            notes.append(
                f"no generated world for {scenario_id}; the S01 history was scaled by "
                f"{multiplier} through the generic scenario layer"
            )
        closed = closed_entities(overrides)["gates"]
        for gate in closed:
            if gate in history:
                history[gate] = history[gate] * 0.0
                notes.append(f"{gate} is closed; its forecast is zero")
        return history, self._covariates.get("S01"), notes

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

        history, covariates, notes = self._scenario_history(scenario_id, overrides)
        for note in notes:
            builder.warn(note)
        entities = self.resolve_entities(request, list(history))
        if not entities:
            builder.warn("no known entities matched the request")
            return builder

        # Only forecast from data the caller could actually have at `as_of`.
        trimmed = {
            entity: series.loc[series.index <= as_of]
            for entity, series in history.items()
            if entity in entities
        }
        trimmed = {e: s for e, s in trimmed.items() if len(s) > steps + 2}
        if not trimmed:
            builder.warn(f"no history at or before {as_of.isoformat()}")
            return builder

        upstream = self.resolve_upstream(request, scenario_id=scenario_id, entity_ids=entities)
        self.record_upstream(builder, upstream)
        weather_note = self._upstream_weather(upstream)

        result = self.forecast_engine.predict(
            trimmed,
            steps,
            future_covariates={e: c for e, c in (covariates or {}).items() if e in trimmed},
        )
        if result.degraded:
            builder.warn(
                f"forecast ran on the {result.backend_used} backend rather than "
                f"{result.backend_requested}"
            )
        for message in result.warnings:
            builder.warn(message)

        self._emit(builder, result, kpis, as_of, state, weather_note)
        return builder

    def _upstream_weather(self, upstream: dict[str, Any]) -> dict[str, float]:
        """The M21 multipliers, reported in details so a reader can see the coupling."""
        resolution = upstream.get("M21")
        if resolution is None:
            return {}
        out: dict[str, float] = {}
        for record in resolution.output.results:
            if record.kpi in (
                "weather_arrival_multiplier",
                "weather_medical_multiplier",
                "rainfall_intensity",
            ):
                out.setdefault(record.kpi, float(record.value))
        return out

    def _emit(
        self,
        builder: OutputBuilder,
        result: ForecastResult,
        kpis: set[str],
        as_of: pd.Timestamp,
        state: State,
        weather: dict[str, float],
    ) -> None:
        """Turn the forecast into contract records."""
        confidence = float(self.require_param("confidence"))
        heavy_rain = weather.get("rainfall_intensity", 0.0)

        for entity in result.entities:
            series = result.values[entity]
            entity_type = self._entity_type(entity)
            zone_id = entity if entity_type is EntityType.ZONE else None

            if "expected_footfall" in kpis:
                for timestamp in series.index:
                    value, low, high = result.at(entity, timestamp)
                    offset = int((timestamp - as_of).total_seconds() // 60)
                    reasons: list[str] = []
                    if heavy_rain > 0:
                        reasons.append(REASON_HEAVY_RAIN)
                    builder.add(
                        entity_type,
                        entity,
                        "expected_footfall",
                        timestamp,
                        self._to_persons_hr(value),
                        zone_id=zone_id,
                        horizon_min=max(0, offset),
                        state=state,
                        lower=None if low is None else self._to_persons_hr(low),
                        upper=None if high is None else self._to_persons_hr(high),
                        quantile_level=result.quantile_level if low is not None else None,
                        confidence=confidence,
                        reason_codes=reasons,
                        details={
                            "backend": result.backend_used,
                            "entries_per_step": round(value, 2),
                            **({"weather": weather} if weather else {}),
                        },
                    )

            if "peak_footfall" in kpis and len(series):
                peak_position = int(np.argmax(series.to_numpy()))
                peak_time = series.index[peak_position]
                peak_value, peak_low, peak_high = result.at(entity, peak_time)
                offset = int((peak_time - as_of).total_seconds() // 60)
                builder.add(
                    entity_type,
                    entity,
                    "peak_footfall",
                    peak_time,
                    self._to_persons_hr(peak_value),
                    zone_id=zone_id,
                    horizon_min=max(0, offset),
                    state=state,
                    lower=None if peak_low is None else self._to_persons_hr(peak_low),
                    upper=None if peak_high is None else self._to_persons_hr(peak_high),
                    quantile_level=result.quantile_level if peak_low is not None else None,
                    confidence=confidence,
                    reason_codes=[REASON_SURGE] if peak_position == 0 else [],
                    details={
                        "backend": result.backend_used,
                        "peak_at": peak_time.isoformat(),
                        "horizon_min": offset,
                    },
                )

    # ------------------------------------------------------------------ backtest
    def backtest(self, *, write: bool = True) -> dict[str, Any]:
        """Rolling-origin backtest of all three backends (docs/03 M01).

        Writes ``data/derived/backtest.json``, which the model card quotes.
        """
        settings = self.param("backtest") or {}
        history = self._history["S01"]
        table = self.forecast_engine.compare_backends(
            history,
            horizon_steps=int(settings.get("horizon_steps", 12)),
            folds=int(settings.get("folds", 7)),
        )
        payload = {
            "model_id": self.model_id,
            "entity": next(iter(history)),
            "horizon_steps": int(settings.get("horizon_steps", 12)),
            "folds": int(settings.get("folds", 7)),
            "note": (
                "Computed on synthetic data. These numbers measure pipeline correctness, "
                "not real-world accuracy. Chronos-2 is zero-shot while LightGBM is fitted "
                "on this very history, so the comparison is not like for like."
            ),
            "backends": {name: metrics.as_dict() for name, metrics in table.items()},
        }
        if write:
            path = self.derived_dir / "backtest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.info("M01 backtest written to %s", path)
        return payload

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
            question="How many people will arrive, by zone, gate and time?",
            engine=self.engine,
            method_summary=(
                f"Darts {backend} over one series per zone and per gate, with hour-of-day, "
                f"day-of-week, snan-flag, temperature and rain as future covariates. "
                f"Quantiles 0.1/0.5/0.9 give the 80 percent band. Falls back to LightGBM "
                f"then NaiveSeasonal."
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
                "Trained context is synthetic. A foundation model has never seen this world, "
                "and the backtest measures pipeline correctness, not real-world accuracy.",
                "Chronos-2 runs zero-shot while LightGBM is fitted on the same history, so "
                "the backend comparison in the model card is not like for like.",
                "Covariates are known exactly because the world is generated. With a real "
                "feed, temperature and rain would themselves be forecasts with their own "
                "error.",
                "Insensitive to S04: the docs/03 M21 arrival multiplier depends on rain "
                "only, so extreme heat changes nothing M01 reads.",
                "Confidence is a fixed heuristic, not a calibrated probability.",
            ],
            license_notes=[
                "Darts: Apache-2.0. Chronos-2 weights: see the autogluon model card.",
                "LightGBM: MIT.",
            ],
        )
