"""Forecasting engine (docs/07 Phase 4, docs/06 section 3).

One interface over three Darts backends, so nine models share the same forecasting code and
the same uncertainty convention:

============  ==========================  =========================================
``chronos2``  Darts ``Chronos2Model``     default: zero-shot foundation model
``lightgbm``  Darts ``LightGBMModel``     fitted quantile-regression baseline
``naive``     Darts ``NaiveSeasonal``     the floor every other backend must beat
============  ==========================  =========================================

**Why Chronos-2 and not TimesFM.** docs/06 section 3 is explicit: ``Chronos2Model`` supports
past and future covariates, ``TimesFM2p5Model`` does not. This world's covariates - the snan
flag, the temperature, the rain - are the whole reason a forecast is interesting here, so a
backend that cannot take them is not useful.

**Why it degrades rather than fails.** Chronos-2 needs torch, pytorch-lightning and a weights
download. On a machine with none of those a forecasting model must still answer; docs/02
section 4 has a ``degraded`` status for exactly this. The engine walks
``chronos2 -> lightgbm -> naive`` and reports which backend actually ran.

**Quantiles.** All three backends are asked for 0.1 / 0.5 / 0.9 through Darts'
``predict_likelihood_parameters=True``, which returns components named ``<c>_q0.100`` and so
on. Those map to ``lower`` / ``value`` / ``upper`` with ``quantile_level: 0.8``, the docs/02
section 4 band convention. NaiveSeasonal is a point forecast and says so in a warning.

**Measured on this machine** (CPU, chronos-2-small, 8 series, 12 steps): weights download
about 25 s once, then ``fit`` 0.7 s and ``predict`` 0.11 s. The fit is cheap because a
foundation model is not trained here - docs/06 forbids fine-tuning on synthetic data - but it
is not free, so a model should fit once in ``load()`` rather than per request.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..contracts.models import IST
from ..errors import EngineError
from ..logging import get_logger
from ..paths import ensure_dir, is_offline, real_cache_dir

log = get_logger(__name__)

#: Backend names, in the order the engine falls back through them.
BACKEND_CHRONOS = "chronos2"
BACKEND_LIGHTGBM = "lightgbm"
BACKEND_NAIVE = "naive"
FALLBACK_ORDER = (BACKEND_CHRONOS, BACKEND_LIGHTGBM, BACKEND_NAIVE)

#: The quantiles the contract needs: lower, value, upper.
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)

#: Steps in a day on the 15-minute grid; the seasonality NaiveSeasonal repeats.
STEPS_PER_DAY = 96

#: Darts names quantile components ``<component>_q0.100``.
_QUANTILE_COMPONENT = re.compile(r"_q(\d+\.\d+)$")

#: Below this the band is treated as degenerate and left alone (a flat series).
_MIN_HALF_WIDTH = 1e-9


@dataclass
class ForecastResult:
    """One backend's forecast for one or more entities."""

    #: entity -> median forecast, indexed by timestamp.
    values: dict[str, pd.Series]
    #: entity -> lower quantile. Empty for a point-forecast backend.
    lower: dict[str, pd.Series] = field(default_factory=dict)
    #: entity -> upper quantile.
    upper: dict[str, pd.Series] = field(default_factory=dict)
    quantile_level: float = 0.8
    backend_used: str = BACKEND_NAIVE
    backend_requested: str = BACKEND_CHRONOS
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    #: True when the band has been conformally widened (see :class:`Calibration`). A model
    #: should say so, because an uncalibrated band understates its own uncertainty.
    calibrated: bool = False

    @property
    def degraded(self) -> bool:
        """True when the backend that ran is not the one that was asked for."""
        return self.backend_used != self.backend_requested

    @property
    def has_bands(self) -> bool:
        return bool(self.lower)

    @property
    def entities(self) -> list[str]:
        return list(self.values)

    def at(self, entity: str, timestamp: pd.Timestamp) -> tuple[float, float | None, float | None]:
        """(value, lower, upper) for one entity at one timestamp."""
        value = float(self.values[entity].loc[timestamp])
        low = self.lower.get(entity)
        high = self.upper.get(entity)
        return (
            value,
            None if low is None else float(low.loc[timestamp]),
            None if high is None else float(high.loc[timestamp]),
        )

    def frame(self, entity: str) -> pd.DataFrame:
        """One entity's forecast as a (timestamp, value, lower, upper) frame."""
        out = pd.DataFrame({"value": self.values[entity]})
        if entity in self.lower:
            out["lower"] = self.lower[entity]
            out["upper"] = self.upper[entity]
        return out


@dataclass
class Calibration:
    """Per-horizon-step widening factors that make the band mean what it says.

    A raw quantile forecast from any of these backends is **too narrow at long horizons**.
    Measured on M01 (Z01, 7 folds): the median error grows 26-fold from step 1 to step 12
    while the band widens only 1.7-fold, so coverage falls from ~100 percent in the first
    hour to 43 percent after it. A record labelled ``quantile_level: 0.8`` that actually
    contains the truth 43 percent of the time is worse than no band at all, because a reader
    in a command centre has no way to know.

    The fix is conformal: forecast a set of held-out origins, measure how far outside the
    band the truth actually fell **at each step of the horizon**, and widen that step until
    the empirical coverage matches the nominal level. Errors grow with horizon, so the
    factors do too - which is exactly what the raw quantiles fail to do.

    The factors are multiplicative around the median rather than additive, so one set of
    numbers serves entities of wildly different magnitude (Z01 carries thousands of people,
    G03 hundreds).
    """

    #: One factor per horizon step. ``lower' = median - factor * (median - lower)``.
    factors: np.ndarray
    #: Nominal coverage the factors target, e.g. 0.8 for the 0.1/0.9 band.
    target: float
    #: Calibration points behind each step's factor (folds x entities).
    samples_per_step: int
    folds: int
    #: Pooled coverage on the calibration set, before and after widening.
    coverage_before: float
    coverage_after: float
    backend: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "target_pct": round(self.target * 100, 1),
            "coverage_before_pct": round(self.coverage_before, 1),
            "coverage_after_pct": round(self.coverage_after, 1),
            "factors": [round(float(f), 3) for f in self.factors],
            "folds": self.folds,
            "samples_per_step": self.samples_per_step,
        }


@dataclass
class BacktestMetrics:
    """Rolling-origin backtest metrics (docs/05 section 7)."""

    backend: str
    mae: float
    mape: float
    pinball: float
    coverage: float
    folds: int
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "mae": round(self.mae, 3),
            "mape_pct": round(self.mape, 2),
            "pinball": round(self.pinball, 4),
            "interval_coverage_pct": (None if np.isnan(self.coverage) else round(self.coverage, 1)),
            "folds": self.folds,
            "elapsed_s": round(self.elapsed_s, 2),
        }


# --------------------------------------------------------------------------- metrics
def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def mape(actual: np.ndarray, forecast: np.ndarray, *, min_actual: float = 1.0) -> float:
    """MAPE with near-zero actuals masked out (docs/05 section 7).

    A crowd series is near zero at 04:00, and dividing by it produces a meaningless
    thousand-percent error, so those points are excluded rather than allowed to dominate.
    """
    mask = np.abs(actual) >= min_actual
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - forecast[mask]) / np.abs(actual[mask])) * 100.0)


def pinball_loss(actual: np.ndarray, forecast: np.ndarray, quantile: float) -> float:
    """``mean(max(q(y - yhat), (q-1)(y - yhat)))`` (docs/05 section 7)."""
    difference = actual - forecast
    return float(np.mean(np.maximum(quantile * difference, (quantile - 1.0) * difference)))


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actuals inside the band, as a percentage."""
    return float(np.mean((actual >= lower) & (actual <= upper)) * 100.0)


# ------------------------------------------------------------------------ the engine
class ForecastEngine:
    """Forecasts many entity series at once, with a documented fallback chain."""

    def __init__(
        self,
        *,
        backend: str = BACKEND_CHRONOS,
        seed: int = 42,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        hub_model_name: str = "autogluon/chronos-2-small",
        input_chunk_length: int = STEPS_PER_DAY,
        lightgbm_lags: int = STEPS_PER_DAY,
        seasonality: int = STEPS_PER_DAY,
        allow_fallback: bool = True,
        max_context_steps: int = 2688,
        calibrate_bands: bool = False,
        calibration_folds: int = 24,
        calibration_stride_steps: int = 24,
        calibration_min_factor: float = 1.0,
        calibration_max_factor: float = 12.0,
    ) -> None:
        """
        Args:
            backend: which backend to try first.
            seed: from ``config.seed``; re-applied on every call so repeats reproduce.
            quantiles: must include 0.5; the outer two become the band.
            hub_model_name: the small model by default, because everything must run on CPU
                (docs/06 sections 2 and 3). ``amazon/chronos-2`` is opt-in through config.
            input_chunk_length: context window Chronos-2 reads, in steps.
            lightgbm_lags: lag structure for the baseline.
            seasonality: NaiveSeasonal K; one day on the 15-minute grid.
            allow_fallback: set False to make a backend failure raise, which is what the
                backend-comparison table needs.
            max_context_steps: history fed to a foundation model. docs/06 section 3 asks for
                at least two weeks of 15-minute history; 2688 steps is four weeks.
            calibrate_bands: conformally widen the band so its stated coverage is true. Off
                by default so a bare engine is cheap; models turn it on in config and pay
                for it once in ``warmup``. See :class:`Calibration`.
            calibration_folds: held-out origins to measure coverage on. 24 origins x the
                number of entities is enough to estimate an 80th percentile per step.
            calibration_stride_steps: spacing between origins. 24 steps is six hours, so 24
                origins span six days. Measured on M01, held out from the evaluation window:
                six-hourly origins give 74.6 percent coverage, two-hourly 74.2, and DAILY
                origins over 20 days only 64.6. Recency beats diversity here because the
                series is not stationary - the event builds to a peak - so origins close to
                now describe the regime the forecast is actually in.
            calibration_min_factor: floor on the widening factor. The default of 1.0 means
                the band is never made NARROWER than the backend's own. That asymmetry is
                deliberate: with a few dozen folds the estimate is noisy, and in a command
                centre an over-wide band is a smaller error than an over-narrow one.
            calibration_max_factor: ceiling, so one pathological fold cannot produce a band
                so wide it is meaningless.
        """
        if 0.5 not in tuple(quantiles):
            raise EngineError("quantiles must include the median, 0.5")
        self.backend = backend
        self.seed = int(seed)
        self.quantiles = tuple(sorted(float(q) for q in quantiles))
        self.hub_model_name = hub_model_name
        self.input_chunk_length = int(input_chunk_length)
        self.lightgbm_lags = int(lightgbm_lags)
        self.seasonality = int(seasonality)
        self.allow_fallback = allow_fallback
        self.max_context_steps = int(max_context_steps)
        self.calibrate_bands = bool(calibrate_bands)
        self.calibration_folds = int(calibration_folds)
        self.calibration_stride_steps = int(calibration_stride_steps)
        self.calibration_min_factor = float(calibration_min_factor)
        self.calibration_max_factor = float(calibration_max_factor)
        #: horizon_steps -> the Calibration measured for it.
        self._calibration: dict[int, Calibration] = {}
        #: Set while calibrate() runs, so its own folds are scored on the RAW band.
        self._suppress_calibration = False
        # Fitted Darts models, keyed by what makes a fit reusable. A Darts global model can
        # predict on any series afterwards, so one fit serves every later call of the same
        # shape. Without this, a live /predict refits on every request: measured 8.9 s for
        # Chronos-2 and 12.6 s for LightGBM, against the docs/02 section 6 target of 10 s.
        self._fitted: dict[tuple, Any] = {}
        self._configure_caches()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None, *, seed: int = 42) -> ForecastEngine:
        """Build from a model's ``params.forecast`` block."""
        block = dict(config or {})
        calibration = dict(block.get("calibration") or {})
        return cls(
            backend=block.get("backend", BACKEND_CHRONOS),
            calibrate_bands=bool(calibration.get("enabled", False)),
            calibration_folds=int(calibration.get("folds", 24)),
            calibration_stride_steps=int(calibration.get("stride_steps", 24)),
            calibration_min_factor=float(calibration.get("min_factor", 1.0)),
            calibration_max_factor=float(calibration.get("max_factor", 12.0)),
            seed=seed,
            quantiles=tuple(block.get("quantiles", DEFAULT_QUANTILES)),
            hub_model_name=block.get("hub_model_name", "autogluon/chronos-2-small"),
            input_chunk_length=int(block.get("input_chunk_length", STEPS_PER_DAY)),
            lightgbm_lags=int(block.get("lightgbm_lags", STEPS_PER_DAY)),
            seasonality=int(block.get("seasonality", STEPS_PER_DAY)),
            allow_fallback=bool(block.get("allow_fallback", True)),
            max_context_steps=int(block.get("max_context_steps", 2688)),
        )

    # ------------------------------------------------------------------ setup
    def _configure_caches(self) -> None:
        """Point every weights cache inside the repo, gitignored (docs/06 section 2).

        Set before any huggingface import, otherwise the hub client has already resolved the
        default cache in the user profile.
        """
        cache = ensure_dir(real_cache_dir() / "hf")
        os.environ.setdefault("HF_HOME", str(cache))
        os.environ.setdefault("TORCH_HOME", str(ensure_dir(real_cache_dir() / "torch")))
        # Windows without developer mode cannot symlink; the hub warns loudly about it.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        if is_offline():
            # Fail fast instead of hanging on a blocked network.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    def _seed_everything(self) -> None:
        """Re-seed on every call, so a second call in one process reproduces the first."""
        try:
            import torch

            torch.manual_seed(self.seed)
        except ImportError:
            pass

    @property
    def quantile_level(self) -> float:
        """Band coverage implied by the outer quantiles, e.g. 0.1/0.9 -> 0.8."""
        return round(self.quantiles[-1] - self.quantiles[0], 4)

    # ------------------------------------------------------------------ conversion
    @staticmethod
    def _strip_tz(index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, Any]:
        """Darts does not support tz-aware indices, so the tz is removed and remembered."""
        if index.tz is None:
            return index, None
        return index.tz_localize(None), index.tz

    @staticmethod
    def _infer_freq(index: pd.DatetimeIndex) -> str:
        inferred = pd.infer_freq(index)
        if inferred:
            return inferred
        if len(index) >= 2:
            minutes = max(1, int((index[1] - index[0]).total_seconds() // 60))
            return f"{minutes}min"
        return "15min"

    def _prepare(
        self,
        history: Mapping[str, pd.Series],
        past_covariates: Mapping[str, pd.DataFrame] | None,
        future_covariates: Mapping[str, pd.DataFrame] | None,
    ) -> tuple[list[str], list[Any], list[Any] | None, list[Any] | None, Any]:
        """Convert the inputs into the parallel lists Darts wants."""
        from darts import TimeSeries

        entities = list(history)
        first_index = pd.DatetimeIndex(history[entities[0]].index)
        naive_index, tz = self._strip_tz(first_index)
        freq = self._infer_freq(naive_index)

        series: list[Any] = []
        for entity in entities:
            values = history[entity].astype("float64").sort_index()
            if len(values) > self.max_context_steps:
                values = values.iloc[-self.max_context_steps :]
            index, _ = self._strip_tz(pd.DatetimeIndex(values.index))
            series.append(
                TimeSeries.from_series(
                    pd.Series(values.to_numpy(), index=index),
                    freq=freq,
                    fill_missing_dates=True,
                )
            )

        def convert(source: Mapping[str, pd.DataFrame] | None) -> list[Any] | None:
            if not source:
                return None
            out: list[Any] = []
            for entity in entities:
                frame = source.get(entity)
                if frame is None or frame.empty:
                    return None
                ordered = frame.sort_index().astype("float64")
                index, _ = self._strip_tz(pd.DatetimeIndex(ordered.index))
                out.append(
                    TimeSeries.from_dataframe(
                        pd.DataFrame(ordered.to_numpy(), index=index, columns=ordered.columns),
                        freq=freq,
                        fill_missing_dates=True,
                    )
                )
            return out

        return entities, series, convert(past_covariates), convert(future_covariates), tz

    # ------------------------------------------------------------------ prediction
    def predict(
        self,
        history: Mapping[str, pd.Series],
        horizon_steps: int,
        *,
        past_covariates: Mapping[str, pd.DataFrame] | None = None,
        future_covariates: Mapping[str, pd.DataFrame] | None = None,
    ) -> ForecastResult:
        """Forecast every entity series ``horizon_steps`` ahead.

        Args:
            history: entity -> a time-indexed series of the quantity to forecast.
            horizon_steps: how many steps ahead, on the series' own grid.
            past_covariates: entity -> covariates known only up to now.
            future_covariates: entity -> covariates known across the forecast window.
        """
        if not history:
            raise EngineError("predict() needs at least one entity series")
        if horizon_steps <= 0:
            raise EngineError(f"horizon_steps must be positive, got {horizon_steps}")

        self._seed_everything()
        started = time.perf_counter()
        collected: list[str] = []

        for backend in self._backend_chain():
            try:
                result = self._run_backend(
                    backend, history, horizon_steps, past_covariates, future_covariates
                )
                result.backend_requested = self.backend
                result.warnings = [*collected, *result.warnings]
                if not self._suppress_calibration:
                    result = self._apply_calibration(result, horizon_steps)
                result.elapsed_s = time.perf_counter() - started
                if result.degraded:
                    log.warning(
                        "forecast fell back from %s to %s", self.backend, result.backend_used
                    )
                return result
            except Exception as exc:
                message = (
                    f"forecast backend {backend!r} unavailable "
                    f"({type(exc).__name__}: {str(exc).splitlines()[0][:160]})"
                )
                log.warning(message)
                collected.append(message)
                if not self.allow_fallback:
                    raise EngineError(message) from exc

        raise EngineError("every forecast backend failed: " + "; ".join(collected))

    @staticmethod
    def _covariate_signature(covariates: list[Any] | None) -> tuple[str, ...] | None:
        """The component names of a prepared covariate list, or None when there are none."""
        if not covariates:
            return None
        return tuple(str(name) for name in covariates[0].components)

    def _cache_key(
        self, backend: str, horizon_steps: int, context: int, past: Any, future: Any
    ) -> tuple:
        """What makes a fitted model reusable for a later call.

        The covariate COMPONENTS are part of the key, not merely whether any were given. A
        Darts global model records how many covariate components it was fitted with and
        rejects a different number at predict time:

            "The provided `historic_future_covariates` must have equal number of components
             as the `historic_future_covariates` used to train the model."

        A model whose covariates are optional - M05 drops its arrivals column when M01 is
        unavailable - would otherwise reuse a model fitted with a different set and fail the
        whole request instead of just re-fitting.
        """
        return (
            backend,
            horizon_steps,
            context,
            self._covariate_signature(past),
            self._covariate_signature(future),
        )

    def warmup(
        self,
        history: Mapping[str, pd.Series],
        horizon_steps: int,
        *,
        past_covariates: Mapping[str, pd.DataFrame] | None = None,
        future_covariates: Mapping[str, pd.DataFrame] | None = None,
    ) -> ForecastResult:
        """Fit and cache now, so the first real request is not the slow one.

        A model calls this from ``load()``. Fetching weights and fitting takes seconds, and
        doing it while a request waits would blow the docs/02 section 6 budget.

        When ``calibrate_bands`` is on this also measures the band calibration, which costs
        one forecast per fold. That is startup time, not request time, and it is the only
        thing that makes ``quantile_level`` an honest number.
        """
        result = self.predict(
            history,
            horizon_steps,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        log.info(
            "forecast warmup: backend %s ready in %.1fs",
            result.backend_used,
            result.elapsed_s,
        )
        if self.calibrate_bands and result.has_bands:
            try:
                self.calibrate(
                    history,
                    horizon_steps,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                )
            except Exception as exc:
                # An uncalibrated band is worse than a calibrated one but better than no
                # service. The model reports `calibrated: false` and says so.
                log.warning("band calibration failed (%s); serving the raw band", exc)
        return result

    # ------------------------------------------------------------------ calibration
    def calibrate(
        self,
        history: Mapping[str, pd.Series],
        horizon_steps: int,
        *,
        folds: int | None = None,
        stride_steps: int | None = None,
        past_covariates: Mapping[str, pd.DataFrame] | None = None,
        future_covariates: Mapping[str, pd.DataFrame] | None = None,
    ) -> Calibration:
        """Measure how wide the band really needs to be, one factor per horizon step.

        Walks back through ``folds`` origins, forecasts from each, and compares the band
        against what actually happened. For every held-out point the *required* multiplier
        is how far the truth sat beyond the median relative to the band's own half-width::

            r = (y - median) / (upper - median)      when y is above the median
                (median - y) / (median - lower)      when y is below it
                0                                    when y is already inside

        A point inside the band needs ``r <= 1``. The factor for a step is then the
        ``target`` quantile of those ratios, which by construction makes that fraction of
        the calibration points fall inside the widened band. This is conformalised quantile
        regression, adapted to a multiplicative form so one factor serves every entity.

        Entities are pooled, because a factor per entity per step would be estimated from
        ``folds`` points alone. Pooling assumes the backend is miscalibrated in the same way
        across series, which is what the diagnosis showed: the failure is horizon-driven,
        not entity-driven.
        """
        folds = int(folds if folds is not None else self.calibration_folds)
        stride = int(stride_steps if stride_steps is not None else self.calibration_stride_steps)
        if horizon_steps <= 0:
            raise EngineError("calibrate() needs a positive horizon")
        if folds < 2:
            raise EngineError(f"calibration needs at least 2 folds, got {folds}")

        entities = list(history)
        series = {e: history[e].astype("float64").sort_index() for e in entities}
        shortest = min(len(s) for s in series.values())
        needed = horizon_steps + stride * folds
        if shortest < needed + horizon_steps:
            raise EngineError(
                f"calibration needs at least {needed + horizon_steps} steps of history, "
                f"the shortest series has {shortest}"
            )

        # ratios[step] collects one value per (fold, entity) held-out point.
        ratios: list[list[float]] = [[] for _ in range(horizon_steps)]
        inside_before = 0
        total = 0
        backend_used = self.backend

        for fold in range(folds):
            end = shortest - (folds - fold) * stride
            if end <= horizon_steps:
                continue
            context = {e: s.iloc[:end] for e, s in series.items()}
            self._suppress_calibration = True
            try:
                result = self.predict(
                    context,
                    horizon_steps,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                )
            except Exception as exc:  # a bad fold must not lose the whole calibration
                log.warning("calibration fold %d failed (%s); skipping", fold, exc)
                continue
            finally:
                self._suppress_calibration = False
            backend_used = result.backend_used
            if not result.lower:
                raise EngineError(
                    f"backend {backend_used!r} produces no band, so there is nothing to calibrate"
                )
            for entity in entities:
                truth = series[entity].iloc[end : end + horizon_steps].to_numpy()
                if len(truth) < horizon_steps:
                    continue
                median = result.values[entity].to_numpy()[:horizon_steps]
                low = result.lower[entity].to_numpy()[:horizon_steps]
                high = result.upper[entity].to_numpy()[:horizon_steps]
                for step in range(horizon_steps):
                    ratios[step].append(
                        self._required_factor(truth[step], median[step], low[step], high[step])
                    )
                    inside_before += int(low[step] <= truth[step] <= high[step])
                    total += 1

        counts = [len(r) for r in ratios]
        if not total or min(counts) == 0:
            raise EngineError("calibration produced no usable folds")

        target = self.quantile_level
        factors = np.empty(horizon_steps, dtype="float64")
        for step in range(horizon_steps):
            sample = np.asarray(ratios[step], dtype="float64")
            # The finite-sample conformal quantile: with n points, the level that guarantees
            # `target` coverage is ceil((n+1)*target)/n, which is slightly above `target`.
            n = sample.size
            level = min(1.0, np.ceil((n + 1) * target) / n)
            factors[step] = float(np.quantile(sample, level, method="higher"))
        factors = np.clip(factors, self.calibration_min_factor, self.calibration_max_factor)
        # Widening must never shrink as the horizon grows: a later step is strictly harder to
        # forecast, and a dip would be sampling noise rather than signal.
        factors = np.maximum.accumulate(factors)

        inside_after = sum(
            int(ratio <= factors[step]) for step in range(horizon_steps) for ratio in ratios[step]
        )
        calibration = Calibration(
            factors=factors,
            target=target,
            samples_per_step=min(counts),
            folds=folds,
            coverage_before=100.0 * inside_before / total,
            coverage_after=100.0 * inside_after / total,
            backend=backend_used,
        )
        self._calibration[horizon_steps] = calibration
        log.info(
            "band calibration (%s, %d steps): coverage %.1f%% -> %.1f%% against a %.0f%% "
            "target; factors %.2f to %.2f",
            backend_used,
            horizon_steps,
            calibration.coverage_before,
            calibration.coverage_after,
            target * 100,
            factors[0],
            factors[-1],
        )
        return calibration

    @staticmethod
    def _required_factor(truth: float, median: float, low: float, high: float) -> float:
        """How much this band would have to be widened to contain ``truth``."""
        if truth > median:
            half = high - median
            return 0.0 if half <= _MIN_HALF_WIDTH else float((truth - median) / half)
        if truth < median:
            half = median - low
            return 0.0 if half <= _MIN_HALF_WIDTH else float((median - truth) / half)
        return 0.0

    def _apply_calibration(self, result: ForecastResult, horizon_steps: int) -> ForecastResult:
        """Widen a result's band with the factors measured for this horizon."""
        calibration = self._calibration.get(horizon_steps)
        if calibration is None or not result.lower:
            return result
        factors = calibration.factors
        for entity, median in result.values.items():
            low = result.lower.get(entity)
            high = result.upper.get(entity)
            if low is None or high is None:
                continue
            span = min(len(median), len(factors))
            scale = np.ones(len(median), dtype="float64")
            scale[:span] = factors[:span]
            scale[span:] = factors[-1]  # a longer request keeps the last measured factor
            centre = median.to_numpy()
            result.lower[entity] = pd.Series(
                centre - scale * (centre - low.to_numpy()), index=median.index
            )
            result.upper[entity] = pd.Series(
                centre + scale * (high.to_numpy() - centre), index=median.index
            )
        result.calibrated = True
        return result

    def calibration_for(self, horizon_steps: int) -> Calibration | None:
        """The calibration in force for a horizon, if one has been measured."""
        return self._calibration.get(horizon_steps)

    def clear_cache(self) -> None:
        """Drop fitted models. Call when the underlying history changes materially."""
        self._fitted.clear()

    def _backend_chain(self) -> list[str]:
        """The backend to try, then the ones to fall back to."""
        if not self.allow_fallback:
            return [self.backend]
        chain = [self.backend]
        start = FALLBACK_ORDER.index(self.backend) if self.backend in FALLBACK_ORDER else -1
        chain.extend(b for b in FALLBACK_ORDER[start + 1 :] if b != self.backend)
        if BACKEND_NAIVE not in chain:
            chain.append(BACKEND_NAIVE)
        return chain

    def _run_backend(
        self,
        backend: str,
        history: Mapping[str, pd.Series],
        horizon_steps: int,
        past_covariates: Mapping[str, pd.DataFrame] | None,
        future_covariates: Mapping[str, pd.DataFrame] | None,
    ) -> ForecastResult:
        if backend == BACKEND_CHRONOS:
            return self._run_chronos(history, horizon_steps, past_covariates, future_covariates)
        if backend == BACKEND_LIGHTGBM:
            return self._run_lightgbm(history, horizon_steps, past_covariates, future_covariates)
        if backend == BACKEND_NAIVE:
            return self._run_naive(history, horizon_steps)
        raise EngineError(f"unknown forecast backend {backend!r}")

    # ------------------------------------------------------------------ backends
    def _run_chronos(
        self,
        history: Mapping[str, pd.Series],
        horizon_steps: int,
        past_covariates: Mapping[str, pd.DataFrame] | None,
        future_covariates: Mapping[str, pd.DataFrame] | None,
    ) -> ForecastResult:
        """Darts Chronos2Model, zero-shot with covariates.

        ``fit`` is required by the Darts API even though nothing is trained: docs/06 forbids
        fine-tuning on synthetic data, and no ``enable_finetuning`` argument is passed, so
        this only registers the series. It costs under a second after the weights are cached.
        """
        from darts.models import Chronos2Model
        from darts.utils.likelihood_models.torch import QuantileRegression

        entities, series, past, future, tz = self._prepare(
            history, past_covariates, future_covariates
        )
        context = min(self.input_chunk_length, min(len(s) for s in series) - horizon_steps)
        if context < 2:
            raise EngineError(
                f"not enough history for Chronos-2: need more than {horizon_steps + 2} steps"
            )
        key = self._cache_key(BACKEND_CHRONOS, horizon_steps, context, past, future)
        model = self._fitted.get(key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model is None:
                model = Chronos2Model(
                    input_chunk_length=context,
                    output_chunk_length=horizon_steps,
                    hub_model_name=self.hub_model_name,
                    likelihood=QuantileRegression(quantiles=list(self.quantiles)),
                )
                model.fit(series, past_covariates=past, future_covariates=future)
                self._fitted[key] = model
            forecasts = model.predict(
                n=horizon_steps,
                series=series,
                past_covariates=past,
                future_covariates=future,
                predict_likelihood_parameters=True,
                verbose=False,
            )
        return self._collect(entities, forecasts, BACKEND_CHRONOS, tz)

    def _run_lightgbm(
        self,
        history: Mapping[str, pd.Series],
        horizon_steps: int,
        past_covariates: Mapping[str, pd.DataFrame] | None,
        future_covariates: Mapping[str, pd.DataFrame] | None,
    ) -> ForecastResult:
        """Darts LightGBMModel with quantile regression, fitted on the given history."""
        from darts.models import LightGBMModel

        entities, series, past, future, tz = self._prepare(
            history, past_covariates, future_covariates
        )
        shortest = min(len(s) for s in series)
        lags = self.lightgbm_lags
        if shortest <= lags + horizon_steps:
            # Not enough history for the configured lag; use what there is.
            lags = max(2, (shortest - horizon_steps) // 2)
        key = self._cache_key(BACKEND_LIGHTGBM, horizon_steps, lags, past, future)
        model = self._fitted.get(key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model is None:
                model = LightGBMModel(
                    lags=lags,
                    lags_past_covariates=lags if past else None,
                    lags_future_covariates=(0, 1) if future else None,
                    output_chunk_length=horizon_steps,
                    likelihood="quantile",
                    quantiles=list(self.quantiles),
                    random_state=self.seed,
                    verbose=-1,
                )
                model.fit(series, past_covariates=past, future_covariates=future)
                self._fitted[key] = model
            forecasts = model.predict(
                n=horizon_steps,
                series=series,
                past_covariates=past,
                future_covariates=future,
                predict_likelihood_parameters=True,
            )
        return self._collect(entities, forecasts, BACKEND_LIGHTGBM, tz)

    def _run_naive(self, history: Mapping[str, pd.Series], horizon_steps: int) -> ForecastResult:
        """NaiveSeasonal: repeat the last season. The floor every backend must beat."""
        from darts.models import NaiveSeasonal

        entities, series, _, _, tz = self._prepare(history, None, None)
        forecasts = []
        for one in series:
            k = min(self.seasonality, max(1, len(one) - 1))
            model = NaiveSeasonal(K=k)
            model.fit(one)
            forecasts.append(model.predict(horizon_steps))
        return self._collect(entities, forecasts, BACKEND_NAIVE, tz)

    def _collect(
        self, entities: list[str], forecasts: Any, backend: str, tz: Any
    ) -> ForecastResult:
        """Darts output -> the dicts of pandas Series the models consume.

        With ``predict_likelihood_parameters=True`` Darts returns one component per quantile,
        named ``<component>_q0.100``. A point-forecast backend returns a single component.
        """
        if not isinstance(forecasts, list):
            forecasts = [forecasts]
        values: dict[str, pd.Series] = {}
        lower: dict[str, pd.Series] = {}
        upper: dict[str, pd.Series] = {}
        result_warnings: list[str] = []

        for entity, forecast in zip(entities, forecasts, strict=True):
            index = pd.DatetimeIndex(forecast.time_index)
            index = index.tz_localize(tz if tz is not None else IST)
            data = forecast.values()
            by_quantile: dict[float, np.ndarray] = {}
            for position, component in enumerate(forecast.components):
                match = _QUANTILE_COMPONENT.search(str(component))
                if match:
                    by_quantile[round(float(match.group(1)), 3)] = data[:, position]

            if by_quantile:
                median = by_quantile.get(0.5)
                if median is None:  # no median component: take the middle one
                    ordered = sorted(by_quantile)
                    median = by_quantile[ordered[len(ordered) // 2]]
                low = by_quantile.get(round(self.quantiles[0], 3))
                high = by_quantile.get(round(self.quantiles[-1], 3))
                values[entity] = pd.Series(median, index=index)
                if low is not None and high is not None:
                    # Quantile crossing is possible in a fitted quantile regression; the
                    # contract requires lower <= value <= upper, so they are ordered here.
                    lower[entity] = pd.Series(np.minimum(low, median), index=index)
                    upper[entity] = pd.Series(np.maximum(high, median), index=index)
            else:
                values[entity] = pd.Series(data[:, 0], index=index)

        if not lower:
            result_warnings.append(
                f"forecast backend {backend!r} produced a point forecast with no uncertainty band"
            )
        return ForecastResult(
            values=values,
            lower=lower,
            upper=upper,
            quantile_level=self.quantile_level,
            backend_used=backend,
            warnings=result_warnings,
        )

    # ------------------------------------------------------------------ backtest
    def backtest(
        self,
        history: Mapping[str, pd.Series],
        *,
        horizon_steps: int = 12,
        folds: int = 7,
        stride_steps: int = STEPS_PER_DAY,
    ) -> BacktestMetrics:
        """Rolling-origin backtest over the last ``folds`` origins (docs/03 M01).

        Each fold cuts the history at an origin, forecasts ``horizon_steps`` ahead and scores
        against what actually happened: MAE, MAPE, pinball loss at the median, and 80 percent
        interval coverage (docs/05 section 7).
        """
        started = time.perf_counter()
        entity = next(iter(history))
        full = history[entity].astype("float64").sort_index()
        needed = horizon_steps + stride_steps
        if len(full) < needed * 2:
            raise EngineError(
                f"backtest needs at least {needed * 2} steps of history, got {len(full)}"
            )

        actuals: list[np.ndarray] = []
        medians: list[np.ndarray] = []
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        backend_used = self.backend

        for fold in range(folds):
            end = len(full) - (folds - fold) * stride_steps
            if end <= needed:
                continue
            truth = full.iloc[end : end + horizon_steps]
            if len(truth) < horizon_steps:
                continue
            result = self.predict({entity: full.iloc[:end]}, horizon_steps)
            backend_used = result.backend_used
            actuals.append(truth.to_numpy())
            medians.append(result.values[entity].to_numpy()[: len(truth)])
            if entity in result.lower:
                lows.append(result.lower[entity].to_numpy()[: len(truth)])
                highs.append(result.upper[entity].to_numpy()[: len(truth)])

        if not actuals:
            raise EngineError("backtest produced no usable folds")

        actual = np.concatenate(actuals)
        median = np.concatenate(medians)
        coverage = (
            interval_coverage(actual, np.concatenate(lows), np.concatenate(highs))
            if lows
            else float("nan")
        )
        return BacktestMetrics(
            backend=backend_used,
            mae=mae(actual, median),
            mape=mape(actual, median),
            pinball=pinball_loss(actual, median, 0.5),
            coverage=coverage,
            folds=len(actuals),
            elapsed_s=time.perf_counter() - started,
        )

    def compare_backends(
        self,
        history: Mapping[str, pd.Series],
        *,
        horizon_steps: int = 12,
        folds: int = 7,
        backends: Sequence[str] = FALLBACK_ORDER,
    ) -> dict[str, BacktestMetrics]:
        """Backtest each backend in turn, for the M01 model-card table (docs/03 M01)."""
        out: dict[str, BacktestMetrics] = {}
        for backend in backends:
            engine = ForecastEngine(
                backend=backend,
                seed=self.seed,
                quantiles=self.quantiles,
                hub_model_name=self.hub_model_name,
                input_chunk_length=self.input_chunk_length,
                lightgbm_lags=self.lightgbm_lags,
                seasonality=self.seasonality,
                allow_fallback=False,
                max_context_steps=self.max_context_steps,
            )
            try:
                out[backend] = engine.backtest(history, horizon_steps=horizon_steps, folds=folds)
            except Exception as exc:
                log.warning("backend %s could not be backtested: %s", backend, exc)
        return out


# ------------------------------------------------------------------ covariate helpers
def calendar_covariates(
    index: pd.DatetimeIndex,
    *,
    snan_days: Sequence[Any] | None = None,
    extra: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """The future covariates every forecasting model shares (docs/03 M01).

    Hour of day as sine and cosine, so midnight sits next to 23:00 rather than 23 units away;
    day of week; and the snan-day flag. ``extra`` adds model-specific series such as
    temperature or rain, aligned onto the same index.
    """
    frame = pd.DataFrame(index=index)
    minutes = index.hour * 60 + index.minute
    frame["hour_sin"] = np.sin(2.0 * np.pi * minutes / (24 * 60))
    frame["hour_cos"] = np.cos(2.0 * np.pi * minutes / (24 * 60))
    frame["day_of_week"] = index.dayofweek.astype(float)
    snan = {pd.Timestamp(day).date() for day in (snan_days or [])}
    frame["is_snan_day"] = [1.0 if day in snan else 0.0 for day in index.date]
    if extra:
        for name, series in extra.items():
            aligned = series.reindex(index).astype("float64")
            frame[name] = aligned.ffill().bfill().fillna(0.0)
    return frame.astype("float64")


def series_by_entity(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    value_column: str,
    time_column: str = "timestamp",
) -> dict[str, pd.Series]:
    """Split a long table into one time-indexed series per entity."""
    out: dict[str, pd.Series] = {}
    for entity, group in frame.groupby(entity_column):
        series = group.set_index(time_column)[value_column].astype("float64").sort_index()
        out[str(entity)] = series[~series.index.duplicated(keep="last")]
    return out
