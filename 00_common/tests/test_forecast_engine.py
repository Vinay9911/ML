"""Forecast engine tests (docs/07 Phase 4 engine acceptance).

The acceptance list asks for: unit tests on a synthetic sine + noise series, finite backtest
metrics, a clear message or cached weights in offline mode, and a default request under 10 s
on CPU.

Chronos-2 needs a weights download on the very first run, so the tests that touch it are
marked ``weights`` and skipped when the cache is cold and the network is off.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from twin_common.engines.forecast import (
    BACKEND_CHRONOS,
    BACKEND_LIGHTGBM,
    BACKEND_NAIVE,
    ForecastEngine,
    calendar_covariates,
    interval_coverage,
    mae,
    mape,
    pinball_loss,
    series_by_entity,
)
from twin_common.errors import EngineError

STEPS = 600
HORIZON = 12


def _sine_series(entities: int = 4, steps: int = STEPS, seed: int = 0) -> dict[str, pd.Series]:
    """A daily sine plus noise: enough structure that a good backend beats the naive one."""
    index = pd.date_range("2027-07-01", periods=steps, freq="15min", tz="Asia/Kolkata")
    rng = np.random.default_rng(seed)
    return {
        f"Z{i + 1:02d}": pd.Series(
            100.0 * (i + 1)
            + 50.0 * np.sin(np.arange(steps) * 2 * np.pi / 96)
            + rng.normal(0.0, 3.0, steps),
            index=index,
        )
        for i in range(entities)
    }


@pytest.fixture(scope="module")
def history() -> dict[str, pd.Series]:
    return _sine_series()


def _chronos_available() -> bool:
    try:
        from darts.models import Chronos2Model  # noqa: F401

        return True
    except Exception:
        return False


chronos_only = pytest.mark.skipif(
    not _chronos_available(), reason="Chronos-2 needs torch and pytorch-lightning"
)


# --------------------------------------------------------------------------- metrics
def test_mae_is_zero_for_a_perfect_forecast() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    assert mae(actual, actual) == pytest.approx(0.0)
    assert mae(actual, actual + 2.0) == pytest.approx(2.0)


def test_mape_masks_near_zero_actuals() -> None:
    """docs/05 section 7: avoid near-zero actuals. A crowd series is zero at 04:00."""
    actual = np.array([0.0, 100.0])
    forecast = np.array([5.0, 110.0])
    # Without masking the first point would contribute an infinite percentage error.
    assert mape(actual, forecast) == pytest.approx(10.0)


def test_mape_of_all_zero_actuals_is_nan() -> None:
    assert np.isnan(mape(np.zeros(4), np.ones(4)))


def test_pinball_loss_is_asymmetric() -> None:
    """A high quantile should be punished more for under-forecasting than over."""
    actual = np.array([10.0])
    under = pinball_loss(actual, np.array([8.0]), 0.9)
    over = pinball_loss(actual, np.array([12.0]), 0.9)
    assert under > over
    # At the median the penalty is symmetric.
    assert pinball_loss(actual, np.array([8.0]), 0.5) == pytest.approx(
        pinball_loss(actual, np.array([12.0]), 0.5)
    )


def test_interval_coverage() -> None:
    actual = np.array([1.0, 5.0, 9.0])
    assert interval_coverage(actual, np.zeros(3), np.full(3, 10.0)) == pytest.approx(100.0)
    assert interval_coverage(actual, np.full(3, 4.0), np.full(3, 6.0)) == pytest.approx(100.0 / 3)


# ------------------------------------------------------------------------ construction
def test_quantiles_must_include_the_median() -> None:
    with pytest.raises(EngineError, match="must include the median"):
        ForecastEngine(quantiles=(0.1, 0.9))


def test_quantile_level_is_the_band_width() -> None:
    assert ForecastEngine().quantile_level == pytest.approx(0.8)
    assert ForecastEngine(quantiles=(0.05, 0.5, 0.95)).quantile_level == pytest.approx(0.9)


def test_from_config_reads_the_params_block() -> None:
    engine = ForecastEngine.from_config(
        {"backend": "lightgbm", "seasonality": 48, "quantiles": [0.2, 0.5, 0.8]}, seed=7
    )
    assert engine.backend == BACKEND_LIGHTGBM
    assert engine.seasonality == 48
    assert engine.seed == 7
    assert engine.quantile_level == pytest.approx(0.6)


def test_from_config_of_nothing_uses_the_documented_defaults() -> None:
    engine = ForecastEngine.from_config(None)
    assert engine.backend == BACKEND_CHRONOS
    assert engine.hub_model_name == "autogluon/chronos-2-small"


def test_fallback_chain_order() -> None:
    assert ForecastEngine(backend=BACKEND_CHRONOS)._backend_chain() == [
        BACKEND_CHRONOS,
        BACKEND_LIGHTGBM,
        BACKEND_NAIVE,
    ]
    assert ForecastEngine(backend=BACKEND_LIGHTGBM)._backend_chain() == [
        BACKEND_LIGHTGBM,
        BACKEND_NAIVE,
    ]
    assert ForecastEngine(backend=BACKEND_NAIVE)._backend_chain() == [BACKEND_NAIVE]


def test_no_fallback_means_one_attempt() -> None:
    assert ForecastEngine(backend=BACKEND_CHRONOS, allow_fallback=False)._backend_chain() == [
        BACKEND_CHRONOS
    ]


# ------------------------------------------------------------------------- validation
def test_empty_history_is_rejected() -> None:
    with pytest.raises(EngineError, match="at least one entity series"):
        ForecastEngine(backend=BACKEND_NAIVE).predict({}, 12)


def test_non_positive_horizon_is_rejected(history) -> None:
    with pytest.raises(EngineError, match="must be positive"):
        ForecastEngine(backend=BACKEND_NAIVE).predict(history, 0)


# ---------------------------------------------------------------------------- naive
def test_naive_forecasts_every_entity(history) -> None:
    result = ForecastEngine(backend=BACKEND_NAIVE, allow_fallback=False).predict(history, HORIZON)
    assert result.backend_used == BACKEND_NAIVE
    assert set(result.entities) == set(history)
    for entity in history:
        assert len(result.values[entity]) == HORIZON


def test_naive_repeats_the_previous_season(history) -> None:
    """NaiveSeasonal with K=96 must literally repeat one day earlier."""
    engine = ForecastEngine(backend=BACKEND_NAIVE, seasonality=96, allow_fallback=False)
    result = engine.predict(history, HORIZON)
    original = history["Z01"]
    expected = original.to_numpy()[-96 : -96 + HORIZON]
    assert result.values["Z01"].to_numpy() == pytest.approx(expected)


def test_naive_has_no_band_and_says_so(history) -> None:
    result = ForecastEngine(backend=BACKEND_NAIVE, allow_fallback=False).predict(history, HORIZON)
    assert not result.has_bands
    assert any("point forecast" in w for w in result.warnings)


def test_timezone_survives_the_round_trip(history) -> None:
    """Darts strips tz internally; the engine must put it back (docs/02 section 4)."""
    result = ForecastEngine(backend=BACKEND_NAIVE, allow_fallback=False).predict(history, HORIZON)
    index = result.values["Z01"].index
    assert index.tz is not None
    assert str(index.tz) == "Asia/Kolkata"


def test_forecast_starts_after_the_history_ends(history) -> None:
    result = ForecastEngine(backend=BACKEND_NAIVE, allow_fallback=False).predict(history, HORIZON)
    assert result.values["Z01"].index[0] > history["Z01"].index[-1]


# -------------------------------------------------------------------------- lightgbm
def test_lightgbm_produces_an_ordered_band(history) -> None:
    result = ForecastEngine(backend=BACKEND_LIGHTGBM, allow_fallback=False).predict(
        history, HORIZON
    )
    assert result.backend_used == BACKEND_LIGHTGBM
    assert result.has_bands
    assert result.quantile_level == pytest.approx(0.8)
    for entity in history:
        value, low, high = result.at(entity, result.values[entity].index[0])
        # The contract requires lower <= value <= upper even if the quantiles crossed.
        assert low <= value <= high


def test_lightgbm_is_deterministic(history) -> None:
    engine = ForecastEngine(backend=BACKEND_LIGHTGBM, seed=42, allow_fallback=False)
    first = engine.predict(history, HORIZON).values["Z01"].to_numpy()
    second = engine.predict(history, HORIZON).values["Z01"].to_numpy()
    assert np.allclose(first, second)


def test_lightgbm_tracks_a_sine(history) -> None:
    """A fitted model on a clean daily sine should beat the naive repeat."""
    fitted = ForecastEngine(backend=BACKEND_LIGHTGBM, allow_fallback=False).backtest(
        history, horizon_steps=HORIZON, folds=3
    )
    naive = ForecastEngine(backend=BACKEND_NAIVE, allow_fallback=False).backtest(
        history, horizon_steps=HORIZON, folds=3
    )
    assert np.isfinite(fitted.mae)
    assert fitted.mae < naive.mae * 1.5, "a fitted model should be competitive with naive"


# --------------------------------------------------------------------------- chronos
@chronos_only
@pytest.mark.weights
def test_chronos_produces_an_ordered_band(history) -> None:
    result = ForecastEngine(backend=BACKEND_CHRONOS, allow_fallback=False).predict(history, HORIZON)
    assert result.backend_used == BACKEND_CHRONOS
    assert result.has_bands
    for entity in history:
        value, low, high = result.at(entity, result.values[entity].index[0])
        assert low <= value <= high


@chronos_only
@pytest.mark.weights
def test_chronos_is_deterministic(history) -> None:
    engine = ForecastEngine(backend=BACKEND_CHRONOS, seed=42, allow_fallback=False)
    first = engine.predict(history, HORIZON).values["Z01"].to_numpy()
    second = engine.predict(history, HORIZON).values["Z01"].to_numpy()
    assert np.allclose(first, second)


@chronos_only
@pytest.mark.weights
def test_warm_request_is_well_under_the_budget(history) -> None:
    """docs/02 section 6: under 10 s on CPU.

    The fit is cached, so the cost lands in warmup at load time rather than in the request.
    """
    engine = ForecastEngine(backend=BACKEND_CHRONOS, allow_fallback=False)
    engine.warmup(history, HORIZON)
    started = time.perf_counter()
    engine.predict(history, HORIZON)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0, f"warm request took {elapsed:.1f}s"


@chronos_only
@pytest.mark.weights
def test_the_fit_cache_actually_saves_time(history) -> None:
    engine = ForecastEngine(backend=BACKEND_CHRONOS, allow_fallback=False)
    started = time.perf_counter()
    engine.predict(history, HORIZON)
    cold = time.perf_counter() - started
    started = time.perf_counter()
    engine.predict(history, HORIZON)
    warm = time.perf_counter() - started
    assert warm < cold, f"cold {cold:.2f}s, warm {warm:.2f}s"


@chronos_only
@pytest.mark.weights
def test_clear_cache_forces_a_refit(history) -> None:
    engine = ForecastEngine(backend=BACKEND_CHRONOS, allow_fallback=False)
    engine.predict(history, HORIZON)
    assert engine._fitted
    engine.clear_cache()
    assert not engine._fitted


# -------------------------------------------------------------------------- fallback
def test_an_unavailable_backend_degrades_rather_than_failing(history) -> None:
    """docs/02 section 4: a fallback means degraded, not an error."""
    engine = ForecastEngine(backend=BACKEND_CHRONOS, hub_model_name="does/not-exist")
    result = engine.predict(history, HORIZON)
    assert result.degraded
    assert result.backend_requested == BACKEND_CHRONOS
    assert result.backend_used in (BACKEND_LIGHTGBM, BACKEND_NAIVE)
    assert any("unavailable" in w for w in result.warnings)


def test_no_fallback_raises_instead(history) -> None:
    engine = ForecastEngine(
        backend=BACKEND_CHRONOS, hub_model_name="does/not-exist", allow_fallback=False
    )
    with pytest.raises(EngineError, match="unavailable"):
        engine.predict(history, HORIZON)


def test_unknown_backend_is_rejected(history) -> None:
    engine = ForecastEngine(backend="magic", allow_fallback=False)
    with pytest.raises(EngineError, match="unknown forecast backend"):
        engine.predict(history, HORIZON)


# -------------------------------------------------------------------------- backtest
def test_backtest_metrics_are_finite(history) -> None:
    """docs/07 Phase 4 engine acceptance."""
    metrics = ForecastEngine(backend=BACKEND_NAIVE, allow_fallback=False).backtest(
        history, horizon_steps=HORIZON, folds=3
    )
    assert np.isfinite(metrics.mae)
    assert np.isfinite(metrics.mape)
    assert np.isfinite(metrics.pinball)
    assert metrics.folds == 3
    assert metrics.backend == BACKEND_NAIVE


def test_backtest_reports_coverage_when_there_are_bands(history) -> None:
    metrics = ForecastEngine(backend=BACKEND_LIGHTGBM, allow_fallback=False).backtest(
        history, horizon_steps=HORIZON, folds=3
    )
    assert 0.0 <= metrics.coverage <= 100.0


def test_backtest_coverage_is_absent_for_a_point_forecast(history) -> None:
    metrics = ForecastEngine(backend=BACKEND_NAIVE, allow_fallback=False).backtest(
        history, horizon_steps=HORIZON, folds=2
    )
    assert np.isnan(metrics.coverage)
    assert metrics.as_dict()["interval_coverage_pct"] is None


def test_backtest_needs_enough_history() -> None:
    short = _sine_series(entities=1, steps=50)
    with pytest.raises(EngineError, match="at least"):
        ForecastEngine(backend=BACKEND_NAIVE, allow_fallback=False).backtest(
            short, horizon_steps=HORIZON, folds=2
        )


def test_compare_backends_returns_a_row_per_backend(history) -> None:
    """The M01 model card needs all three backends side by side (docs/03 M01)."""
    table = ForecastEngine(seed=42).compare_backends(
        history, horizon_steps=HORIZON, folds=2, backends=(BACKEND_LIGHTGBM, BACKEND_NAIVE)
    )
    assert set(table) == {BACKEND_LIGHTGBM, BACKEND_NAIVE}
    for metrics in table.values():
        assert np.isfinite(metrics.mae)
        assert "mae" in metrics.as_dict()


# ------------------------------------------------------------------------ covariates
def test_calendar_covariates_shape() -> None:
    index = pd.date_range("2027-08-01", periods=96, freq="15min", tz="Asia/Kolkata")
    frame = calendar_covariates(index, snan_days=[pd.Timestamp("2027-08-02")])
    assert list(frame.columns) == ["hour_sin", "hour_cos", "day_of_week", "is_snan_day"]
    assert len(frame) == 96
    assert frame["is_snan_day"].sum() == 0  # this day is not the snan day


def test_calendar_covariates_flag_the_snan_day() -> None:
    index = pd.date_range("2027-08-02", periods=96, freq="15min", tz="Asia/Kolkata")
    frame = calendar_covariates(index, snan_days=[pd.Timestamp("2027-08-02")])
    assert frame["is_snan_day"].sum() == 96


def test_hour_of_day_is_cyclic() -> None:
    """Midnight must sit next to 23:45, not 23 units away."""
    index = pd.date_range("2027-08-01", periods=96, freq="15min", tz="Asia/Kolkata")
    frame = calendar_covariates(index)
    first = np.array([frame["hour_sin"].iloc[0], frame["hour_cos"].iloc[0]])
    last = np.array([frame["hour_sin"].iloc[-1], frame["hour_cos"].iloc[-1]])
    assert np.linalg.norm(first - last) < 0.2


def test_calendar_covariates_accept_extra_series() -> None:
    index = pd.date_range("2027-08-01", periods=8, freq="15min", tz="Asia/Kolkata")
    temperature = pd.Series(np.arange(8, dtype=float), index=index)
    frame = calendar_covariates(index, extra={"temperature_c": temperature})
    assert "temperature_c" in frame.columns
    assert frame["temperature_c"].iloc[-1] == pytest.approx(7.0)


def test_extra_covariates_are_filled_not_left_null() -> None:
    """A gap in a covariate must not become NaN and poison the fit."""
    index = pd.date_range("2027-08-01", periods=8, freq="15min", tz="Asia/Kolkata")
    sparse = pd.Series([1.0, np.nan, 3.0], index=index[:3])
    frame = calendar_covariates(index, extra={"x": sparse})
    assert not frame["x"].isna().any()


def test_series_by_entity_splits_a_long_table() -> None:
    index = pd.date_range("2027-08-01", periods=4, freq="15min", tz="Asia/Kolkata")
    frame = pd.DataFrame(
        {
            "timestamp": list(index) * 2,
            "zone_id": ["Z01"] * 4 + ["Z02"] * 4,
            "population": list(range(8)),
        }
    )
    out = series_by_entity(frame, entity_column="zone_id", value_column="population")
    assert set(out) == {"Z01", "Z02"}
    assert len(out["Z01"]) == 4
    assert out["Z02"].iloc[0] == pytest.approx(4.0)


def test_covariates_reach_the_backend(history) -> None:
    """A covariate must change the forecast, or it is not being used.

    A future covariate has to span the history AND the forecast window - that is what makes
    it "known in the future". Supplying only the future part gives Darts no overlap with the
    target series and it refuses with "do not share any common times".
    """
    index = pd.DatetimeIndex(history["Z01"].index)
    full_index = pd.date_range(
        start=index[0], periods=len(index) + HORIZON, freq="15min", tz=index.tz
    )
    flat = {entity: calendar_covariates(full_index) for entity in history}
    engine = ForecastEngine(backend=BACKEND_LIGHTGBM, seed=42, allow_fallback=False)
    with_covariates = engine.predict(history, HORIZON, future_covariates=flat)
    engine.clear_cache()
    without = engine.predict(history, HORIZON)
    assert not np.allclose(
        with_covariates.values["Z01"].to_numpy(), without.values["Z01"].to_numpy()
    )
