# Phase 4 — Forecast engine + 9 forecasting models

Status: engine + M01 done; M05, M06, M15, M17, M18, M19, M20, M22 remaining.

## Goal
`twin_common.engines.forecast` plus **M01** as the reference model, then M05, M06, M15, M17,
M18, M19, M20, M22.

## 1. Engine design (`engines/forecast.py`)

### Backends
| Backend | Class | Role |
|---|---|---|
| `chronos2` | Darts `Chronos2Model` | default. Zero-shot foundation model, past + future covariates |
| `lightgbm` | Darts `LightGBMModel` | quantile-regression baseline, lags 1-8 and 96 |
| `naive_seasonal` | Darts `NaiveSeasonal(K=96)` | the "did we beat yesterday?" floor |

docs/06 section 3 is explicit that **`Chronos2Model` supports covariates and
`TimesFM2p5Model` does not**, so Chronos-2 is the default and TimesFM is not used.

### CPU fallback
`hub_model_name` defaults to `autogluon/chronos-2-small` (28M params) rather than the 120M
`amazon/chronos-2`, because everything must work on CPU (docs/06 section 2). The larger model
is opt-in through config. If the weights cannot be fetched and no cache exists, the engine
falls back to `lightgbm`, then to `naive_seasonal`, and reports which it used so the model
can set `status: degraded`.

### Interface
```python
engine = ForecastEngine(config)  # backend, horizon, quantiles, covariates from config
engine.fit(history)  # no-op for a zero-shot foundation model
result = engine.predict(history, horizon_steps, past_covariates, future_covariates)
#   -> ForecastResult(values, lower, upper, quantile_level, backend_used, degraded, warnings)
metrics = engine.backtest(history, folds)  # MAE, MAPE, pinball, 80% interval coverage
```

- **Multi-series**: one series per entity (Z01-Z08, G01-G04, P1-P3, ...). Darts takes a list
  of TimeSeries, so all entities are forecast in one call rather than a loop.
- **Quantiles**: 0.1 / 0.5 / 0.9 map to `lower` / `value` / `upper` with
  `quantile_level: 0.8`, which is exactly the docs/02 section 4 band convention.
- **Covariates**: past (lagged entries, population) and future (hour sin/cos, day of week,
  `is_snan_day`, snan multiplier, temperature, rain). The future covariates are known for the
  whole window because the world is generated, which is what makes a zero-shot forecast
  meaningful here.

### Caching
Weights under `00_common/data/real_cache/hf` via `HF_HOME` (docs/06 section 2), so the second
run and every offline run need no network. `TWIN_OFFLINE=1` never reaches for weights.

## 2. Determinism
A foundation model on CPU is deterministic given fixed weights and input, but LightGBM needs
`random_state` and torch needs `manual_seed`; both come from `config.seed`. The engine sets
them on every call, not once at import, so a second call in the same process reproduces the
first.

## 3. Model-by-model plan
| Model | Forecasts | Covariates | Derived KPIs |
|---|---|---|---|
| M01 | entries per zone and gate | hour, dow, snan flag, temp, rain | `expected_footfall`, `peak_footfall` |
| M05 | occupied spaces per site | M01 arrivals, hour, snan | `parking_occupancy`, `parking_search_time`, `parking_demand` |
| M06 | boardings per route | M01 arrivals, M05 overflow | `shuttle_demand`, `shuttle_requirement`, `passenger_wait_time` |
| M15 | consumption per zone | population, temperature, hour | `expected_water_demand`, `water_supply_demand_gap`, points, tankers |
| M17 | meals per outlet | market-zone population, hour | `food_demand`, `food_stock_coverage`, `delivery_requirement` |
| M18 | kg per zone | population, food demand | `waste_generation`, `waste_bin_fill_level`, trips, `bin_overflow_time` |
| M19 | kW per zone | temperature, population, hour | `electricity_demand`, `generator_backup_duration`, generators |
| M20 | utilisation per tower | population, active cameras | `bandwidth_utilization`, `network_availability`, `network_capacity_risk` |
| M22 | pm2.5 | traffic volume, wind, hour | `air_quality_index`, `pm25`, `noise_level`, `co2_emissions` |

## 4. Acceptance (docs/07 Phase 4)
**Engine**
- unit tests on a synthetic sine + noise series; backtest metrics finite
- offline mode uses cached weights or raises a clear message
- default M01 `/predict` under 10 s on CPU with the small model

**Each model**
- the full Definition of Done (docs/02 section 9)
- the scenario directions of docs/05 section 1.2
- M01 records all three backends' backtest metrics in its model card

## 5. Risks
1. **torch install size.** `darts` pulls torch, several GB. If the install fails the engine
   must still work through `lightgbm` and `naive_seasonal`, and that path is tested.
2. **Chronos-2 CPU latency.** docs/06 section 3 flags it. Mitigations, in order: the small
   hub model, fewer entities per call, a shorter horizon, and the precomputed-forecast cache.
   If the default request cannot be served in 10 s, the model card says so and the default
   backend changes to `lightgbm`.
3. **Zero-shot on synthetic data.** A foundation model has never seen this world. The
   backtest exists to show whether it actually beats `naive_seasonal`; if it does not, that
   is a finding to report rather than to hide.

## 6. Open questions
1. Whether `amazon/chronos-2` (120M) is worth the CPU cost over `chronos-2-small` for the
   demo. Decide from the M01 backtest table.
2. docs/06 forbids fine-tuning on synthetic data (`enable_finetuning`), so the comparison is
   zero-shot Chronos-2 against a *fitted* LightGBM. That is not a like-for-like contest and
   the model card must say so.


## 7. What the real Darts API turned out to need

The plan above was written from docs/06. Four things differed once the library was installed,
and each is now encoded in the engine:

1. **`Chronos2Model` requires `input_chunk_length` and `output_chunk_length`.** They are not
   optional, and the docstring example in docs/06 omits them.
2. **It requires `fit()` before `predict()`** even though it is zero-shot. With no
   `enable_finetuning` the fit only registers the series - 0.7 s once the weights are cached -
   but it is not free, so the engine caches the fitted model and exposes `warmup()`. Without
   the cache every request paid 8.9 s against a 10 s budget.
3. **Quantiles need an explicit `likelihood=QuantileRegression(quantiles=[...])` plus
   `predict_likelihood_parameters=True`.** `num_samples>1` raises
   "only supported for probabilistic models". All three backends then return components named
   `<c>_q0.100`, which is what the engine parses.
4. **Darts strips timezones** from a DatetimeIndex. The engine removes the tz before building
   a TimeSeries and restores it on the way out, because docs/02 section 4 requires tz-aware
   timestamps.

The dependency chain also needed `pytorch_lightning` and `huggingface_hub`; darts reports a
missing `huggingface_hub` as a generic "(Py)Torch module could not be imported", which is
misleading.

## 8. M01 results

Backtest, Z01, 12-step horizon, 7 folds, CPU:

| Backend | MAE | MAPE | Pinball | 80% coverage | Time |
|---|---:|---:|---:|---:|---:|
| chronos2 (zero-shot) | 194.7 | 24.0% | 97.33 | 61.9% | 1.8 s |
| lightgbm (fitted) | 205.5 | 27.9% | 102.76 | 39.3% | 14.3 s |
| naive (K=96) | 309.5 | 58.4% | 154.77 | - | 0.1 s |

Both learned backends beat the naive floor. Zero-shot Chronos-2 edging out a fitted LightGBM
is suspicious rather than impressive - the synthetic world is cleanly periodic, which suits a
foundation model's priors - and the comparison is not like-for-like because docs/06 forbids
fine-tuning. **Interval coverage is the weak result**: 61.9 percent against a nominal 80
means the bands are too narrow, and that is recorded as a limitation rather than smoothed over.

Default `/predict`: 156 records, 12 series, **1.2 s** warm.

## 9. Two problems M01 exposed that affect all 25 folders

- **`src.model` collides across folders.** Python caches the first `src` it imports, so a
  second model folder silently received the first one's `Model` class. Every folder's conftest
  now purges `src*` from `sys.modules` before importing. The template carries the fix.
- **The stub provider was never installed.** `twin_common.synthetic.stubs.install()` existed
  but nothing called it, so any model with an upstream failed instead of degrading.
  `upstream.resolve_one` now installs the default provider on first use.

## 10. Open question resolved
`chronos-2-small` is the right default: it beat the fitted baseline on this world and answers
in 1.8 s. There is no case for the 120M `amazon/chronos-2` here.
