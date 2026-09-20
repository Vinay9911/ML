# M01 — Footfall Forecast

How many people will arrive, by zone, gate and time?

M01 is the reference model for the shared forecast engine and the busiest node in the
dependency graph — **13 other models read it**. It forecasts 12 series at once (8 zones and
4 gates) on a 15-minute grid.

> All data is **synthetic**. Every output carries `is_synthetic: true` and every number in
> `config.yaml` is a placeholder. The backtest in [model_card.md](model_card.md) measures
> pipeline correctness, not real-world accuracy.

## Run

```bash
# tests
uv run pytest M01_footfall_forecast/tests -q

# the service
uv run uvicorn api:app --app-dir M01_footfall_forecast --port 8001

# one-off from the CLI
cd M01_footfall_forecast
uv run python -m src.run --out outputs/
uv run python -m src.run --scenario S02 --out outputs/
```

First run downloads the Chronos-2 weights (about 25 s) into
`00_common/data/real_cache/hf`. After that it is offline.

## API

```bash
curl localhost:8001/health
curl localhost:8001/metadata

# defaults: as_of = demo_now, horizon 180 min, all 12 series, both KPIs
curl -X POST localhost:8001/predict -H 'content-type: application/json' -d '{}'

curl -X POST localhost:8001/predict -H 'content-type: application/json' -d '{
  "as_of": "2027-08-02T06:00:00+05:30",
  "horizon_min": 60,
  "entity_ids": ["Z01", "G02"],
  "kpis": ["peak_footfall"]
}'

curl -X POST localhost:8001/scenario -H 'content-type: application/json' -d '{
  "scenario_id": "S02",
  "compare_to_baseline": true
}'
```

A default `/predict` returns 156 records in about **1.2 s** once warm.

## Inputs and outputs

| Kind | Value |
|---|---|
| Tables | `footfall_15min` (D01), `gate_entries`, `event_calendar` (D21), `weather_hourly` (D12), `zones` (D03) |
| Upstream | `M21` — rain and heat multipliers, echoed in `details.weather` |

| KPI | Unit | Entity | Meaning |
|---|---|---|---|
| `expected_footfall` | persons/hr | zone, gate | forecast arrivals per step ×4, with the 80% band |
| `peak_footfall` | persons/hr | zone, gate | the maximum of the median forecast in the horizon; `details.peak_at` gives when |

## The forecast engine

Three backends behind one interface, tried in order:

| Backend | What it is | When it runs |
|---|---|---|
| `chronos2` | Darts `Chronos2Model`, zero-shot foundation model | default |
| `lightgbm` | Darts `LightGBMModel`, quantile regression, fitted | when Chronos-2 is unavailable |
| `naive` | `NaiveSeasonal(K=96)` — repeat yesterday | last resort |

A fallback is reported: the response carries `status: degraded` and a warning naming the
backend that actually ran, and every record says which in `details.backend`.

**Why Chronos-2** — docs/06 §3: it is the only foundation backend in Darts that accepts
covariates. `TimesFM2p5Model` does not, and the covariates (snan flag, temperature, rain) are
the point.

**Warm start** — the engine fits once at load. Skipping that made every request pay a 8.9 s
fit against a 10 s budget.

## Configuration

Everything numeric lives in `config.yaml`.

| Key | Controls |
|---|---|
| `params.forecast.backend` | which backend to try first |
| `params.forecast.hub_model_name` | `chronos-2-small` (CPU) or `amazon/chronos-2` (120M) |
| `params.forecast.quantiles` | the band; `[0.1, 0.5, 0.9]` gives `quantile_level: 0.8` |
| `params.forecast.max_context_steps` | how much history the model reads (four weeks) |
| `params.backtest.*` | horizon, folds and stride for `Model.backtest()` |
| `params.steps_per_hour` | 15-min counts → persons/hr |

## Scenarios

Supported: `S01`, `S02`, `S03`, `S05`.

| Scenario | Effect | Asserted in tests |
|---|---|---|
| S02 30% more crowd | total footfall rises | ratio in **[1.2, 1.4]** (docs/03 M01) |
| S03 heavy rain | arrivals fall in the rain window | total down vs S01 |
| S05 gate closure | **G02 forecast is exactly 0**; other gates stay open | G02 = 0, others > 0 |

**S04 is deliberately not supported**, even though the docs/03 M01 card mentions it: the
docs/03 M21 arrival multiplier depends on rain only, so heat changes nothing M01 reads.
M01 returns the documented `insensitive to S04` degraded response rather than an unchanged
baseline that looks like a real answer. See [model_card.md](model_card.md).

## Swapping synthetic data for real data

1. Put gate-counter or CCTV aggregates in `data/replay/footfall_15min.parquet` with the
   **same columns** as D01 (docs/04 §4).
2. Set `data_source: replay` in `config.yaml`.
3. Re-run `Model.backtest()` — the table in the model card must be replaced before anyone
   relies on it.
4. Re-check interval coverage. On synthetic data the bands are **too narrow** (61.9% against
   a nominal 80%), so the quantiles likely need widening or calibrating.
5. Optionally fine-tune once several weeks of real history exist (docs/06 §3 forbids it on
   synthetic data).

No model code changes: `twin_common.io.load_table` hides which source is active.
