# M17 — Food & Supply

**How much food is needed, how long does stock last, how many deliveries?**

Forecasts meals per outlet, then **simulates inventory forward** with the supplier lead time
so the model can say when an outlet runs out - not just what it holds now.

> Two things are unusual here. D17 is **hourly**, not on the 15-minute grid the other
> forecasting models use. And the default horizon is **12 hours, deliberately longer than the
> 6-hour lead time**: inside a window shorter than the lead time, nothing an outlet orders can
> arrive, so replenishment is invisible and S13 has nothing to show.

> All data is **synthetic**. Every output carries `is_synthetic: true` and every number in
> `config.yaml` is a placeholder pending expert review.

## Run

```bash
# tests
uv run pytest M17_food_supply/tests -q

# the service
uv run uvicorn api:app --app-dir M17_food_supply --port 8017

# one-off from the CLI
uv run python -m src.run --as-of "2027-08-02T06:00:00+05:30" --out outputs/
```

## API

```bash
curl localhost:8017/health
curl localhost:8017/metadata

curl -X POST localhost:8017/predict -H 'content-type: application/json' -d '{}'

curl -X POST localhost:8017/predict -H 'content-type: application/json' -d '{
  "as_of": "2027-08-02T06:00:00+05:30",
  "horizon_min": 720,
  "entity_ids": ["FO01", "FO02"]
}'

curl -X POST localhost:8017/scenario -H 'content-type: application/json' -d '{
  "scenario_id": "S13",
  "compare_to_baseline": true
}'
```

Every response validates against `00_common/schemas/model_output.schema.json`.

## Inputs and outputs

| Kind | Value |
|---|---|
| Tables | `food_inventory_hourly` (D17, hourly), `footfall_15min` (D01), `event_calendar` (D21) |
| Upstream models | `M01` arrivals covariate |

| KPI | Unit | Meaning |
|---|---|---|
| `food_demand` | meals/hr | servings expected |
| `food_stock_coverage` | hours | stock divided by consumption rate |
| `delivery_requirement` | trips/day | demand divided by payload |

## Configuration

Everything numeric lives in `config.yaml`. Shared assumptions come from
`00_common/config/assumptions.yaml`; risk bands from `risk_bands.yaml`.

| Key | What it controls |
|---|---|
| `params.forecast.*` | backend, quantiles, context length, fallback, band calibration |
| `params.max_coverage_hours` | reporting cap; an idle outlet has unbounded cover |
| `params.hours_per_day` | scales the hourly rate to the trips/day KPI unit |
| `assumptions.food.lead_time_hours` | how long a delivery takes to arrive |
| `assumptions.food.min_stock_cover_hours` | the buffer below which a stockout is raised |
| `assumptions.food.vehicle_payload_meals` | meals per delivery vehicle |
| `seed` | all randomness; the same seed reproduces the output exactly |
| `data_source` | `synthetic` for the demo; `replay` reads recorded real data |

## Scenarios

Supported: `S01`, `S02`, `S13`. Anything else returns baseline values with
`status: degraded` and an `insensitive to Sxx` warning.

| Scenario | Expected effect |
|---|---|
| `S02` 30 % more crowd | demand up 1.2-1.4x, more delivery trips |
| `S13` supply disruption | lead time x1.5, so coverage falls to ~0.88x and stockout flags double |
| `S06` road closure | **insensitive** - the card makes this optional and dependent on M04 (Phase 7) |

## Swapping synthetic data for real data

1. Put recorded data with the **same columns** (docs/04 section 4) under `data/replay/`.
2. Set `data_source: replay` in `config.yaml`.
3. Recalibrate the values in `params` against the real data, and re-run the tests.

No model code changes: `twin_common.io.load_table` hides which source is active.
