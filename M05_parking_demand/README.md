# M05 — Parking Demand

**How full will each parking site be, and when does it overflow?**

M05 forecasts the *demand* for each lot - the uncapped number of vehicles wanting a bay -
and derives occupancy from it, so **occupancy can exceed 100 %**. That is the overflow
signal: D06 stores `occupied` clipped at capacity, so a forecast of that column could never
warn about the one thing this model exists for. `details.occupied_spaces` has the capped,
physically-parked figure. See `model_card.md`.

> All data is **synthetic**. Every output carries `is_synthetic: true` and every number in
> `config.yaml` is a placeholder pending expert review.

## Run

```bash
# tests
uv run pytest M05_parking_demand/tests -q

# the service
uv run uvicorn api:app --app-dir M05_parking_demand --port 8005

# one-off from the CLI
uv run python -m src.run --as-of "2027-08-02T06:00:00+05:30" --out outputs/
```

## API

```bash
curl localhost:8005/health
curl localhost:8005/metadata

curl -X POST localhost:8005/predict -H 'content-type: application/json' -d '{}'

curl -X POST localhost:8005/predict -H 'content-type: application/json' -d '{
  "as_of": "2027-08-02T06:00:00+05:30",
  "horizon_min": 180,
  "entity_ids": ["P1", "P2"]
}'

curl -X POST localhost:8005/scenario -H 'content-type: application/json' -d '{
  "scenario_id": "S02",
  "compare_to_baseline": true
}'
```

Every response validates against `00_common/schemas/model_output.schema.json`.

## Inputs and outputs

| Kind | Value |
|---|---|
| Tables | `parking_15min` (demand history), `footfall_15min` (arrivals covariate), `event_calendar` (snan flag) |
| Upstream models | `M01` - forecast arrivals for the future half of the covariate |

| KPI | Unit | Meaning |
|---|---|---|
| `parking_occupancy` | % | demand divided by capacity times 100; **may exceed 100** |
| `parking_search_time` | min | arrival to parked |
| `parking_demand` | vehicles | forecast vehicles seeking parking |

## Configuration

Everything numeric lives in `config.yaml`. Shared assumptions come from
`00_common/config/assumptions.yaml`; risk bands from `risk_bands.yaml`.

| Key | What it controls |
|---|---|
| `params.forecast.*` | backend, hub model, quantiles, context length, fallback |
| `params.search_time.*` | the base / k / knee of the search-time curve |
| `params.overflow_pct` | occupancy at which `PARKING_OVERFLOW` is raised |
| `params.max_occupancy_pct` | reporting ceiling; catches a runaway forecast |
| `params.confidence` | the confidence reported on each record |
| `assumptions.transport.parking_capacity` | bays per site - **raised from the docs/04 placeholder**, see the model card |
| `seed` | all randomness; the same seed reproduces the output exactly |
| `data_source` | `synthetic` for the demo; `replay` reads recorded real data |

## Scenarios

Supported: `S01`, `S02`. Anything else returns baseline values with
`status: degraded` and an `insensitive to Sxx` warning.

| Scenario | Expected effect |
|---|---|
| `S02` 30 % more crowd | demand, occupancy and search time all up; horizon total 1.2-1.4x |
| `S06` road closure | **insensitive** - R07 serves no lot (the access links are R04/R05/R06) |
| `S12` bridge closure | **insensitive** - B01 serves no lot |

Closing a parking access link *does* move that site's demand to the others in proportion to
capacity. No scripted scenario does so, but the mechanism is implemented and tested.

## Swapping synthetic data for real data

1. Put recorded data with the **same columns** (docs/04 section 4) under `data/replay/`.
2. Set `data_source: replay` in `config.yaml`.
3. Recalibrate the values in `params` against the real data, and re-run the tests.

No model code changes: `twin_common.io.load_table` hides which source is active.
