# M18 — Waste Generation

**How much waste, when do bins overflow, how many collection trips?**

Forecasts kilograms per bin group, projects each group's fill level until its next
collection, and says when it will overflow.

> `waste_bin_fill_level` is a percentage of the **group** capacity. `bin_capacity_kg` in
> assumptions is ONE bin; `bins_per_group` turns it into a zone's worth. Before that was
> added, every group read 100 % full within one step and both fill KPIs were useless.

> All data is **synthetic**. Every output carries `is_synthetic: true` and every number in
> `config.yaml` is a placeholder pending expert review.

## Run

```bash
# tests
uv run pytest M18_waste_forecast/tests -q

# the service
uv run uvicorn api:app --app-dir M18_waste_forecast --port 8018

# one-off from the CLI
uv run python -m src.run --as-of "2027-08-02T06:00:00+05:30" --out outputs/
```

## API

```bash
curl localhost:8018/health
curl localhost:8018/metadata

curl -X POST localhost:8018/predict -H 'content-type: application/json' -d '{}'

curl -X POST localhost:8018/predict -H 'content-type: application/json' -d '{
  "as_of": "2027-08-02T06:00:00+05:30",
  "horizon_min": 180,
  "entity_ids": ["WB01", "WB05"]
}'

curl -X POST localhost:8018/scenario -H 'content-type: application/json' -d '{
  "scenario_id": "S02",
  "compare_to_baseline": true
}'
```

Every response validates against `00_common/schemas/model_output.schema.json`.

## Inputs and outputs

| Kind | Value |
|---|---|
| Tables | `waste` (D16), `footfall_15min` (D01), `event_calendar` (D21) |
| Upstream models | `M01` arrivals covariate |

| KPI | Unit | Meaning |
|---|---|---|
| `waste_generation` | kg/day | solid waste produced |
| `waste_bin_fill_level` | % | fill divided by capacity times 100 |
| `waste_collection_trips` | trips/day | ceil(kg per day divided by vehicle payload) |
| `bin_overflow_time` | min | minutes until fill reaches 100 percent |

## Configuration

Everything numeric lives in `config.yaml`. Shared assumptions come from
`00_common/config/assumptions.yaml`; risk bands from `risk_bands.yaml`.

| Key | What it controls |
|---|---|
| `params.forecast.*` | backend, quantiles, context length, fallback, band calibration |
| `params.collection_interval_min` | how often a group is emptied on the normal round |
| `params.collection_delay_multiplier` | stretches the round; what S06/S12 would drive |
| `params.no_overflow_minutes` | sentinel when a bin is not projected to overflow |
| `assumptions.waste.bin_capacity_kg` | capacity of ONE bin |
| `assumptions.waste.bins_per_group` | bins in a zone group - **added by this build** |
| `assumptions.waste.vehicle_payload_kg` | kg per collection vehicle |
| `seed` | all randomness; the same seed reproduces the output exactly |
| `data_source` | `synthetic` for the demo; `replay` reads recorded real data |

## Scenarios

Supported: `S01`, `S02`. Anything else returns baseline values with
`status: degraded` and an `insensitive to Sxx` warning.

| Scenario | Expected effect |
|---|---|
| `S02` 30 % more crowd | generation up 1.2-1.4x, bins fill faster, more trips |
| `S06` / `S12` closures | **insensitive** - the card wants collection delayed, but only M04 can say by how much (Phase 7) |

Delaying the round *does* leave bins fuller. Pass `waste_collection_delay_multiplier` in the
scenario overrides; the mechanism is implemented and tested.

## Swapping synthetic data for real data

1. Put recorded data with the **same columns** (docs/04 section 4) under `data/replay/`.
2. Set `data_source: replay` in `config.yaml`.
3. Recalibrate the values in `params` against the real data, and re-run the tests.

No model code changes: `twin_common.io.load_table` hides which source is active.
