# M06 — Transit / Shuttle Demand

**How many passengers and shuttles are needed per route?**

M06 forecasts boardings per shuttle route, converts them to a vehicle requirement, and
reports how long riders wait. It is the first model with **two upstreams** — M01 (arrivals)
and M05 (parking overflow).

> **Finding:** the fleet in `resources.yaml` (80 vehicles) is 4.6x too small on an ordinary
> day and 18x at the snan peak. That is reported, not silently fixed — `available` is M24's
> input. See `model_card.md`.

> All data is **synthetic**. Every output carries `is_synthetic: true` and every number in
> `config.yaml` is a placeholder pending expert review.

## Run

```bash
# tests
uv run pytest M06_transit_demand/tests -q

# the service
uv run uvicorn api:app --app-dir M06_transit_demand --port 8006

# one-off from the CLI
uv run python -m src.run --as-of "2027-08-02T06:00:00+05:30" --out outputs/
```

## API

```bash
curl localhost:8006/health
curl localhost:8006/metadata

curl -X POST localhost:8006/predict -H 'content-type: application/json' -d '{}'

curl -X POST localhost:8006/predict -H 'content-type: application/json' -d '{
  "as_of": "2027-08-02T06:00:00+05:30",
  "horizon_min": 180,
  "entity_ids": ["SH1", "SH2"]
}'

curl -X POST localhost:8006/scenario -H 'content-type: application/json' -d '{
  "scenario_id": "S02",
  "compare_to_baseline": true
}'
```

Every response validates against `00_common/schemas/model_output.schema.json`.

## Inputs and outputs

| Kind | Value |
|---|---|
| Tables | `transit_15min` (boardings), `footfall_15min` (covariate), `event_calendar` |
| Upstream models | `M01` arrivals covariate · `M05` parking overflow (context only) |

| KPI | Unit | Meaning |
|---|---|---|
| `shuttle_demand` | persons/hr | passengers per hour |
| `shuttle_requirement` | vehicles | ceil(demand times round_trip over 60, divided by capacity times load factor) |
| `passenger_wait_time` | min | headway divided by 2 |

## Configuration

Everything numeric lives in `config.yaml`. Shared assumptions come from
`00_common/config/assumptions.yaml`; risk bands from `risk_bands.yaml`.

| Key | What it controls |
|---|---|
| `params.forecast.*` | backend, quantiles, context length, fallback, band calibration |
| `params.apply_overload_to_wait` | stretch the wait when the fleet cannot carry demand |
| `params.max_overload_factor` | ceiling on that stretch |
| `params.min_buses_when_demand` | a route with any demand needs at least one vehicle |
| `assumptions.transport.*` | round trip, bus capacity, load factor |
| `resources.shuttle_bus.available` | fleet size (owned by M24) |
| `seed` | all randomness; the same seed reproduces the output exactly |
| `data_source` | `synthetic` for the demo; `replay` reads recorded real data |

## Scenarios

Supported: `S01`, `S02`. Anything else returns baseline values with
`status: degraded` and an `insensitive to Sxx` warning.

| Scenario | Expected effect |
|---|---|
| `S02` 30 % more crowd | demand, vehicles and wait all up; horizon total 1.2-1.4x |
| `S06` road closure | **insensitive** — no shuttle route closes and M04 does not exist yet |
| `S12` bridge closure | **insensitive** — same reason |

A longer round trip *does* raise the vehicle requirement and the wait. Pass `round_trip_min`
in the scenario overrides, or add M04 upstream once it exists.

## Swapping synthetic data for real data

1. Put recorded data with the **same columns** (docs/04 section 4) under `data/replay/`.
2. Set `data_source: replay` in `config.yaml`.
3. Recalibrate the values in `params` against the real data, and re-run the tests.

No model code changes: `twin_common.io.load_table` hides which source is active.
