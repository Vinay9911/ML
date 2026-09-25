# M20 — Network Capacity

**Will communications (CCTV backhaul, command links) overload or fail?**

Forecasts bandwidth per tower, converts it to utilisation, and turns that into a capacity
risk. Availability is reported separately, because a tower can be perfectly uncongested and
still be off.

> **Finding: this world does not run out of bandwidth.** Peak utilisation at the busiest
> tower is 27.5 %, so `network_capacity_risk` sits near zero. The binding constraint here is
> power-driven *availability* (S08), not congestion. The risk threshold was left where an
> operator would want it rather than lowered to make the number move.

> D19 is on a **5-minute** grid, unlike the 15-minute grid the rest of the project uses.

> All data is **synthetic**. Every output carries `is_synthetic: true` and every number in
> `config.yaml` is a placeholder pending expert review.

## Run

```bash
# tests
uv run pytest M20_network_capacity/tests -q

# the service
uv run uvicorn api:app --app-dir M20_network_capacity --port 8020

# one-off from the CLI
uv run python -m src.run --as-of "2027-08-02T06:00:00+05:30" --out outputs/
```

## API

```bash
curl localhost:8020/health
curl localhost:8020/metadata

curl -X POST localhost:8020/predict -H 'content-type: application/json' -d '{}'

curl -X POST localhost:8020/predict -H 'content-type: application/json' -d '{
  "as_of": "2027-08-02T06:00:00+05:30",
  "horizon_min": 180,
  "entity_ids": ["NT01", "NT02"]
}'

curl -X POST localhost:8020/scenario -H 'content-type: application/json' -d '{
  "scenario_id": "S08",
  "compare_to_baseline": true
}'
```

Every response validates against `00_common/schemas/model_output.schema.json`.

## Inputs and outputs

| Kind | Value |
|---|---|
| Tables | `network_5min`, `assets`, `camera_registry` |
| Upstream models | `M01` arrivals covariate |

| KPI | Unit | Meaning |
|---|---|---|
| `network_availability` | % | available divided by expected times 100 |
| `bandwidth_utilization` | % | used divided by capacity times 100 |
| `network_capacity_risk` | % | 100 times sigmoid of k times (utilization minus utilization0) |

## Configuration

Everything numeric lives in `config.yaml`. Shared assumptions come from
`00_common/config/assumptions.yaml`; risk bands from `risk_bands.yaml`.

| Key | What it controls |
|---|---|
| `params.forecast.*` | backend, quantiles, context length, fallback, band calibration |
| `params.step_minutes` | D19 is 5-minute |
| `params.risk_midpoint_pct` | utilisation at which capacity risk is 50 % |
| `params.risk_steepness` | how sharply risk climbs around that midpoint |
| `assumptions.network.*` | tower capacities, camera bitrate, active-user share |
| `seed` | all randomness; the same seed reproduces the output exactly |
| `data_source` | `synthetic` for the demo; `replay` reads recorded real data |

## Scenarios

Supported: `S01`, `S02`, `S08`. Anything else returns baseline values with
`status: degraded` and an `insensitive to Sxx` warning.

| Scenario | Expected effect |
|---|---|
| `S02` 30 % more crowd | utilisation up ~1.27x |
| `S08` power failure | NT02 (on SS01) loses power: availability falls to 2/3, `GRID_OUTAGE` raised |
| `S09` CCTV failure | camera backhaul removed, utilisation falls, `CCTV_BLIND_SPOT` flagged |

## Swapping synthetic data for real data

1. Put recorded data with the **same columns** (docs/04 section 4) under `data/replay/`.
2. Set `data_source: replay` in `config.yaml`.
3. Recalibrate the values in `params` against the real data, and re-run the tests.

No model code changes: `twin_common.io.load_table` hides which source is active.
