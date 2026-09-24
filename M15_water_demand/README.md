# M15 — Water Demand

**How much water is needed by zone, is there a gap, and how many points and tankers?**

Forecasts consumption per zone, compares it with planned pump supply, and sizes drinking
water points and tanker runs. A **positive gap means a shortage**; a surplus is reported as a
negative number rather than clipped.

> All data is **synthetic**. Every output carries `is_synthetic: true` and every number in
> `config.yaml` is a placeholder pending expert review.

## Run

```bash
# tests
uv run pytest M15_water_demand/tests -q

# the service
uv run uvicorn api:app --app-dir M15_water_demand --port 8015

# one-off from the CLI
uv run python -m src.run --as-of "2027-08-02T06:00:00+05:30" --out outputs/
```

## API

```bash
curl localhost:8015/health
curl localhost:8015/metadata

curl -X POST localhost:8015/predict -H 'content-type: application/json' -d '{}'

curl -X POST localhost:8015/predict -H 'content-type: application/json' -d '{
  "as_of": "2027-08-02T06:00:00+05:30",
  "horizon_min": 180,
  "entity_ids": ["Z01", "Z05"]
}'

curl -X POST localhost:8015/scenario -H 'content-type: application/json' -d '{
  "scenario_id": "S02",
  "compare_to_baseline": true
}'
```

Every response validates against `00_common/schemas/model_output.schema.json`.

## Inputs and outputs

| Kind | Value |
|---|---|
| Tables | `weather_hourly`, `water_15min`, `assets` |
| Upstream models | `M01` arrivals covariate - `M21` temperature for the heat term |

| KPI | Unit | Meaning |
|---|---|---|
| `expected_water_demand` | L/hr | consumption per time |
| `water_supply_demand_gap` | L/hr | demand minus supply |
| `drinking_water_point_requirement` | points | population divided by point service capacity |
| `water_tanker_requirement` | vehicles | ceil(positive gap times hours divided by tanker payload) |

## Configuration

Everything numeric lives in `config.yaml`. Shared assumptions come from
`00_common/config/assumptions.yaml`; risk bands from `risk_bands.yaml`.

| Key | What it controls |
|---|---|
| `params.forecast.*` | backend, quantiles, context length, fallback, band calibration |
| `params.tanker_cover_hours` | hours of shortfall one tanker dispatch covers |
| `params.carry_supply_forward` | supply is a planned quantity, not a forecast |
| `assumptions.water.*` | litres per person, heat multiplier, persons per point, tanker payload |
| `assumptions.water.pump_outage_supply_factor` | output while a substation is down |
| `seed` | all randomness; the same seed reproduces the output exactly |
| `data_source` | `synthetic` for the demo; `replay` reads recorded real data |

## Scenarios

Supported: `S01`, `S02`, `S04`, `S08`. Anything else returns baseline values with
`status: degraded` and an `insensitive to Sxx` warning.

| Scenario | Expected effect |
|---|---|
| `S02` 30 % more crowd | demand up 1.2-1.4x across the whole horizon |
| `S04` extreme heat | demand up - but only ~1 hour ahead, and barely at all at the 06:00 demo instant (see the model card) |
| `S08` power failure | a pump loses output, so the gap widens and tankers are called |

> The S04 effect is 0.12 % at 06:00 but 8.5 % at 13:00: the demo instant is a cool morning.
> It also decays with horizon. Both limits are measured and documented in `model_card.md`.

## Swapping synthetic data for real data

1. Put recorded data with the **same columns** (docs/04 section 4) under `data/replay/`.
2. Set `data_source: replay` in `config.yaml`.
3. Recalibrate the values in `params` against the real data, and re-run the tests.

No model code changes: `twin_common.io.load_table` hides which source is active.
