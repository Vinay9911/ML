# M19 — Power Load

**What electrical load is expected, and how long can backup last?**

Forecasts kW per substation (reported in **MW**), sizes the backup generators the critical
load needs, and counts down how long the fuel lasts. `details.load_utilization` is published
for M12 (fire risk), which reads electrical utilisation as a driver.

> **S08 is the scenario this model exists for.** A downed substation puts its zones on backup,
> raises `GRID_OUTAGE`, and starts a finite duration countdown.

> All data is **synthetic**. Every output carries `is_synthetic: true` and every number in
> `config.yaml` is a placeholder pending expert review.

## Run

```bash
# tests
uv run pytest M19_power_load/tests -q

# the service
uv run uvicorn api:app --app-dir M19_power_load --port 8019

# one-off from the CLI
uv run python -m src.run --as-of "2027-08-02T06:00:00+05:30" --out outputs/
```

## API

```bash
curl localhost:8019/health
curl localhost:8019/metadata

curl -X POST localhost:8019/predict -H 'content-type: application/json' -d '{}'

curl -X POST localhost:8019/predict -H 'content-type: application/json' -d '{
  "as_of": "2027-08-02T06:00:00+05:30",
  "horizon_min": 180,
  "entity_ids": ["SS01", "SS02"]
}'

curl -X POST localhost:8019/scenario -H 'content-type: application/json' -d '{
  "scenario_id": "S08",
  "compare_to_baseline": true
}'
```

Every response validates against `00_common/schemas/model_output.schema.json`.

## Inputs and outputs

| Kind | Value |
|---|---|
| Tables | `weather_hourly`, `power_15min`, `assets` |
| Upstream models | `M01` arrivals covariate - `M21` temperature for the cooling term |

| KPI | Unit | Meaning |
|---|---|---|
| `electricity_demand` | MW | forecast event load |
| `generator_backup_duration` | hours | fuel divided by consumption rate |
| `backup_generator_requirement` | units | ceil(critical load times redundancy divided by unit kW) |

## Configuration

Everything numeric lives in `config.yaml`. Shared assumptions come from
`00_common/config/assumptions.yaml`; risk bands from `risk_bands.yaml`.

| Key | What it controls |
|---|---|
| `params.forecast.*` | backend, quantiles, context length, fallback, band calibration |
| `params.kw_per_mw` | D18 is kW; the KPI unit is MW |
| `params.critical_load_fraction` | share that must stay up on backup |
| `params.assumed_load_fraction` | what a generator carries, which sets its burn rate |
| `params.max_backup_hours` | reporting cap; past a day refuelling is the real constraint |
| `assumptions.power.*` | generator rating, tank size, burn rate, redundancy, cooling per degree |
| `seed` | all randomness; the same seed reproduces the output exactly |
| `data_source` | `synthetic` for the demo; `replay` reads recorded real data |

## Scenarios

Supported: `S01`, `S02`, `S04`, `S08`. Anything else returns baseline values with
`status: degraded` and an `insensitive to Sxx` warning.

| Scenario | Expected effect |
|---|---|
| `S02` 30 % more crowd | load up |
| `S04` extreme heat | load up through the cooling term (~1.04x) |
| `S08` power failure | SS01 on backup, `GRID_OUTAGE` raised, fuel countdown starts |

## Swapping synthetic data for real data

1. Put recorded data with the **same columns** (docs/04 section 4) under `data/replay/`.
2. Set `data_source: replay` in `config.yaml`.
3. Recalibrate the values in `params` against the real data, and re-run the tests.

No model code changes: `twin_common.io.load_table` hides which source is active.
