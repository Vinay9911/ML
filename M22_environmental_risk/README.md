# M22 — Environmental Risk

**Where will air quality or noise stress increase, and what are the emissions?**

Forecasts venue PM2.5, converts it to an Indian National AQI, models noise from crowd density
per zone, and totals CO2 from electricity and generator diesel.

> **D13 is the one table in this project built from genuinely REAL data** - cached Open-Meteo
> air-quality history - so its rows carry `is_synthetic: false`.

> **The NAQI breakpoints are placeholders** transcribed from docs/05 and must be verified
> against the published CPCB table before any real use. The CO2 traffic term is zero because
> M04 (Phase 7) does not exist yet, and the response says so.

> All data is **synthetic**. Every output carries `is_synthetic: true` and every number in
> `config.yaml` is a placeholder pending expert review.

## Run

```bash
# tests
uv run pytest M22_environmental_risk/tests -q

# the service
uv run uvicorn api:app --app-dir M22_environmental_risk --port 8022

# one-off from the CLI
uv run python -m src.run --as-of "2027-08-02T06:00:00+05:30" --out outputs/
```

## API

```bash
curl localhost:8022/health
curl localhost:8022/metadata

curl -X POST localhost:8022/predict -H 'content-type: application/json' -d '{}'

curl -X POST localhost:8022/predict -H 'content-type: application/json' -d '{
  "as_of": "2027-08-02T06:00:00+05:30",
  "horizon_min": 360,
  "entity_ids": ["EVENT", "Z01"]
}'

curl -X POST localhost:8022/scenario -H 'content-type: application/json' -d '{
  "scenario_id": "S08",
  "compare_to_baseline": true
}'
```

Every response validates against `00_common/schemas/model_output.schema.json`.

## Inputs and outputs

| Kind | Value |
|---|---|
| Tables | `air_quality_hourly` (D13, REAL data), `footfall_15min` (D01), `weather_hourly` (D12), `power_15min` (D18), `event_calendar` (D21) |
| Upstream models | `M21` wind - `M04` vehicle-km (**not built yet**; the traffic term is zero) |

| KPI | Unit | Meaning |
|---|---|---|
| `air_quality_index` | index | composite AQI (Indian NAQI) |
| `pm25` | ug/m3 | fine particulate concentration |
| `noise_level` | dB | ambient noise |
| `co2_emissions` | tCO2e/day | from fuel, electricity and traffic factors |

## Configuration

Everything numeric lives in `config.yaml`. Shared assumptions come from
`00_common/config/assumptions.yaml`; risk bands from `risk_bands.yaml`.

| Key | What it controls |
|---|---|
| `params.naqi.*` | index and concentration breakpoints - **PLACEHOLDERS, verify with CPCB** |
| `params.noise.*` | base dB, reference density, reporting ceiling |
| `params.co2.*` | emission factors per vehicle-km, kWh and diesel litre - **all placeholders** |
| `params.forecast.*` | backend, quantiles, context length, fallback, band calibration |
| `seed` | all randomness; the same seed reproduces the output exactly |
| `data_source` | `synthetic` for the demo; `replay` reads recorded real data |

## Scenarios

Supported: `S01`, `S15`. Anything else returns baseline values with
`status: degraded` and an `insensitive to Sxx` warning.

| Scenario | Expected effect |
|---|---|
| `S02` 30 % more crowd | noise up (~1.01x - decibels are logarithmic), CO2 up ~1.07x |
| `S08` power failure | CO2 up: the generators burn diesel |
| `S15` fire | answered and flagged, but D13 is venue-level so 'PM2.5 near Z03' cannot be resolved - see the model card |

## Swapping synthetic data for real data

1. Put recorded data with the **same columns** (docs/04 section 4) under `data/replay/`.
2. Set `data_source: replay` in `config.yaml`.
3. Recalibrate the values in `params` against the real data, and re-run the tests.

No model code changes: `twin_common.io.load_table` hides which source is active.
