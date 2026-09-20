# M21 — Weather Impact

What is the weather now and over the next hours, and how does it change risk elsewhere?

M21 is the root of the dependency graph. It has no upstream, and **M01, M03, M07, M12, M15,
M19, M22 and M25 all read it**. Its job is to turn a weather series into the handful of
numbers the rest of the twin reasons about — a heat index, a per-zone flooding probability,
and three multipliers that couple weather to arrivals, medical demand and traffic speed.

> All data is **synthetic**. Every output carries `is_synthetic: true`. The observations
> underneath are real Open-Meteo history for a past year shifted onto the event dates, so
> they are plausible but they are not a forecast. Every coefficient in `config.yaml` is a
> placeholder pending expert review.

## Run

```bash
# tests
uv run pytest M21_weather_impact/tests -q

# the service
uv run uvicorn api:app --app-dir M21_weather_impact --port 8021

# one-off from the CLI
cd M21_weather_impact
uv run python -m src.run --out outputs/
uv run python -m src.run --scenario S04 --out outputs/
```

## API

```bash
curl localhost:8021/health
# {"status":"ok","model_id":"M21","model_version":"0.1.0","data_source":"synthetic"}

curl localhost:8021/metadata

# defaults: as_of = demo_now, horizon = 180 min, all zones, all KPIs
curl -X POST localhost:8021/predict -H 'content-type: application/json' -d '{}'

curl -X POST localhost:8021/predict -H 'content-type: application/json' -d '{
  "as_of": "2027-08-02T06:00:00+05:30",
  "horizon_min": 60,
  "entity_ids": ["Z01"],
  "kpis": ["waterlogging_probability"]
}'

curl -X POST localhost:8021/scenario -H 'content-type: application/json' -d '{
  "scenario_id": "S03",
  "compare_to_baseline": true
}'
```

Every response validates against `schemas/model_output.schema.json`.

## Inputs and outputs

| Kind | Value |
|---|---|
| Tables | `weather_hourly` (D12), `zones` (D03) |
| Upstream models | none — M21 is the root |

| KPI | Unit | Entity | How it is computed |
|---|---|---|---|
| `temperature` | degC | EVENT | read from D12, interpolated to the 15-min grid |
| `heat_index` | degC | EVENT | NOAA Rothfusz regression, recomputed from the interpolated inputs |
| `rainfall_intensity` | mm/hr | EVENT | read from D12 |
| `waterlogging_probability` | % | zone | `100·sigmoid(k·(rain − drainage) + bonus·low_lying + offset)` |
| `weather_arrival_multiplier` | ratio | EVENT | `1 − k_rain_arrival · min(rain, cap)/cap` |
| `weather_medical_multiplier` | ratio | EVENT | `1 + k_heat_medical · max(0, HI − hi_threshold)` |
| `weather_traffic_speed_multiplier` | ratio | EVENT | `1 − k_rain_speed · min(rain, cap)/cap` |

Reason codes raised: `HEAT_STRESS`, `HEAT_DRY`, `HEAVY_RAIN`, `WATERLOGGING`.

## Why the heat index is recomputed, not interpolated

D12 is hourly and the twin runs on 15 minutes. Temperature and humidity are interpolated
linearly, but the heat index is then **recomputed** from those interpolated values rather
than interpolated itself. The NOAA regression is non-linear, so interpolating it would give
a number that does not match its own reported inputs. `tests/test_formulas.py` asserts the
two agree.

## Configuration

Everything numeric is in `config.yaml` under `params`. The five response coefficients
deliberately mirror `00_common/config/assumptions.yaml → weather_response`, because the
synthetic generator uses the same values to shape the world; a test fails if they drift
apart.

| Key | Controls |
|---|---|
| `rain_cap_mm_hr` | rainfall at which the rain effects saturate |
| `k_rain_arrival` | how hard rain suppresses arrivals (M01 reads this) |
| `k_rain_speed` | how hard rain suppresses traffic speed (M04 reads this) |
| `k_heat_medical`, `hi_threshold_c` | when and how fast heat raises medical demand (M07) |
| `waterlogging_k`, `_low_lying_bonus`, `_offset` | the flooding curve; the offset sets the dry-weather baseline near 2%, not 50% |
| `heavy_rain_mm_hr`, `dry_humidity_pct`, `waterlogging_alert_pct` | when each reason code is raised |
| `confidence` | fixed: this is a replay, not a sampled forecast |

## Scenarios

Supported: `S01`, `S03`, `S04`. Everything else returns baseline values with
`status: degraded` and an `insensitive to Sxx` warning.

| Scenario | Override | Effect |
|---|---|---|
| S03 heavy rain | `rain_mm_hr: 50` in 06:00–09:00 | waterlogging up (max 80% on low-lying zones), arrival multiplier 1.00 → 0.55, traffic speed multiplier down |
| S04 extreme heat | `+6 degC`, humidity floor 60% | heat index +13.4 degC, medical multiplier up, `HEAT_STRESS` raised |

## Swapping synthetic data for real data

1. Point `data/replay/weather_hourly.parquet` at a licensed forecast feed (IMD or similar)
   with the **same columns** as D12 (docs/04 section 4).
2. Set `data_source: replay` in `config.yaml`.
3. Recalibrate `waterlogging_*` against observed flooding, and the three multipliers against
   observed attendance, presentations and probe speeds.
4. For real waterlogging, replace the logistic with a hydraulic model (PySWMM) fed by site
   rain gauges — the current curve has no notion of ponding, runoff or river level.

No model code changes: `twin_common.io.load_table` hides which source is active.
