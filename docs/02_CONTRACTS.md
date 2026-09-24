# 02 — Contracts (MUST follow exactly)

These contracts are what make 25 folders integrable. Change them only with explicit human approval.

## 1. Repository layout
```
event-twin/
├── CLAUDE.md
├── PROGRESS.md
├── pyproject.toml                 # dev tooling only (ruff, pytest config)
├── docs/                          # these specs
├── plans/                         # PHASE_<n>.md plans written in plan mode
├── scripts/
│   ├── new_model.py               # scaffold a model folder from the template + registry
│   ├── slice_world.py             # copy the tables a model needs into its data/synthetic/
│   ├── refresh_samples.py         # regenerate data/sample_upstream/ in dependency order
│   ├── validate_all.py            # run every folder's tests + contract validation
│   ├── run_all.py                 # start all 25 APIs on ports 8001–8025
│   ├── scenario_e2e.py            # run scenarios through all models in dependency order
│   └── package_delivery.py        # vendor twin_common, pin requirements, zip each folder
├── templates/model_folder/        # source template used by new_model.py
├── 00_common/
│   ├── pyproject.toml             # package "twin_common" with extras
│   ├── twin_common/
│   │   ├── contracts/             # pydantic models + JSON Schema export
│   │   ├── config/                # yaml loading/merging
│   │   ├── io/                    # table registry, parquet IO, data_source adapters
│   │   ├── upstream/              # upstream output resolution
│   │   ├── api/                   # create_app() FastAPI factory
│   │   ├── output/                # OutputBuilder, risk levels, reason codes
│   │   ├── scenarios/             # scenario loading + generic adjustments
│   │   ├── engines/               # forecast, formula, rules, mlclf, vision, network, optimize, location, pedsim
│   │   ├── synthetic/             # world generator + stubs
│   │   ├── real/                  # Open-Meteo, OSM fetchers with cache + fallback
│   │   ├── testing/               # shared contract test helpers
│   │   └── logging.py
│   ├── config/                    # assumptions.yaml, world.yaml, scenarios.yaml, kpi_registry.yaml,
│   │                              # model_registry.yaml, risk_bands.yaml, resources.yaml
│   ├── schemas/                   # exported JSON Schemas (model_output.schema.json, …)
│   ├── data/
│   │   ├── world/<scenario_id>/   # generated synthetic world (gitignored, regenerable)
│   │   ├── real_cache/            # cached API/OSM downloads
│   │   └── bundled/               # small offline fallbacks committed to git
│   └── tests/
├── M01_footfall_forecast/
├── …
└── M25_overall_risk/
```

## 2. Model folder template (identical for all 25)
```
Mxx_<name>/
├── README.md               # purpose, run, API examples, config, swap-to-real-data
├── model_card.md           # method, assumptions, metrics, limitations, licenses
├── config.yaml             # ALL parameters for this model (standard keys in §8)
├── requirements.txt        # twin_common extra(s) + model-specific deps
├── Dockerfile              # python:3.11-slim, installs requirements, runs uvicorn on the model port
├── api.py                  # `app = create_app(Model)` — no logic here
├── src/
│   ├── __init__.py
│   ├── model.py            # class Model(TwinModel): load(), predict(), scenario(), metadata()
│   ├── features.py         # model-specific data preparation (optional)
│   └── run.py              # CLI: python -m src.run --as-of … --scenario S02 --out outputs/
├── schemas/
│   ├── input.schema.json   # exported from contracts (PredictRequest/ScenarioRequest)
│   └── output.schema.json  # exported from contracts (ModelOutput)
├── data/
│   ├── synthetic/          # slices of the synthetic world this model reads
│   ├── sample_upstream/    # Mxx__S01.json etc. for every upstream dependency
│   └── derived/            # precomputed artifacts (e.g. vision time series, fitted params)
├── outputs/
│   ├── sample_output.json          # /predict result for the default request
│   └── sample_scenario_<Sxx>.json  # at least one /scenario result
└── tests/
    ├── test_contract.py    # output validates; KPI coverage; bounds; determinism
    ├── test_scenarios.py   # expected directions from docs/05 §1.2
    └── test_api.py         # /health /metadata /predict /scenario via TestClient
```
Rules: `api.py` stays thin; model logic in `src/`; heavy engines in `twin_common.engines`.
Optional extra endpoints are allowed only under `/x/...` and must be documented in README.

## 3. Names, IDs and ports
| ID | Folder | Port |
|---|---|---|
| M01 | M01_footfall_forecast | 8001 |
| M02 | M02_crowd_density_flow | 8002 |
| M03 | M03_hotspot_risk | 8003 |
| M04 | M04_traffic_forecast | 8004 |
| M05 | M05_parking_demand | 8005 |
| M06 | M06_transit_demand | 8006 |
| M07 | M07_medical_demand | 8007 |
| M08 | M08_ambulance_staging | 8008 |
| M09 | M09_security_incident | 8009 |
| M10 | M10_cctv_anomaly | 8010 |
| M11 | M11_evacuation_sim | 8011 |
| M12 | M12_fire_risk | 8012 |
| M13 | M13_vip_route | 8013 |
| M14 | M14_route_diversion | 8014 |
| M15 | M15_water_demand | 8015 |
| M16 | M16_toilet_sanitation | 8016 |
| M17 | M17_food_supply | 8017 |
| M18 | M18_waste_forecast | 8018 |
| M19 | M19_power_load | 8019 |
| M20 | M20_network_capacity | 8020 |
| M21 | M21_weather_impact | 8021 |
| M22 | M22_environmental_risk | 8022 |
| M23 | M23_asset_failure | 8023 |
| M24 | M24_resource_requirement | 8024 |
| M25 | M25_overall_risk | 8025 |

Entity ID formats (regex in `twin_common.contracts.ids`):
| entity_type | Pattern | Examples |
|---|---|---|
| event | `EVENT` | EVENT |
| sector | `S[A-Z]` | SA |
| zone | `Z\d{2}` | Z01 |
| gate | `G\d{2}` | G02 |
| exit | `E\d{2}` | E03 |
| bridge / key link | `B\d{2}` | B01 |
| road_segment | `R\d{2}` (key links mapped to OSM edges in key_links.csv) | R07 |
| intersection | `J\d{2}` | J04 |
| route | `[A-Z]+\d+` | VIP1, EVAC2 |
| parking_site | `P\d` | P2 |
| facility | `H\d{2}` hospital, `MP\d{2}` medical post, `PP\d{2}` police post, `FS\d{2}` fire station, `SH\d{2}` shelter, `AS\d{2}` ambulance staging, **`FO\d{2}` food outlet** | H01, MP03, FO01 |
| service_point | `TC\d{2}` toilet cluster, `WP\d{2}` water point, `WB\d{2}` waste bin group | TC04 |
| asset | `SS\d{2}` substation, `GEN\d{2}` generator, `NT\d{2}` network tower, `PMP\d{2}` pump | GEN02 |
| camera | `CAM\d{2}` | CAM07 |
| resource_pool | resource type name | police |

## 4. ModelOutput (response body of /predict and /scenario)
Implemented as pydantic v2 models in `twin_common.contracts`; JSON Schema exported to `00_common/schemas/`.

```json
{
  "schema_version": "1.0",
  "model_id": "M03",
  "model_name": "Hotspot / Crush Risk",
  "model_version": "0.1.0",
  "run_id": "2f1c7b9e-6b4a-4d7e-9a57-0d6c5a3e9e11",
  "generated_at": "2027-08-02T06:00:04+05:30",
  "as_of": "2027-08-02T06:00:00+05:30",
  "scenario_id": "S01",
  "scenario_overrides": {},
  "data_source": "synthetic",
  "is_synthetic": true,
  "status": "ok",
  "warnings": [],
  "upstream": [
    {"model_id": "M01", "run_id": "…", "source": "sample", "generated_at": "2027-08-02T05:59:50+05:30"}
  ],
  "results": [
    {
      "entity_type": "zone",
      "entity_id": "Z01",
      "zone_id": "Z01",
      "kpi": "crush_risk_score",
      "timestamp": "2027-08-02T06:45:00+05:30",
      "horizon_min": 45,
      "state": "forecast",
      "value": 72.4,
      "unit": "score_0_100",
      "lower": 61.0,
      "upper": 81.5,
      "quantile_level": 0.8,
      "baseline_value": null,
      "delta": null,
      "risk_level": "red",
      "reason_codes": ["DENSITY_RISING_FAST", "ACCUMULATION", "HEAT_STRESS"],
      "resource_type": null,
      "recommendation": null,
      "confidence": 0.7,
      "details": {"density_p_m2": 4.3}
    }
  ]
}
```

Field rules:
- `schema_version`: "1.0". `model_version`: semver from config.
- `generated_at`, `as_of`, `timestamp`: timezone-aware, +05:30. `timestamp` = time the value refers to.
- `horizon_min`: 0 for current state; `state` ∈ {current, forecast, scenario}. `/scenario` results use `state: "scenario"`.
- `data_source` ∈ {synthetic, replay, live}. `is_synthetic` true whenever any input row is synthetic.
- `status` ∈ {ok, degraded, error}. Use `degraded` + a warning when a fallback (sample/stub/cached) was used.
- `upstream[].source` ∈ {inline, url, sample, stub}.
- `entity_type` from the table in §3. `zone_id` = parent zone when the entity is inside a zone, else null.
- `kpi` must exist in `kpi_registry.yaml`; `unit` must equal the registry unit.
- Probabilities are expressed in `%` (0–100) as in the study. `confidence` is 0–1.
- `lower ≤ value ≤ upper` when bands exist; `quantile_level` = band coverage (0.8 = 10th–90th percentile).
- `baseline_value` and `delta` (= value − baseline_value) are filled only in `/scenario` responses when
  `compare_to_baseline` is true.
- `risk_level` ∈ {green, amber, red, critical, null}. Derived via `risk_bands.yaml`, never ad hoc.
- `reason_codes`: UPPER_SNAKE_CASE, from the reason code list in docs/05 §4. Mandatory when risk_level is amber or worse.
- `resource_type`: one of the 15 resource types in docs/05 §5 when the KPI is resource-specific.
- `recommendation` (optional): `{action, resource_type, quantity, target_entity_id, from_entity_id, rationale, requires_approval}`;
  `requires_approval` is ALWAYS true in this project.
- `details`: small JSON object; route geometry may be GeoJSON LineString; keep under ~50 KB per record.

## 5. KPI and unit rules
- All KPI names, units, owner model, direction (`higher_is_worse` true/false) and example thresholds are defined
  in `00_common/config/kpi_registry.yaml`, generated from docs/05 §2.
- Allowed units: persons, persons/hr, persons/min, persons/m², persons/m²/min, m, min, m/s, %, ratio, score_0_100,
  vehicles, vehicles/hr, vehicles/km, km/h, cases/hr, cases, units/day, units, beds, posts, points, routes, teams,
  personnel, incidents/hr, L/hr, MLD, kg/day, meals/hr, hours, trips/day, cycles/day, MW, dB, µg/m³, index,
  tCO2e/day, °C, mm/hr.

## 6. API contract (FastAPI via `twin_common.api.create_app`)
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | /health | — | `{"status":"ok","model_id":"M01","model_version":"0.1.0","data_source":"synthetic"}` |
| GET | /metadata | — | `Metadata` |
| POST | /predict | `PredictRequest` | `ModelOutput` |
| POST | /scenario | `ScenarioRequest` | `ModelOutput` |

`PredictRequest`:
```json
{
  "as_of": "2027-08-02T06:00:00+05:30",
  "horizon_min": 180,
  "entity_ids": ["Z01", "Z03"],
  "kpis": ["crush_risk_score"],
  "include_current": true,
  "upstream": {"M01": { "...ModelOutput..." : "..." }}
}
```
All fields optional. Defaults: `as_of` = `demo_now` from config; `horizon_min` = `default_horizon_min`;
all entities; all KPIs owned by the model.

`ScenarioRequest` = `PredictRequest` + `{"scenario_id": "S02", "overrides": {}, "compare_to_baseline": true}`.
Unknown `scenario_id` → 404. Scenario not in `scenarios_supported` → 200 with `status: "degraded"` and a warning
that the model is insensitive to it (values equal baseline).

`Metadata`:
```json
{
  "model_id": "M01", "model_name": "Footfall Forecast", "model_version": "0.1.0",
  "question": "How many people will arrive, by zone and time?",
  "engine": "forecast", "method_summary": "Chronos-2 zero-shot with covariates; LightGBM baseline",
  "kpis": [{"kpi": "expected_footfall", "unit": "persons/hr", "description": "…"}],
  "upstream": ["M21"], "inputs": ["footfall_15min", "event_calendar", "weather_hourly"],
  "horizons_min": [15, 60, 180], "scenarios_supported": ["S01", "S02", "S03", "S04", "S05"],
  "limitations": ["Synthetic training context", "…"], "license_notes": ["Chronos-2 weights: …"]
}
```
Errors: 422 validation errors (FastAPI default); 404 unknown entity or scenario; 500 → JSON `{"status":"error","detail":…}`.
CORS: allow all origins (demo). Response time target: < 10 s on CPU for default request.

## 7. Upstream resolution (in `twin_common.upstream`)
For each upstream model ID listed in `config.yaml → upstream`, resolve in this order:
1. `request.upstream[ID]` (inline) — must validate as ModelOutput.
2. Env var `UPSTREAM_<ID>_URL` (e.g. `UPSTREAM_M01_URL=http://localhost:8001`): call `/scenario` if a scenario is
   requested else `/predict`, same `as_of`, timeout from config.
3. `data/sample_upstream/<ID>__<scenario_id>.json`.
4. `data/sample_upstream/<ID>__S01.json` with warning "baseline upstream used for scenario".
5. Stub generated from the synthetic world by `twin_common.synthetic.stubs` (status degraded).
Record the chosen source in `ModelOutput.upstream`.

## 8. Standard `config.yaml` keys
```yaml
model_id: M01
model_name: Footfall Forecast
model_version: 0.1.0
port: 8001
engine: forecast
data_source: synthetic          # synthetic | replay | live
seed: 42
demo_now: "2027-08-02T06:00:00+05:30"
horizons_min: [15, 60, 180]
default_horizon_min: 180
upstream: [M21]
inputs: [footfall_15min, event_calendar, weather_hourly]
kpis: [expected_footfall, peak_footfall]
scenarios_supported: [S01, S02, S03, S04, S05]
upstream_timeout_s: 5
params: {}                      # model-specific numbers only here
```
`data_source: live` must raise a clear `NotImplementedError("live connector for <table> not configured")`
through the `twin_common.io` adapter interface — never fake live data.

## 9. Definition of Done (per model)
1. Folder matches §2 exactly; config has all §8 keys.
2. Clean install works: new venv + `pip install -r requirements.txt` (with local `00_common`).
3. `pytest` passes: contract validity, every KPI in the model card appears, bounds, determinism (same seed → identical
   JSON except run_id/generated_at), scenario directions from docs/05 §1.2.
4. `/health`, `/metadata`, `/predict`, `/scenario` return valid responses (show curl or TestClient output).
5. `outputs/sample_output.json` and ≥1 `outputs/sample_scenario_<Sxx>.json` saved and valid.
6. Runs with `TWIN_OFFLINE=1` using cached/bundled/sample data.
7. No numeric thresholds/rates/weights in `src/` (math constants excepted) — grep check.
8. `is_synthetic` and `data_source` correct in outputs.
9. README.md and model_card.md complete (templates §11–§12).
10. No imports from other model folders.
11. Default `/predict` < 10 s on CPU (vision/pedsim serve precomputed results; recompute via CLI).
12. `model-reviewer` subagent reports no blocking gaps.

## 10. Delivery packaging (`scripts/package_delivery.py`)
- Copies `00_common/twin_common` into `Mxx_*/vendor/twin_common` and rewrites `requirements.txt` to pinned versions
  (from the working environment) without the editable local path.
- Produces `delivery/Mxx_<name>.zip` for each model plus `delivery/INTEGRATION_GUIDE.md`
  (ports, endpoints, curl examples, dependency DAG as a Mermaid diagram, scenario list).
- Verifies each zip in a fresh temporary venv: install, run tests, call /health.
- Confirm with the manager whether he wants vendored copies (default) or a shared `00_common`.

## 11. README.md template
```
# Mxx — <Model Name>
One-paragraph purpose (the operational question it answers).
## Run
## API (curl examples for /health, /metadata, /predict, /scenario)
## Inputs (tables + upstream models) and outputs (KPIs with units)
## Configuration (key params and where they come from)
## Scenarios supported and expected effects
## Swapping synthetic data for real data (exact tables/columns, what to recalibrate)
```

## 12. model_card.md template
```
# Model card — Mxx
- Question / decision supported
- Method (engine, algorithm, library + version, license)
- Inputs and features
- Assumptions (link to config keys) — all demo placeholders unless stated
- Evaluation (metrics on synthetic backtest or test data; state clearly they are not real-world accuracy)
- Limitations and failure modes
- Real-data readiness: what must be recalibrated / retrained
- PDF traceability: KPI names and dataset IDs (D01–D25) covered
```

## 13. `00_common/config/model_registry.yaml` (authoritative copy)
```yaml
M01: {folder: M01_footfall_forecast,   port: 8001, engine: forecast, phase: 4, upstream: [M21]}
M02: {folder: M02_crowd_density_flow,  port: 8002, engine: vision,   phase: 6, upstream: []}
M03: {folder: M03_hotspot_risk,        port: 8003, engine: rules,    phase: 5, upstream: [M01, M02, M21]}
M04: {folder: M04_traffic_forecast,    port: 8004, engine: network,  phase: 7, upstream: [M01]}
M05: {folder: M05_parking_demand,      port: 8005, engine: forecast, phase: 4, upstream: [M01]}
M06: {folder: M06_transit_demand,      port: 8006, engine: forecast, phase: 4, upstream: [M01, M05]}
M07: {folder: M07_medical_demand,      port: 8007, engine: formula,  phase: 5, upstream: [M01, M21]}
M08: {folder: M08_ambulance_staging,   port: 8008, engine: location, phase: 7, upstream: [M07, M04]}
M09: {folder: M09_security_incident,   port: 8009, engine: rules,    phase: 5, upstream: [M01, M02, M23]}
M10: {folder: M10_cctv_anomaly,        port: 8010, engine: vision,   phase: 6, upstream: []}
M11: {folder: M11_evacuation_sim,      port: 8011, engine: pedsim,   phase: 8, upstream: [M01]}
M12: {folder: M12_fire_risk,           port: 8012, engine: rules,    phase: 5, upstream: [M19, M21, M04]}
M13: {folder: M13_vip_route,           port: 8013, engine: network,  phase: 7, upstream: [M04, M01]}
M14: {folder: M14_route_diversion,     port: 8014, engine: network,  phase: 7, upstream: [M04]}
M15: {folder: M15_water_demand,        port: 8015, engine: forecast, phase: 4, upstream: [M01, M21]}
M16: {folder: M16_toilet_sanitation,   port: 8016, engine: optimize, phase: 7, upstream: [M01]}
M17: {folder: M17_food_supply,         port: 8017, engine: forecast, phase: 4, upstream: [M01]}
M18: {folder: M18_waste_forecast,      port: 8018, engine: forecast, phase: 4, upstream: [M01]}
M19: {folder: M19_power_load,          port: 8019, engine: forecast, phase: 4, upstream: [M01, M21]}
M20: {folder: M20_network_capacity,    port: 8020, engine: forecast, phase: 4, upstream: [M01]}
M21: {folder: M21_weather_impact,      port: 8021, engine: formula,  phase: 3, upstream: []}
M22: {folder: M22_environmental_risk,  port: 8022, engine: forecast, phase: 4, upstream: [M21, M04]}
M23: {folder: M23_asset_failure,       port: 8023, engine: mlclf,    phase: 5, upstream: []}
M24: {folder: M24_resource_requirement,port: 8024, engine: optimize, phase: 7, upstream: [M01, M03, M06, M07, M08, M09, M12, M15, M16, M17, M18, M19, M20]}
M25: {folder: M25_overall_risk,        port: 8025, engine: rules,    phase: 9, upstream: [M03, M04, M07, M09, M12, M14, M19, M20, M21, M23, M24]}
```
The dependency graph is acyclic. `scripts/refresh_samples.py` must topologically sort it.
