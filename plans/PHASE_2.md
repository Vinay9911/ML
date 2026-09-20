# Phase 2 — Synthetic world generator + real data fetchers

Status: implemented.

## Goal
One consistent, scenario-aware world under `00_common/data/world/<Sxx>/`, so that every model reads the
same crowd state and scenarios change all KPIs coherently (docs/04 section 1 principle 1).

## 1. Module design
| Module | Writes | Notes |
|---|---|---|
| `synthetic/grid.py` | - | time grid helpers: 15-min index, day slicing, hour profiles |
| `synthetic/layout.py` | `zones` (D03 + zones.geojson), `assets`, `camera_registry`, `key_links` (D24) | schematic metric layout around `venue_center`, UTM 43N -> WGS84; honours a user `zones.geojson` |
| `synthetic/calendar.py` | `event_calendar` (D21) | daily multiplier = base x weekday x event-day x scenario `footfall_multiplier` |
| `synthetic/weather.py` (in `real/`) | `weather_hourly` (D12), `air_quality_hourly` (D13) | Open-Meteo cache -> bundled -> synthetic diurnal; S03/S04 overrides applied after |
| `synthetic/crowd_flow.py` | `footfall_15min` (D01) | the compartment model, the heart of the world |
| `synthetic/derived.py` | D04, D06, D07, D08, D09, D10, D11, D14, D15, D16, D17, D18, D19 | all per-person rates from `assumptions.yaml` |
| `synthetic/assets_telemetry.py` | `asset_maintenance` (D20) | AI4I-shaped telemetry + rule-based failure modes |
| `synthetic/roster.py` | `resource_roster` (D22) | `resources.available` x scenario multiplier, split by shift |
| `synthetic/misc.py` | `signal_cycles` (D05), `sop_logs` (D23), `lessons_learned` (D25) | small placeholder tables |
| `synthetic/validate.py` | - | the docs/04 section 8 invariants; generation FAILS on violation |
| `synthetic/report.py` | `00_common/reports/world_<Sxx>.html` | static PNGs embedded, matplotlib only |
| `synthetic/generate.py` | `world_manifest.json` | CLI `--scenario --seed --offline --out` |

## 2. Crowd compartment model (docs/04 section 5.5)
State: `population[zone]` at each 15-min step.

```
arrivals(t)        = daily_arrivals x hourly_profile[hour]/4 x weather_arrival_multiplier x lognormal(sigma)
gate_entries(t,g)  = arrivals(t) x gate_split[g]          (S05: closed gate share redistributed by
                                                           gate_reassignment, closed gate gets 0)
                     capped at gate_throughput_p_min x 15
inflow to zone     = sum of entries at gates in that zone
outflow demand(z)  = population[z] / mean_dwell_steps(zone_type)
actual outflow(z)  = min(demand, downstream link/exit capacity for this step)
                     -> blocked flow stays in the zone = ACCUMULATION
population(z,t+1)  = population(z,t) + inflow - actual_outflow
density(z,t)       = population(z,t) / usable_area_m2
```
Splits come from `world.yaml -> routing` (inbound and outbound phases, each summing to 1, already
asserted by a Phase 1 test). Closed links/exits get capacity 0 (S05, S12, S15).

Invariant by construction: `entries` and `exits` are recorded as the flows actually applied, so
`population(t+1) = population(t) + entries - exits` holds exactly, which is the docs/04 section 8 check.

## 3. Determinism
One `numpy.random.default_rng(seed)` per generator step, seeded as
`seed_for(step_name, scenario_id, base_seed)` = a stable hash, so adding a step never shifts the
random stream of an earlier one. Two runs with the same seed must produce identical parquet bytes;
the manifest records per-table sha256 and the test compares them.

## 4. Real fetchers (`twin_common/real/`) — cache first, always a fallback
| Source | Module | Cache | Fallback chain |
|---|---|---|---|
| Open-Meteo archive (weather) | `weather.py` | `data/real_cache/weather_<year>.parquet` | bundled sample -> synthetic diurnal + monsoon events |
| Open-Meteo air quality | `weather.py` | `data/real_cache/aq_<year>.parquet` | bundled sample -> synthetic AQ model |
| OSM drive network (OSMnx v2) | `roads.py` | `data/real_cache/roads.graphml` | synthetic grid graph with the same R01-R10 / J01-J06 IDs |
| AI4I 2020 (UCI 601) | `ai4i.py` | `data/real_cache/ai4i2020.parquet` | rule-based regenerator using the published failure-mode rules |

`TWIN_OFFLINE=1` skips the network entirely and goes straight to cache -> bundled -> synthetic.
Rows from a real source carry `is_synthetic=False`; the fallbacks carry `True`. Bundled fallbacks are
committed to `00_common/data/bundled/` so a fresh clone generates offline.

Network use is Phase 2's only outward call. Open-Meteo free tier is non-commercial, CC BY 4.0
attribution (docs/06 section 3) — recorded in the manifest and the report.

## 5. Scenario application, step by step
| Scenario | Applied in | How |
|---|---|---|
| S02 `footfall_multiplier` | calendar | daily arrivals x 1.3 |
| S03 `rain_mm_hr` + window | weather | rain forced inside the window; arrival multiplier falls |
| S04 `temperature_offset_c`, `humidity_pct_min` | weather | temperature + 6, humidity floored at 60, heat index recomputed |
| S05 `closed_gates` | crowd_flow | G02 entries 0; share redistributed by `gate_reassignment` |
| S06 / S12 `closed_links` | layout + derived | link capacity 0; traffic probe reassignment |
| S07 `vip_convoy` | calendar | `vip_flag` on the affected session |
| S08 `substations_down` + window | derived (power) | `grid_available=false` for SS01 zones, generators on, fuel falling |
| S09 `camera_outage_share` | layout | 20 percent of cameras `status=offline` (deterministic pick) |
| S10 / S11 `resource_availability_multiplier` | roster | availability scaled per resource type |
| S13 `food_lead_time_multiplier` | derived (food) | lead time x 1.5 |
| S14 `medical_rate_multiplier` | derived (medical) | lambda x 2 |
| S15 `fire_zone`, `blocked_exits`, `start` | crowd_flow + derived | E03 capacity 0 from `start`; Z03 evacuation demand |

## 6. Validation invariants (docs/04 section 8) — generation fails on violation
1. no NaN in required columns; timestamps continuous on the grid; tz-aware
2. population >= 0; entries, exits >= 0; population continuity per zone (+/- 1 rounding)
3. density <= 9 persons/m2; S01 peak-day ghat max reaches amber, not sustained critical
4. sum of gate entries per day = calendar arrivals (+/- 1 percent)
5. medical and security counts non-negative integers; severe <= total
6. scenario sanity: S02 ~ 1.3x S01 (+/- 2 percent); S05 G02 = 0; S08 SS01 grid down in window;
   S09 active cameras ~ 80 percent; S12 B01 flow 0
7. every row `is_synthetic` true, except weather/AQ/roads/AI4I rows that came from a real source

## 7. Acceptance (docs/07 Phase 2)
- `generate --scenario S01` and `--scenario S02` each finish under 2 min on CPU; validation passes
- S02/S01 daily arrivals ratio in [1.28, 1.32]
- S05 G02 entries = 0 and Z05 peak density > S01
- S08, S09, S12 invariants pass
- `TWIN_OFFLINE=1 generate --scenario S01` succeeds from bundled/cached data
- sanity report written; peak-day ghat density reaches amber in S01
- deterministic: two runs, same seed -> identical file hashes (manifest timestamps excepted)

## 8. Decisions taken where the docs needed resolving
All five are recorded in the config files next to the value they changed, and all are reversible.

**D3 - `gate_throughput_p_min` 60 -> per-gate widths.** 60 persons/min is one turnstile: at the
project's own `exit_specific_flow_p_per_m_s` of 1.3 it implies a gate 0.8 m wide. Four such gates pass
3,600 people per 15-min step against a snan peak of 10,206, so arrivals were capped at 92 percent, the
ghats never reached amber and the S02/S01 ratio was squeezed to 1.22. Gates now carry individual widths
in `world.yaml`: G01 1500, G02 900, G03 250, G04 600 persons/min.

**D4 - added `max_holding_density_p_m2` (jam density).** The compartment model had no back-pressure, so
a sealed zone filled without limit: S12 drove Z06 to 302 persons/m2 against the docs/04 section 8 sanity
limit of 9. Inflow is now capped by the space the receiving zone actually has, allocated to a fixed
point so a zone cannot accept people it will not be able to shed.

**D5 - `snan_day_multiplier` 4.0 -> 2.8.** The two docs/04 placeholder sets are mutually inconsistent.
The section 2.1 zone areas give the ghats 41,067 persons/hour of safe throughput; 120000 x 4.0 shaped by
`hourly_profile_snan` demands 58,318, which is 1.42x over. No link width fixes that - widening the ghat
approach puts the ghats at 5.9 p/m2 (sustained critical), narrowing it moves the jam into the Z03 temple
corridor, which is worse. At 2.8 the world reproduces the docs/01 section 5 demo story exactly: Ghat A is
3.96 p/m2 (amber) at demo_now 06:00 and 4.15 (red) by 06:45.

**D6 - zone link widths and a `closure_map` added to `world.yaml` routing.** Pedestrian capacities had
no config home, and a closed link ID like B01 had no defined effect on pedestrian movement. Both now live
in config, so no capacity is hard-coded in `src/`.

**D7 - corridor zones are generated portrait.** The docs/04 section 2.1 adjacency chain runs north-south;
generating corridors landscape pointed every camera across a 19 m strip and collapsed coverage to 3.8
percent in Z06. Coverage is now 42-78 percent per zone.

## 9. Open questions
1. `venue_center` is still the docs/04 default (Ramkund, Nashik 20.0063 / 73.7926), `verified_on_map: false`.
   Supply real coordinates or a `00_common/config/zones.geojson` to replace the schematic layout.
2. Every value changed under D3-D5 is a placeholder that a police / medical / event-safety reviewer should
   confirm. The gate widths and the snan multiplier are the two that most change the demo.
3. `osmnx` is not installed, so `real/roads.py` currently serves the synthetic grid fallback. Install the
   `network` extra before Phase 7 to fetch the real OSM graph.
