# Progress

Update at the end of every phase and every model. Status: ⬜ not started · 🟨 in progress · ✅ done (DoD met) · ⛔ blocked

## Phases
| Phase | Scope | Status | Commit | Notes |
|---|---|---|---|---|
| 0 | Bootstrap | ✅ | chore: bootstrap repository | Python 3.11.16 via uv-managed CPython; ruff+pytest green |
| 1 | twin_common core | ✅ | feat(common): contracts, config, io, api factory | 423 tests; 93 KPIs asserted; graph acyclic; schemas exported |
| 2 | Synthetic world + fetchers | ✅ | feat(common): synthetic world generator | 6 scenarios, 123-126 checks each, ~3s; 569 tests; offline OK |
| 3 | Template, tooling, M21 | ✅ | feat(M21): weather impact + model template tooling | 11/11 DoD; validate_all 21 checks |
| 4 | Forecast engine + 9 models | ✅ | feat(M20,M22): network capacity and environmental risk | engine + all 9 models; conformal band calibration; common random numbers |
| 5 | Formula/rules/ML + 5 models | ⬜ | | |
| 6 | Vision + 2 models | ⬜ | | |
| 7 | Network/optimization + 6 models | ⬜ | | |
| 8 | Pedestrian simulation + M11 | ⬜ | | |
| 9 | M25, e2e, delivery | ⬜ | | |

## Models
| ID | Folder | Engine | Phase | Status | Tests | Reviewer | Notes |
|---|---|---|---|---|---|---|---|
| M01 | M01_footfall_forecast | forecast | 4 | ✅ | 42 pass | self | chronos2 beats naive (MAE 190 vs 335); bands now conformally calibrated |
| M02 | M02_crowd_density_flow | vision | 6 | ⬜ | | | |
| M03 | M03_hotspot_risk | rules | 5 | ⬜ | | | |
| M04 | M04_traffic_forecast | network | 7 | ⬜ | | | |
| M05 | M05_parking_demand | forecast | 4 | ✅ | 57 pass | model-reviewer: no blocking gaps | forecasts uncapped demand, so occupancy can exceed 100%; parking capacity raised 4,800 -> 23,200 |
| M06 | M06_transit_demand | forecast | 4 | ✅ | 58 pass | self | first two-upstream model; fleet in resources.yaml is 4.6-18x too small - reported, not changed (M24 owns it) |
| M07 | M07_medical_demand | formula | 5 | ⬜ | | | |
| M08 | M08_ambulance_staging | location | 7 | ⬜ | | | |
| M09 | M09_security_incident | rules | 5 | ⬜ | | | |
| M10 | M10_cctv_anomaly | vision | 6 | ⬜ | | | |
| M11 | M11_evacuation_sim | pedsim | 8 | ⬜ | | | |
| M12 | M12_fire_risk | rules | 5 | ⬜ | | | |
| M13 | M13_vip_route | network | 7 | ⬜ | | | |
| M14 | M14_route_diversion | network | 7 | ⬜ | | | |
| M15 | M15_water_demand | forecast | 4 | ✅ | 60 pass | self | S04 heat response is real but decays with horizon; trustworthy ~1 h |
| M16 | M16_toilet_sanitation | optimize | 7 | ⬜ | | | |
| M17 | M17_food_supply | forecast | 4 | ✅ | 55 pass | self | hourly grid; inventory simulated forward so S13 lead time actually bites (coverage 0.88x) |
| M18 | M18_waste_forecast | forecast | 4 | ✅ | 49 pass | self | bins_per_group added: a group is a zone's bins, not one 120 kg bin |
| M19 | M19_power_load | forecast | 4 | ✅ | 53 pass | self | generator refuelling added to the world; D18 rows are (substation, zone) and must be summed |
| M20 | M20_network_capacity | forecast | 4 | ✅ | 55 pass | self | 5-min grid; finding: this world never runs out of bandwidth (peak 27.5%), so capacity risk ~0 |
| M21 | M21_weather_impact | formula | 3 | ✅ | 52 pass | self | 11/11 DoD; NOAA heat index verified vs 15 NWS chart cells |
| M22 | M22_environmental_risk | forecast | 4 | ✅ | 53 pass | self | uses the one REAL table (D13); NAQI breakpoints + CO2 factors are placeholders; CO2 traffic term zero until M04 |
| M23 | M23_asset_failure | mlclf | 5 | ⬜ | | | |
| M24 | M24_resource_requirement | optimize | 7 | ⬜ | | | |
| M25 | M25_overall_risk | rules | 9 | ⬜ | | | |

## Decisions log
| Date | Decision | Why | Approved by |
|---|---|---|---|
| 2026-09-24 | Generators now refuel between outages in the world generator | Fuel was a cumulative drain over the whole month, and the S08 outage window recurs daily, so 31 x 3 h emptied the tank long before the demo day: `generator_backup_duration` read 0 for the one scenario it exists for. Now resets whenever the grid is back; S08 fuel falls 394 -> 298 L across the outage. | Vinay |
| 2026-09-24 | `waste.bins_per_group: 50` added | `bin_capacity_kg` is ONE bin, but D16 models a bin GROUP per zone and the generator divided a whole zone's waste by a single bin. Every group read 100% full within one step, so `waste_bin_fill_level` and `bin_overflow_time` carried no information. 50 x 120 kg puts the busiest group at 103% of the snan peak. All 10 worlds regenerated. | Vinay |
| 2026-09-24 | `facility` entity pattern extended with `FO` (food outlet) | The generator emits FO01/FO02 for D17 and the docs/03 M17 card asks for per-outlet reporting, but no entity type in the docs/02 §3 ID table covered them - an inconsistency between docs/02 and docs/04. `facility` is the closest fit (staffed, stocked, fixed). docs/02 §3 and contracts/ids.py both updated. | Vinay |
| 2026-09-25 | Food deliveries now arrive AFTER the lead time in the world generator | The lead time sized the order but the delivery landed in the same step, so a longer lead time only made orders bigger and left the outlet BETTER off - the opposite of a supply disruption, against the docs/05 S13 assertion of "coverage down". S13 mean stock now falls 3,128 -> 2,518 meals. | Vinay |
| 2026-09-25 | Added `GET /inputs` to the shared API factory, and `scripts/serve_all.py` | The four contract endpoints say what a model produces, never what it consumes, which makes a model hard to explain to anyone who did not write it. `/inputs` returns the real tables, real row counts, a real sample around `as_of`, and the full params block. `serve_all.py` runs every built model on one port so the live simulation page needs one command rather than ten. Both additive; the contract endpoints are untouched. | Vinay |
| 2026-09-24 | Synthetic world switched to COMMON RANDOM NUMBERS across scenarios | `rng_for` seeded on the scenario, so every scenario drew different noise. With `arrival_noise_sigma` 0.08 that swamped small signals: S04's 1.8% water uplift came out 0.2% NEGATIVE. Scenario deltas are now attributable to the intervention alone - S02 is exactly 1.3000x. All 9 worlds regenerated. | Vinay |
| 2026-09-24 | Conformal band calibration enabled on every forecasting model | A band labelled 80% held the truth ~50% of the time because error grows far faster across the horizon than the quantiles do. Calibrated at startup from held-out origins. | Vinay |
| 2026-09-24 | `water.pump_outage_supply_factor` lifted from the generator into assumptions.yaml | It was a literal in `derived.py`; M15 needed the same value, and two copies could drift. Value unchanged. | Vinay |
| 2026-09-21 | Installed every remaining engine extra (network, opt, vision, pedsim) ahead of Phases 6-8 | All resolve on Windows/Python 3.11 with no change to the CPU torch build, so the later phases start unblocked. jupedsim needed no WSL, contrary to the docs/06 §2 warning. | Vinay |
| 2026-09-21 | Accept PySide6 (LGPL-3.0 OR GPL-2.0 OR GPL-3.0) as a transitive dependency of jupedsim | jupedsim hard-requires it for its visualizer. We elect the LGPL-3.0 option, which CLAUDE.md allows; no twin code imports PySide6. | Vinay |
| 2026-09-21 | `transport.parking_capacity` raised from 4,800 to 23,200 bays | The docs/04 §6 placeholder put an ordinary day at 140-180% occupancy and the snan peak at 583%, so `parking_occupancy` would read critical at every hour and its <90% threshold was unreachable. Sized for the design peak instead: 121% at the snan peak, 31% on an ordinary day. | Vinay |
| 2026-09-21 | M05 forecasts uncapped demand rather than the `occupied` column | D06 clips `occupied` at capacity, so a model trained on it can never predict the overflow M05 exists to warn about. Occupancy is therefore demand/capacity and may exceed 100%, which `risk_bands.yaml` already anticipates (critical = 100-9999). | Vinay |
| 2026-09-21 | Road graph switched from the synthetic grid to real OSM data | Installing osmnx exposed two bugs in `real/roads.py` (see below). Fixed; 3,098 nodes / 7,510 edges now cached for Phase 7. | Vinay |

## Open questions
- **Shuttle fleet** (`resources.shuttle_bus.available` = 80) needs 369 vehicles on an ordinary day
  and 1,447 at the snan peak. Confirm when M24 is built: is the fleet genuinely this short, or is
  `bus_mode_share: 0.35` too high for a venue with only two shuttle routes?
- **Tower capacities** (`assumptions.network.tower_capacity_mbps`) are over-provisioned: peak
  utilisation is 27.5 %, so M20's capacity-risk KPI carries no signal. Validate before relying on it.
- **NAQI breakpoints and CO2 emission factors** in M22 are transcription placeholders. Verify the
  breakpoints against the published CPCB table and replace the factors with official values before
  any figure leaves the demo.
- **Crowd videos** are needed before Phase 6 (M02, M10) and must be supplied by the project owner.
