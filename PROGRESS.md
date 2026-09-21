# Progress

Update at the end of every phase and every model. Status: ⬜ not started · 🟨 in progress · ✅ done (DoD met) · ⛔ blocked

## Phases
| Phase | Scope | Status | Commit | Notes |
|---|---|---|---|---|
| 0 | Bootstrap | ✅ | chore: bootstrap repository | Python 3.11.16 via uv-managed CPython; ruff+pytest green |
| 1 | twin_common core | ✅ | feat(common): contracts, config, io, api factory | 423 tests; 93 KPIs asserted; graph acyclic; schemas exported |
| 2 | Synthetic world + fetchers | ✅ | feat(common): synthetic world generator | 6 scenarios, 123-126 checks each, ~3s; 569 tests; offline OK |
| 3 | Template, tooling, M21 | ✅ | feat(M21): weather impact + model template tooling | 11/11 DoD; validate_all 21 checks |
| 4 | Forecast engine + 9 models | 🟨 | | engine + M01 done; M05 M06 M15 M17 M18 M19 M20 M22 to go |
| 5 | Formula/rules/ML + 5 models | ⬜ | | |
| 6 | Vision + 2 models | ⬜ | | |
| 7 | Network/optimization + 6 models | ⬜ | | |
| 8 | Pedestrian simulation + M11 | ⬜ | | |
| 9 | M25, e2e, delivery | ⬜ | | |

## Models
| ID | Folder | Engine | Phase | Status | Tests | Reviewer | Notes |
|---|---|---|---|---|---|---|---|
| M01 | M01_footfall_forecast | forecast | 4 | ✅ | 42 pass | self | chronos2 beats naive (MAE 195 vs 310); coverage 62% is under-dispersed |
| M02 | M02_crowd_density_flow | vision | 6 | ⬜ | | | |
| M03 | M03_hotspot_risk | rules | 5 | ⬜ | | | |
| M04 | M04_traffic_forecast | network | 7 | ⬜ | | | |
| M05 | M05_parking_demand | forecast | 4 | ⬜ | | | |
| M06 | M06_transit_demand | forecast | 4 | ⬜ | | | |
| M07 | M07_medical_demand | formula | 5 | ⬜ | | | |
| M08 | M08_ambulance_staging | location | 7 | ⬜ | | | |
| M09 | M09_security_incident | rules | 5 | ⬜ | | | |
| M10 | M10_cctv_anomaly | vision | 6 | ⬜ | | | |
| M11 | M11_evacuation_sim | pedsim | 8 | ⬜ | | | |
| M12 | M12_fire_risk | rules | 5 | ⬜ | | | |
| M13 | M13_vip_route | network | 7 | ⬜ | | | |
| M14 | M14_route_diversion | network | 7 | ⬜ | | | |
| M15 | M15_water_demand | forecast | 4 | ⬜ | | | |
| M16 | M16_toilet_sanitation | optimize | 7 | ⬜ | | | |
| M17 | M17_food_supply | forecast | 4 | ⬜ | | | |
| M18 | M18_waste_forecast | forecast | 4 | ⬜ | | | |
| M19 | M19_power_load | forecast | 4 | ⬜ | | | |
| M20 | M20_network_capacity | forecast | 4 | ⬜ | | | |
| M21 | M21_weather_impact | formula | 3 | ✅ | 52 pass | self | 11/11 DoD; NOAA heat index verified vs 15 NWS chart cells |
| M22 | M22_environmental_risk | forecast | 4 | ⬜ | | | |
| M23 | M23_asset_failure | mlclf | 5 | ⬜ | | | |
| M24 | M24_resource_requirement | optimize | 7 | ⬜ | | | |
| M25 | M25_overall_risk | rules | 9 | ⬜ | | | |

## Decisions log
| Date | Decision | Why | Approved by |
|---|---|---|---|
| 2026-09-21 | Installed every remaining engine extra (network, opt, vision, pedsim) ahead of Phases 6-8 | All resolve on Windows/Python 3.11 with no change to the CPU torch build, so the later phases start unblocked. jupedsim needed no WSL, contrary to the docs/06 §2 warning. | Vinay |
| 2026-09-21 | Accept PySide6 (LGPL-3.0 OR GPL-2.0 OR GPL-3.0) as a transitive dependency of jupedsim | jupedsim hard-requires it for its visualizer. We elect the LGPL-3.0 option, which CLAUDE.md allows; no twin code imports PySide6. | Vinay |
| 2026-09-21 | Road graph switched from the synthetic grid to real OSM data | Installing osmnx exposed two bugs in `real/roads.py` (see below). Fixed; 3,098 nodes / 7,510 edges now cached for Phase 7. | Vinay |

## Open questions
-
