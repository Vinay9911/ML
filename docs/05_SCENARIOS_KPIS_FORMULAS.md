# 05 — Scenarios, KPI Catalogue, Formulas & Metrics

## 1. Scenarios S01–S15

### 1.1 `00_common/config/scenarios.yaml`
```yaml
S01: {name: Normal event day, overrides: {}}
S02: {name: 30% more crowd, overrides: {footfall_multiplier: 1.3}}
S03: {name: Heavy rain, overrides: {rain_mm_hr: 50, window: ["06:00", "09:00"]}}
S04: {name: Extreme heat, overrides: {temperature_offset_c: 6, humidity_pct_min: 60}}
S05: {name: Gate closure, overrides: {closed_gates: [G02]}}
S06: {name: Road closure, overrides: {closed_links: [R07]}}
S07: {name: VIP movement at peak, overrides: {vip_convoy: {route_id: VIP1, start: "07:00", duration_min: 30}}}
S08: {name: Power failure, overrides: {substations_down: [SS01], window: ["05:30", "08:30"]}}
S09: {name: CCTV failure, overrides: {camera_outage_share: 0.2}}
S10: {name: Ambulance fleet -20%, overrides: {resource_availability_multiplier: {ambulance: 0.8}}}
S11: {name: Police deployment -10%, overrides: {resource_availability_multiplier: {police: 0.9}}}
S12: {name: Bridge unavailable, overrides: {closed_links: [B01]}}
S13: {name: Food supply disruption, overrides: {food_lead_time_multiplier: 1.5}}
S14: {name: Mass medical surge, overrides: {medical_rate_multiplier: 2.0}}
S15: {name: Evacuation event (fire), overrides: {fire_zone: Z03, blocked_exits: [E03], start: "07:30"}}
```
Scenarios may be combined in a request (`overrides` merged left→right), e.g. S02 + S04 for the demo.

### 1.2 Expected directions (used by `tests/test_scenarios.py`)
| Scenario | Study's expected shift | Models that MUST change | Assertion (vs S01, same as_of/horizon) |
|---|---|---|---|
| S02 | Density, queues, V/C, medical, toilets ↑ | M01, M03, M04, M05, M06, M07, M15, M16, M17, M18, M19, M20, M24, M25 | primary KPI ↑ (M01 total 1.2–1.4×) |
| S03 | Waterlogging, travel time, evacuation time ↑ | M21, M01, M04, M14, M11, M07 | waterlogging ↑; arrivals ↓; travel_time ↑ |
| S04 | Heat risk, water, medical ↑ | M21, M07, M15, M19, M03, M12, M25 | heat_index ↑; cases ↑; water ↑ |
| S05 | Density, queue, spillover, delay ↑ | M01, M02(world), M03, M11, M24 | G02 entries 0; Z05/Z04 risk ↑ |
| S06 | Delay, queue, emergency availability ↓ | M04, M14, M08, M12 | alternate travel_time ↑; emergency availability ≤ S01 |
| S07 | Route conflict, delay ↑ | M13, M04, M09 | conflict computed; corridor delay ↑ |
| S08 | Power, camera uptime, network ↓ | M19, M20, M23, M12, M15 | backup duration finite; availability ↓; fire risk ↑ |
| S09 | Blind spots ↑ | M23, M09, M10 | camera_availability ≈ −20%; security_risk ↑ |
| S10 | Response time, hospital load ↑ | M08, M24, M25 | response_time ↑; coverage ↓ |
| S11 | Coverage ↓, response ↑ | M09, M24, M25 | response ↑; police gap ↑ |
| S12 | Route capacity, evacuation, density ↑ | M04, M14, M11, M03 | B01 flow 0; Z02/Z06 evacuation_time ↑ |
| S13 | Stock cover ↓ | M17, M24 | coverage ↓ |
| S14 | Cases, ambulance demand, hospital load ↑ | M07, M08, M24, M25 | cases ≈ 2× (±5%) |
| S15 | Evacuation time, exits, hospital load ↑ | M11, M03, M07, M12, M22, M24, M25 | blocked_route_count ≥ 1; overall risk ↑ |

Models not listed for a scenario return baseline values with `status: degraded` and warning "insensitive to Sxx".

## 2. KPI catalogue — all 93 study KPIs + extensions (source for `kpi_registry.yaml`)
Columns: kpi · definition / formula · unit · example threshold from study (placeholder) · priority · owner.
Direction: `higher_is_worse` true unless marked (↓bad).

### Crowd (17)
| kpi | definition / formula | unit | example threshold | prio | owner |
|---|---|---|---|---|---|
| expected_footfall | forecast people arriving per period | persons/hr | model error band | Critical | M01 |
| peak_footfall | max persons/hr in operating window | persons/hr | safe operating range | Critical | M01 |
| zone_population | people inside zone | persons | < safe capacity | Critical | M02 |
| crowd_density | population ÷ usable area | persons/m² | < site safe threshold (bands §3) | Critical | M02 |
| density_trend | Δdensity ÷ Δtime | persons/m²/min | stable/declining near capacity | Critical | M02 |
| inflow_rate | entries per minute | persons/min | < gate/route capacity | High | M02 |
| outflow_rate | exits per minute (↓bad) | persons/min | enough to keep safe density | High | M02 |
| inflow_outflow_ratio | inflow ÷ outflow | ratio | < 1 preferred at peak | Critical | M02 |
| queue_length | detected queue | persons | < site threshold | High | M02 |
| queue_wait_time | avg/percentile wait | min | < 15 | High | M02 |
| pedestrian_speed | distance ÷ time (↓bad when dense) | m/s | site baseline | High | M02 |
| counterflow_risk | probability of opposing streams conflict | % | site threshold | High | M10 |
| crowd_spillover_risk | probability of overflow to adjacent zones | % | < 20 | Critical | M03 |
| bottleneck_probability | likelihood of capacity bottleneck | % | < 20 | Critical | M03 |
| crush_risk_score | composite crowd safety risk (study: stampede/crush risk) | score_0_100 | < 40 | Critical | M03 |
| crowd_dispersion_time | time to return below safe density | min | site-specific | High | M11 |
| zone_capacity_utilization | population ÷ safe capacity × 100 | % | < 80 | Critical | M02 |

### Traffic (14)
| kpi | definition / formula | unit | example threshold | prio | owner |
|---|---|---|---|---|---|
| vehicle_volume | vehicles crossing a point per time | vehicles/hr | < road capacity | High | M04 |
| traffic_density | vehicles per km | vehicles/km | < saturation | High | M04 |
| average_speed | mean vehicle speed (↓bad) | km/h | > operating floor | Medium | M04 |
| travel_time | arrival − departure | min | < route SLA | High | M04 |
| delay | actual − free-flow time | min | < 10 | High | M04 |
| vehicle_queue_length | queue extent | m | < spillback | High | M04 |
| volume_capacity_ratio | volume ÷ capacity | ratio | < 0.80 | Critical | M04 |
| intersection_saturation | demand ÷ capacity × 100 | % | < 90 | High | M04 |
| route_congestion_index | normalized weighted index | score_0_100 | < 60 | High | M04 |
| diversion_delay | scenario − baseline travel time | min | < 15 | High | M14 |
| emergency_route_availability | available ÷ required × 100 (↓bad) | % | 100 | Critical | M14 |
| parking_occupancy | occupied ÷ total × 100 | % | < 90 | High | M05 |
| parking_search_time | arrival to parked | min | < 15 | Medium | M05 |
| shuttle_demand | passengers per hour | persons/hr | model-specific | High | M06 |

### Medical (9)
| kpi | definition / formula | unit | example threshold | prio | owner |
|---|---|---|---|---|---|
| expected_medical_cases | predicted cases per time | cases/hr | within capacity | Critical | M07 |
| peak_medical_demand | max cases/hr | cases/hr | < post capacity | Critical | M07 |
| ambulance_requirement | demand ÷ dispatch capacity | vehicles | available ≥ required | Critical | M08 |
| ambulance_response_time | arrival − dispatch | min | < 8 | Critical | M08 |
| hospital_load | occupied ÷ available × 100 | % | < 80 | Critical | M07 |
| first_aid_post_requirement | cases ÷ service capacity | posts | demand covered | High | M08 |
| icu_bed_requirement | critical cases vs ICU capacity | beds | demand covered | Critical | M07 |
| medicine_consumption | units per time | units/day | stock-covered | High | M07 |
| critical_case_probability | chance of severe surge | % | site-specific | High | M07 |

### Security (8)
| kpi | definition / formula | unit | example threshold | prio | owner |
|---|---|---|---|---|---|
| security_incident_rate | incidents per time | incidents/hr | < baseline | High | M09 |
| security_risk_score | composite security risk | score_0_100 | < 40 | Critical | M09 |
| predicted_incident_probability | future incident probability | % | < 20 | High | M09 |
| cctv_coverage | covered ÷ required × 100 (↓bad) | % | 100 critical | Critical | M23 |
| camera_availability | healthy ÷ total × 100 (↓bad) | % | ≥ 99 | Critical | M23 |
| unauthorized_entry_count | access violations | incidents/hr | 0 critical | Critical | M09 |
| lost_person_count | lost/missing cases per time | cases/hr | minimize | Medium | M09 |
| security_response_time | arrival − dispatch | min | < 5–10 | Critical | M09 |

### Emergency & evacuation (9)
| kpi | definition / formula | unit | example threshold | prio | owner |
|---|---|---|---|---|---|
| evacuation_population | people in hazard impact zone | persons | within route/shelter capacity | Critical | M11 |
| evacuation_time | last evacuee − start | min | < safe available time | Critical | M11 |
| exit_capacity | max persons through exit per min (↓bad) | persons/min | > demand | Critical | M11 |
| exit_utilization | flow ÷ capacity × 100 | % | < 80 | High | M11 |
| blocked_route_count | unavailable evacuation routes | routes | 0 | Critical | M11 |
| safe_shelter_capacity | available ÷ displaced × 100 (↓bad) | % | demand covered | High | M11 |
| fire_risk_score | probability/severity of fire | score_0_100 | < 40 | Critical | M12 |
| fire_response_time | fire dispatch to arrival | min | < 8 | Critical | M12 |
| emergency_access_availability | usable ÷ required emergency capacity × 100 (↓bad) | % | 100 | Critical | M14 |

### VIP & route diversion (4)
| kpi | definition / formula | unit | example threshold | prio | owner |
|---|---|---|---|---|---|
| vip_route_clearance_time | time to make VIP route safe | min | protocol window | High | M13 |
| vip_route_conflict_probability | conflict with public flows | % | < 10 | High | M13 |
| route_diversion_effectiveness | (baseline − scenario delay) ÷ baseline × 100 (↓bad) | % | positive | High | M14 |
| alternate_route_capacity | capacity − demand on backup (↓bad) | vehicles/hr | positive margin | Critical | M14 |

### Water, sanitation & supply (12)
| kpi | definition / formula | unit | example threshold | prio | owner |
|---|---|---|---|---|---|
| expected_water_demand | consumption per time | L/hr | within supply | High | M15 |
| water_supply_demand_gap | demand − supply | L/hr | ≤ 0 | Critical | M15 |
| drinking_water_point_requirement | population ÷ point service capacity | points | demand covered | High | M15 |
| mobile_toilet_requirement | users ÷ target users per toilet | units | demand covered | High | M16 |
| toilet_utilization | actual ÷ capacity × 100 | % | < 80 | High | M16 |
| cleaning_frequency | cleaning cycles needed | cycles/day | ≥ plan | High | M16 |
| waste_generation | solid waste produced | kg/day | within collection capacity | High | M18 |
| waste_bin_fill_level | fill ÷ capacity × 100 | % | < 80 | High | M18 |
| sewage_generation | wastewater generated | MLD | within treatment capacity | High | M16 |
| food_demand | servings expected | meals/hr | within supply | Medium | M17 |
| food_stock_coverage | stock ÷ consumption rate (↓bad) | hours | ≥ min cover | High | M17 |
| delivery_requirement | demand ÷ payload | trips/day | within logistics capacity | Medium | M17 |

### Utilities, weather & environment (12)
| kpi | definition / formula | unit | example threshold | prio | owner |
|---|---|---|---|---|---|
| electricity_demand | forecast event load | MW | within available capacity | Critical | M19 |
| generator_backup_duration | fuel ÷ consumption rate (↓bad) | hours | ≥ emergency need | Critical | M19 |
| network_availability | available ÷ expected × 100 (↓bad) | % | > 99.9 | Critical | M20 |
| bandwidth_utilization | used ÷ capacity × 100 | % | < 80 | High | M20 |
| temperature | ambient temperature | °C | site threshold | High | M21 |
| heat_index | combined heat/humidity (NOAA) | °C | site threshold | Critical | M21 |
| rainfall_intensity | rain volume per time | mm/hr | site threshold | Critical | M21 |
| waterlogging_probability | flooding probability | % | < 20 | Critical | M21 |
| air_quality_index | composite AQI (Indian NAQI) | index | applicable standard | High | M22 |
| pm25 | fine particulate concentration | µg/m³ | applicable standard | High | M22 |
| noise_level | ambient noise | dB | site threshold | Medium | M22 |
| co2_emissions | fuel/electricity/traffic factors | tCO2e/day | baseline/target | Medium | M22 |

### Resource optimization & overall risk (8)
| kpi | definition / formula | unit | example threshold | prio | owner |
|---|---|---|---|---|---|
| asset_capacity_utilization | actual ÷ design × 100 | % | < 80 normal | High | M23 |
| asset_availability | available ÷ total × 100 (↓bad) | % | > 99 critical | Critical | M23 |
| resource_requirement | demand ÷ service capacity | units | demand covered | Critical | M24 |
| resource_gap | required − available | units | ≤ 0 | Critical | M24 |
| resource_coverage | allocated ÷ required × 100 (↓bad) | % | ≥ 100 critical | Critical | M24 |
| response_capacity | teams able to respond within SLA (↓bad) | teams | above minimum | Critical | M24 |
| overall_event_risk | Σ(weight × normalized risk) | score_0_100 | < 40 | Critical | M25 |
| decision_confidence | calibrated recommendation confidence (↓bad) | % | > 80 preferred | High | M25 |

### Extensions (added by this implementation; mark `extension: true` in registry)
| kpi | unit | owner |
|---|---|---|
| parking_demand | vehicles | M05 |
| shuttle_requirement | vehicles | M06 |
| passenger_wait_time | min | M06 |
| anomaly_score | score_0_100 | M10 |
| fire_vehicle_requirement | vehicles | M12 |
| vip_recommended_departure_offset_min | min | M13 |
| water_tanker_requirement | vehicles | M15 |
| cleaning_staff_requirement | personnel | M16 |
| waste_collection_trips | trips/day | M18 |
| bin_overflow_time | min (↓bad) | M18 |
| backup_generator_requirement | units | M19 |
| network_capacity_risk | % | M20 |
| weather_arrival_multiplier | ratio | M21 |
| weather_medical_multiplier | ratio | M21 |
| weather_traffic_speed_multiplier | ratio (↓bad) | M21 |
| asset_failure_probability | % | M23 |

Registry check: 17+14+9+8+9+4+12+12+8 = 93 study KPIs. Test in `00_common/tests` must assert this count.

## 3. Risk bands (`risk_bands.yaml`)
```yaml
score_0_100: {green: [0, 40], amber: [40, 60], red: [60, 80], critical: [80, 100]}
crowd_density_p_m2: {green: [0, 2.0], amber: [2.0, 4.0], red: [4.0, 5.0], critical: [5.0, 99]}
probability_pct: {green: [0, 20], amber: [20, 40], red: [40, 70], critical: [70, 100]}
utilization_pct: {green: [0, 80], amber: [80, 90], red: [90, 100], critical: [100, 9999]}
```
Density rationale (for model cards): crowd-safety literature (Fruin level of service as summarized by G. Keith Still)
places flow breakdown around 2–3 persons/m² and significant crush risk above about 4 persons/m²; Still recommends an
alert when anyone is exposed above 4 persons/m² for more than 6 minutes. These are placeholders to be set per site.

## 4. Reason codes (allowed list; extend only via this doc)
Crowd: DENSITY_HIGH, DENSITY_RISING_FAST, ACCUMULATION, NEAR_CAPACITY, SUSTAINED_EXPOSURE, EXIT_SATURATED,
COUNTERFLOW, SURGE, SUDDEN_DISPERSAL, STALLED_BUILDUP, SPILLOVER_RISK, BOTTLENECK
Weather: HEAT_STRESS, HEAVY_RAIN, WATERLOGGING, HEAT_DRY
Mobility: CONGESTION, ROAD_CLOSED, BRIDGE_CLOSED, EMERGENCY_CORRIDOR_BLOCKED, PARKING_OVERFLOW, VIP_CONFLICT
Medical: MEDICAL_SURGE, HOSPITAL_LOAD_HIGH, RESPONSE_SLA_BREACH, FLEET_SHORTAGE
Security: CROWD_DENSITY, CCTV_BLIND_SPOT, SENSITIVE_SITE, ACCESS_VIOLATIONS
Utilities/assets: GRID_OUTAGE, ELECTRICAL_OVERLOAD, GENERATOR_OPERATION, NETWORK_OVERLOAD, ASSET_FAILURE_RISK
Supply: WATER_GAP, STOCKOUT_RISK, BIN_OVERFLOW, COOKING_DENSITY, ACCESS_DELAY
Resources/overall: RESOURCE_SHORTFALL, CROWD_RISK, MEDICAL_LOAD, FIRE_RISK, MOBILITY_RISK, UTILITY_RISK, WEATHER_RISK,
ASSET_RISK, SECURITY_RISK, SYNTHETIC_DATA, UPSTREAM_FALLBACK

## 5. Resource types (15) — from the study's resource plan
| resource_type | planning logic (study) | unit | primary driver | granularity | objective |
|---|---|---|---|---|---|
| police | predicted crowd × zone risk / service ratio | personnel | crowd population/density | zone × time | max safety coverage, min travel |
| crpf | high-risk population / service rate | personnel | security risk | sector × time | cover critical zones + reserve |
| ambulance | medical demand / dispatch capacity | vehicles | medical demand | hotspot × time | meet response SLA |
| medical_team | cases / cases-per-team | teams | medical demand | zone × shift | coverage + workload |
| fire_vehicle | risk-weighted demand / station coverage | vehicles | fire risk | zone/sector | min arrival time |
| mobile_toilet | users / users per toilet | units | footfall | zone × hotspot | meet sanitation ratio |
| cleaning_staff | usage × cycles / staff capacity | personnel | toilet/waste usage | zone × shift | hygiene SLA |
| water_point | population / point capacity | points | population | zone × hotspot | min queue/shortage |
| water_tanker | supply gap / tanker payload | vehicles | water gap | zone/depot | min stockout + travel |
| shuttle_bus | passenger demand / bus capacity | vehicles | transit demand | route × time | min waiting/congestion |
| traffic_police | junction workload / staff capacity | personnel | traffic volume | intersection/route | maintain flow |
| waste_vehicle | waste volume / payload | vehicles | waste generation | zone/depot | min overflow + route time |
| food_vehicle | demand / payload | vehicles | food demand | depot/outlet | prevent stockout |
| backup_generator | critical load / generator capacity | units | power load | critical facility | redundancy/runtime |
| network_capacity | data demand / capacity | Mbps | CCTV/IoT load | zone/site | latency/availability |

## 6. Formula reference
| Metric | Formula | Note |
|---|---|---|
| Zone capacity utilization | population / safe_capacity × 100 | use usable area |
| Resource gap | required − available | positive = shortage |
| Resource coverage | allocated / required × 100 | 100% target for critical functions |
| Inflow/outflow | inflow_rate / outflow_rate | > 1 accumulation; guard outflow = 0 → cap at ratio_max |
| Volume/capacity | demand / capacity | ≥ 1 saturation |
| Travel time delay | actual − free-flow | |
| BPR travel time | t0 × (1 + α (V/C)^β), α = 0.15, β = 4 (params) | classic volume-delay function |
| Medical demand | Poisson λ = population × rate × multipliers | GLM Poisson / NegBin with log-population offset when fitted |
| Ambulance requirement | demand_rate × cycle_time / target_utilization | include turnaround |
| Toilet requirement | users / target_users_per_toilet | adjust for dwell/cleaning |
| Water gap | forecast demand − available supply | positive = shortage |
| Waste trips | waste_volume / vehicle_payload | include route time |
| Evacuation (flow model) | premovement + max_e N_e / (specific_flow × width_e) | cross-check for JuPedSim |
| Heat index | NOAA algorithm: Rothfusz regression in °F with NOAA low-HI formula and humidity adjustments | test against NOAA chart values |
| AQI (India) | sub-index linear interpolation between CPCB breakpoints; AQI = max sub-index | breakpoints in config; verify with CPCB |
| Overall risk | 100 × Σ w_d · r_d with floor rules | weights sum to 1 |
| Logistic probability | 100 / (1 + exp(−z)) | all k, z₀ params |

Indian NAQI placeholders (verify before real use):
```yaml
naqi_categories: [[0,50,Good],[51,100,Satisfactory],[101,200,Moderate],[201,300,Poor],[301,400,Very Poor],[401,500,Severe]]
naqi_breakpoints:
  pm2_5_24h: [[0,30],[31,60],[61,90],[91,120],[121,250],[251,380]]
  pm10_24h:  [[0,50],[51,100],[101,250],[251,350],[351,430],[431,510]]
```

## 7. Evaluation metrics
| Metric | Formula | Use |
|---|---|---|
| MAE | mean(|actual − forecast|) | forecasts (scale-dependent) |
| MAPE | mean(|actual − forecast| / actual) × 100 | avoid near-zero actuals (mask < min_actual) |
| Pinball loss | mean(max(q(y−ŷ_q), (q−1)(y−ŷ_q))) | quantile forecasts |
| Interval coverage | share of actuals inside [lower, upper] | uncertainty sanity |
| Precision | TP / (TP + FP) | false-alert burden (M10, M23) |
| Recall | TP / (TP + FN) | missed incidents |
| AUC | area under ROC | classifiers, always with calibration curve |
| Poisson deviance | 2 Σ (y log(y/μ) − (y − μ)) | count models (M07) |
State in every model card that metrics on synthetic data measure pipeline correctness, not real-world accuracy.
