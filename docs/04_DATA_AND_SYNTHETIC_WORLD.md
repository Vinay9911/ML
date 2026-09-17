# 04 — Data Contracts & Synthetic World Generator

## 1. Principles
1. **One world, many views.** A single generator simulates the event; every table is derived from the same crowd
   state, so scenarios change all KPIs consistently.
2. **Real where cheap.** Weather/air quality (Open-Meteo, cached), roads (OpenStreetMap via OSMnx, cached), crowd video
   (public clips supplied by the user), asset-failure training data (AI4I 2020). Everything else synthetic.
3. **Same schema as real.** Tables use the column names real feeds will use, so swapping is table-by-table.
4. **Deterministic and scenario-aware.** `generate --scenario Sxx --seed 42` → `00_common/data/world/Sxx/`.
5. **Everything flagged.** Every row has `is_synthetic` (bool) and `source` (str).
6. **All numbers are placeholders** in `assumptions.yaml`, to be validated by domain experts.

## 2. Spatial model (default schematic venue)
The user may supply a real `00_common/config/zones.geojson` (drawn in geojson.io). If absent, generate a schematic
layout around `venue_center` in `world.yaml` using local metric offsets (pyproj UTM 43N) converted to WGS84.

`venue_center` default: approximate Ramkund area, Nashik (lat 20.0063, lon 73.7926) — VERIFY on a map; any
coordinates work for the demo.

### 2.1 Zones (8)
| zone_id | name | type | area_m2 | usable_share | low_lying | notes |
|---|---|---|---|---|---|---|
| Z01 | Ghat A (river front) | ghat | 12000 | 0.70 | true | main bathing ghat |
| Z02 | Ghat B | ghat | 10000 | 0.70 | true | second ghat |
| Z03 | Temple approach corridor | corridor | 6000 | 0.80 | false | cooking stalls, fire risk |
| Z04 | Main pedestrian approach | corridor | 9000 | 0.85 | false | |
| Z05 | Gate plaza / holding area | holding | 15000 | 0.80 | false | gates G02, G03 |
| Z06 | Bridge B01 approach | bridge | 3000 | 0.90 | true | links Z02 to far bank |
| Z07 | Market & food street | market | 8000 | 0.60 | false | gate G04 |
| Z08 | Transit & parking hub | hub | 20000 | 0.50 | false | gate G01 |

`usable_area_m2 = area_m2 × usable_share`; `safe_capacity = usable_area_m2 × design_density_p_m2`.
Adjacency: Z08–Z05, Z05–Z04, Z05–Z07, Z04–Z03, Z04–Z07, Z03–Z01, Z03–Z02, Z02–Z06.

### 2.2 Other entities (generate in `assets.csv` with zone_id, lat, lon, capacity, status)
- Gates G01 (Z08), G02 (Z05), G03 (Z05), G04 (Z07); throughput persons/min param.
- Exits E01 (Z05), E02 (Z08), E03 (Z03 side lane), E04 (Z01 emergency), E05 (Z06 far bank), E06 (Z07); width_m.
- Bridge B01 (Z06). Key road segments R01–R10 and intersections J01–J06 → `key_links.csv` mapping to OSM edge/node IDs
  (nearest to schematic positions) or synthetic grid IDs.
- Parking P1, P2, P3; shuttle routes SH1 (P2→G01), SH2 (P3→G04).
- Hospitals H01 (district hospital, outside venue), H02; medical posts MP01–MP04; police posts PP01–PP04;
  fire station FS01; shelters SH01–SH02; ambulance staging candidates AS01–AS06.
- Toilet cluster candidate sites TC01–TC08; water points WP01–WP10 candidates; waste bin groups WB01–WB08.
- Substations SS01 (feeds Z01–Z04), SS02 (Z05–Z08); generators GEN01–GEN04; pumps PMP01–PMP03; network towers NT01–NT03.
- Cameras CAM01–CAM24: zone_id, mode (sparse/dense), bitrate_mbps, tower_id, fov polygon (simple sector), status.
- VIP OD: VIP1 from city entry node to Z01 viewpoint; emergency corridors EC1 (H01 ↔ Z03), EC2 (FS01 ↔ Z05).

## 3. Time model
- Grid 15 min (network telemetry 5 min), timezone Asia/Kolkata.
- `history_start: 2027-07-04T00:00+05:30` (28 days of normal context with weekly pattern and two small festival spikes).
- Event days: 2027-08-01 (arrival day ×`arrival_day_multiplier`), 2027-08-02 (peak bathing day ×`snan_day_multiplier`),
  2027-08-03 (post-peak ×`post_peak_multiplier`). The peak date mirrors a reported Amrit Snan date — verify official
  dates; the demo only needs "a peak day".
- `demo_now: 2027-08-02T06:00:00+05:30`.
- Weather: Open-Meteo historical for the same month/day in `weather_source_year: 2025`, shifted to 2027.

## 4. Data tables (Parquet unless noted) — mapped to study datasets D01–D25
All tables include `is_synthetic: bool`, `source: str`. Timestamps tz-aware.

| ID | Table | Grain | Key columns | Used by |
|---|---|---|---|---|
| D01 | footfall_15min | zone/gate × 15 min | timestamp, zone_id, gate_id, entries, exits, population | M01, M03, M24 |
| D02 | vision_timeseries (derived) | camera × 15 s/15 min | timestamp, camera_id, zone_id, count, density, in_count, out_count, speed_m_s, direction_deg | M02, M10 |
| D03 | zones.geojson | zone | zone_id, name, type, area_m2, usable_area_m2, safe_capacity, adjacency, low_lying, drainage_capacity_mm_hr, attributes{} | all |
| D04 | traffic_probe_15min | road segment × 15 min | timestamp, link_id, speed_kmh, travel_time_s, volume_proxy | M04 |
| D05 | signal_cycles | intersection × cycle | timestamp, junction_id, cycle_s, green_s, queue_veh | M04 (optional) |
| D06 | parking_15min | site × 15 min | timestamp, parking_id, entries, exits, occupied, capacity | M05 |
| D07 | transit_15min | route × 15 min | timestamp, route_id, boardings, alightings, vehicles_active, avg_wait_min | M06 |
| D08 | medical_incidents | event | incident_id, timestamp, zone_id, lat, lon, category, severity(1–5), outcome, transferred_to | M07 |
| D09 | ambulance_dispatch | incident × dispatch | incident_id, vehicle_id, dispatch_ts, arrival_ts, hospital_ts, from_site | M08 |
| D10 | security_incidents | event | incident_id, timestamp, zone_id, type, severity, response_min | M09 |
| D11 | access_control | gate × event | timestamp, gate_id, credential_class, result(granted/denied/forced) | M09 |
| D12 | weather_hourly | venue × hour | timestamp, temperature_c, humidity_pct, rain_mm, wind_kmh, visibility_m (nullable), heat_index_c | M01, M07, M15, M19, M21 |
| D13 | air_quality_hourly | venue × hour | timestamp, pm2_5, pm10, no2, co, noise_db | M22 |
| D14 | water_15min | zone/asset × 15 min | timestamp, zone_id, asset_id, consumption_l, supply_l, storage_l | M15 |
| D15 | toilet_usage | cluster × 15 min | timestamp, cluster_id, uses, queue_persons, clean_status, units_active | M16 |
| D16 | waste | bin group × 15 min | timestamp, bin_group_id, zone_id, kg_added, fill_pct, collected_kg | M18 |
| D17 | food_inventory_hourly | outlet × hour | timestamp, outlet_id, zone_id, meals_sold, stock_meals, deliveries, lead_time_h | M17 |
| D18 | power_15min | asset/zone × 15 min | timestamp, asset_id, zone_id, kw, kwh, voltage, grid_available, generator_on, fuel_l | M19 |
| D19 | network_5min | tower × 5 min | timestamp, tower_id, bandwidth_used_mbps, capacity_mbps, latency_ms, packet_loss_pct, up | M20 |
| D20 | asset_maintenance | asset × hour + work orders | timestamp, asset_id, asset_type, ambient_temp_k, process_temp_k, speed_rpm, torque_nm, wear_min, failure(0/1), failure_mode | M23 |
| D21 | event_calendar | day/session | date, session, start, end, zone_id, expected_attendance_multiplier, is_snan_day, vip_flag | M01, M04, M09, M13 |
| D22 | resource_roster | resource × shift × zone | shift_start, shift_end, resource_type, zone_id, quantity_available, status | M24 |
| D23 | sop_logs | incident × task | incident_id, sop_id, task, t_notify, t_complete, outcome | (future learning loop) |
| D24 | road_network.graphml + key_links.csv | graph | nodes, edges(length, lanes, maxspeed, capacity), key link IDs | M04, M08, M12, M13, M14 |
| D25 | lessons_learned.csv | event × KPI | event_name, kpi, outcome, notes (placeholder rows) | documentation only |

Also: `assets.csv`, `camera_registry.csv`, `world_manifest.json` (tables, rows, time range, seed, scenario, hashes).

## 5. Generator pipeline (`twin_common.synthetic.generate`)
1. **Layout** → zones.geojson, assets.csv, camera_registry.csv, key_links.csv.
2. **Calendar** → D21; daily multiplier = base × weekday factor × event-day multiplier × scenario `footfall_multiplier`.
3. **Weather** → D12/D13 from `twin_common.real.weather` (cache → fallback synthetic diurnal model with monsoon rain
   events). Apply scenario weather overrides (S03 rain window, S04 temperature offset).
4. **Arrivals** → per 15 min: daily arrivals × normalized hourly profile (normal or snan) / 4, split across gates by
   `gate_split`, multiplied by weather arrival multiplier; add lognormal noise (σ param). S05 reassigns closed gate
   share to open gates by `gate_reassignment`.
5. **Crowd flow (compartment model)** → population per zone per step:
   - Each zone is a tank. Route graph (world.yaml `routing`): Z08→Z05; Z05→Z04 or Z07; Z04→Z03; Z03→Z01/Z02 (split);
     return: Z01/Z02→Z03/Z06→Z04→Z05→exits; Z07→Z04/E06.
   - Outflow demand = population / mean_dwell_steps(zone type); actual outflow = min(demand, downstream exit/link
     capacity per step); blocked flow stays (accumulation). Closed links/exits have capacity 0 (S05, S12, S15).
   - Record entries, exits, population → D01. Density = population / usable_area.
6. **Derived domains** (all per zone × step unless noted, λ with Poisson/lognormal noise, all rates from assumptions):
   - Medical incidents (D08) λ from M07 formula; severity by `severe_share`; transfers by share; dispatch records (D09)
     with travel times from free-flow graph × congestion factor.
   - Security (D10) and access control (D11) by base rates × density factors.
   - Parking (D06): car arrivals = arrivals × car_mode_share ÷ persons_per_car, split to P1–P3 by capacity; dwell
     from bathing duration distribution.
   - Transit (D07): boardings = arrivals × bus_mode_share by route split.
   - Water (D14), toilets (D15), waste (D16), food (D17) from per-person-present rates with heat multipliers.
   - Power (D18) = base_kw + population × watts_per_person/1000 + cooling term; S08 sets grid_available=false for SS01
     zones and turns generators on (fuel decreasing).
   - Network (D19) = cameras × bitrate + population × active_user_share × kbps/1000; S09 sets 20% cameras offline.
   - Traffic probes (D04) from a quick BPR pass on key links (full model is M04).
   - Assets (D20): AI4I-like telemetry per asset with rule-based failure modes; failures more likely under S04 heat
     and S08 generator overload.
   - Roster (D22) from `resources.available` × scenario availability multipliers (S10, S11) split by shift.
7. **Validate** (`twin_common.synthetic.validate`) — fail generation if any invariant breaks (see §8).
8. **Write** Parquet + manifest + a sanity report `00_common/reports/world_<scenario>.html` (matplotlib/plotly-free
   static PNGs embedded is fine) showing arrivals, zone populations, density vs bands, medical cases, weather.

Performance target: one scenario world (31 days × 8 zones) in < 2 minutes on CPU.

## 6. `assumptions.yaml` defaults (ALL placeholders — validate with experts)
```yaml
crowd:
  normal_day_arrivals_sector: 120000
  arrival_day_multiplier: 1.8
  snan_day_multiplier: 4.0
  post_peak_multiplier: 1.8
  weekend_multiplier: 1.2
  arrival_noise_sigma: 0.08
  hourly_profile_normal: [1,1,1,2,4,7,9,9,8,6,5,4,4,4,4,5,6,7,6,4,3,2,1,1]
  hourly_profile_snan:   [3,6,9,12,13,12,10,8,6,4,3,2,2,2,2,2,2,2,2,1,1,1,1,1]
  gate_split: {G01: 0.40, G02: 0.30, G03: 0.20, G04: 0.10}
  gate_reassignment: {G02: {G03: 0.7, G01: 0.3}}
  gate_throughput_p_min: 60
  mean_dwell_min: {ghat: 45, corridor: 12, holding: 25, bridge: 6, market: 40, hub: 20}
  design_density_p_m2: 2.0
  density_bands_p_m2: {amber: 2.0, red: 4.0, critical: 5.0}
  sustained_exposure: {density_p_m2: 4.0, minutes: 6}
  exit_specific_flow_p_per_m_s: 1.3
  walking_speed_m_s: 1.2
  premovement_min: 2
medical:
  presentations_per_1000_attendees_per_day: 1.0
  hospital_transfer_share: 0.03
  severe_share: 0.05
  icu_share_of_transfers: 0.15
  heat_index_threshold_c: 32
  heat_rate_increase_per_c: 0.04
  medicine_units_per_case: 3
  hospital_beds: {H01: 400, H02: 150}
  hospital_baseline_occupancy: 0.65
  ambulance_cycle_time_min: 60
  ambulance_target_utilization: 0.6
  response_sla_min: 8
  dispatch_delay_min: 2
  first_aid_post_cases_per_hr_capacity: 12
water:
  litres_per_person_per_hour_present: 0.5
  heat_multiplier_per_c_above_30: 0.03
  persons_per_water_point: 1500
  tanker_payload_l: 10000
sanitation:
  uses_per_person_per_hour_present: 0.2
  users_per_toilet_per_hour: 12
  uses_per_cleaning: 50
  cleaning_minutes: 8
  shift_minutes: 480
  litres_per_use: 5
  walking_radius_m: 250
waste:
  kg_per_person_per_hour_present: 0.05
  bin_capacity_kg: 120
  vehicle_payload_kg: 3000
food:
  meals_per_person_per_hour_market: 0.15
  vehicle_payload_meals: 2000
  lead_time_hours: 6
  min_stock_cover_hours: 8
transport:
  car_mode_share: 0.25
  bus_mode_share: 0.35
  persons_per_car: 3.5
  bus_capacity: 50
  bus_load_factor: 0.85
  round_trip_min: 40
  lane_capacity_veh_hr: 1800
  bpr_alpha: 0.15
  bpr_beta: 4
  vehicle_length_m: 7
  parking_capacity: {P1: 1500, P2: 2500, P3: 800}
security:
  base_incidents_per_100k_per_hr: 2.0
  lost_persons_per_100k_per_hr: 5.0
  unauthorized_entries_per_secure_gate_per_hr: 0.2
  patrol_walk_speed_m_s: 1.0
  dispatch_delay_min: 2
  response_sla_min: 8
power:
  base_kw_per_zone: 150
  watts_per_person: 5
  cooling_kw_per_c_above_30_per_zone: 20
  generator_unit_kw: 250
  generator_fuel_l: 400
  generator_l_per_hr_at_full_load: 60
  redundancy_factor: 1.25
network:
  camera_bitrate_mbps: 4
  active_user_share: 0.3
  kbps_per_active_user: 50
  tower_capacity_mbps: {NT01: 2000, NT02: 2000, NT03: 1000}
environment:
  vehicle_ef_kgco2_per_km: 0.2
  grid_ef_kgco2_per_kwh: 0.7
  diesel_ef_kgco2_per_l: 2.7
  noise_base_db: 60
resources:
  police_per_1000_persons: 2.0
  crpf_per_high_risk_zone: 30
  traffic_police_per_intersection_peak: 4
  medical_team_cases_per_hr: 6
  risk_multiplier: {green: 1.0, amber: 1.25, red: 1.5, critical: 2.0}
  available:
    police: 600
    crpf: 150
    ambulance: 20
    medical_team: 30
    fire_vehicle: 6
    mobile_toilet: 400
    cleaning_staff: 250
    water_point: 60
    water_tanker: 15
    shuttle_bus: 80
    traffic_police: 120
    waste_vehicle: 10
    food_vehicle: 8
    backup_generator: 8
    network_capacity: 5000   # Mbps
```

## 7. Real data fetchers (`twin_common.real`) — cache first, always a fallback
| Source | Module | Cache path | Fallback |
|---|---|---|---|
| Open-Meteo historical weather API | `weather.py` | `data/real_cache/weather_<year>.parquet` | bundled sample → synthetic diurnal model |
| Open-Meteo air-quality API | `weather.py` | `data/real_cache/aq_<year>.parquet` | synthetic AQ model |
| OpenStreetMap drive network (OSMnx v2) | `roads.py` | `data/real_cache/roads.graphml` | synthetic grid graph with same key link IDs |
| AI4I 2020 (UCI id 601) | `ai4i.py` | `data/real_cache/ai4i2020.parquet` | rule-based regenerator |
| Model weights (Chronos-2, RF-DETR, LWCC) | library caches | set `HF_HOME` etc. under `data/real_cache/` | clear error + instructions |
`TWIN_OFFLINE=1` disables all network calls. Open-Meteo free tier is for non-commercial use (attribution CC BY 4.0).

## 8. Validation invariants (generation fails if violated)
- No NaN in required columns; timestamps continuous on the grid; tz-aware.
- population ≥ 0; entries, exits ≥ 0; population(t+1) = population(t) + entries − exits (±1 rounding) per zone.
- density ≤ 9 persons/m² (physical sanity); S01 peak-day max density in ghats reaches amber but not sustained critical.
- Sum of gate entries per day = calendar arrivals (±1%).
- Medical/security counts are non-negative integers; severe ≤ total.
- Scenario sanity: S02 daily arrivals ≈ 1.3 × S01 (±2%); S05 G02 entries = 0; S08 SS01 zones grid_available=false in
  window; S09 active cameras ≈ 80%; S12 B01 flow = 0.
- Every row `is_synthetic` is true (except rows from real sources: weather/AQ/roads/AI4I, which are false).

## 9. Slicing into model folders
`scripts/slice_world.py Mxx --scenarios S01 S02 …` reads `config.yaml → inputs` and copies only those tables
(time-filtered to history window + event days) into `Mxx_*/data/synthetic/<scenario>/`. Keep each folder's synthetic
slice small (target < 20 MB); larger artifacts are regenerated by script, not committed.

## 10. Swapping to real data later
`twin_common.io.load_table(name, data_source)`:
- `synthetic` → `data/synthetic/<scenario>/<name>.parquet`
- `replay` → `data/replay/<name>.parquet` (real recorded data in the same schema)
- `live` → registered connector class implementing `read(name, start, end) -> DataFrame` (not implemented in demo)
Column-level schema checks (pandera-free, simple dict of dtypes in `twin_common.io.schemas`) run on every load.
Model code never knows which mode is active.
