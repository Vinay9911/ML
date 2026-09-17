# 03 — Model Specs (one card per model)

How to use: read only the card you are building plus the KPI/formula rows it references in docs/05.
Every number mentioned as a "param" goes in the model `config.yaml` (or shared `assumptions.yaml`), never in code.
Dataset IDs (D01–D25) refer to docs/04 §4. Scenario IDs (S01–S15) refer to docs/05 §1.

Card format: Question · Engine/phase · KPIs owned · Method · Inputs · Upstream · Scenarios · Tests · Real-data swap.

---

## M01 — Footfall Forecast
- **Question:** How many people will arrive / be present, by zone, gate and time?
- **Engine / phase:** `engines.forecast` · Phase 4 (reference model for the forecast engine) · Priority Critical
- **KPIs:** `expected_footfall` (persons/hr), `peak_footfall` (persons/hr)
- **Method:**
  - Backends compared in model_card: (a) Darts `Chronos2Model` zero-shot, quantiles 0.1/0.5/0.9, with past covariates
    (lagged entries, population) and future covariates (hour_of_day sin/cos, day_of_week, is_snan_day, snan_multiplier,
    temperature_c, rain_mm); (b) Darts `LightGBMModel` (lags 1–8 and 96, same covariates, quantile regression);
    (c) `NaiveSeasonal(K=96)`. Default backend = Chronos-2; CPU fallback hub model `autogluon/chronos-2-small`.
  - One series per zone (Z01–Z08) and per gate (G01–G04); 15-min grid; forecast horizon default 180 min (12 steps),
    configurable up to 24 h; `peak_footfall` = max of median forecast within the horizon (and its timestamp in details).
  - Convert per-15-min counts to persons/hr (×4).
  - Backtest: rolling origin over the last 7 synthetic days, report MAE, MAPE, pinball loss, 80% interval coverage.
- **Inputs:** D01 `footfall_15min`, D21 `event_calendar`, D12 `weather_hourly` (resampled to 15 min).
- **Upstream:** M21 (optional: heat/rain arrival multipliers in `details`).
- **Scenarios:** S02 ↑ (~+30%), S03 ↓ arrivals during rain window, S04 slight ↓ midday, S05 redistribution between gates
  (total ≈ unchanged, G02 → 0). Apply via the synthetic world for that scenario when available, else the generic
  scenario layer (`twin_common.scenarios.apply_adjustments`).
- **Tests:** contract; lower ≤ value ≤ upper; S02 total expected_footfall over horizon is 1.2–1.4× S01; S05 G02 = 0;
  determinism; backtest metrics file written to `data/derived/backtest.json`.
- **Real-data swap:** replace D01 with gate counter/CCTV aggregates; rerun backtest; optionally fine-tune
  (Darts `enable_finetuning`) once ≥ several weeks of real history exist.

## M02 — Crowd Density & Flow
- **Question:** How many people are in each zone right now, how dense, and how are they moving?
- **Engine / phase:** `engines.vision` · Phase 6 · Critical
- **KPIs:** `zone_population`, `crowd_density`, `density_trend`, `inflow_rate`, `outflow_rate`, `inflow_outflow_ratio`,
  `queue_length`, `queue_wait_time`, `pedestrian_speed`, `zone_capacity_utilization`
- **Method (two pipelines, chosen per camera in `camera_registry.csv → mode`):**
  - `sparse` (gates, queues): RF-DETR (`rfdetr`, COCO person class) → `trackers.ByteTrackTracker` →
    `supervision.LineZone` (in/out counts) and `supervision.PolygonZone` (occupancy). Speed from track displacement
    converted to metres via homography.
  - `dense` (ghats, plazas): `lwcc` DM-Count with QNRF weights, `resize_img=False`, count inside ROI polygon
    (crop to ROI bounding box and mask). Density = count / ROI ground area (m²).
  - Mean speed and direction in dense ROIs from OpenCV Farneback optical flow (shared with M10).
  - Camera geometry: `scripts/calibrate_camera.py` (click 4 ground points with known distances → `homography.json`).
    Fallback: `roi_ground_area_m2` and `metres_per_pixel` from camera_registry.
  - Processing is OFFLINE precompute (`python -m src.run precompute --video …`) sampling `params.sample_fps` (default 1),
    writing `data/derived/vision_timeseries.parquet` (15 s and 15 min aggregates per camera and zone).
    `/predict` serves the precomputed series aligned to `as_of` (loop the clip over the demo day).
  - Zone aggregation: zone_population = sum over cameras mapped to the zone × `coverage_scale` (param, <1 if partial
    coverage); zones with no camera → population from synthetic world with `details.source = "synthetic_fallback"`.
  - `density_trend` = slope of density over the last `params.trend_window_min`; `queue_wait_time` = queue persons /
    gate throughput (persons/min, assumptions); `zone_capacity_utilization` = population / safe_capacity × 100.
- **Inputs:** videos in `data/video/` with `SOURCES.md` (license + origin), `camera_registry.csv`, `zones.geojson`, D02 schema.
- **Upstream:** none.
- **Scenarios:** vision reflects observed video only; for S02/S05 return synthetic-world zone values with
  `state: scenario` and `details.source = "synthetic_world"`.
- **Tests:** synthetic test video (OpenCV-drawn moving discs, known counts and line crossings) → crossing count exact in
  simple case; density = population/area; utilization formula; precompute → serve path; CPU mode works.
- **Real-data swap:** point `source_uri` to RTSP/recorded real camera; calibrate homographies; annotate 100–300 frames
  (head points) to measure MAE; fine-tune the dense counter if error is high.

## M03 — Hotspot / Crush Risk
- **Question:** Which zones are becoming dangerous, and why?
- **Engine / phase:** `engines.rules` · Phase 5 · Critical
- **KPIs:** `crush_risk_score` (score_0_100), `crowd_spillover_risk` (%), `bottleneck_probability` (%)
- **Method:**
  - Sub-scores clipped to [0,1] (all breakpoints are params):
    `s_density = (d − d_amber)/(d_critical − d_amber)`; `s_trend = trend / trend_max`;
    `s_accum = (ratio − 1)/(ratio_max − 1)`; `s_util = (u − u_start)/(1 − u_start)`;
    `s_heat = (HI − hi_start)/(hi_max − hi_start)`.
  - `crush_risk_score = 100 × Σ wᵢ·sᵢ` (weights sum to 1).
  - Override: sustained exposure (density > `exposure.density_p_m2` for ≥ `exposure.minutes`) → score = max(score, 80)
    and reason `SUSTAINED_EXPOSURE`.
  - `crowd_spillover_risk = 100 × sigmoid(k₁(u − u₀) + k₂(ratio − 1))` for zones with adjacent zones below capacity
    reduced by neighbour spare capacity share.
  - `bottleneck_probability` for gates/exits/bridge: `100 × sigmoid(k(demand_flow / capacity_flow − 0.9))`.
  - Current state from M02 (density, trend, in/out); forecast state from M01 population ÷ usable area, trend from
    forecast differences.
  - Reason codes: DENSITY_HIGH, DENSITY_RISING_FAST, ACCUMULATION, NEAR_CAPACITY, HEAT_STRESS, SUSTAINED_EXPOSURE,
    EXIT_SATURATED.
- **Inputs:** `zones.geojson` (usable area, safe capacity, adjacency), exits/gates capacities.
- **Upstream:** M02 (current), M01 (forecast), M21 (heat index).
- **Scenarios:** S02 ↑, S04 ↑ (heat), S05 ↑ for zones adjacent to G02, S12 ↑ Z06/Z02, S15 ↑ Z03.
- **Tests:** monotonic in density; exposure override; reason codes present when amber+; bands from risk_bands.yaml.
- **Real-data swap:** re-tune breakpoints and weights with crowd-safety experts; later train logistic regression/GBM on
  logged near-miss events and compare against the rules.

## M04 — Traffic Forecast
- **Question:** Where and when will roads congest, and how long will trips take?
- **Engine / phase:** `engines.network` · Phase 7 · High/Critical
- **KPIs:** `vehicle_volume`, `traffic_density`, `average_speed`, `travel_time`, `delay`, `vehicle_queue_length`,
  `volume_capacity_ratio`, `intersection_saturation`, `route_congestion_index`
- **Method (demo = macroscopic network model, pure Python):**
  - Road graph: OSMnx v2 drive network around venue (cached GraphML, D24); fallback synthetic grid graph.
  - Capacity per edge = lanes × `lane_capacity_veh_hr` (assumptions); free-flow time from length / speed limit.
  - Demand per 15-min step: background profile per edge class + event trips = M01 arrivals × car/bus mode share ÷
    occupancy, from city entry nodes (`entry_nodes` in world.yaml) to parking sites; assign with incremental
    all-or-nothing shortest paths (`params.assignment_iterations`).
  - BPR travel time: `t = t0 × (1 + α (V/C)^β)`; `delay = t − t0`; `average_speed = length / t`;
    `traffic_density = V / speed`; queue when V > C: `(V − C) × Δt × vehicle_length_m`;
    `intersection_saturation = Σ approach volume / node capacity × 100`;
    `route_congestion_index = 100 × clip(w₁·(V/C) + w₂·(delay/t0))`.
  - Report for key links R01–R10, key intersections J01–J06 and routes listed in world.yaml.
  - Optional experimental add-on (Linux/WSL only): SUMO corridor simulation via `eclipse-sumo`/`traci`; never required.
- **Inputs:** D24 road graph + `key_links.csv`, parking sites, D21 calendar.
- **Upstream:** M01.
- **Scenarios:** S02 ↑ volumes; S06 remove R07 → alternates ↑ travel_time; S07 temporary closure of VIP1 corridor;
  S12 remove B01 → rerouting.
- **Tests:** V/C increases with demand; closure increases travel time on at least one alternate; no negative/NaN
  speeds; graph fallback works offline.
- **Real-data swap:** calibrate capacities and BPR parameters with loop/CCTV counts and probe speeds; later forecast
  per-edge speed series with Chronos-2 or a spatio-temporal model.

## M05 — Parking Demand
- **Question:** How full will each parking site be, and when does it overflow?
- **Engine / phase:** `engines.forecast` · Phase 4 · High
- **KPIs:** `parking_occupancy` (%), `parking_search_time` (min), `parking_demand` (vehicles, extension)
- **Method:** forecast occupied spaces per site (Chronos-2, covariates: M01 forecast arrivals, hour, is_snan_day);
  occupancy = occupied / capacity × 100; search time = `base_min + k × max(0, occ − occ_knee)/(1 − occ_knee)`;
  overflow flag when occupancy ≥ `overflow_pct` → reason `PARKING_OVERFLOW`, recommend shuttle diversion.
- **Inputs:** D06 `parking_15min`, parking capacities. **Upstream:** M01.
- **Scenarios:** S02 ↑; S06/S12 may shift demand between sites.
- **Tests:** occupancy within [0, 100+overflow]; S02 ↑; search time monotonic in occupancy.
- **Real-data swap:** real entry/exit counts; reconcile occupancy drift daily.

## M06 — Transit / Shuttle Demand
- **Question:** How many passengers and shuttles are needed per route?
- **Engine / phase:** `engines.forecast` · Phase 4 · High
- **KPIs:** `shuttle_demand` (persons/hr), `shuttle_requirement` (vehicles, ext.), `passenger_wait_time` (min, ext.)
- **Method:** forecast boardings per route (Chronos-2, covariates M01 arrivals, M05 overflow); buses =
  `ceil(demand × round_trip_min/60 ÷ (bus_capacity × load_factor))`; wait ≈ headway/2 with headway = round_trip / buses
  (capped by available fleet from resources.yaml); `recommendation` when required > available.
- **Inputs:** D07 `transit_15min`, routes in world.yaml. **Upstream:** M01, M05.
- **Scenarios:** S02 ↑; S06/S12 ↑ round trip time via M04 details if present.
- **Tests:** requirement integer ≥ 0; S02 ↑; wait decreases when fleet increases.
- **Real-data swap:** real boardings/GPS; update round-trip times from observed trips.

## M07 — Medical Demand
- **Question:** How many medical cases will occur, where, and will hospitals cope?
- **Engine / phase:** `engines.formula` · Phase 5 · Critical
- **KPIs:** `expected_medical_cases` (cases/hr), `peak_medical_demand` (cases/hr), `critical_case_probability` (%),
  `medicine_consumption` (units/day), `hospital_load` (%), `icu_bed_requirement` (beds)
- **Method:**
  - Rate λ(zone,t) = population × base_rate × f_heat × f_humidity × f_density, where
    base_rate derives from `presentations_per_1000_attendees_per_day` (literature order of magnitude ≈ 1 per 1,000),
    `f_heat = 1 + heat_rate_increase_per_c × max(0, HI − heat_index_threshold_c)`, `f_density` piecewise from density bands.
  - Bands from Poisson quantiles (scipy). `peak_medical_demand` = max upper quantile in horizon.
  - If synthetic history D08 exists: fit statsmodels GLM (Poisson; switch to Negative Binomial if dispersion > param)
    with log(population) offset and covariates heat_index, density, hour; use fitted model when it beats the formula
    on holdout deviance, and record both in model_card.
  - `critical_case_probability = 100 × P(N_severe ≥ severe_threshold)` with N_severe ~ Poisson(λ × severe_share).
  - `hospital_load = (baseline_occupied + transfers)/beds × 100`, transfers = cases × hospital_transfer_share;
    `icu_bed_requirement = ceil(transfers × icu_share_of_transfers)`; `medicine_consumption = cases/day × units_per_case`.
- **Inputs:** D08 `medical_incidents`, hospital capacities. **Upstream:** M01 (population forecast), M21 (heat index, humidity).
- **Scenarios:** S02 ↑, S04 ↑, S14 ×2 (±5%), S15 ↑ Z03 and hospital_load.
- **Tests:** λ ≥ 0; S14 doubles expected cases; heat monotonic; hospital_load uses correct beds.
- **Real-data swap:** fit GLM on real medical logs (restricted data, aggregated); recalibrate transfer and ICU shares.

## M08 — Ambulance & First-Aid Staging
- **Question:** How many ambulances and first-aid posts are needed, and where should they be staged?
- **Engine / phase:** `engines.location` (spopt) · Phase 7 · Critical
- **KPIs:** `ambulance_requirement` (vehicles), `ambulance_response_time` (min), `first_aid_post_requirement` (posts)
- **Method:**
  - Requirement = `ceil(λ_transport × ambulance_cycle_time_min/60 ÷ ambulance_target_utilization)`.
  - Staging: spopt `MCLP` — demand points = zone centroids weighted by M07 cases; candidate sites = staging points in
    world.yaml; cost matrix = network travel time from M04 congested times (fallback: free-flow OSM times);
    service radius = response SLA minutes; facilities p = available ambulances (resources.yaml × scenario multiplier).
  - Response time per zone = travel time from assigned site + dispatch delay; report 90th percentile in details.
  - First-aid posts: spopt `LSCP` (or MCLP with capacity check) using walking-distance radius and cases/hr capacity.
  - Recommendations: move/stage N ambulances at site X (`requires_approval: true`).
- **Inputs:** staging candidates, D09 schema for sample dispatch history. **Upstream:** M07, M04.
- **Scenarios:** S10 −20% fleet → coverage ↓, response ↑; S06/S12 → response ↑; S14 → requirement ↑.
- **Tests:** solver status optimal; coverage non-increasing when fleet reduced; response ≥ dispatch delay.
- **Real-data swap:** CAD/GPS dispatch history to calibrate cycle time and travel times; hypercube/queueing later.

## M09 — Security Incident Risk
- **Question:** Where is a security incident or lost-person case more likely, and can teams respond in time?
- **Engine / phase:** `engines.rules` · Phase 5 · Critical/High
- **KPIs:** `security_incident_rate` (incidents/hr), `security_risk_score` (score_0_100),
  `predicted_incident_probability` (%), `unauthorized_entry_count` (incidents/hr), `lost_person_count` (cases/hr),
  `security_response_time` (min)
- **Method:** Poisson rates: λ_inc = base × (population/100k) × f_density × f_hour × f_event; probability over horizon
  = 100 × (1 − exp(−λ·Δt)); lost persons λ similarly with its own base; unauthorized entries from D11 synthetic rate
  per secure gate; response time = walking distance from nearest police post / patrol speed + dispatch delay (police
  availability multiplier from scenario); risk score = weighted(incident prob, density sub-score, CCTV coverage gap from
  M23, sensitive-site flag, response-time breach). Reason codes: CROWD_DENSITY, CCTV_BLIND_SPOT, RESPONSE_SLA_BREACH,
  SENSITIVE_SITE, ACCESS_VIOLATIONS.
- **Inputs:** D10 `security_incidents`, D11 `access_control`, police posts. **Upstream:** M01, M02, M23.
- **Scenarios:** S09 ↑ risk (coverage gap), S11 ↑ response time, S02 ↑ incidents, S07 ↑ at VIP corridor zones.
- **Tests:** probabilities 0–100; S09 and S11 directions; reason codes.
- **Real-data swap:** fit rates on real incident logs (taxonomy normalized); later classification model with
  calibration.

## M10 — CCTV Anomaly
- **Question:** Is crowd motion on a camera abnormal (surge, counterflow, sudden dispersal, stalled build-up)?
- **Engine / phase:** `engines.vision` (optical flow) · Phase 6 · Critical
- **KPIs:** `anomaly_score` (score_0_100, ext.), `counterflow_risk` (%)
- **Method:** Farneback optical flow in each ROI at `sample_fps`; features per window: mean magnitude, magnitude z-score
  vs rolling baseline, direction entropy, share of vectors opposing dominant direction, divergence; rules (params):
  SURGE (z-score > z_surge), COUNTERFLOW (opposing share > c_counter), SUDDEN_DISPERSAL (divergence > d_disp and
  magnitude spike), STALLED_BUILDUP (magnitude low while M02 density rising). `anomaly_score = 100 × max rule strength`;
  `counterflow_risk = 100 × sigmoid(k(opposing_share − c₀))`. Events carry camera_id, zone_id, start/end in details.
  Precompute offline like M02.
- **Inputs:** videos + camera_registry; optional `data/labels/events.csv` (human-labelled time ranges).
- **Upstream:** none (optionally reads M02 derived density for STALLED_BUILDUP when present).
- **Scenarios:** observed-video model; S09 → cameras offline flagged with `status: degraded` for affected zones.
- **Tests:** synthetic video with scripted counterflow and surge segments → correct classes in correct windows;
  precision/recall computed against labels when labels exist.
- **Real-data swap:** tune thresholds per camera on real footage; later weakly-supervised VAD or VLM narration.

## M11 — Evacuation Simulation
- **Question:** How fast can a zone be cleared, where are the bottlenecks, and is shelter capacity enough?
- **Engine / phase:** `engines.pedsim` (JuPedSim) · Phase 8 · Critical
- **KPIs:** `evacuation_population` (persons), `evacuation_time` (min), `exit_capacity` (persons/min),
  `exit_utilization` (%), `blocked_route_count` (routes), `safe_shelter_capacity` (%), `crowd_dispersion_time` (min)
- **Method:**
  - Walkable area = zone polygon(s) minus obstacles, projected to metric CRS (pyproj/shapely, valid polygons only);
    exits as JuPedSim exit stages; `CollisionFreeSpeedModel`; agents spawned from population (M01 at scenario time)
    up to `max_agents` (param); record per-exit flow and time of last agent.
  - Flow-model fallback and cross-check: `T = premovement + max_e (N_e / (specific_flow × width_e))`, N_e assigned by
    nearest exit; used when population > max_agents (report `details.method = "flow_model"`) or JuPedSim unavailable.
  - `crowd_dispersion_time`: flow model time for density to fall below `density_bands.amber` after opening exits.
  - `safe_shelter_capacity = available shelter capacity / displaced persons × 100`.
  - Runs are cached by (zone, scenario, population bucket) in `data/derived/`; `/scenario` returns cached or runs if
    under `max_runtime_s`.
- **Inputs:** zones.geojson, exits (width, status), shelters. **Upstream:** M01.
- **Scenarios:** S15 fire in Z03 with E03 blocked → evacuation_time ↑ and blocked_route_count ≥ 1; S12 bridge closed
  → Z06/Z02 ↑; S02 ↑.
- **Tests:** blocking an exit increases time; doubling agents increases time; sim and flow model within factor
  `params.crosscheck_factor` on a simple rectangle; runtime cap respected.
- **Real-data swap:** calibrate speeds/specific flows with observed trajectories; validate against drills.

## M12 — Fire Risk
- **Question:** Where is fire risk high, and can fire units respond in time?
- **Engine / phase:** `engines.rules` · Phase 5 · Critical
- **KPIs:** `fire_risk_score` (score_0_100), `fire_response_time` (min), `fire_vehicle_requirement` (vehicles, ext.)
- **Method:** weighted sub-scores: cooking/stall density (world.yaml zone attributes), electrical load utilization
  (M19 details), heat index and low humidity (M21), wind speed, tent density, generator operation (S08);
  response time = travel time from nearest fire station (M04 congested or free-flow fallback) + turnout;
  vehicles = `ceil(Σ risk-weighted demand ÷ unit_coverage_capacity)`. Reason codes: ELECTRICAL_OVERLOAD, HEAT_DRY,
  COOKING_DENSITY, GENERATOR_OPERATION, ACCESS_DELAY.
- **Inputs:** zone attributes, fire stations. **Upstream:** M19, M21, M04.
- **Scenarios:** S04 ↑, S08 ↑, S06/S12 ↑ response time, S15 Z03 critical.
- **Tests:** monotonic sub-scores; S04 and S08 directions.
- **Real-data swap:** fire incident history and electrical inspection data to re-weight.

## M13 — VIP Route Optimizer
- **Question:** Which route and departure time minimize conflict with public flows?
- **Engine / phase:** `engines.network` · Phase 7 · High
- **KPIs:** `vip_route_clearance_time` (min), `vip_route_conflict_probability` (%),
  `vip_recommended_departure_offset_min` (min, ext.)
- **Method:** candidate routes = NetworkX k-shortest simple paths (k param) between VIP origin/destination (world.yaml)
  on the M04 graph; edge cost = congested travel time × (1 + λ × crowd_exposure), exposure from densities of zones
  intersecting an edge buffer (M01 forecast); evaluate departure times every 15 min in a window;
  conflict probability = 100 × sigmoid(k(exposure_max − e₀)); clearance time = length_km × clearance_min_per_km +
  setup_min. Output best route (GeoJSON in details) and alternatives ranked.
- **Inputs:** VIP OD pairs and window. **Upstream:** M04, M01.
- **Scenarios:** S07 (the VIP event itself), S02 ↑ conflict, S06/S12 constrain routes.
- **Tests:** chosen route has minimum score among candidates; closing a link removes routes using it.
- **Real-data swap:** security protocol constraints and real closure logs.

## M14 — Route Diversion Optimizer
- **Question:** Which diversion plan minimizes disruption while keeping emergency corridors open?
- **Engine / phase:** `engines.network` · Phase 7 · High/Critical
- **KPIs:** `diversion_delay` (min), `route_diversion_effectiveness` (%), `alternate_route_capacity` (vehicles/hr),
  `emergency_route_availability` (%), `emergency_access_availability` (%)
- **Method:** for a closure set (scenario), re-run the shared BPR assignment from `engines.network`; evaluate candidate
  diversion plans (turn restrictions / preferred alternates listed in world.yaml, plus automatic k-shortest alternates);
  choose plan minimizing total vehicle delay subject to emergency corridors (world.yaml) keeping V/C ≤ limit;
  `diversion_delay = scenario travel time − baseline`; `effectiveness = (delay_no_plan − delay_plan)/delay_no_plan × 100`;
  `alternate_route_capacity = capacity − demand`; emergency availability = usable/required corridor capacity × 100.
- **Inputs:** closures, corridors, alternates. **Upstream:** M04.
- **Scenarios:** S06, S07, S12 (primary); S03 waterlogged links from M21 details optional.
- **Tests:** effectiveness ≥ 0 for chosen plan; emergency corridor constraint respected; closure → delay > 0.
- **Real-data swap:** calibrated M04; police-approved diversion catalogue.

## M15 — Water Demand
- **Question:** How much water is needed by zone, is there a gap, and how many points/tankers?
- **Engine / phase:** `engines.forecast` · Phase 4 · High/Critical
- **KPIs:** `expected_water_demand` (L/hr), `water_supply_demand_gap` (L/hr), `drinking_water_point_requirement`
  (points), `water_tanker_requirement` (vehicles, ext.)
- **Method:** forecast consumption (Chronos-2, covariates population forecast, temperature, hour); supply from world
  assets (pump/tank capacities, S08 affects pumps); gap = demand − supply; points = `ceil(peak population ÷
  persons_per_water_point)`; tankers = `ceil(max(gap,0) × hours ÷ tanker_payload_l)`.
- **Inputs:** D14 `water_15min`, water assets. **Upstream:** M01, M21.
- **Scenarios:** S02 ↑, S04 ↑, S08 ↑ gap.
- **Tests:** gap sign; S04 demand ↑; integer requirements.
- **Real-data swap:** meter data; leak reconciliation; real pump curves.

## M16 — Toilets & Sanitation
- **Question:** How many toilets, where, how often cleaned, by how many staff?
- **Engine / phase:** `engines.optimize` (PuLP) · Phase 7 · High
- **KPIs:** `mobile_toilet_requirement` (units), `toilet_utilization` (%), `cleaning_frequency` (cycles/day),
  `sewage_generation` (MLD), `cleaning_staff_requirement` (personnel, ext.)
- **Method:** demand uses/hr per zone = population × uses_per_person_per_hour_present; requirement per zone =
  `ceil(peak uses/hr ÷ users_per_toilet_per_hour)`; placement MILP: integer units per candidate cluster site
  (world.yaml), minimize total units + λ × walking distance, s.t. each zone's demand covered by clusters within walking
  radius, site capacity limits, total ≤ available (S-aware); utilization = uses / (units × users_per_toilet_per_hour) ×
  100; cycles/day = uses/day ÷ uses_per_cleaning; staff = cycles × cleaning_minutes ÷ shift_minutes; sewage = uses ×
  litres_per_use ÷ 1e6 per day.
- **Inputs:** D15 `toilet_usage`, candidate sites. **Upstream:** M01.
- **Scenarios:** S02 ↑; S03 may disable low-lying sites (flag in world.yaml).
- **Tests:** solver optimal; coverage constraint satisfied; S02 requirement ↑.
- **Real-data swap:** usage/inspection logs; field-validated site capacities.

## M17 — Food & Supply
- **Question:** How much food is needed, how long does stock last, how many deliveries?
- **Engine / phase:** `engines.forecast` · Phase 4 · Medium/High
- **KPIs:** `food_demand` (meals/hr), `food_stock_coverage` (hours), `delivery_requirement` (trips/day)
- **Method:** forecast meals per outlet (Chronos-2, covariates market-zone population, hour); inventory simulation per
  depot/outlet with lead time (S13 multiplies lead time); coverage = stock ÷ forecast consumption rate;
  trips = `ceil(replenishment_meals ÷ vehicle_payload_meals)`; reason `STOCKOUT_RISK` when coverage < min cover.
- **Inputs:** D17 `food_inventory_hourly`. **Upstream:** M01.
- **Scenarios:** S02 ↑ demand; S13 ↓ coverage / stockout risk; S06 ↑ lead time optional.
- **Tests:** coverage non-negative; S13 direction.
- **Real-data swap:** POS and inventory feeds; supplier lead times.

## M18 — Waste Generation
- **Question:** How much waste, when do bins overflow, how many collection trips?
- **Engine / phase:** `engines.forecast` · Phase 4 · High
- **KPIs:** `waste_generation` (kg/day), `waste_bin_fill_level` (%), `waste_collection_trips` (trips/day, ext.),
  `bin_overflow_time` (min, ext.)
- **Method:** forecast kg per zone (Chronos-2, covariates population, food demand if available); bin fill projection =
  current fill + inflow × t until collection; overflow time when fill ≥ 100%; trips = `ceil(kg/day ÷ vehicle_payload_kg)`.
- **Inputs:** D16 `waste`. **Upstream:** M01.
- **Scenarios:** S02 ↑; S06/S12 delay collection (longer overflow risk).
- **Tests:** fill within [0, 100] after collection reset; S02 ↑.
- **Real-data swap:** weighbridge and bin sensor data.

## M19 — Power Load
- **Question:** What electrical load is expected, and how long can backup last?
- **Engine / phase:** `engines.forecast` · Phase 4 · Critical
- **KPIs:** `electricity_demand` (MW), `generator_backup_duration` (hours), `backup_generator_requirement` (units, ext.)
- **Method:** forecast kW per zone/facility (Chronos-2, covariates temperature, population, hour); backup duration =
  fuel_l ÷ (l_per_hr_at_full_load × load_fraction); generators = `ceil(critical_load_kw × redundancy_factor ÷
  generator_unit_kw)`; `details.load_utilization` for M12.
- **Inputs:** D18 `power_15min`, power assets. **Upstream:** M01, M21.
- **Scenarios:** S04 ↑ load; S08 substation SS01 down → affected facilities on backup, duration countdown, reason
  `GRID_OUTAGE`.
- **Tests:** MW conversion; S08 → backup duration finite and decreasing with load.
- **Real-data swap:** smart meter/BMS telemetry; generator telemetry.

## M20 — Network Capacity
- **Question:** Will communications (CCTV backhaul, command links) overload or fail?
- **Engine / phase:** `engines.forecast` · Phase 4 · Critical/High
- **KPIs:** `bandwidth_utilization` (%), `network_availability` (%), `network_capacity_risk` (%, ext.)
- **Method:** demand Mbps per tower = Σ active camera bitrate + population × active_user_share × kbps/1000;
  forecast utilization (Chronos-2, covariates population, cameras active); availability from synthetic outage events;
  risk = 100 × sigmoid(k(util − util₀)).
- **Inputs:** D19 `network_5min`, camera_registry, towers. **Upstream:** M01.
- **Scenarios:** S02 ↑; S08 ↓ availability for towers on SS01; S09 ↓ camera load but availability flags.
- **Tests:** utilization formula; S08 direction.
- **Real-data swap:** NMS telemetry.

## M21 — Weather Impact  (first model built; reference for template)
- **Question:** What is the weather now/next hours, and how does it change risk elsewhere?
- **Engine / phase:** `engines.formula` · Phase 3 · Critical
- **KPIs:** `temperature` (°C), `heat_index` (°C), `rainfall_intensity` (mm/hr), `waterlogging_probability` (%),
  `weather_arrival_multiplier`, `weather_medical_multiplier`, `weather_traffic_speed_multiplier` (ratio, ext.)
- **Method:** hourly weather from cached Open-Meteo historical data for the venue (same calendar dates of
  `weather_source_year`, shifted to event year), interpolated to 15 min; heat index = NOAA algorithm (Rothfusz
  regression with NOAA adjustments, computed in °F then converted); waterlogging probability per low-lying zone/link =
  100 × sigmoid(a × (rain_mm_hr − drainage_capacity_mm_hr) + b × low_lying_flag); multipliers:
  arrival = 1 − k_rain_arrival × min(rain, rain_cap)/rain_cap; medical = 1 + k_heat × max(0, HI − hi₀);
  traffic speed = 1 − k_rain_speed × min(rain, rain_cap)/rain_cap. "Forecast" for demo = replay of the stored series
  (optional live Open-Meteo forecast behind a flag).
- **Inputs:** D12 `weather_hourly`, zone low-lying flags and drainage capacity. **Upstream:** none.
- **Scenarios:** S03 rain 50 mm/hr in window → waterlogging ↑, multipliers change; S04 temperature offset → heat_index ↑.
- **Tests:** heat index matches NOAA reference values within tolerance (table of 5 known points written from the NOAA
  heat index chart during implementation); S03/S04 directions; offline cached path.
- **Real-data swap:** IMD/licensed forecast feed; site rain gauges; hydraulic model (PySWMM) for waterlogging later.

## M22 — Environmental Risk
- **Question:** Where will air quality or noise stress increase, and what are emissions?
- **Engine / phase:** `engines.forecast` · Phase 4 · High/Medium
- **KPIs:** `air_quality_index` (index), `pm25` (µg/m³), `noise_level` (dB), `co2_emissions` (tCO2e/day)
- **Method:** background PM2.5/PM10/NO2/CO from cached Open-Meteo air-quality history; local increments from traffic
  (M04 vehicle-km × factor) and crowd density; forecast pm25 with Chronos-2 (covariates traffic volume, wind, hour);
  AQI = Indian National AQI sub-index method with breakpoints in config (verify against CPCB before real use);
  noise = base_db + 10·log10(1 + density/density_ref) (+ traffic term); CO2 = vehicle-km × EF + kWh × grid factor +
  generator litres × diesel factor (all factors params, placeholders).
- **Inputs:** D13 `air_quality_hourly`. **Upstream:** M21 (wind), M04 (vehicle volumes).
- **Scenarios:** S02 ↑ noise/CO2; S08 ↑ CO2 (generators); S15 ↑ PM2.5 near Z03.
- **Tests:** AQI sub-index piecewise-linear correctness on breakpoints; directions.
- **Real-data swap:** calibrated AQ and noise sensors; official emission factors.

## M23 — Asset Failure
- **Question:** Which assets may fail, and what is current asset and camera availability?
- **Engine / phase:** `engines.mlclf` · Phase 5 · Critical/High
- **KPIs:** `asset_failure_probability` (%, ext.), `asset_availability` (%), `asset_capacity_utilization` (%),
  `camera_availability` (%), `cctv_coverage` (%)
- **Method:** LightGBM classifier trained on AI4I 2020 (UCI id 601, CC BY 4.0; download once via `ucimlrepo`, cache;
  offline fallback: regenerate AI4I-like data using its published failure-mode rules) with
  `CalibratedClassifierCV`; event assets (generators, pumps, transformers) get synthetic telemetry mapped to the same
  feature semantics (ambient temp, process temp, speed, torque-like load, wear hours) — clearly documented as a
  stand-in; availability = healthy/total × 100; utilization = actual/design × 100; camera availability from
  camera_registry status + synthetic outages; cctv_coverage = area of critical zones covered by camera FOV polygons ÷
  critical area × 100 (shapely).
- **Inputs:** D20 `asset_maintenance`, asset registry, camera_registry. **Upstream:** none.
- **Scenarios:** S08 ↓ availability (substation assets), S09 camera availability −20% and coverage ↓.
- **Tests:** held-out AUC reported; probabilities 0–100; S09 direction.
- **Real-data swap:** CMMS work orders + IoT telemetry; retrain on real failures (survival model later).

## M24 — Resource Requirement
- **Question:** How many of each resource are needed per zone and time, what is the gap, and how to allocate?
- **Engine / phase:** `engines.optimize` (PuLP) · Phase 7 · Critical
- **KPIs (per resource_type):** `resource_requirement` (units), `resource_gap` (units), `resource_coverage` (%),
  `response_capacity` (teams)
- **Resource types (15):** police, crpf, ambulance, medical_team, fire_vehicle, mobile_toilet, cleaning_staff,
  water_point, water_tanker, shuttle_bus, traffic_police, waste_vehicle, food_vehicle, backup_generator, network_capacity.
- **Method:**
  - Requirements taken from upstream where they exist: ambulance (M08), fire_vehicle (M12), mobile_toilet &
    cleaning_staff (M16), water_point & water_tanker (M15), shuttle_bus (M06), waste_vehicle (M18), food_vehicle (M17),
    backup_generator (M19), network_capacity (M20); own formulas: police = population/1000 × police_per_1000 ×
    risk_multiplier(M03/M09 level); crpf = high-risk zones × crpf_per_high_risk_zone; traffic_police = key intersections
    at peak × per_intersection; medical_team = expected cases / medical_team_cases_per_hr.
  - Allocation MILP per shift: variables units(type, zone) integer; minimize Σ risk_weight(zone) × unmet(type, zone) +
    μ × relocation distance; s.t. Σ zones units ≤ available(type) × scenario multiplier; units + unmet ≥ required;
    per-zone max capacity.
  - gap = required − available (positive = shortage); coverage = allocated/required × 100;
    response_capacity = teams whose travel time to zone ≤ SLA.
  - Recommendations: add/move quantity per type and zone, `requires_approval: true`, rationale with upstream reasons.
- **Upstream:** M01, M03, M06, M07, M08, M09, M12, M15, M16, M17, M18, M19, M20 (no M25 — avoids a cycle).
- **Scenarios:** S02 ↑ gaps; S10/S11 availability ↓ → coverage ↓; S14 medical ↑; S15 Z03 priority ↑.
- **Tests:** gap arithmetic exact; allocations never exceed availability; higher-risk zones served first under shortage.
- **Real-data swap:** live rosters (D22); agreed service ratios per SOP.

## M25 — Overall Event Risk
- **Question:** What is the overall risk now/next hour, what drives it, and how confident are we?
- **Engine / phase:** `engines.rules` · Phase 9 · Critical/High
- **KPIs:** `overall_event_risk` (score_0_100), `decision_confidence` (%)
- **Method:** normalize domain risks to [0,1]: crowd (M03 crush_risk_score), medical (M07 hospital_load and
  critical_case_probability), security (M09 security_risk_score), fire (M12), mobility/emergency access (M04
  route_congestion_index, M14 emergency availability shortfall), utilities (M19 backup shortfall, M20 capacity risk),
  weather (M21 heat/waterlogging), assets (M23 availability shortfall), resources (M24 coverage shortfall);
  `overall = 100 × Σ w_d r_d` (weights params) per zone and for EVENT; floor rules: any domain critical → overall ≥ 80,
  any red → ≥ 60. `decision_confidence = 100 × freshness × (1 − mean normalized band width) × upstream_share`,
  labelled "heuristic, not calibrated" in model_card. Reason codes = top 3 domain contributors
  (e.g. CROWD_RISK, MEDICAL_LOAD, RESOURCE_SHORTFALL). `/scenario` returns baseline vs scenario deltas per domain in
  details.
- **Upstream:** M03, M04, M07, M09, M12, M14, M19, M20, M21, M23, M24.
- **Scenarios:** all; S15 and S14 must be ≥ S01; S01 lowest.
- **Tests:** floors applied; weights sum to 1; contributors sorted; S15 > S01.
- **Real-data swap:** calibrate weights against expert rankings of historical situations; add reliability curves.
