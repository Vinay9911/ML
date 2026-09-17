# 01 — Project Brief

## 1. Context
A Digital Twin Command Center for a mega religious gathering (Kumbh Mela scale: tens of millions of
visitors, peak "bathing days", river ghats, temporary city). The twin is a live virtual copy of the event:
zones, gates, exits, bridges, roads, ghats, hospitals, medical posts, police posts, toilets, water points,
waste points, power and network assets, weather — plus the people, vehicles and resources moving through them.

Operating loop: **Observe → Predict → Simulate → Optimize → Act (human approval) → Learn → Observe**.

## 2. Problem being solved
Crowd crushes, traffic gridlock that blocks ambulances, heat/rain-driven medical surges, flooding of low
roads, shortages of water/toilets/staff in specific sectors. These build up in minutes and are usually handled
reactively. The twin must:
1. Detect building risk early (15–60 min ahead).
2. Let operators test decisions safely ("what if we close Gate G02?").
3. Recommend resource quantity, location and timing, with reasons and confidence, for human approval.

## 3. What this repository delivers (demo stage)
- 25 independent model services (M01–M25), each a folder with an identical structure and API.
- `00_common/`: shared contracts, engines, synthetic world generator, configs.
- The frontend is NOT in scope. Another developer integrates the 25 services.
- No real operational data exists yet. The synthetic world must be realistic, internally consistent,
  scenario-aware, and replaceable by real data table-by-table later.

## 4. The 25 models at a glance
| ID | Model | Kind of work | Engine |
|---|---|---|---|
| M01 | Footfall forecast | Forecasting | forecast |
| M02 | Crowd density & flow | Computer vision | vision |
| M03 | Hotspot / crush risk | Risk rules | rules |
| M04 | Traffic forecast | Network model (BPR assignment) | network |
| M05 | Parking demand | Forecasting | forecast |
| M06 | Transit / shuttle demand | Forecast + formula | forecast |
| M07 | Medical demand | Statistical count model | formula |
| M08 | Ambulance & first-aid staging | Location optimization | location |
| M09 | Security incident risk | Risk rules + Poisson | rules |
| M10 | CCTV anomaly | Optical-flow rules | vision |
| M11 | Evacuation simulation | Pedestrian simulation | pedsim |
| M12 | Fire risk | Risk rules | rules |
| M13 | VIP route optimizer | Graph optimization | network |
| M14 | Route diversion optimizer | Graph optimization | network |
| M15 | Water demand | Forecast + formula | forecast |
| M16 | Toilets & sanitation | Integer optimization | optimize |
| M17 | Food & supply | Forecast + inventory | forecast |
| M18 | Waste generation | Forecast + formula | forecast |
| M19 | Power load | Forecast + formula | forecast |
| M20 | Network capacity | Forecast + formula | forecast |
| M21 | Weather impact | Physics formulas + multipliers | formula |
| M22 | Environmental risk | Forecast + formula | forecast |
| M23 | Asset failure | Small ML classifier | mlclf |
| M24 | Resource requirement | Integer optimization | optimize |
| M25 | Overall event risk | Weighted aggregation | rules |

Only a few models are "true ML". Most are simulation, optimization, formulas or rules. That is intentional:
explainable, needs no labeled data, and is defensible to authorities.

## 5. Demo story (the 5 minutes everything must support)
1. 06:00 on a peak bathing day. Map shows zone densities; Ghat A (Z01) is amber and rising.
2. M01/M03 forecast Z01 reaching red within ~40 minutes; alert shows reason codes
   (DENSITY_RISING_FAST, ACCUMULATION, HEAT_STRESS).
3. Operator runs S05 "close Gate G02" vs baseline; M01/M03/M11 show redistribution and evacuation time.
4. M24 recommends +medical teams and +police for Z01/Z03 with quantities; each recommendation
   `requires_approval: true`.
5. Operator runs S02 + S04 ("30% more crowd + extreme heat"); M07 medical, M15 water, M16 toilets, M25 overall
   risk all change consistently because they share one synthetic world.
Every output is visibly labelled synthetic.

## 6. Out of scope for the demo
Frontend, authentication, databases (Parquet/GeoJSON files are enough), Kafka/streaming,
real-time camera ingestion at scale, face recognition or any personal identification, production MLOps.

## 7. Implementation readiness checklist (from the study)
1. Zone/asset hierarchy: event → venue → sector → zone → asset.
2. Baseline capacity model: safe crowd capacity, road capacity, medical capacity, service rates.
3. Historical data (multiple event cycles, normalized for event type and weather) — synthetic for now.
4. Real-time feeds (CCTV, IoT, traffic, medical, security, weather, utilities) — replay/synthetic for now.
5. Domain models (crowd, traffic, medical, security, evacuation, resources).
6. Twin state: current + forecast + scenario, linked to GIS.
7. Scenario validation: replay events and expert what-if tests.
8. Decision thresholds: risk levels, escalation, SOP triggers, confidence limits.
9. Dashboards: operational, management, GIS, resource, scenario views (frontend team).
10. Learning loop: forecast vs actual, retrain/recalibrate.

## 8. Known errors in the original study (do not copy)
- Summary says "KPI domains: 93" — there are 9 domains and 93 KPIs.
- Its domain summary omits "VIP & Route Diversion" (4 KPIs).
- "Emergency Route Availability" and "Emergency Access Availability" overlap; both are kept but owned by M14.

## 9. Glossary
- **Ghat**: riverbank steps/platform used for bathing; highest crowd density locations.
- **Amrit Snan / Shahi Snan**: principal bathing days with extreme peaks.
- **Zone**: smallest crowd management polygon (Z01…). **Sector**: group of zones.
- **ICCC**: Integrated Command & Control Centre.
- **CRPF**: Central Reserve Police Force (paramilitary).
- **SOP**: standard operating procedure triggered by thresholds.
- **Twin state**: current (observed), forecast (predicted), scenario (what-if) values per entity and time.
- **Stub**: contract-valid placeholder output generated from the synthetic world for a model not yet built.
