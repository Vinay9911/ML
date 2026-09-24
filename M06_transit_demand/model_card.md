# Model card — M06 Transit / Shuttle Demand

## Question / decision supported
**How many passengers and shuttles are needed per route?**

The decision is a dispatch one: how many vehicles to put on each shuttle route, and how long
riders will wait if you do not. M06 is the first model with **two upstreams** (M01 and M05),
so it is also the first real test of the dependency chain.

## Method
- **Engine:** `twin_common.engines.forecast`
- **Algorithm:** Darts `Chronos2Model` (zero-shot, `autogluon/chronos-2-small`) → `LightGBMModel`
  → `NaiveSeasonal`, with conformal band calibration
- **Library + version:** darts 0.47.0, lightgbm 4.7.0
- **Licence:** Darts Apache-2.0, LightGBM MIT; Chronos-2 weights per the autogluon model card

| Step | What happens |
|---|---|
| Forecast | boardings per route from D07, one series per route |
| Covariates | calendar (hour sin/cos, day of week, snan flag) + arrivals, future half from **M01** |
| Vehicles | `ceil(demand × round_trip_min/60 ÷ (bus_capacity × bus_load_factor))` |
| Deployment | the fleet is **shared** between routes in proportion to what each needs |
| Wait | `headway/2` with `headway = round_trip ÷ deployed`, stretched when the fleet cannot cope |

Every input to the fleet arithmetic comes from `assumptions.transport`, and the fleet size
from `resources.yaml` (`shuttle_bus.available`), so M06 and M24 cannot disagree about how
many vehicles exist.

### Why the wait formula has an extra term
The card gives `wait ≈ headway/2`, "capped by available fleet". Taken literally that has a
perverse property: cap a hopelessly oversubscribed route at the available fleet and the
computed wait gets **shorter**, because fewer buses on the road only ever shows up as
headway. What actually happens to a passenger is that full buses go past.

So the wait is multiplied by an overload factor — how many bus-loads of demand exist per
bus-load of capacity. A route carrying twice what it can hold reports roughly twice the
wait. Set `apply_overload_to_wait: false` to get the card's plain formula back.

This is a **first-order approximation**. When demand exceeds capacity for a sustained period
the real queue grows without bound and no steady-state wait exists; the factor is capped at
`max_overload_factor` so the number stays finite and readable. The route is flagged
`CONGESTION` either way.

## Inputs and features
| Source | Detail |
|---|---|
| Tables | `transit_15min` (D07, boardings history), `footfall_15min` (D01, covariate past), `event_calendar` (D21, snan flag) |
| Upstream | **M01** — forecast arrivals for the covariate's future half; **M05** — parking overflow, reported as context |
| Shared config | `transport.round_trip_min`, `transport.bus_capacity`, `transport.bus_load_factor`, `resources.shuttle_bus.available` |

## Assumptions
| Config key | Value | Meaning | Source |
|---|---|---|---|
| `transport.round_trip_min` | 40 | one loop, including dwell | `assumptions.yaml` |
| `transport.bus_capacity` | 50 | seats + standing per vehicle | `assumptions.yaml` |
| `transport.bus_load_factor` | 0.85 | usable share of capacity | `assumptions.yaml` |
| `shuttle_bus.available` | 80 | fleet size — **see the finding below** | `resources.yaml` |
| `min_buses_when_demand` | 1 | a route with any demand needs a vehicle | `config.yaml` |
| `apply_overload_to_wait` | true | stretch the wait when the fleet cannot cope | `config.yaml` |
| `max_overload_factor` | 20.0 | ceiling on that stretch | `config.yaml` |
| `confidence` | 0.45 | heuristic; lowest of the three built models — M06 is two hops downstream | `config.yaml` |

### Finding: the shuttle fleet is an order of magnitude too small
| | Vehicles needed | Fleet | Ratio |
|---|---:|---:|---:|
| Ordinary day peak | 369 | 80 | **4.6×** |
| Snan-day peak | 1,447 | 80 | **18.1×** |

**This was deliberately not "fixed".** Unlike the parking capacity M05 changed, `available`
is **M24's input** and is read by several models that do not exist yet; changing it here
would pre-empt a model whose whole job is resource planning. And unlike `parking_occupancy`,
`shuttle_requirement` is a raw vehicle count with no band, so it stays informative when it
exceeds the fleet. M06 reports the true requirement, raises `CONGESTION`, and recommends the
shortfall. **The gap is the finding, not a bug.**

It should be confirmed when M24 is built: either the fleet is genuinely this undersized, or
`bus_mode_share: 0.35` is too high for a venue with only two shuttle routes.

## Evaluation
> These metrics are computed on **synthetic** data. They measure pipeline correctness, not
> real-world accuracy.

Default 3-hour request at the demo instant, both routes:

| KPI | Range across the horizon |
|---|---|
| `shuttle_demand` | 21,613 – 54,194 persons/hr |
| `shuttle_requirement` | 340 – 851 vehicles |
| `passenger_wait_time` | 4.4 – 11.1 min |

Bands are conformally calibrated by the shared engine (see
`twin_common.engines.forecast.Calibration`); every record carries `details.band_calibrated`.

`/predict` returns 72 records in **1.40 s** warm, against the 10 s budget. Startup is 13.3 s,
which includes the fit and the band calibration.

## Limitations and failure modes
- **The fleet is 4.6–18× too small** in this world. See the finding above. Wait times are
  therefore dominated by the overload factor rather than by headway.
- **Wait past capacity is approximate.** A sustained excess of demand over capacity produces
  an unbounded queue; the reported number is a first-order estimate with a cap.
- **Round-trip time is one constant** for every route and every hour. M04 will supply a
  congestion-dependent value. Until then a scenario override (`round_trip_min`) moves it, and
  that path is tested.
- **M05's parking overflow is reported, not added to demand.** The split between drivers who
  park elsewhere and drivers who switch to a shuttle is not something M06 can know, and
  inventing a share would be a number with no evidence behind it.
- **The fleet is split in proportion to need**, which assumes vehicles are interchangeable and
  can be repositioned instantly.
- **Insensitive to S06 and S12.** The card hedges "via M04 details if present"; M04 is a
  Phase 7 model and is not present, and neither scenario closes a shuttle route.
- Confidence is a fixed heuristic, not a calibrated probability.

## Real-data readiness
| Step | What is needed |
|---|---|
| Data | Real boardings (ticketing or APC) and vehicle GPS replacing D07. GPS also gives observed round-trip times, which is the single biggest accuracy win available here. |
| Recalibration | `round_trip_min` per route and per hour from GPS; `bus_load_factor` from observed crush loads; the fleet size from the operator. |
| Retraining | Not required — Chronos-2 is zero-shot. Refit LightGBM and re-run the band calibration once a few weeks of real boardings exist. |
| Validation | Compare `passenger_wait_time` against observed queue times at the two stops; the overload term is the part most likely to be wrong. |

## Traceability
- KPIs: `shuttle_demand`, `shuttle_requirement`, `passenger_wait_time`
- Datasets: `D07`, `D01`, `D21`
- Upstream: `M01`, `M05`
- Model card in the study: M06
