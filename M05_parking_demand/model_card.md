# Model card — M05 Parking Demand

## Question / decision supported
**How full will each parking site be, and when does it overflow?**

The decision this supports is a diversion: three hours' warning that a lot will not hold the
vehicles heading for it is enough time to redirect arrivals and run a shuttle from an
overflow area. Two things follow from that. The model has to be able to say "more vehicles
are coming than there are bays", and it has to say it before the lot is full.

## Method
- **Engine:** `twin_common.engines.forecast`
- **Algorithm:** Darts `Chronos2Model` (zero-shot, `autogluon/chronos-2-small`), falling back
  to `LightGBMModel` then `NaiveSeasonal`
- **Library + version:** darts 0.47.0, lightgbm 4.7.0
- **Licence:** Darts Apache-2.0, LightGBM MIT; Chronos-2 weights per the autogluon model card

### What is forecast, and why it is not what the card says
The docs/03 M05 card asks for a forecast of "occupied spaces per site". M05 forecasts
**demand** instead — the uncapped running total of vehicles wanting a bay — and derives
occupancy from it.

The reason is in the data. D06 writes `occupied` clipped to `capacity`, so a model trained
on that column saturates at 100 % and can never predict an overflow, which is the one thing
this model exists to warn about. The uncapped series is recovered from the same table as
`cumsum(entries - exits)`, which is valid because D06 records arrival *demand* in `entries`
rather than admitted vehicles.

**Occupancy is therefore demand ÷ capacity and may exceed 100 %.** That is deliberate, and
the rest of the contract already expects it: the `utilization_pct` band in `risk_bands.yaml`
has a `critical` range of 100–9999, and the card's own test asks for occupancy "within
[0, 100 + overflow]". `details.occupied_spaces` carries the physically parked count,
`min(demand, capacity)`, for anyone who wants the capped view.

### Covariates
Calendar always: hour sine and cosine, day of week, the snan flag. Arrivals are added as a
fourth covariate whose **past half comes from D01 and whose future half comes from M01's
forecast**. Arrivals are not known ahead, so taking the future half from the generated world
would be peeking at data the caller could not have had at `as_of`. With no M01 available the
covariate is dropped entirely and the response says so and marks itself degraded.

That covariate is not decorative. Running S02 against M01's *baseline* sample under-forecast
the horizon total by 14.5 %; with M01's S02 sample the error is 3.5 %.

## Inputs and features
| Source | Detail |
|---|---|
| Tables | `parking_15min` (D06, the demand history), `footfall_15min` (D01, the arrivals covariate's past), `event_calendar` (D21, the snan flag) |
| Upstream models | `M01` — forecast arrivals for the future half of the covariate |
| Shared config | `transport.parking_capacity`, `transport.persons_per_car`, `transport.bus_capacity`, `transport.bus_load_factor` |

## Assumptions
Every value is a **placeholder** pending expert review, and every one is a config key rather
than a literal in the code.

| Config key | Value | Meaning | Source |
|---|---|---|---|
| `transport.parking_capacity` | P1 7300, P2 12000, P3 3900 | bays per site | `assumptions.yaml` — **changed, see below** |
| `search_time.base_min` | 3.0 | minutes to park an empty lot | `config.yaml` |
| `search_time.k_min` | 12.0 | extra minutes at full | `config.yaml` |
| `search_time.occupancy_knee` | 0.8 | where search time starts to grow | `config.yaml` |
| `overflow_pct` | 100.0 | occupancy at which `PARKING_OVERFLOW` is raised | `config.yaml` |
| `max_occupancy_pct` | 400.0 | reporting ceiling, catches a runaway forecast | `config.yaml` |
| `confidence` | 0.5 | heuristic, lower than M01's because M05 inherits M01's error | `config.yaml` |

### Changed assumption: parking capacity
The docs/04 §6 placeholder was `{P1: 1500, P2: 2500, P3: 800}`, 4,800 bays in total. That is
inconsistent with the arrival assumptions printed directly above it in the same block.

Cars present at once = arrivals × `car_mode_share` ÷ `persons_per_car`, accumulated over the
car dwell. Against 4,800 bays that peaks at **140–180 % on an ordinary day** and **583 % on
the snan morning**. The lot would read `critical` at every hour of every day, the "< 90 %"
threshold in docs/05 §2 would be unreachable, and search time — monotonic in occupancy —
would saturate with it. The KPI would carry no information.

Capacity was raised to **23,200 bays**, sized for the design peak the way event parking is
actually planned. Snan-morning demand of 27,995 vehicles then reads 121 % — a real overflow
that fires `PARKING_OVERFLOW` — while an ordinary day sits near 31 %. The 15 : 25 : 8 split
between the three sites is kept from the placeholder. **Still a placeholder**: replace with
the real surveyed bay count per site.

### Search-time curve
`search = base + k × max(0, occ − knee) / (1 − knee)`, with `occ` as a fraction. Calibrated
against the docs/05 §2 threshold of 15 minutes: 9 minutes at 90 % (green), exactly 15 at
100 %. It is **not** capped at full, so an oversubscribed lot keeps getting worse — which is
what keeps the KPI monotonic in occupancy, as the card's test requires.

## Evaluation
> These metrics are computed on **synthetic** data. They measure pipeline correctness, not
> real-world accuracy. Nothing here is evidence the model would perform this way at a real
> event.

Rolling-origin backtest, site P1 demand, 12-step horizon, 7 folds, CPU:

| Backend | MAE (vehicles) | MAPE | Pinball | 80 % coverage | Time |
|---|---:|---:|---:|---:|---:|
| chronos2 (zero-shot) | 236.7 | 23.9 % | 118.33 | 27.4 % | 1.3 s |
| lightgbm (fitted) | 235.4 | 21.2 % | 117.72 | 52.4 % | 6.9 s |
| naive (K=96) | 354.4 | 37.6 % | 177.18 | — | 0.03 s |

Both learned backends clear the naive floor by roughly a third. Unlike M01, the fitted
LightGBM edges out zero-shot Chronos-2 here, which is the expected ordering — the demand
series is a cumulative quantity with a sharper event-driven shape than raw footfall.

**Interval coverage was the weak result, and is now calibrated.** The raw Chronos-2 band held
the truth 27.4 % of the time against a nominal 80 % — far too narrow, because the error grows
much faster across the horizon than the quantiles do. M05 enables the engine's **conformal band
calibration** (`params.forecast.calibration`): at startup it forecasts 24 held-out origins,
measures the real miss rate at each horizon step and widens that step to match.

| | Raw | Calibrated |
|---|---:|---:|
| Coverage on the calibration set | 51.4 % | **89.9 %** |

Conformal calibration guarantees *at least* the stated coverage, not exactly it, so
over-covering is expected and is the safe direction. Every record carries
`details.band_calibrated`. The median remains untouched — only the band moves.

Horizon-window accuracy against the generated truth, summed over the three sites for the
default 3-hour request:

| Run | Forecast | Truth | Error |
|---|---:|---:|---:|
| S01 baseline | 308,111 | 307,324 | +0.3 % |
| S02 | 380,030 | 393,654 | −3.5 % |

## Limitations and failure modes
- **Occupancy exceeds 100 %** by design. A reader who expects a percentage of a physical
  capacity will misread it; `details.occupied_spaces` is the capped figure.
- **The uncertainty bands are conformally calibrated** (51 % → 90 % coverage against a nominal
  80 %). They now over-cover slightly, which is the safe direction. The factors are measured
  once at startup from recent history, so a sudden regime change would leave them stale until
  the service restarts.
- **Capacity is a changed placeholder.** Every occupancy number moves proportionally when the
  real bay counts arrive. The ranking between sites will not move, because it is set by the
  capacity ratio.
- **Demand splits between sites in proportion to capacity**, because that is what the
  generator does. Real drivers choose by distance, price and signage, so the per-site split
  is the least trustworthy part of the output. The total across sites is sounder than any one
  site.
- **Without M01 the arrivals covariate is dropped** and only the calendar drives the forecast.
  The response warns and marks itself degraded rather than substituting world data.
- **Insensitive to S06 and S12.** The docs/03 card hedges that they "may shift demand between
  sites". The reallocation mechanism is implemented and tested, but S06 closes R07 and S12
  closes B01, while the parking access links are R04/R05/R06 — so neither scenario touches a
  lot. Close R04, R05 or R06 and the reallocation fires.
- `entries` in D06 is arrival demand, not admitted vehicles, so it is inconsistent with the
  clipped `occupied` column in the same table. M05 relies on that reading; a real feed of
  barrier counts would mean admitted vehicles, and the demand series would have to be
  reconstructed from the queue instead.
- Confidence is a fixed heuristic, not a calibrated probability.

## Real-data readiness
| Step | What is needed |
|---|---|
| Data | Real entry/exit counts per site replacing D06. Barrier counts give *admitted* vehicles, so turn-aways must be captured separately or demand will be censored at capacity — the exact problem this model works around in the synthetic world. |
| Recalibration | `transport.parking_capacity` from a survey; `search_time.*` from observed arrival-to-parked times, which is the one curve here that can be fitted directly from a real feed. |
| Retraining | Not required — Chronos-2 is zero-shot. Refit LightGBM once a few weeks of real history exist, and re-run the backtest before trusting the bands. |
| Reconciliation | docs/03 M05 asks for daily occupancy-drift reconciliation: cumulative entries minus exits drifts from the true occupancy count, so it needs a daily reset against a physical count. |

## Traceability
- KPIs: `parking_occupancy`, `parking_search_time`, `parking_demand`
- Datasets: `D06`, `D01`, `D21`
- Upstream: `M01`
- Model card in the study: M05
