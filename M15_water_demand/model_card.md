# Model card — M15 Water Demand

## Question / decision supported
**How much water is needed by zone, is there a gap, and how many points and tankers?**

The decisions are where to put drinking-water points before the crowd arrives, and when to
send a tanker to a zone that is about to run short.

## Method
- **Engine:** `twin_common.engines.forecast`
- **Algorithm:** Darts `Chronos2Model` (zero-shot, `autogluon/chronos-2-small`) → `LightGBMModel`
  → `NaiveSeasonal`, with conformal band calibration
- **Library + version:** darts 0.47.0, lightgbm 4.7.0
- **Licence:** Darts Apache-2.0, LightGBM MIT; Chronos-2 weights per the autogluon model card

| Step | What happens |
|---|---|
| Forecast | consumption per zone from D14, one series per zone |
| Covariates | calendar, **temperature** from D12, and arrivals whose future half comes from M01 |
| Supply | carried forward from the last observed value — see below |
| Gap | `demand − supply`; **positive means a shortage** |
| Points | `ceil(peak population ÷ persons_per_water_point)`, sized once against the horizon peak |
| Tankers | `ceil(max(gap, 0) × cover_hours ÷ tanker_payload_l)` |

### Why supply is carried forward rather than forecast
A pump's design output does not vary with the crowd, so supply is a **planned** quantity.
Reading it across the forecast window is no more "peeking" than reading the calendar. A
scenario that downs a substation scales the affected zones by
`water.pump_outage_supply_factor` — a value lifted out of the generator during this work so
M15 and the generator cannot drift apart.

### Why population is recovered rather than forecast
`consumption = population × litres_per_person_per_hour × heat_factor`, so population is
recovered by inverting that relation. Forecasting it separately would allow the water-point
requirement and the demand forecast to contradict each other; this way they cannot.

## Inputs and features
| Source | Detail |
|---|---|
| Tables | `water_15min` (D14), `footfall_15min` (D01), `weather_hourly` (D12), `event_calendar` (D21) |
| Upstream | **M01** — arrivals covariate; **M21** — the temperature that drives the heat term |
| Shared config | `water.*` in `assumptions.yaml`, `pumps` in `world.yaml` |

## Assumptions
| Config key | Value | Meaning | Source |
|---|---|---|---|
| `litres_per_person_per_hour_present` | 0.5 | consumption per person present | `assumptions.yaml` |
| `heat_multiplier_per_c_above_30` | 0.03 | +3 % demand per °C over 30 | `assumptions.yaml` |
| `persons_per_water_point` | 1500 | people one point serves | `assumptions.yaml` |
| `tanker_payload_l` | 10000 | litres per tanker | `assumptions.yaml` |
| `pump_outage_supply_factor` | 0.2 | output while the substation is down | `assumptions.yaml` — **moved here from the generator** |
| `tanker_cover_hours` | 4.0 | hours of shortfall one dispatch covers | `config.yaml` |
| `confidence` | 0.5 | heuristic | `config.yaml` |

## Evaluation
> These metrics are computed on **synthetic** data. They measure pipeline correctness, not
> real-world accuracy.

Default 3-hour request at the demo instant, all eight zones:

| KPI | Range |
|---|---|
| `expected_water_demand` | 904 – 17,802 L/hr |
| `water_supply_demand_gap` | −8,471 – +8,427 L/hr |
| `water_tanker_requirement` | 0 – 4 vehicles |
| `drinking_water_point_requirement` | 2 – 24 points |

Bands are conformally calibrated by the shared engine; every record carries
`details.band_calibrated`. `/predict` returns 296 records in **1.81 s** warm.

### The S04 heat response, and where it stops being trustworthy
This is the card's headline scenario, and it has two real limits.

**Time of day.** Measured on the generated worlds, S04 raises water demand over a three-hour
window by **0.12 % at 06:00, 7.0 % at 10:00 and 8.5 % at 13:00**. The demo instant is a cool
morning: even a +6 °C offset only lifts the peak from 24.6 °C to 30.6 °C, barely across the
30 °C the multiplier starts from. There is nothing wrong with the model — there is very
little heat effect to find at 06:00.

**Horizon.** Even at midday the uplift decays across the horizon:

| Horizon | 15 min | 60 min | 90 min | 180 min |
|---|---:|---:|---:|---:|
| S04 ÷ baseline | **1.100** | 1.029 | 0.988 | 0.871 |

against a ground truth of about 1.085 throughout. The zero-shot backend reverts towards its
own seasonal shape and loses a small covariate signal. **The scenario response is trustworthy
for roughly the first hour.** A 30 % signal like S02 survives the whole horizon; an 8 % one
does not. The tests assert the direction over the near horizon and separately pin the decay,
so it cannot quietly turn into a false claim.

## Limitations and failure modes
- **The S04 response decays with horizon** (see above). Treat heat scenarios as a one-hour
  signal, not a three-hour one.
- **Supply is design output, carried forward.** Real pump curves vary with pressure and head,
  and **leakage is not modelled at all** — a real network loses a substantial share between
  pump and tap, so the real gap would be worse than reported.
- **Storage is not drawn down.** A zone with a positive gap is reported short even when its
  tanks could cover it for hours, which makes the tanker requirement conservative.
- **Population inherits the litres-per-person assumption.** It is exactly consistent with the
  demand forecast, but both move together if that rate is wrong.
- **Water points are sized, not placed.** Where they go is M16's and M24's work.
- Confidence is a fixed heuristic, not a calibrated probability.

## Real-data readiness
| Step | What is needed |
|---|---|
| Data | Meter readings per zone replacing D14, and real pump telemetry for supply. |
| Recalibration | `litres_per_person_per_hour_present` and the heat multiplier from metered consumption against measured crowd; `persons_per_water_point` from observed queueing. |
| Leak reconciliation | The single biggest gap between this model and reality. Pumped volume minus metered consumption gives the loss rate, which should become an explicit term. |
| Retraining | Not required — Chronos-2 is zero-shot. Refit LightGBM and re-run the band calibration once real history exists. |

## Traceability
- KPIs: `expected_water_demand`, `water_supply_demand_gap`,
  `drinking_water_point_requirement`, `water_tanker_requirement`
- Datasets: `D14`, `D01`, `D12`, `D21`
- Upstream: `M01`, `M21`
- Model card in the study: M15
