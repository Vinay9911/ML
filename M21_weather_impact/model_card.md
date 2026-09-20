# Model card — M21 Weather Impact

## Question / decision supported
What is the weather now and over the next hours, and how does it change risk elsewhere?

Operationally, M21 answers two things an operator acts on: *is it about to flood a ghat?*
and *should I expect fewer arrivals and more medical presentations?* It is the root of the
dependency graph — M01, M03, M07, M12, M15, M19, M22 and M25 all consume it — so an error
here propagates widely.

## Method
| Aspect | Detail |
|---|---|
| Engine | `twin_common.engines.formula` (no fitting, no training) |
| Heat index | NOAA/NWS Rothfusz regression with the two NWS adjustments, computed in °F and converted |
| Waterlogging | logistic in rainfall against a **per-zone** drainage capacity plus a low-lying bonus |
| Multipliers | the three linear/clipped forms in the docs/03 M21 card |
| Forecast | a **replay** of the stored series, not a prediction (docs/03 M21 allows this for the demo) |
| Library | numpy, pandas; no model weights |
| Licence | NOAA heat index: public domain. Open-Meteo data: CC BY 4.0, free tier non-commercial |

The heat index and the three multipliers are implemented **once**, in
`twin_common.engines.formula` and `twin_common.synthetic.weather`, and shared with the
synthetic generator. That is deliberate: if M21 and the generator each had their own copy,
a downstream model could read a multiplier that differs from the one that actually shaped
the world it is reading. `tests/test_formulas.py` asserts the coefficients have not drifted.

## Inputs and features
| Source | Detail |
|---|---|
| Tables | `weather_hourly` (D12), `zones` (D03: `drainage_capacity_mm_hr`, `low_lying`) |
| Upstream models | none |
| Features | temperature, relative humidity, rainfall, wind; per-zone drainage and elevation flag |

## Assumptions
Every value is a **placeholder** and every one is a config key, never a literal in the code.

| Config key | Value | Meaning |
|---|---|---|
| `params.rain_cap_mm_hr` | 20.0 | rainfall at which rain effects saturate |
| `params.k_rain_arrival` | 0.45 | heavy rain cuts arrivals by up to 45% |
| `params.k_rain_speed` | 0.30 | heavy rain cuts traffic speed by up to 30% |
| `params.k_heat_medical` | 0.04 | +4% medical demand per °C of heat index above the threshold |
| `params.hi_threshold_c` | 32.0 | heat index at which medical demand starts to rise |
| `params.waterlogging_k` | 0.12 | steepness of the flooding curve in rainfall |
| `params.waterlogging_low_lying_bonus` | 1.1 | logit bonus for a low-lying zone |
| `params.waterlogging_offset` | −3.9 | sets the dry-weather baseline near 2% rather than 50% |
| `params.heavy_rain_mm_hr` | 10.0 | when `HEAVY_RAIN` is raised |
| `params.dry_humidity_pct` | 35.0 | when hot weather is `HEAT_DRY` rather than `HEAT_STRESS` alone |
| `params.waterlogging_alert_pct` | 20.0 | when `WATERLOGGING` is raised on a zone |
| `params.confidence` | 0.8 | fixed; see Limitations |

`k_heat_medical` and `hi_threshold_c` intentionally equal
`assumptions.medical.heat_rate_increase_per_c` and `heat_index_threshold_c`, which M07 uses,
so the two models agree on when heat begins to matter.

## Evaluation
> These checks run on **synthetic** data and on a published reference table. They measure
> **implementation correctness**, not real-world accuracy. Nothing here is evidence that the
> waterlogging curve or the three multipliers predict anything at a real event — none of
> them has been fitted to observed outcomes.

| Check | Result |
|---|---|
| Heat index vs the published NWS chart | 15 reference cells, all within **1.5 °F** |
| Heat index internal consistency | reported value recomputes from its own reported inputs (±0.15 °C) |
| Waterlogging monotonic in rainfall | passes |
| Waterlogging responds to drainage and elevation | passes (low-lying > well-drained at equal rain) |
| Waterlogging in dry weather | < 5% for every zone |
| Coefficients match the shared assumptions | passes |
| S03 direction (docs/05 §1.2) | waterlogging ↑, arrivals ↓, traffic speed ↓ |
| S04 direction (docs/05 §1.2) | heat index ↑ (+13.4 °C), medical multiplier ↑ |
| Determinism | identical JSON across runs apart from `run_id` / `generated_at` |
| Contract | validates against `ModelOutput`; 52 tests pass, also with `TWIN_OFFLINE=1` |

There is no accuracy metric (MAE, MAPE, coverage) because M21 does not forecast: it replays
a stored series. Adding a real forecast feed would make those metrics meaningful, and the
model card must be updated with them at that point.

## Limitations and failure modes
- **Not a forecast.** The "forecast" horizon is a replay of a stored series. There is no
  forecast uncertainty, which is why no record carries `lower`/`upper` bands and why
  `confidence` is a fixed 0.8 rather than a calibrated number. It is labelled heuristic.
- **Waterlogging is not hydrology.** A logistic in rainfall against a single drainage rate
  per zone. It has no notion of ponding, runoff, upstream catchment, drain blockage or river
  level, all of which dominate real flooding at a riverside site.
- **The multipliers are unfitted.** The arrival, medical and speed couplings are plausible
  shapes with placeholder coefficients. Real crowds may be far less rain-sensitive at a
  religious event with fixed auspicious bathing times than the 45% figure assumes.
- **Venue-wide weather.** One weather series for the whole site. A convective downpour over
  one ghat is not representable.
- **Shifted history.** The observations are real Open-Meteo data for the same calendar days
  of a past year, shifted onto the event year. Plausible, but it is not what the weather
  will be.

## Real-data readiness
| Step | What is needed |
|---|---|
| Data | a licensed forecast feed (IMD or similar) in the D12 schema; site rain gauges for observed rainfall |
| Recalibration | all three multipliers against observed attendance, presentations and probe speeds; the waterlogging curve against recorded flooding incidents |
| Replacement | for credible flooding, a hydraulic model (PySWMM) rather than the logistic |
| Retraining | not applicable — M21 fits nothing |
| New capability | forecast uncertainty, so the bands and `confidence` become meaningful |

## Traceability
- **KPIs:** `temperature`, `heat_index`, `rainfall_intensity`, `waterlogging_probability`,
  and the extensions `weather_arrival_multiplier`, `weather_medical_multiplier`,
  `weather_traffic_speed_multiplier` (docs/05 §2).
- **Datasets:** D12 `weather_hourly`, D03 `zones`.
- **Study card:** M21, docs/03.
- **Scenarios:** S03, S04 (docs/05 §1.1, directions in §1.2).
