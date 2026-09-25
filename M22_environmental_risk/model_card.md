# Model card — M22 Environmental Risk

## Question / decision supported
**Where will air quality or noise stress increase, and what are the emissions?**

Two different jobs. Air quality and noise feed operational decisions during the event; the
CO2 figure is a reporting number for afterwards. They are in one model because they share
the same inputs, but they have very different reliability — see the limitations.

## Method
- **Engine:** `twin_common.engines.forecast`
- **Algorithm:** Darts `Chronos2Model` (zero-shot) → `LightGBMModel` → `NaiveSeasonal`, plus
  three deterministic formulas
- **Library + version:** darts 0.47.0, lightgbm 4.7.0
- **Licence:** Darts Apache-2.0, LightGBM MIT; Open-Meteo air-quality data CC BY 4.0
  (free tier is non-commercial); the NAQI method is CPCB's

| KPI | How |
|---|---|
| `pm25` | forecast from D13 with wind and calendar covariates |
| `air_quality_index` | Indian NAQI: piecewise-linear sub-index per pollutant, **AQI = max** |
| `noise_level` | `base_db + 10·log10(1 + density ÷ density_ref)`, per zone |
| `co2_emissions` | `vehicle-km × EF + kWh × grid EF + diesel litres × EF`, in tCO2e/day |

### D13 is the one table built from real data
Every other table in this project is synthetic. D13 comes from **cached Open-Meteo
air-quality history**, so its rows carry `is_synthetic: false`. It is also **venue-level and
hourly**, which constrains what this model can say — see the S15 limitation below.

### Why the AQI is a maximum, not an average
The NAQI takes the **worst** pollutant sub-index, not the mean. Averaging would let a clean
pollutant mask a dangerous one — a day with harmless PM10 and severe PM2.5 would report as
moderate. A test asserts the maximum rule explicitly against an averaged alternative.

## Inputs and features
| Source | Detail |
|---|---|
| Tables | `air_quality_hourly` (D13, **real**), `footfall_15min` (D01), `weather_hourly` (D12), `power_15min` (D18), `event_calendar` (D21) |
| Upstream | **M21** — wind; **M04** — vehicle-km, **not built yet** |
| Shared config | `power.*` in `assumptions.yaml` |

## Assumptions
**Every number in this section is a placeholder, and two of them are the kind that should not
be published without replacement.**

| Config key | Value | Meaning | Status |
|---|---|---|---|
| `naqi.index_breakpoints` | 50/100/200/300/400/500 | NAQI category edges | **VERIFY AGAINST CPCB** |
| `naqi.concentration_breakpoints.pm2_5` | 30/60/90/120/250/380 | µg/m³ per band | **VERIFY AGAINST CPCB** |
| `naqi.concentration_breakpoints.pm10` | 50/100/250/350/430/510 | µg/m³ per band | **VERIFY AGAINST CPCB** |
| `noise.base_db` | 55.0 | ambient with no crowd | placeholder |
| `noise.density_ref_p_m2` | 1.0 | reference density | placeholder |
| `co2.kg_per_vehicle_km` | 0.18 | traffic factor | **placeholder — official factor needed** |
| `co2.kg_per_kwh` | 0.71 | India grid average | **placeholder — official factor needed** |
| `co2.kg_per_diesel_litre` | 2.68 | generator diesel | **placeholder — official factor needed** |

## Evaluation
> These metrics are computed on **synthetic** crowd and power data against **real** air-quality
> history. They measure pipeline correctness, not real-world accuracy.

Default 6-hour request at the demo instant, 66 records in **0.27 s** warm:

| KPI | Range |
|---|---|
| `pm25` | 30.5 – 32.2 µg/m³ |
| `air_quality_index` | 50.8 – 53.7 (Satisfactory) |
| `noise_level` | 55.4 – 62.1 dB |
| `co2_emissions` | 23.6 – 31.0 tCO2e/day |

| Scenario | Effect |
|---|---|
| S02 | noise **1.011×**, CO2 **1.074×** |
| S08 | CO2 **1.015×** (generators burning diesel) |

The noise ratio being ~1.01 for a 30 % larger crowd is correct, not a bug: decibels are
logarithmic. A test asserts it stays under 1.1 for exactly this reason.

## Limitations and failure modes
- **THE NAQI BREAKPOINTS ARE PLACEHOLDERS.** They are transcribed from docs/05 and have not
  been checked against the published CPCB table. The official method also uses **24-hour
  averages** for PM2.5 and PM10; this uses the instantaneous forecast, which reads differently
  during a sharp episode. Do not present an AQI from this model as a regulatory figure.
- **The CO2 traffic term is ZERO** because M04 does not exist yet. The reported figure is
  electricity and generator diesel only, and every record says so in
  `details.traffic_term`. The real total will be materially higher.
- **Every emission factor is a placeholder.** Do not publish a tCO2e number from this model.
- **S15 cannot be represented properly.** The card says PM2.5 rises "near Z03", but D13 is a
  single venue-level series — there is no "near Z03" to move. The scenario is answered, the
  fire is flagged in the reason codes and a warning, and that is the honest limit. Inventing a
  per-zone plume would be fabricating spatial detail the input does not contain.
- **Only PM2.5 and PM10 contribute to the AQI.** The official index also covers NO2, SO2, CO,
  O3 and NH3, any of which could be the binding sub-index on a given day. D13 carries NO2 and
  CO, so adding them is a config change rather than a redesign.
- **Noise comes from crowd density alone.** Public address, generators, traffic and music are
  not modelled, and at a real event those usually dominate the measured level.

## Real-data readiness
| Step | What is needed |
|---|---|
| Data | Calibrated AQ sensors placed per zone — the single change that would unlock the spatial questions this model currently cannot answer — and real noise meters. |
| Breakpoints | Replace the NAQI table with the CPCB published values and switch to 24-hour averaging. |
| Factors | Replace all three emission factors with official published values. |
| Upstream | Add M04 once it exists so the traffic term stops being zero. |

## Traceability
- KPIs: `air_quality_index`, `pm25`, `noise_level`, `co2_emissions`
- Datasets: `D13` (real), `D01`, `D12`, `D18`, `D21`
- Upstream: `M21`; `M04` pending
- Model card in the study: M22
