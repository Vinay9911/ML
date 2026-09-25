# Model card — M19 Power Load

## Question / decision supported
**What electrical load is expected, and how long can backup last?**

Two decisions: how much backup generation to have staged, and — once the grid goes — how long
there is before the lights go out. The second is the one that matters in the room, and it is
why `generator_backup_duration` has to be a real countdown rather than a nominal figure.

## Method
- **Engine:** `twin_common.engines.forecast`
- **Algorithm:** Darts `Chronos2Model` (zero-shot) → `LightGBMModel` → `NaiveSeasonal`, with
  conformal band calibration
- **Library + version:** darts 0.47.0, lightgbm 4.7.0
- **Licence:** Darts Apache-2.0, LightGBM MIT; Chronos-2 weights per the autogluon model card

| Step | What happens |
|---|---|
| Forecast | kW per substation from D18, reported in **MW** with kW kept in `details` |
| Covariates | calendar, temperature, and arrivals whose future half comes from M01 |
| Critical load | `kw × critical_load_fraction` — what must stay up when the grid goes |
| Generators | `ceil(critical_kw × redundancy_factor ÷ generator_unit_kw)` |
| Backup duration | `fuel_l ÷ (l_per_hr_at_full_load × load_fraction)`, capped at 24 h |
| Handoff | `details.load_utilization` published for M12, whose card reads it as a fire driver |

### A data-shape trap worth recording
**D18 carries one row per (substation, zone) pair**, so SS01 appears four times at every
timestamp — once for each zone it feeds. Splitting the table on `asset_id` alone keeps only
the last row and silently reports **a quarter of the substation's load**, with no error
anywhere. M19 aggregates first: load *sums* across the zones a substation feeds, while fuel is
a property of its generator and must not be summed. A test asserts the SS01 series is larger
than a single zone's baseline, so this cannot regress quietly.

### Changed behaviour: generators now refuel
`generator_backup_duration` initially read **0 hours for every asset in S08** — the one
scenario the KPI exists for.

The cause was in the world generator: fuel was a single cumulative drain across the whole
month (`generator_fuel - cumsum(burn)`), and the S08 outage window recurs *daily*. Thirty-one
days × three hours of running emptied a 400 L tank long before the demo day, so by 2027-08-02
the tank was dry and the countdown was zero.

Generators now refuel whenever the grid is back, which is what actually happens between
outages. S08 fuel falls from 394 L to 298 L across the outage window — a real countdown.

## Inputs and features
| Source | Detail |
|---|---|
| Tables | `power_15min` (D18), `footfall_15min` (D01), `weather_hourly` (D12), `event_calendar` (D21) |
| Upstream | **M01** — arrivals covariate; **M21** — temperature driving the cooling term |
| Shared config | `power.*` in `assumptions.yaml`, `substations` in `world.yaml` |

## Assumptions
| Config key | Value | Meaning | Source |
|---|---|---|---|
| `base_kw_per_zone` | 150 | standing load per zone | `assumptions.yaml` |
| `watts_per_person` | 5 | load added per person present | `assumptions.yaml` |
| `cooling_kw_per_c_above_30_per_zone` | 20 | the S04 mechanism | `assumptions.yaml` |
| `generator_unit_kw` | 250 | one generator's rating | `assumptions.yaml` |
| `generator_fuel_l` | 400 | tank size | `assumptions.yaml` |
| `generator_l_per_hr_at_full_load` | 60 | burn at full load | `assumptions.yaml` |
| `redundancy_factor` | 1.25 | N+ spare capacity | `assumptions.yaml` |
| `critical_load_fraction` | 0.6 | share that must stay up | `config.yaml` |
| `assumed_load_fraction` | 0.75 | what a set actually carries | `config.yaml` |
| `max_backup_hours` | 24 | reporting cap | `config.yaml` |

## Evaluation
> These metrics are computed on **synthetic** data. They measure pipeline correctness, not
> real-world accuracy.

Default 3-hour request at the demo instant, two substations, 72 records:

| KPI | Range |
|---|---|
| `electricity_demand` | 0.74 – 1.08 MW |
| `backup_generator_requirement` | 3 – 4 units |
| `generator_backup_duration` | 8.89 hours |

| Scenario | Effect |
|---|---|
| S04 extreme heat | demand **1.041×** through the cooling term |
| S08 power failure | 36 `GRID_OUTAGE` flags; SS01 on backup, SS02 untouched |

`/predict` returns 72 records in **0.47 s** warm.

## Limitations and failure modes
- **The critical load is one fraction of the whole asset**, a placeholder for a per-circuit
  schedule. Which circuits are genuinely life-safety is a design decision, not a ratio.
- **Backup duration assumes a constant load fraction.** Real consumption varies with load,
  ambient temperature and set condition. Past a day, refuelling — not tank size — is the
  binding constraint, which is why the figure is capped at 24 h.
- **Fuel does not deplete across the forecast horizon.** The countdown is the runtime from the
  last observed level, not a live burn-down.
- **The cooling uplift is a flat kW per degree per asset**, applied only when a scenario moves
  the temperature. It does not vary by what the asset actually feeds.
- **Only two substations exist in this world**, so there is no meaningful spatial detail.
- Confidence is a fixed heuristic, not a calibrated probability.

## Real-data readiness
| Step | What is needed |
|---|---|
| Data | Smart-meter or BMS telemetry per feeder replacing D18, and generator telemetry (fuel level, running hours, load) — the latter turns the countdown from an estimate into a measurement. |
| Recalibration | `critical_load_fraction` replaced by the real essential-circuit schedule; burn rates from generator logs. |
| Retraining | Not required — Chronos-2 is zero-shot. Refit LightGBM and re-run the band calibration once real history exists. |
| Handoff | M12 consumes `details.load_utilization`; keep that key stable. |

## Traceability
- KPIs: `electricity_demand`, `generator_backup_duration`, `backup_generator_requirement`
- Datasets: `D18`, `D01`, `D12`, `D21`
- Upstream: `M01`, `M21` · Downstream: `M12` reads `details.load_utilization`
- Model card in the study: M19
