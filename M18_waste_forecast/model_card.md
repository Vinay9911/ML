# Model card — M18 Waste Generation

## Question / decision supported
**How much waste, when do bins overflow, how many collection trips?**

The decision is whether to bring a collection forward. Waste on the ground at a ghat is both
a public-health problem and a slip hazard in a dense crowd, so the useful output is not "this
bin is 80 % full" but "this bin reaches full in 120 minutes, before the next round".

## Method
- **Engine:** `twin_common.engines.forecast`
- **Algorithm:** Darts `Chronos2Model` (zero-shot) → `LightGBMModel` → `NaiveSeasonal`, with
  conformal band calibration, plus a deterministic fill projection
- **Library + version:** darts 0.47.0, lightgbm 4.7.0
- **Licence:** Darts Apache-2.0, LightGBM MIT; Chronos-2 weights per the autogluon model card

| Step | What happens |
|---|---|
| Forecast | kg added per bin group from D16, one series per group |
| Covariates | calendar, and arrivals whose future half comes from M01 |
| Fill | `current fill + inflow × t`, **reset to zero on each collection round** |
| Overflow time | the first step the projection reaches 100 % |
| Trips | `ceil(kg per day ÷ vehicle_payload_kg)` |

### Changed assumption: `bins_per_group`
`bin_capacity_kg: 120` is **one bin**. But D16 and `world.yaml` model a bin **group** per zone
(WB01…WB08), and the generator was dividing a whole zone's waste by a single bin's capacity.

The consequence was that every group read **100 % full within one 15-minute step**, in every
zone and every scenario. `waste_bin_fill_level` was pinned at 100 and `bin_overflow_time` at
the first step — both KPIs carried no information whatsoever.

`bins_per_group: 50` was added, so a group holds 50 × 120 = 6,000 kg. Sized from the generated
world: over a 4-hour round the busiest group (WB01, the main ghat) accumulates 6,204 kg at the
snan peak, which is 103 % — one group overflowing at the peak and the rest below.

| | Before | After |
|---|---|---|
| WB01 fill, mean / max | 86.9 % / 100 % | **8.2 % / 96.8 %** |
| WB06 fill, mean / max | 22.3 % / 100 % | **0.5 % / 4.9 %** |

Both the generator and M18 read the same two keys, so they cannot drift apart. **Still a
placeholder**: replace with the surveyed bin count per zone.

### Why fill is clipped at 100 %
A bin cannot be more than full. Reporting "137 %" would be arithmetic, not information — what
an operator needs is *when* it crossed, which is exactly what `bin_overflow_time` says. A bin
that is not projected to overflow reports `no_overflow_minutes` (999) rather than null, so a
dashboard series stays numeric, and rather than 0, which reads as "overflowing right now".

## Inputs and features
| Source | Detail |
|---|---|
| Tables | `waste` (D16), `footfall_15min` (D01), `event_calendar` (D21) |
| Upstream | **M01** — arrivals covariate |
| Shared config | `waste.*` in `assumptions.yaml`, `waste_bin_groups` in `world.yaml` |

## Assumptions
| Config key | Value | Meaning | Source |
|---|---|---|---|
| `kg_per_person_per_hour_present` | 0.05 | waste per person present | `assumptions.yaml` |
| `bin_capacity_kg` | 120 | capacity of **one** bin | `assumptions.yaml` |
| `bins_per_group` | 50 | bins in a zone group | `assumptions.yaml` — **added** |
| `vehicle_payload_kg` | 3000 | kg per collection vehicle | `assumptions.yaml` |
| `collection_interval_min` | 240 | the normal round | `config.yaml` |
| `no_overflow_minutes` | 999 | sentinel for "not overflowing" | `config.yaml` |
| `confidence` | 0.5 | heuristic | `config.yaml` |

## Evaluation
> These metrics are computed on **synthetic** data. They measure pipeline correctness, not
> real-world accuracy.

Default 3-hour request at the demo instant, eight bin groups, 296 records:

| KPI | Range |
|---|---|
| `waste_generation` | 2,174 – 41,889 kg/day |
| `waste_bin_fill_level` | 2.6 – 100 % |
| `waste_collection_trips` | 1 – 14 trips/day |
| `bin_overflow_time` | 120 min (one group) – 999 (the rest) |

S02 raises generation by exactly **1.300×**, which is the common-random-numbers change
working: before it, scenario comparisons carried several percent of noise.

`/predict` returns 296 records in **1.64 s** warm.

## Limitations and failure modes
- **Collection is a fixed round at a fixed interval**, not a schedule. A real operation
  collects on a route with travel time between stops, so a "bring it forward" recommendation
  does not account for where the vehicle currently is.
- **The 4-hour round is longer than the 3-hour default horizon**, so in a default request no
  collection falls inside the window and the reset never fires. It is exercised at a longer
  horizon in the tests. If rounds get shorter than the horizon this matters more.
- **Fill is clipped at 100 %.** Waste on the ground past that point is not quantified.
- **Opening fill is the last observed level**, so sensor drift propagates through the whole
  projected horizon.
- **Insensitive to S06 and S12**: the card wants collection delayed under a closure, but only
  M04 (Phase 7) can say by how much. The delay override is wired and tested.
- **Trips are quoted per day from a 15-minute rate**, which assumes the forecast step is
  representative of the day.
- Confidence is a fixed heuristic, not a calibrated probability.

## Real-data readiness
| Step | What is needed |
|---|---|
| Data | Weighbridge tickets for generation and bin-level fill sensors replacing D16. |
| Recalibration | `bins_per_group` from a survey — the single most load-bearing number here; `kg_per_person_per_hour_present` from weighbridge totals against crowd counts. |
| Routing | Replace the fixed round with the actual collection schedule and vehicle positions, which is where the recommendation becomes actionable rather than indicative. |
| Retraining | Not required — Chronos-2 is zero-shot. Refit LightGBM and re-run the band calibration once real history exists. |

## Traceability
- KPIs: `waste_generation`, `waste_bin_fill_level`, `waste_collection_trips`, `bin_overflow_time`
- Datasets: `D16`, `D01`, `D21`
- Upstream: `M01`
- Model card in the study: M18
