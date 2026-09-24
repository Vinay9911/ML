# Model card — M17 Food & Supply

## Question / decision supported
**How much food is needed, how long does stock last, how many deliveries?**

The decision is whether to place an order **now**. That framing sets almost every design
choice below, because an order is only useful if it arrives before the shelf is empty.

## Method
- **Engine:** `twin_common.engines.forecast`
- **Algorithm:** Darts `Chronos2Model` (zero-shot, `autogluon/chronos-2-small`) → `LightGBMModel`
  → `NaiveSeasonal`, with conformal band calibration, plus a deterministic inventory simulation
- **Library + version:** darts 0.47.0, lightgbm 4.7.0
- **Licence:** Darts Apache-2.0, LightGBM MIT; Chronos-2 weights per the autogluon model card

| Step | What happens |
|---|---|
| Forecast | meals sold per outlet from D17 — **an hourly table**, unlike every other model here |
| Covariates | calendar, and arrivals from M01 resampled onto the hourly grid |
| Inventory | stock walked forward: `stock − meals + arrivals`, with orders placed when cover runs short |
| Coverage | `stock ÷ consumption rate`, capped for reporting |
| Deliveries | `ceil(demand × 24 ÷ vehicle_payload_meals)`, quoted per day to match the KPI unit |
| Stockout | raised when cover is below the minimum **or** below the lead time |

### Why the stock is simulated rather than snapshotted
The first implementation held stock at its last observed level. Coverage was then
`stock ÷ rate`, and **S13 did nothing at all** — stretching the lead time changes neither the
stock on hand nor the rate it is eaten, so the number was identical. docs/05 asserts
"coverage down" for S13, and it was right to.

What a lead time actually changes is **when a replenishment lands**. Stock is now walked
forward across the horizon: an order placed at hour *t* arrives at *t + lead_time*, and if
that falls outside the window the shelf simply keeps draining. Under S13 the delivery lands
three hours later, and coverage falls to **0.88×** with stockout flags doubling.

### Why the default horizon is 12 hours, not 6
With a 6-hour lead time, **nothing ordered inside a 6-hour window can arrive inside it**. The
stock path is then identical whatever the lead time is, and S13 has nothing to show. A
12-hour horizon lets a delivery land at h+6 under S01 and h+9 under S13 — which is exactly
the difference the scenario is about. A test pins this so the horizon and the lead time
cannot drift apart.

## Inputs and features
| Source | Detail |
|---|---|
| Tables | `food_inventory_hourly` (D17), `footfall_15min` (D01), `event_calendar` (D21) |
| Upstream | **M01** — arrivals covariate, resampled 15-min → hourly |
| Shared config | `food.*` in `assumptions.yaml` |

## Assumptions
| Config key | Value | Meaning | Source |
|---|---|---|---|
| `meals_per_person_per_hour_market` | 0.15 | consumption rate driving D17 | `assumptions.yaml` |
| `vehicle_payload_meals` | 2000 | meals per delivery vehicle | `assumptions.yaml` |
| `lead_time_hours` | 6 | supplier lead time | `assumptions.yaml` |
| `min_stock_cover_hours` | 8 | buffer below which a stockout is raised | `assumptions.yaml` |
| `max_coverage_hours` | 72 | reporting cap | `config.yaml` |
| `confidence` | 0.45 | heuristic | `config.yaml` |

### Contract extension: `FO` added to the `facility` entity type
The generator emits outlet IDs `FO01`/`FO02` for D17, and the docs/03 M17 card asks for
"meals per outlet" — but **no entity type in the docs/02 §3 ID table covered them**. That is
an inconsistency between docs/02 and docs/04, not a design choice.

`facility` was extended from `(?:H|MP|PP|FS|SH|AS)\d{2}` to include `FO`, because a food
outlet is the same kind of thing as the medical posts already in that pattern: a staffed,
stocked, fixed location. docs/02 §3 and `contracts/ids.py` both updated, and the decision is
in the `PROGRESS.md` log.

## Evaluation
> These metrics are computed on **synthetic** data. They measure pipeline correctness, not
> real-world accuracy.

Default 12-hour request at the demo instant, both outlets, 72 records:

| KPI | Range |
|---|---|
| `food_demand` | 164 – 1,869 meals/hr |
| `food_stock_coverage` | 8.1 – 63.3 hours |
| `delivery_requirement` | 2 – 23 trips/day |

| | S01 | S13 |
|---|---:|---:|
| Total coverage across the horizon | 1,019 h | **896 h** (0.88×) |
| Stockout flags | 3 | **6** |

Bands are conformally calibrated; every record carries `details.band_calibrated`.
`/predict` returns 72 records in **1.36 s** warm.

## Limitations and failure modes
- **The reorder policy is deliberately simple**: order enough whole vehicle loads to cover the
  lead time plus the minimum buffer, and never order while an order is in flight. A real
  operation batches, splits and expedites. This is a model of *when stock runs out*, not a
  replenishment plan.
- **Opening stock is the last observed level.** Any drift between the recorded level and what
  is physically on the shelf propagates through the whole simulated horizon.
- **Only two outlets exist in this world**, both in the market zone, so the per-outlet split
  carries very little information. The totals are sounder than either outlet alone.
- **Coverage is capped** at 72 h; an outlet with stock and no demand reports the cap rather
  than infinity.
- **Deliveries are quoted per day from an hourly rate**, which assumes the forecast hour is
  representative of the day.
- **Insensitive to S06.** The card makes the lead-time increase optional and dependent on M04,
  a Phase 7 model. The override is wired and tested — adding M04 upstream is all that is needed.
- Confidence is a fixed heuristic, not a calibrated probability.

## Real-data readiness
| Step | What is needed |
|---|---|
| Data | POS transactions and a live inventory feed replacing D17. |
| Recalibration | `lead_time_hours` per supplier from purchase-order history — this is the single most load-bearing number in the model, and it is currently one constant for everyone. |
| Policy | Replace the reorder rule with the operator's actual policy; the simulation is structured so only that function changes. |
| Retraining | Not required — Chronos-2 is zero-shot. Refit LightGBM and re-run the band calibration once real sales history exists. |

## Traceability
- KPIs: `food_demand`, `food_stock_coverage`, `delivery_requirement`
- Datasets: `D17`, `D01`, `D21`
- Upstream: `M01`
- Model card in the study: M17
