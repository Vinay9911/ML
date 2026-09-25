# Model card — M20 Network Capacity

## Question / decision supported
**Will communications — CCTV backhaul and command links — overload or fail?**

Everything else in the command centre depends on this one working. If the backhaul dies, the
operators lose the cameras, the dashboards and each other, which is why the KPIs separate
*congestion* from *availability*: a tower can be perfectly uncongested and still be off.

## Method
- **Engine:** `twin_common.engines.forecast`
- **Algorithm:** Darts `Chronos2Model` (zero-shot) → `LightGBMModel` → `NaiveSeasonal`, with
  conformal band calibration
- **Library + version:** darts 0.47.0, lightgbm 4.7.0
- **Licence:** Darts Apache-2.0, LightGBM MIT; Chronos-2 weights per the autogluon model card

| Step | What happens |
|---|---|
| Forecast | bandwidth used per tower from D19 — **a 5-minute table**, not the 15-minute grid |
| Covariates | calendar, and arrivals whose future half comes from M01 |
| Utilisation | `used ÷ capacity × 100`, clipped at 100 |
| Risk | `100 × sigmoid(k × (utilisation − midpoint))` |
| Availability | recent uptime share, zeroed for a tower whose substation is down |

## Finding: this world does not run out of bandwidth

| Tower | Mean util | Peak util | Capacity |
|---|---:|---:|---:|
| NT01 | 4.9 % | 19.8 % | 2,000 Mbps |
| NT02 | 5.5 % | **27.5 %** | 2,000 Mbps |
| NT03 | 4.0 % | 11.7 % | 1,000 Mbps |

`network_capacity_risk` therefore sits between **0.04 % and 0.71 %** across the default
request — effectively zero.

**That is the correct answer for these capacities, and the threshold was deliberately not
lowered to make the number move.** A network running at 27 % genuinely is low-risk; reporting
a manufactured risk would be worse than reporting none. The binding constraint in this world
is power-driven *availability* (S08), not congestion.

The honest read is that `tower_capacity_mbps` in `assumptions.network` is over-provisioned
relative to the crowd assumptions, and it should be validated before anyone draws a conclusion
from the risk KPI. A test asserts risk stays under 20 % so that if the capacities or the demand
assumptions ever change, this section is flagged as stale rather than quietly becoming wrong.

## Inputs and features
| Source | Detail |
|---|---|
| Tables | `network_5min` (D19), `footfall_15min` (D01), `event_calendar` (D21), `cameras` |
| Upstream | **M01** — arrivals covariate |
| Shared config | `network.*` in `assumptions.yaml`, `network_towers` and `cameras` in `world.yaml` |

## Assumptions
| Config key | Value | Meaning | Source |
|---|---|---|---|
| `tower_capacity_mbps` | NT01 2000, NT02 2000, NT03 1000 | link capacity — **see the finding** | `assumptions.yaml` |
| `camera_bitrate_mbps` | 4 | per camera stream | `assumptions.yaml` |
| `active_user_share` | 0.3 | share of the crowd on the network | `assumptions.yaml` |
| `kbps_per_active_user` | 50 | per active user | `assumptions.yaml` |
| `risk_midpoint_pct` | 70 | utilisation at which risk is 50 % | `config.yaml` |
| `risk_steepness` | 0.12 | ~11 % risk at 50 % util, ~92 % at 90 % | `config.yaml` |

## Evaluation
Default 3-hour request at the demo instant, three towers, **324 records in 0.71 s** warm.

| Scenario | Effect |
|---|---|
| S02 | utilisation **1.267×** |
| S08 | availability **0.667×** (NT02 only), utilisation 0.503× |
| S09 | utilisation 0.957×, `CCTV_BLIND_SPOT` raised |

S08 takes down **only NT02**, which is the tower `world.yaml` puts on SS01 — flagging the
other two would be a false alarm, and a test pins that.

## Limitations and failure modes
- **Capacity risk is near zero throughout.** See the finding above; validate the tower
  capacities before reading anything into it.
- **Availability is the recent uptime share carried forward**, not a forecast. An outage that
  has not started is invisible unless a scenario declares it.
- **S09 removes camera backhaul load but does not model the loss of coverage itself**, which
  is M02's and M10's work. The `CCTV_BLIND_SPOT` code is a pointer to them, not an analysis.
- **Each tower is an independent link.** Shared backhaul, contention and failover between
  towers are not modelled, so a real correlated failure would be worse than anything here.
- Confidence is a fixed heuristic, not a calibrated probability.

## Real-data readiness
| Step | What is needed |
|---|---|
| Data | NMS telemetry per link replacing D19 — throughput, errors and up/down transitions. |
| Recalibration | `tower_capacity_mbps` from the actual circuits (the number that matters most here), and `active_user_share` from real attachment counts. |
| Modelling | Add backhaul topology so correlated failures and failover can be represented. |
| Retraining | Not required — Chronos-2 is zero-shot. Refit LightGBM and re-run the band calibration once real history exists. |

## Traceability
- KPIs: `bandwidth_utilization`, `network_availability`, `network_capacity_risk`
- Datasets: `D19`, `D01`, `D21`
- Upstream: `M01`
- Model card in the study: M20
