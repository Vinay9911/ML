# 06 — Tech Stack, Licenses & Gotchas

Verified against project documentation in September 2026. Do not pin versions in this document; pin them in the
lock/requirements after the first successful install. Run a license audit (`pip-licenses`) in Phase 9.

## 1. Approved dependencies
| Area | Package | License (verify in audit) | Used for | `twin_common` extra |
|---|---|---|---|---|
| Core | fastapi, uvicorn | MIT / BSD | model APIs | core |
| Core | pydantic v2 | MIT | contracts | core |
| Core | pandas, numpy, pyarrow, scipy | BSD / Apache | tables, Parquet, stats | core |
| Core | pyyaml, httpx | MIT / BSD | config, upstream calls, API fetchers | core |
| Dev | pytest, ruff, pip-licenses | MIT | tests, lint, license audit | dev |
| Geo | shapely 2, pyproj, geopandas | BSD / MIT | polygons, CRS, areas | geo |
| Network | osmnx (v2 API), networkx | MIT / BSD | road graph, shortest paths | network |
| Forecast | darts (`Chronos2Model`, `LightGBMModel`, `NaiveSeasonal`) | Apache-2.0 | forecasting engine | forecast |
| Forecast | lightgbm | MIT | baseline + M23 classifier | forecast / mlclf |
| Stats | statsmodels | BSD | GLM Poisson/NegBin (M07) | formula |
| ML | scikit-learn, ucimlrepo | BSD / (check) | calibration, AI4I download | mlclf |
| Optimization | pulp (CBC bundled) | permissive (check) | MILP (M16, M24) | opt |
| Optimization | ortools | Apache-2.0 | optional alternative solver/routing | opt |
| Location | spopt | BSD-3 (PySAL) | MCLP / LSCP (M08) | opt |
| Vision | opencv-python-headless | Apache-2.0 | frames, optical flow, homography | vision |
| Vision | rfdetr | Apache-2.0 core models (Plus/XL variants are a different license — do not use them) | person detection | vision |
| Vision | trackers | Apache-2.0 | ByteTrack/SORT tracking | vision |
| Vision | supervision | MIT | LineZone, PolygonZone, annotators | vision |
| Vision | lwcc | MIT wrapper; wrapped model licenses inherited | dense crowd counting | vision |
| Vision | torch, torchvision | BSD | backends | vision / forecast |
| Pedestrian sim | jupedsim | LGPL-3.0 | evacuation (M11) | pedsim |
| Traffic sim (optional) | eclipse-sumo, traci, sumolib | EPL-2.0 | experimental corridor sim | sumo (optional) |

Not allowed: `ultralytics` (AGPL-3.0; commercial use needs an enterprise license), `boxmot` (AGPL), `sdv`
(Business Source License, non-commercial), any GPL-only package without approval.

## 2. Environment
- Python **3.11** (compatible with jupedsim, rfdetr, trackers ≥3.10 requirements and darts). Use `uv`.
- CPU-only must work for everything. GPU (CUDA) is optional and only speeds up vision precompute and Chronos-2.
- Windows: most packages work natively. If `eclipse-sumo` or JuPedSim wheels fail, use WSL2 (Ubuntu). SUMO is optional.
- Set model/data caches inside the repo (gitignored): `HF_HOME=00_common/data/real_cache/hf`, `TORCH_HOME=…/torch`.
- `.gitignore`: `.venv/`, `**/data/video/*` (except SOURCES.md), `00_common/data/world/`, `00_common/data/real_cache/`,
  model weights, `delivery/`, `__pycache__/`.

## 3. Gotchas (read before coding the relevant engine)
### Forecasting
- In Darts, **`Chronos2Model` supports past and future covariates; `TimesFM2p5Model` does not support covariates.**
  Default to Chronos2Model. TimesFM may be used only as an optional univariate comparison.
- Smaller Chronos-2 for CPU: `hub_model_name="autogluon/chronos-2-small"` (28M params). Default `amazon/chronos-2` (120M).
- Darts `enable_finetuning` exists for foundation models — do not fine-tune on synthetic data for the demo.
- Foundation models need enough context: provide ≥ 2 weeks of 15-min history (the world has 28 days).
- Probabilistic output: request quantiles and map 0.1/0.5/0.9 to lower/value/upper; set `quantile_level: 0.8`.

### Vision
- **`supervision.ByteTrack` is deprecated (removal planned) — use `trackers.ByteTrackTracker` with `.update(detections)`.**
- `trackers` does not read video or detect; combine: OpenCV frames → rfdetr detections → `sv.Detections` → tracker.
- `sv.LineZone` needs tracker IDs; `sv.PolygonZone` counts per frame.
- LWCC: set `resize_img=False` for dense crowds (the default resizes large images, hurting dense counts); weights
  download on first use → cache; the package is small and old — if it breaks with current torch, vendor its model code
  and weights loader into `engines/vision_dense/` keeping its MIT notice and each model's license.
- Avoid P2PNet/APGCC for the demo: research code tied to old PyTorch/Python versions.
- Detection-based counting fails in very dense crowds (occlusion) → that is why dense ROIs use density-map counting.
- Precompute vision offline; APIs must not run GPU inference per request.
- Never add face recognition or identity features.

### Network / traffic
- OSMnx v2 removed v1 deprecated APIs (e.g. separate north/south/east/west args → `bbox` tuple). Check the v2 docs for
  argument order; always cache the graph to GraphML and load from cache.
- `eclipse-sumo` wheels are not always available for Windows for every release — keep SUMO optional; M04 uses the
  pure-Python BPR assignment in `engines.network`.
- Project graphs to a metric CRS before computing lengths/buffers.

### Optimization
- PuLP ships CBC; always check `LpStatus[prob.status] == "Optimal"` and return `status: degraded` otherwise.
- Keep MILPs small (8 zones × ≤ 15 types × ≤ 3 shifts); set a solver time limit param.
- spopt `MCLP`/`LSCP` take a cost matrix; build it from network travel times (minutes) and a service radius.

### Pedestrian simulation
- JuPedSim needs valid, simple polygons in metres; fix with `shapely.make_valid` and check `is_valid`.
- Cap agents (`max_agents`) and runtime; fall back to the flow model and say so in `details.method`.
- Cache runs by (zone, scenario, population bucket).

### Data & APIs
- Open-Meteo free tier is non-commercial with CC BY 4.0 attribution; cache everything; commercial use needs a plan.
- `TWIN_OFFLINE=1` must bypass all network calls (tests run offline).
- Timezones: keep tz-aware timestamps; never mix naive and aware in pandas joins.
- Seeds: numpy `default_rng(seed)`, torch manual_seed where applicable.

## 4. Useful references (for humans; Claude fetches only if needed)
- Darts foundation models examples: unit8co.github.io/darts (Foundation Model Examples notebook)
- Chronos: github.com/amazon-science/chronos-forecasting
- RF-DETR: github.com/roboflow/rf-detr · trackers: github.com/roboflow/trackers · supervision docs
- LWCC: github.com/tersekmatija/lwcc
- JuPedSim docs: jupedsim.org · SUMO docs: sumo.dlr.de/docs
- spopt: pysal.org/spopt · OSMnx: osmnx.readthedocs.io
- Open-Meteo: open-meteo.com/en/docs · AI4I 2020: archive.ics.uci.edu/dataset/601
