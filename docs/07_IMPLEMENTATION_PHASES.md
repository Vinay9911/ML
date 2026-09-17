# 07 — Implementation Phases

Ten phases. Each phase = plan (plan mode) → implement → verify with evidence → review → commit → stop.
Estimated durations assume one developer supervising Claude Code; adjust freely.

## Standard phase loop (applies to every phase)
1. Fresh context (`/clear`), plan mode on.
2. Paste the phase's **Plan prompt**. Claude writes `plans/PHASE_<n>.md` (files, interfaces, tests, risks, questions).
3. Human reviews/edits the plan, answers questions, approves.
4. Paste the **Build prompt**. Claude implements, runs the acceptance commands and shows the output.
5. Paste the **Review prompt**. Fix blocking gaps only.
6. Claude updates `PROGRESS.md`, commits, and stops.

Generic Review prompt (use at the end of every phase):
```
Use a subagent to review the Phase <n> changes against plans/PHASE_<n>.md, docs/02_CONTRACTS.md and the acceptance
checks in docs/07 Phase <n>. Report only gaps that break a requirement, a contract, a test, determinism, offline mode
or licensing. No style suggestions. Then fix the reported blocking gaps and re-run the acceptance checks.
```

---

## Phase 0 — Repository bootstrap (½ day)
**Goal:** working toolchain and skeleton.
**Tasks:** `git init`; root `pyproject.toml` (ruff: line-length 100, target py311; pytest config); `.gitignore` per docs/06
§2; folders `scripts/ templates/ plans/ 00_common/`; `PROGRESS.md` from the kit; verify `uv`, Python 3.11.
**Acceptance:** `uv run python --version` shows 3.11.x · `uv run ruff check .` passes · `uv run pytest -q` runs (0 tests ok).

**Plan prompt:**
```
Read CLAUDE.md and the Phase 0 section of docs/07_IMPLEMENTATION_PHASES.md. Plan the repository bootstrap:
list every file you will create and its content outline. Write the plan to plans/PHASE_0.md and wait for approval.
```
**Build prompt:**
```
Implement plans/PHASE_0.md. Run the Phase 0 acceptance commands and paste their output. Update PROGRESS.md and commit
with message "chore: bootstrap repository". Stop after that.
```

---

## Phase 1 — `00_common` core: contracts, config, IO, API factory (2–3 days)
**Goal:** the shared package every model depends on.
**Tasks:**
- `00_common/pyproject.toml`: package `twin_common`; extras `dev, geo, network, forecast, formula, mlclf, opt, vision,
  pedsim, all` (dependencies from docs/06 §1).
- `twin_common.contracts`: pydantic v2 models `ModelOutput`, `ResultRecord`, `UpstreamRef`, `Recommendation`,
  `PredictRequest`, `ScenarioRequest`, `Metadata`, enums (state, status, data_source, risk_level, entity_type), ID
  regexes; validators for rules in docs/02 §4 (tz-aware, lower ≤ value ≤ upper, registry KPI/unit match,
  requires_approval true, reason codes when amber+). `export_schemas()` → `00_common/schemas/*.json`.
- `twin_common.config`: load/merge yaml (shared + model config + env overrides), typed access.
- Config files in `00_common/config/`: `assumptions.yaml` (docs/04 §6), `world.yaml` (layout/time/routing from docs/04
  §2–3), `scenarios.yaml` (docs/05 §1.1), `risk_bands.yaml` (docs/05 §3), `resources.yaml` (docs/05 §5),
  `kpi_registry.yaml` (docs/05 §2: 93 study KPIs + extensions), `model_registry.yaml` (docs/02 §13),
  `reason_codes.yaml` (docs/05 §4).
- `twin_common.io`: table registry with column dtype schemas (docs/04 §4), `load_table/save_table`, data_source adapters
  (synthetic/replay/live-stub), `is_synthetic` enforcement.
- `twin_common.upstream`: resolution order docs/02 §7 (stub hook left as interface; implemented in Phase 3).
- `twin_common.output`: `OutputBuilder` (add results, risk level from bands, reason code validation, deltas vs baseline).
- `twin_common.scenarios`: load scenarios, merge overrides, generic `apply_adjustments` for multiplier-type overrides.
- `twin_common.api.create_app(ModelClass)`: the 4 endpoints, CORS, error handling, logging.
- `twin_common.testing`: `assert_valid_output`, `assert_kpi_coverage`, `assert_deterministic`, `api_smoke`.
- `TwinModel` base class/protocol: `load(config)`, `predict(req)`, `scenario(req)`, `metadata()`.
**Acceptance:**
- `uv pip install -e "00_common[dev]"` works; `uv run pytest 00_common/tests -q` passes.
- Test asserts registry has exactly 93 non-extension KPIs and every owner is a valid model ID; every model in
  model_registry owns ≥ 1 KPI; upstream graph is acyclic.
- A dummy model built with `create_app` passes `api_smoke` and produces a valid `ModelOutput`.
- JSON Schemas exported.

**Plan prompt:**
```
We are starting Phase 1. Read CLAUDE.md, docs/02_CONTRACTS.md, docs/05_SCENARIOS_KPIS_FORMULAS.md sections 1–5, and
the Phase 1 section of docs/07. Do not read docs/03 yet. In plan mode, propose the twin_common package structure,
the pydantic models with all fields and validators, the yaml config files you will generate and how you will
transcribe them from the docs, and the tests. List open questions. Write plans/PHASE_1.md and wait.
```
**Build prompt:**
```
Implement plans/PHASE_1.md. Transcribe config yaml files exactly from the docs (no invented values). Run the Phase 1
acceptance checks and show the output, including the KPI count assertion. Update PROGRESS.md, commit
"feat(common): contracts, config, io, api factory", and stop.
```

---

## Phase 2 — Synthetic world generator + real data fetchers (3–4 days)
**Goal:** one consistent, scenario-aware world with validation.
**Tasks:** implement docs/04 §2–§8: `synthetic/layout.py`, `calendar.py`, `crowd_flow.py`, `derived.py`, `assets.py`,
`roster.py`, `generate.py` (CLI: `--scenario`, `--seed`, `--offline`), `validate.py`, `report.py`;
`real/weather.py` (Open-Meteo historical weather + air quality, cache, fallback), `real/roads.py` (OSMnx v2 graph,
cache, synthetic grid fallback with same key IDs), `real/ai4i.py` (UCI 601 download, cache, rule-based regenerator).
Small bundled fallbacks in `00_common/data/bundled/` so offline generation works on a fresh clone.
**Acceptance:**
- `generate --scenario S01` and `--scenario S02` each finish < 2 min on CPU; validation passes.
- S02/S01 daily arrivals ratio in [1.28, 1.32]; S05 G02 entries = 0 and Z05 peak density > S01; S08, S09, S12
  invariants from docs/04 §8 pass.
- `TWIN_OFFLINE=1 generate --scenario S01` succeeds using bundled/cached data.
- Sanity report written; peak-day ghat density reaches amber in S01.
- Deterministic: two runs with the same seed produce identical file hashes (except manifest timestamps).

**Plan prompt:**
```
Phase 2. Read CLAUDE.md, docs/04_DATA_AND_SYNTHETIC_WORLD.md fully, docs/05 section 1, docs/06 sections 3 (Network,
Data & APIs), and Phase 2 in docs/07. In plan mode design the generator modules, the crowd compartment model
equations, how each scenario override is applied in each step, caching/fallback for each real source, and the
validation checks. Write plans/PHASE_2.md with open questions and wait.
```
**Build prompt:**
```
Implement plans/PHASE_2.md. Run generation for S01, S02, S05, S08, S09, S12 plus the offline run, show the validation
summaries and the S02/S01 ratio, and attach the path of the sanity report. Update PROGRESS.md, commit
"feat(common): synthetic world generator", and stop.
```

---

## Phase 3 — Model template, scaffolder, sample tooling + M21 (2 days)
**Goal:** a proven folder pattern and the first complete model.
**Tasks:** `templates/model_folder/` per docs/02 §2; `scripts/new_model.py` (reads model_registry + kpi_registry,
fills config/README/model_card/tests skeletons); `scripts/slice_world.py`; `twin_common.synthetic.stubs` (contract-valid
stub outputs per model from world data, `source: stub`); `scripts/refresh_samples.py` (topological order: real model if
built, else stub; writes `data/sample_upstream/<ID>__<Sxx>.json`); `scripts/validate_all.py`; `engines/formula.py`
basics; build **M21** completely (docs/03 card).
**Acceptance:** `new_model.py M21` creates the full tree; M21 Definition of Done (docs/02 §9) shown item by item with
evidence; `/scenario` S03 raises waterlogging_probability and S04 raises heat_index; heat index reference-value test
passes; `validate_all.py` runs M21 + common tests; `refresh_samples.py` writes stubs for all 25 IDs.

**Plan prompt:**
```
Phase 3. Read CLAUDE.md, docs/02_CONTRACTS.md, the M21 card in docs/03, the M21 KPI rows and heat index/logistic
formulas in docs/05, and Phase 3 in docs/07. Plan the template files, the scaffolder logic, the stub generator, the
sample refresh ordering, and M21 implementation + tests. Write plans/PHASE_3.md and wait.
```
**Build prompt:**
```
Implement plans/PHASE_3.md. Then run /build-model M21 steps 6–10 to verify, print the Definition of Done checklist
with evidence, run the model-reviewer subagent on M21_weather_impact, fix blocking gaps, update PROGRESS.md, commit
"feat(M21): weather impact + model template tooling", and stop.
```

---

## Phase 4 — Forecast engine + 9 forecasting models (5–7 days)
**Goal:** `engines.forecast` and M01 (reference), then M05, M06, M15, M17, M18, M19, M20, M22.
**Engine tasks:** `ForecastEngine(config)` with backends `chronos2`, `lightgbm`, `naive_seasonal`; multi-series (one per
entity); past/future covariates; quantiles → value/lower/upper; `backtest()` returning MAE, MAPE, pinball, coverage;
CPU fallback to chronos-2-small; weights cache; scenario layer hook; timing log.
**Acceptance (engine):** unit tests on a synthetic sine+noise series; backtest metrics finite; offline mode uses cached
weights or raises a clear message; default M01 `/predict` < 10 s on CPU with the small model.
**Acceptance (each model):** Definition of Done; scenario directions from docs/05 §1.2; M01 records all three backends'
backtest metrics in model_card.

**Plan prompt (engine + M01):**
```
Phase 4a. Read CLAUDE.md, docs/06 section 3 (Forecasting), the M01 card in docs/03, M01 KPI rows and metrics in
docs/05, and Phase 4 in docs/07. Plan engines/forecast.py (API, backends, covariate handling, quantiles, backtest,
caching, CPU fallback) and M01 on top of it. Write plans/PHASE_4a.md and wait.
```
**Build prompt (engine + M01):**
```
Implement plans/PHASE_4a.md, then complete M01 with /build-model M01. Show engine tests, M01 backtest table for the
three backends, DoD evidence and reviewer result. Commit "feat(M01): forecast engine + footfall forecast" and stop.
```
**Then, one model per session (repeat for M05, M06, M15, M17, M18, M19, M20, M22):**
```
/build-model M05
```

---

## Phase 5 — Formula, statistical, rules and small-ML models (3–4 days)
**Goal:** `engines.formula` (Poisson helpers), `engines.rules` (weighted scores, logistic probabilities, overrides,
reason codes), `engines.mlclf` (LightGBM + calibration + AUC report); models in order **M07, M23, M03, M09, M12**.
**Acceptance:** DoD per model; monotonicity tests for every rule sub-score; M03 sustained-exposure override test; M07 S14
≈ 2× cases (±5%); M23 held-out AUC and calibration plot saved; M09 S09 and S11 directions; M12 S04 and S08 directions.

**Plan prompt:**
```
Phase 5. Read CLAUDE.md, Phase 5 in docs/07, and the cards M07, M23, M03, M09, M12 in docs/03 (only those), plus
their KPI rows, risk bands and reason codes in docs/05. Plan the three engines and the build order with tests.
Write plans/PHASE_5.md and wait.
```
**Build:** implement engines per plan, then `/build-model M07`, `/build-model M23`, `/build-model M03`,
`/build-model M09`, `/build-model M12` (clear context between models).

---

## Phase 6 — Vision engine: M02 and M10 (4–6 days; may run in parallel after Phase 1)
**Human prerequisite:** put 2–4 legally usable crowd clips in `M02_crowd_density_flow/data/video/` and document source
and license in `SOURCES.md` (M10 reuses them via config path). Mark one sparse (gate/queue) and one dense (ghat/plaza).
**Tasks:** `engines/vision.py`: frame sampler; `SparsePipeline` (rfdetr → sv.Detections → trackers.ByteTrackTracker →
LineZone/PolygonZone); `DensePipeline` (lwcc DM-Count QNRF, `resize_img=False`, ROI masking); `FlowAnalyzer` (Farneback
features); homography helpers + `scripts/calibrate_camera.py`; precompute CLI writing D02; synthetic test-video
generator (moving discs with scripted crossings, counterflow and surge segments).
**Acceptance:** tests pass on synthetic video (exact line-crossing counts in the simple case; M10 detects scripted
COUNTERFLOW and SURGE windows); precompute runs on at least one real clip on CPU; APIs serve precomputed data under
10 s; DoD for M02 and M10.

**Plan prompt:**
```
Phase 6. Read CLAUDE.md, docs/06 section 3 (Vision), cards M02 and M10 in docs/03, their KPI rows in docs/05, and
Phase 6 in docs/07. Before planning, check installed versions of rfdetr, trackers, supervision and lwcc and read their
current APIs from the installed packages (help()/source), not from memory. Plan the engine, precompute flow, camera
calibration fallback and the synthetic test video. Write plans/PHASE_6.md and wait.
```
**Build:** implement the engine, then `/build-model M02` and `/build-model M10`.

---

## Phase 7 — Network & optimization: M04, M14, M13, M16, M08, M24 (5–7 days)
**Tasks:** `engines/network.py` (graph load from cache, capacities, BPR incremental assignment, closures, k-shortest
paths, exposure-weighted costs, travel-time matrices); `engines/optimize.py` (PuLP helpers: capacitated set cover,
allocation with weighted unmet demand, time limits, status handling); `engines/location.py` (spopt MCLP/LSCP wrappers).
Build order: **M04 → M14 → M13 → M16 → M08 → M24**.
**Acceptance:** DoD per model; S06 and S12 increase travel time on alternates; M14 chosen plan keeps emergency corridors
within V/C limit; M08 coverage decreases under S10; M16 coverage constraints satisfied; M24 gap arithmetic exact,
allocations ≤ availability, highest-risk zones served first under shortage; all solvers report Optimal or degrade.

**Plan prompt:**
```
Phase 7. Read CLAUDE.md, docs/06 section 3 (Network/traffic, Optimization), cards M04, M14, M13, M16, M08, M24 in
docs/03, their KPI rows and resource types in docs/05, and Phase 7 in docs/07. Plan engines/network.py,
engines/optimize.py, engines/location.py with interfaces and tests, then the per-model builds. Write
plans/PHASE_7.md and wait.
```
**Build:** engines first, then `/build-model M04`, `/build-model M14`, `/build-model M13`, `/build-model M16`,
`/build-model M08`, `/build-model M24`.

---

## Phase 8 — Pedestrian simulation: M11 (+ optional SUMO) (3–5 days)
**Tasks:** `engines/pedsim.py` (walkable geometry in metres, exits, spawn from population, CollisionFreeSpeedModel run
with caps, per-exit flows, flow-model fallback, run cache); M11; optional `M04` SUMO experiment flagged
`experimental: true` in config and excluded from tests unless `TWIN_ENABLE_SUMO=1`.
**Acceptance:** blocking an exit increases evacuation_time; doubling agents increases time; sim vs flow-model within
`crosscheck_factor` on a rectangle; runtime cap respected; S15 → blocked_route_count ≥ 1; DoD for M11.

**Plan prompt:**
```
Phase 8. Read CLAUDE.md, docs/06 section 3 (Pedestrian simulation), the M11 card in docs/03, its KPI rows and the
evacuation formula in docs/05, and Phase 8 in docs/07. Inspect the installed jupedsim package API before planning.
Plan engines/pedsim.py, caching, fallback and tests. Write plans/PHASE_8.md and wait.
```
**Build:** implement, then `/build-model M11`.

---

## Phase 9 — M25, end-to-end scenarios, delivery packaging (3–4 days)
**Tasks:** build **M25**; `scripts/run_all.py` (start 25 APIs on 8001–8025, health check, clean shutdown);
`scripts/scenario_e2e.py` (topological run of all models for S01, S02, S02+S04, S05, S12, S14, S15 passing upstream
outputs inline; writes `reports/scenario_<id>.json` and a markdown comparison table of key KPIs vs S01);
`scripts/package_delivery.py` (docs/02 §10); license audit to `reports/licenses.md` failing on AGPL/GPL-only/BSL;
root `README.md` + `delivery/INTEGRATION_GUIDE.md` (ports, endpoints, curl examples, Mermaid dependency DAG, scenario
list, how to switch `UPSTREAM_<ID>_URL`).
**Acceptance:** all 25 `/health` OK via run_all; scenario_e2e completes; M25 overall_event_risk S15 > S01 and S14 > S01;
each delivery zip installs and passes its tests in a fresh temporary venv; license report clean; `validate_all.py` green.

**Plan prompt:**
```
Phase 9. Read CLAUDE.md, the M25 card in docs/03, docs/02 sections 7 and 10, and Phase 9 in docs/07. Plan M25, the
run_all / scenario_e2e / package_delivery scripts, the license audit and the integration guide. Write
plans/PHASE_9.md and wait.
```
**Build:** implement, `/build-model M25`, then run the full acceptance list and show the evidence.

---

## Phase summary
| Phase | Delivers | Models |
|---|---|---|
| 0 | Toolchain, skeleton | — |
| 1 | `twin_common` contracts/config/IO/API | — |
| 2 | Synthetic world + real fetchers | — |
| 3 | Template, scaffolder, stubs, samples | M21 |
| 4 | Forecast engine | M01, M05, M06, M15, M17, M18, M19, M20, M22 |
| 5 | Formula, rules, small ML | M07, M23, M03, M09, M12 |
| 6 | Vision | M02, M10 |
| 7 | Network & optimization | M04, M14, M13, M16, M08, M24 |
| 8 | Pedestrian simulation | M11 |
| 9 | Overall risk, e2e, delivery | M25 |
Total: 25 models.
