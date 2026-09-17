# Event Digital Twin — 25 Model Services (Demo Stage)

## What this repo is
AI/analytics layer for a mega-event (Kumbh-scale) Digital Twin Command Center.
Deliverable: 25 independent model folders `M01_…` to `M25_…` plus `00_common/`.
A separate developer integrates them into a frontend through a fixed JSON/API contract.
There is NO real data yet. Everything runs on a synthetic world plus a few real public
sources (Open-Meteo weather/air quality, OpenStreetMap roads, public crowd video, AI4I 2020 dataset).

## Source of truth — read on demand, never all at once
- `docs/01_PROJECT_BRIEF.md` — purpose, scope, demo story, glossary
- `docs/02_CONTRACTS.md` — repo layout, folder template, output JSON, API, IDs, ports, Definition of Done
- `docs/03_MODEL_SPECS.md` — one card per model; read ONLY the card you are building
- `docs/04_DATA_AND_SYNTHETIC_WORLD.md` — data tables, venue layout, generator, assumptions
- `docs/05_SCENARIOS_KPIS_FORMULAS.md` — S01–S15, all 93 KPIs, formulas, thresholds, metrics
- `docs/06_TECH_STACK_AND_GOTCHAS.md` — approved libraries, licenses, known pitfalls
- `docs/07_IMPLEMENTATION_PHASES.md` — phases, tasks, acceptance checks
Precedence when documents disagree: docs/02 > docs/03 > docs/05 > other docs > original PDF.
The original PDF (if present in `docs/reference/`) is background only — do not read it unless asked.

## Hard rules
- IMPORTANT: every model response must validate against `twin_common.contracts.ModelOutput`. Never invent another shape.
- Every data row and every output carries `is_synthetic`. Never drop it.
- No numeric thresholds, rates or weights in `src/` code. They live in the model `config.yaml` or `00_common/config/*.yaml`.
- All randomness is seeded from config (`seed`). Same inputs → same outputs.
- Offline-safe: every network fetch (APIs, OSM, model weights, datasets) caches to disk and has a fallback.
- Model folders never import from other model folders. Shared code lives only in `00_common/twin_common`.
  Upstream model outputs are consumed as JSON (inline request → env URL → sample file → stub).
- Allowed licenses: Apache-2.0, MIT, BSD, LGPL, EPL-2.0, CC-BY data. Do NOT add `ultralytics`, `boxmot` (AGPL) or `sdv` (BSL).
  Ask before adding any dependency not listed in docs/06.
- Tracking: use `trackers.ByteTrackTracker` (supervision's `sv.ByteTrack` is deprecated).
- Forecasting with covariates: use Darts `Chronos2Model` (Darts `TimesFM2p5Model` has no covariate support).
- If a spec is ambiguous or a library will not install, STOP and ask. Do not silently substitute.

## Environment & commands
- Python 3.11, package manager `uv`, formatter/linter `ruff`, tests `pytest`.
- Setup: `uv venv --python 3.11` then `uv pip install -e "00_common[dev]"` (add extras per engine: forecast, vision, network, opt, pedsim).
- One folder tests: `uv run pytest M01_footfall_forecast/tests -q`
- All checks: `uv run python scripts/validate_all.py`
- Lint/format: `uv run ruff check . --fix` and `uv run ruff format .`
- Run a model API: `uv run uvicorn api:app --app-dir M01_footfall_forecast --port 8001`
- Generate synthetic world: `uv run python -m twin_common.synthetic.generate --scenario S01`
- Scaffold a model folder: `uv run python scripts/new_model.py M05`
- Build one model end-to-end: `/build-model M05` (skill in `.claude/skills/build-model`)

## Workflow
- Start each phase in plan mode. Write the plan to `plans/PHASE_<n>.md` and wait for approval.
- Build one model at a time. "Done" = Definition of Done in docs/02 §9, shown with evidence (command + output).
- Before marking a model done, run the `model-reviewer` subagent and fix blocking gaps only.
- End of phase: run `scripts/validate_all.py`, update `PROGRESS.md`, commit (`feat(M05): …`), stop and report.
- When compacting, preserve: current phase, model in progress, files modified, failing tests, open questions.

## Code conventions
- Type hints everywhere; pydantic v2 for all request/response models; pandas + Parquet for tables; pathlib for paths.
- Timestamps: timezone-aware ISO 8601 in Asia/Kolkata (+05:30). Time grid: 15 minutes.
- KPI names are snake_case and must exist in `00_common/config/kpi_registry.yaml`.
- Use `twin_common.logging.get_logger`; no `print` in library code.
