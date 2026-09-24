# Event Digital Twin — AI/analytics layer

Twenty-five independent model services for a **Digital Twin Command Centre** at a mega
religious gathering (Kumbh Mela scale — tens of millions of visitors, peak bathing days,
river ghats, a temporary city).

Each service answers **one operational question** — "how many people will arrive at Ghat A in
the next three hours?", "when does car park P2 overflow?" — and returns it in an identical
JSON shape, so the dashboard team writes one parser for all twenty-five.

> **All data is synthetic.** There is no real operational feed yet. Every row and every
> response carries `is_synthetic: true`, and every rate in `00_common/config/` is a
> placeholder pending expert review. The synthetic world is internally consistent and
> scenario-aware, and each table can be swapped for a real feed without changing model code.

## Status

| | |
|---|---|
| Phases complete | **4 of 10** (0–3); Phase 4 in progress |
| Models complete | **3 of 25** — M01, M05, M21 |
| Tests | **790 passing** |
| Repository checks | `validate_all.py` 32 passed, 0 failed |

`PROGRESS.md` is the live tracker: per-phase status, per-model status, and a decisions log
recording every assumption that was changed and why.

## What is here

```
├── 00_common/          the shared package `twin_common` — contracts, config, engines,
│                       synthetic world generator, API factory. Everything else depends on it.
├── M01_footfall_forecast/   ┐
├── M05_parking_demand/      ├─ one self-contained service per model, identical structure
├── M21_weather_impact/      ┘
├── docs/               the specification: brief, contracts, 25 model cards, data model,
│                       scenarios and KPIs, tech stack, implementation phases
├── scripts/            scaffolding, world slicing, sample refresh, repository validation
├── templates/          the model-folder template `new_model.py` renders
├── plans/              the approved plan for each phase
├── CLAUDE.md           project rules loaded by Claude Code every session
└── START_HERE.md       human guide to running the build
```

Every model folder looks the same:

```
M05_parking_demand/
├── config.yaml         EVERY number the model uses — no thresholds live in code
├── src/model.py        the model itself
├── src/run.py          CLI: run once, write JSON
├── api.py              FastAPI app: /health /metadata /predict /scenario
├── tests/              contract, behaviour, scenario and API tests
├── outputs/            sample responses, valid against the schema
├── schemas/            the JSON schemas the responses satisfy
├── model_card.md       method, assumptions, evaluation, limitations, real-data readiness
└── README.md           what it does and how to run it
```

## Quick start

```bash
uv venv --python 3.11
uv pip install -e "00_common[dev,geo,network,forecast,formula,mlclf,opt,vision,pedsim]"

uv run pytest -q                              # the whole suite
uv run python scripts/validate_all.py         # contracts, schemas, isolation, licences
uv run uvicorn api:app --app-dir M05_parking_demand --port 8005
```

Then:

```bash
curl -X POST localhost:8005/predict -H 'content-type: application/json' -d '{}'
curl -X POST localhost:8005/scenario -H 'content-type: application/json' \
     -d '{"scenario_id": "S02", "compare_to_baseline": true}'
```

Everything works offline after the first run: weather, road graph and model weights are
cached under `00_common/data/real_cache/`. Set `TWIN_OFFLINE=1` to forbid network access
entirely — services degrade and say so rather than failing.

## How it fits together

```
     weather / OSM roads / AI4I          synthetic world generator
     (real public sources, cached)   +   (31 days x 8 zones, 6 scenarios)
                        │
                        ▼
              ┌──────────────────┐
              │  twin_common     │  contracts · config · engines · IO · upstream
              └──────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
      M21 ──────────▶ M01 ──────────▶ M05 ──▶ M06 ──▶ ... ──▶ M25
    weather        footfall         parking
```

Models never import each other. A model reads an upstream model's **JSON output**, resolved
in a fixed order: inline in the request → `UPSTREAM_<ID>_URL` → a sample file → a stub. So any
service runs alone, and the response always says which source it actually used.

## The nine engine families

| Engine | Models | What it actually is |
|---|---|---|
| `forecast` | 9 | Chronos-2 transformer foundation model (zero-shot) → LightGBM → naive seasonal |
| `rules` | 4 | threshold and weighted-score logic; no learning |
| `network` | 3 | shortest paths on a real OpenStreetMap graph + the BPR volume-delay function |
| `vision` | 2 | RF-DETR detection + ByteTrack tracking + LWCC crowd counting; optical flow |
| `formula` | 2 | published deterministic equations (NOAA heat index) and a Poisson/NegBin GLM |
| `optimize` | 2 | mixed-integer linear programming (PuLP / OR-Tools) |
| `location` | 1 | MCLP / LSCP facility location (spopt) |
| `pedsim` | 1 | agent-based pedestrian simulation (JuPedSim) |
| `mlclf` | 1 | the one conventional supervised classifier (LightGBM on the AI4I 2020 dataset) |

Most of this is deterministic engineering rather than machine learning — deliberately. An
operator has to be able to argue with a recommendation before approving it, and every
recommendation carries `requires_approval: true`.

## Rules that hold everywhere

- Every response validates against `twin_common.contracts.ModelOutput`. No exceptions.
- Every row and every response carries `is_synthetic`.
- No numeric threshold, rate or weight in `src/` — they live in `config.yaml` or
  `00_common/config/*.yaml`.
- All randomness comes from `seed`. Same inputs, same outputs.
- Every network fetch caches to disk and has a fallback.
- Licences: Apache-2.0, MIT, BSD, LGPL, EPL-2.0, CC-BY data only.

## Where to read next

| You want | Read |
|---|---|
| A plain-language tour of the project | **`docs/EXPLAINER.html`** — open it in a browser |
| The current state of the build | `PROGRESS.md` |
| The output contract and API | `docs/02_CONTRACTS.md` |
| What one model does and why | that model's `model_card.md` |
| Why an assumption was changed | the decisions log in `PROGRESS.md` |

`docs/EXPLAINER.html` is a living document: it covers workflow, technologies, and every model
that has actually been built, and **it is updated as each new model lands**. If it disagrees
with `PROGRESS.md`, `PROGRESS.md` is right.
