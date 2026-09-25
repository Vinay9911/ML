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
| Phases complete | **5 of 10** — 0, 1, 2, 3 and **4** |
| Models complete | **10 of 25** — M01, M05, M06, M15, M17, M18, M19, M20, M21, M22 |
| Tests | **1,282 passing** |
| Repository checks | `validate_all.py` 73 passed, 0 failed |

`PROGRESS.md` is the live tracker: per-phase status, per-model status, and a decisions log
recording every assumption that was changed and why.

## What is here

```
├── 00_common/          the shared package `twin_common` — contracts, config, engines,
│                       synthetic world generator, API factory. Everything else depends on it.
├── M01_footfall_forecast/   ┐
├── M05_parking_demand/      │
├── M06_transit_demand/      │
├── M15_water_demand/        ├─ one self-contained service per model, identical structure
├── M17_food_supply/         │  (10 of 25 built so far)
├── M18_waste_forecast/      │
├── M19_power_load/          │
├── M20_network_capacity/    │
├── M21_weather_impact/      │
├── M22_environmental_risk/  ┘
├── docs/               the specification: brief, contracts, 25 model cards, data model,
│                       scenarios and KPIs, tech stack, implementation phases
├── scripts/            scaffolding, world slicing, sample refresh, repository validation,
│                       and serve_all.py — every built model on one port for the live demo
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
    weather        footfall         parking    shuttles
                  (13 models                  (reads BOTH
                   read this)                  M01 and M05)
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
| **To see the models actually run** | **`docs/SIMULATION.html`** — start the services first, see below |
| A plain-language tour of the project | **`docs/EXPLAINER.html`** — open it in a browser |
| What every table field and every output number means | **`docs/DATA_GUIDE.html`** |
| The current state of the build | `PROGRESS.md` |
| The output contract and API | `docs/02_CONTRACTS.md` |
| What one model does and why | that model's `model_card.md` |
| Why an assumption was changed | the decisions log in `PROGRESS.md` |

### About `docs/SIMULATION.html` — the live model map

```bash
uv run python scripts/serve_all.py      # one process, every built model, port 8000
# then open docs/SIMULATION.html in a browser
```

A page that runs the real models and shows the whole pipeline: the input tables flowing in, the
model working, the output coming out. Click the input box to browse the actual rows being read;
click the model box for a plain-English explanation of the algorithm; click the output for every
record with its risk colour, any recommended actions, and the raw JSON the dashboard receives.
A scenario selector re-runs any model under a what-if.

**Nothing on the page is illustrative.** Inputs are read from the generated world on disk,
outputs are what the model actually returned, and the timings are measured. Models that are not
built yet appear as greyed-out roadmap tiles, so the page grows by itself as the project does —
it discovers what exists from `/models` rather than having a hard-coded list.

Two things worth knowing before demoing it:

- **Warm-up takes about 2½ minutes** for all ten models (M01 alone is ~50 s: it reads a month of
  history, fits a forecaster and calibrates its uncertainty bands). The page enables each model
  as it becomes ready, so you can start after the first one.
- **It needs the services running on the same machine.** Opening the file with nothing running
  shows a banner telling you the command, not a broken page — but it cannot show live numbers to
  someone who does not have the repo. It also cannot be published as a shareable web page: a
  hosted page is not allowed to call `localhost`, so a published copy could never reach the
  services. Share `docs/EXPLAINER.html` instead, or screen-share the simulation.

### About `docs/DATA_GUIDE.html` — the field guide

A reference for anyone reading the data: every input table with each field explained, every
output KPI with its unit and what it actually tells you, and the risk-band tables that decide
the colours. It also covers the three conventions that confuse people first — why the dates are
in 2027, why a timestamp repeats, and why `06:15` is the morning.

Every figure in it was read from the running models or the generated world rather than written
from memory.

### About `docs/EXPLAINER.html`

Open it in any browser — it is a single self-contained file, no server or build step needed.

It is written for someone who has never seen this repository: what the project is, how the
pieces fit together, what technology sits behind each model, how to run one, what you can ask
it and what you get back. Every model that has actually been built gets its own section with a
runnable example and an honest list of what it cannot do.

**It is a living document and is updated every time a model is finished** — a new section
appears, the counts in its header move, and the "still to build" table shrinks. This is part of
the per-model workflow in `CLAUDE.md`, not a manual step someone has to remember.

If it ever disagrees with `PROGRESS.md`, `PROGRESS.md` is right.
