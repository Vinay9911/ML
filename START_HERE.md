# START HERE — How to use this kit with Claude Code in VS Code

This file is for you (the human). Claude Code reads `CLAUDE.md` automatically and the `docs/` on demand.

> **This file describes how to RUN the build.** For what the project is and what state it is in,
> read `README.md`, then `docs/EXPLAINER.html` for a plain-language tour, then `PROGRESS.md`
> for the live status. Sections 1–4 below describe the kit as it shipped; the build has since
> added `00_common/`, `scripts/`, `templates/`, `plans/` and one folder per completed model.

## 1. What is in the kit
```
event-twin-claude-kit/
├── README.md                         ← what the project is, current status (added during the build)
├── START_HERE.md                     ← you are here (human guide)
├── CLAUDE.md                         ← Claude Code's persistent project memory (loaded every session)
├── PROGRESS.md                       ← tracker Claude updates after each phase/model
├── docs/
│   ├── 01_PROJECT_BRIEF.md           ← purpose, scope, demo story, glossary
│   ├── 02_CONTRACTS.md               ← folder template, JSON output, API, ports, Definition of Done
│   ├── 03_MODEL_SPECS.md             ← 25 model cards (method, inputs, KPIs, scenarios, tests)
│   ├── 04_DATA_AND_SYNTHETIC_WORLD.md← data tables, venue layout, generator, assumptions
│   ├── 05_SCENARIOS_KPIS_FORMULAS.md ← S01–S15, all 93 KPIs, formulas, risk bands, metrics
│   ├── 06_TECH_STACK_AND_GOTCHAS.md  ← approved libraries, licenses, known pitfalls
│   └── 07_IMPLEMENTATION_PHASES.md   ← Phases 0–9 with copy-paste prompts and acceptance checks
└── .claude/
    ├── skills/build-model/SKILL.md   ← `/build-model M05` builds one model end to end
    └── agents/model-reviewer.md      ← independent reviewer subagent
```

## 2. Do you need to give Claude Code the PDF?
**No.** Everything Claude Code needs from the PDF is already transcribed into `docs/`, corrected, and turned into
decisions (the PDF alone describes *what*; the docs also fix *how*, *with which libraries*, and *in what format*).

| PDF page | PDF section | Where it lives now |
|---|---|---|
| 1 | Executive summary, domains, readiness checklist | docs/01 §1–§8 |
| 2 | Master KPI catalogue (93 KPIs) | docs/05 §2 (all 93, with owners) |
| 3 | AI/ML model catalogue (M01–M25) | docs/03 (rewritten per model) |
| 4 | Training data catalogue (D01–D25) | docs/04 §4 |
| 5 | Resource forecasting (15 resource types) | docs/05 §5 and M24 card |
| 6 | What-if scenarios (S01–S15) | docs/05 §1 |
| 7 | GIS & environmental layer | docs/04 §2 and model cards |
| 8 | Data → AI → twin → decision architecture | docs/02 and docs/04 (demo-sized) |
| 9 | KPI formula reference | docs/05 §6–§7 |
| 10 | Governance controls | docs/02 §4, §9 and CLAUDE.md rules |
| 11 | Reference sources | docs/05 §3 notes, docs/06 §4 |

Optional: keep the PDF in `docs/reference/` for your own traceability. CLAUDE.md tells Claude not to read it unless
you ask, because parts of it are superseded (e.g. model families changed, a few errors fixed) and reading it would
waste context. If you ever want Claude to cross-check against the original, give it **pages 2, 3 and 6**
(KPIs, models, scenarios) — those are the only pages where a line-by-line comparison is useful.

## 3. Prerequisites
- VS Code with the Claude Code extension (or Claude Code CLI in the VS Code terminal), signed in.
- Git, Python 3.11, and `uv` installed.
- 16 GB RAM recommended. NVIDIA GPU optional (only speeds up vision precompute and Chronos-2).
- Windows: fine throughout. Every dependency in docs/06 — including JuPedSim, OR-Tools, spopt,
  RF-DETR and the vision stack — installed natively on Windows 11 / Python 3.11 with no WSL2
  needed, contrary to the caution below. WSL2 remains the fallback only if `eclipse-sumo` is
  wanted, and SUMO is optional.
- Internet on first runs (downloads of weather, OSM roads, model weights); afterwards everything works offline.

## 4. Setup (15 minutes)
1. Create an empty folder, e.g. `event-twin/`, and copy **all** kit contents into it (including the hidden `.claude/`).
2. Optionally copy the PDF to `event-twin/docs/reference/`.
3. `git init` and make a first commit of the kit.
4. Open the folder in VS Code and start Claude Code.
5. Do NOT run `/init` (it would generate a new CLAUDE.md; ours is already written).
6. Ask: `Summarize CLAUDE.md in 5 bullets and list the docs you can read.` — confirms the memory file is loaded.
   (`/context` also shows what is loaded.)

## 5. Before Phase 1 — confirm with your manager (1 short meeting)
- The JSON output format and the 4 endpoints (docs/02 §4 and §6).
- Folder names and ports 8001–8025 (docs/02 §3).
- Delivery format: 25 standalone zips with shared code vendored inside (default) or 25 folders + `00_common`.
- That Parquet/GeoJSON files (no database) are fine for the demo.

## 6. What you must prepare yourself
- **Crowd videos (before Phase 6):** 2–4 clips you are allowed to use; one gate/queue view, one dense crowd view.
  Put them in `M02_crowd_density_flow/data/video/` with a `SOURCES.md` (link + license).
- **Venue location (before Phase 2):** the demo uses approximate Ramkund, Nashik coordinates by default. Change
  `venue_center` in `00_common/config/world.yaml` if needed. Optionally draw real zones at geojson.io and save as
  `00_common/config/zones.geojson` using the zone IDs Z01–Z08.
- **Assumptions review (any time):** every rate in `assumptions.yaml` is a placeholder. A 30-minute review with a
  police/medical/event-safety person makes the demo far more credible.

## 7. How to run each phase (repeat 10 times)
1. Start fresh: `/clear` (and optionally `/rename phase-3`).
2. Turn on plan mode: press **Shift+Tab** until the status shows plan mode.
3. Open `docs/07_IMPLEMENTATION_PHASES.md`, copy the phase's **Plan prompt**, paste it.
4. Read `plans/PHASE_<n>.md`. Edit it if needed (Ctrl+G opens the plan in your editor). Answer its questions.
5. Approve the plan, then paste the **Build prompt**.
6. Check the evidence Claude shows (test output, commands). Don't accept "done" without output.
7. Paste the generic **Review prompt** (top of docs/07). Let it fix blocking gaps.
8. Confirm `PROGRESS.md` is updated and a commit exists. Then `/clear` before the next phase.

For single models after the engines exist, just run: `/build-model M05` (one model per fresh context).

## 8. Working tips
- If you correct Claude twice on the same issue, `/clear` and restart the task with a sharper prompt.
- Press **Esc** to stop Claude mid-action; **Esc Esc** or `/rewind` to go back to a checkpoint.
- Keep one session per phase; resume later with `claude --continue` or `--resume`.
- Commit after every model. Git is your real safety net.
- When Claude wants to add a new library, check docs/06 first; say no to AGPL/BSL packages.
- Parallel work is possible after Phase 1 (e.g. Phase 6 vision in a separate git worktree/session).

## 9. Suggested schedule
| Week | Phases | Result you can show |
|---|---|---|
| 1 | 0, 1, 2 | Contracts + synthetic world with plots |
| 2 | 3, start 4 | First model (M21) + template; M01 forecasting |
| 3 | finish 4 | 10 models done |
| 4 | 5 | 15 models done |
| 5 | 6 (parallel) + start 7 | Vision working on real clips |
| 6 | finish 7 | 23 models done |
| 7 | 8, 9 | All 25 + end-to-end scenarios + delivery zips |

## 10. Troubleshooting
| Problem | What to do |
|---|---|
| Claude ignores a rule | Check CLAUDE.md is loaded (`/context`); put "IMPORTANT" on that one rule; keep the file short |
| Library install fails | Ask Claude to stop and report; try WSL2; check docs/06 fallback (SUMO optional, LWCC vendoring) |
| Chronos-2 slow on CPU | Use `autogluon/chronos-2-small` in config; reduce horizon/entities for the demo |
| Model takes ~25 s to start | Expected: it fits once and calibrates its uncertainty bands at startup so requests stay fast. Set `params.forecast.calibration.enabled: false` to skip it |
| A band looks too wide | It is conformally calibrated to its stated coverage; the raw backend band was far too narrow. See any forecasting model's card |
| Vision too slow | Lower `sample_fps`, shorter clips, precompute once, serve cached results |
| JuPedSim geometry errors | Ask Claude to validate polygons with shapely and simplify geometry |
| Context gets messy | `/clear`, then restart with the phase prompt; the plan file keeps continuity |
| Output rejected by frontend | Run `scripts/validate_all.py`; contract changes need your manager's approval first |
