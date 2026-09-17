# Phase 0 — Repository bootstrap

Status: implemented.

## Goal
Working toolchain and skeleton so every later phase has a place to put files and a passing lint/test baseline.

## Environment decisions (recorded)
| Item | Decision | Why |
|---|---|---|
| Python | 3.11.16 via `uv`-managed CPython (`uv venv --python 3.11`) | Machine only had 3.13; docs/06 §2 requires 3.11 (jupedsim/darts/rfdetr) |
| Package manager | `uv` 0.12.5 | CLAUDE.md |
| Venv location | `.venv/` at repo root | gitignored |
| Heavy extras | installed per phase, not up front | torch/darts/jupedsim are GB-scale; Phases 0–3 need core+dev+geo only |

## Files created
| File | Content outline |
|---|---|
| `.gitignore` | docs/06 §2 list: `.venv/`, `**/data/video/*` (keep SOURCES.md), `00_common/data/world/`, `00_common/data/real_cache/`, weights, `delivery/`, `__pycache__/`, build/test caches |
| `pyproject.toml` (root) | dev tooling ONLY — no package. `[tool.ruff]` line-length 100, target py311, lint rule selection; `[tool.pytest.ini_options]` testpaths, markers (`slow`, `network`, `sumo`), strict markers |
| `.python-version` | `3.11` so `uv` picks the right interpreter automatically |
| `scripts/.gitkeep`, `templates/model_folder/.gitkeep`, `plans/`, `reports/` | directory skeleton from docs/02 §1 |
| `00_common/{twin_common,config,schemas,tests,data/{bundled,real_cache,world}}` | package skeleton, filled in Phase 1 |

## Not in scope for Phase 0
No `twin_common` code, no config yaml content, no models. Phase 0 only proves the toolchain.

## Acceptance checks (docs/07 Phase 0)
1. `uv run python --version` → 3.11.x
2. `uv run ruff check .` → passes
3. `uv run pytest -q` → runs, 0 tests collected, exit ok

## Open questions
1. Delivery format (vendored `twin_common` per zip vs shared `00_common`) — docs/02 §10 default is vendored; confirm before Phase 9.
2. `venue_center` coordinates — docs/04 §2 default (Ramkund, Nashik 20.0063/73.7926) used until a real `zones.geojson` is supplied.
3. Push target `https://github.com/Vinay9911/ML` — remote added, push deferred until the user confirms.
