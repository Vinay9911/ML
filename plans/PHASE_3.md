# Phase 3 - Model template, scaffolder, sample tooling + M21

Status: implemented.

## Goal
A proven folder pattern and the first complete model, so the remaining 24 are mechanical.

## What was built
| Piece | Path | Notes |
|---|---|---|
| Folder template | `templates/model_folder/` | 13 `.tmpl` files covering the whole docs/02 section 2 layout |
| Scaffolder | `scripts/new_model.py` | derives config, KPIs, ports, upstream and scenarios from the registries, so a folder is correct by construction |
| World slicer | `scripts/slice_world.py` | copies only the tables a model declares; M21 slice is 132 KB against the 20 MB target |
| Stub generator | `twin_common/synthetic/stubs.py` | a contract-valid ModelOutput for any of the 25, derived from the world |
| Sample refresh | `scripts/refresh_samples.py` | topological order, real model where built, stub otherwise; also writes a 25-model catalogue |
| Repo validator | `scripts/validate_all.py` | 7 groups of checks, non-zero exit on failure |
| First model | `M21_weather_impact/` | complete, all 11 Definition of Done items evidenced |

## M21 design notes
- The NOAA heat index and the three weather multipliers live in `twin_common` and are shared
  with the synthetic generator, so a model reading M21 cannot see a multiplier different from
  the one that shaped the world it is reading. A test asserts the coefficients match.
- The heat index is **recomputed** from interpolated temperature and humidity rather than
  interpolated itself, because the regression is non-linear. A test asserts the reported
  value recomputes from its own reported inputs.
- Waterlogging uses a per-zone drainage rate, so the same rainfall floods a low-lying ghat
  and not the well-drained hub.

## Changes to Phase 1 code that Phase 3 forced
- `TwinModel.as_predict()` - every scenario implementation needs to narrow a ScenarioRequest
  to its baseline PredictRequest; it was going to be re-derived in 25 folders.
- `TwinModel.handle_scenario()` - the "insensitive to Sxx" response is a contract requirement
  of the MODEL (docs/05 section 1.2), not of the HTTP layer. It used to live only in
  `create_app`, so `scripts/scenario_e2e.py` and any direct caller would have bypassed it.
- pytest `--import-mode=importlib` and no `tests/__init__.py`, because every model folder has
  its own `tests` package and the default import mode collides on the module name.

## Acceptance (docs/07 Phase 3)
- `new_model.py M21` creates the full tree - done, 22 files
- M21 Definition of Done shown item by item with evidence - 11/11 pass
- `/scenario` S03 raises waterlogging and S04 raises heat_index - both asserted
- heat index reference-value test passes - 15 NWS chart cells within 1.5 degF
- `validate_all.py` runs M21 + common tests - 21 checks pass, 0 fail
- `refresh_samples.py` writes stubs for all 25 IDs - 25 in the catalogue

## Open questions
1. The model card records no accuracy metric because M21 replays a stored series rather than
   forecasting. Metrics become meaningful only with a real forecast feed.
2. `confidence` is a fixed 0.8, labelled heuristic. It should become a calibrated number once
   there is forecast uncertainty to report.
