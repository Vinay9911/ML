---
name: build-model
description: Build or finish one Event Digital Twin model folder (M01–M25) end to end following the repo contracts. Use when asked to build, scaffold, complete or fix a specific model ID.
disable-model-invocation: true
---
Build model $ARGUMENTS.

1. **Read narrowly.** CLAUDE.md rules; docs/02_CONTRACTS.md §2, §4, §6–§9; ONLY the $ARGUMENTS card in
   docs/03_MODEL_SPECS.md; the KPI rows owned by $ARGUMENTS and any formulas/bands/reason codes it references in
   docs/05. Do not read other model cards.
2. **Check dependencies.** Look up $ARGUMENTS in `00_common/config/model_registry.yaml`. Confirm the required engine
   exists in `twin_common.engines`; if not, stop and say which phase must build it. For each upstream ID, ensure
   `data/sample_upstream/<ID>__S01.json` exists (run `uv run python scripts/refresh_samples.py --for $ARGUMENTS`).
3. **Scaffold if needed.** If the folder is missing: `uv run python scripts/new_model.py $ARGUMENTS`, then
   `uv run python scripts/slice_world.py $ARGUMENTS`.
4. **Short plan** (in the conversation): files to change, KPIs and their formulas, engine calls, config keys with
   values from assumptions.yaml or the card, scenarios supported, tests. If plan mode is on, wait for approval.
5. **Implement** `src/model.py` (+ `features.py` if needed) using shared engines. All numbers in `config.yaml`.
   No imports from other model folders. Set `is_synthetic`, `data_source`, upstream refs, reason codes, bands.
6. **Tests** in `tests/`: contract validity; every owned KPI present with registry unit; bounds (lower ≤ value ≤ upper,
   % within 0–100 unless the KPI allows overflow); determinism; scenario directions from docs/05 §1.2; API smoke.
7. **Run and capture evidence:**
   - `uv run pytest <folder>/tests -q`
   - `TWIN_OFFLINE=1 uv run pytest <folder>/tests -q`
   - start the API on its port and call /health, /metadata, /predict and /scenario for each supported scenario
     (or use TestClient); save `outputs/sample_output.json` and `outputs/sample_scenario_<Sxx>.json`
   - grep `src/` for stray numeric literals used as thresholds/rates
8. **Docs.** Fill README.md and model_card.md from the templates in docs/02 §11–§12 (method, assumptions with config
   keys, metrics with the synthetic-data caveat, limitations, license notes, real-data swap).
9. **Definition of Done.** Print docs/02 §9 as a checklist with ✅/❌ and the evidence line for each item.
10. **Review.** Run the `model-reviewer` subagent on the folder; fix blocking gaps only; re-run step 7 if code changed.
11. **Finish.** Refresh downstream samples: `uv run python scripts/refresh_samples.py --from $ARGUMENTS`.
    Update PROGRESS.md; commit `feat($ARGUMENTS): <short description>`; stop and summarize in ≤ 10 lines.
