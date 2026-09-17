# Phase 1 — `00_common` core: contracts, config, IO, API factory

Status: implemented.

## 1. Package layout
```
00_common/
├── pyproject.toml                  # package "twin_common", extras per docs/06 §1
├── twin_common/
│   ├── __init__.py                 # version, re-exports
│   ├── logging.py                  # get_logger (no print in library code)
│   ├── paths.py                    # config_dir/data_dir/world_dir/is_offline resolution
│   ├── errors.py                   # TwinError hierarchy
│   ├── model.py                    # TwinModel ABC: load/predict/scenario/metadata
│   ├── contracts/
│   │   ├── __init__.py             # re-export everything
│   │   ├── enums.py                # State, Status, DataSource, RiskLevel, EntityType, UpstreamSource, Engine
│   │   ├── ids.py                  # entity-id regexes + validate_entity_id()
│   │   ├── registry.py             # KPI/reason-code/band lookups used by validators (cached)
│   │   ├── models.py               # ModelOutput, ResultRecord, UpstreamRef, Recommendation,
│   │   │                           # PredictRequest, ScenarioRequest, Metadata, KpiInfo
│   │   └── export.py               # export_schemas() -> 00_common/schemas/*.json
│   ├── config/__init__.py          # load_yaml, load_shared, load_model_config, deep_merge, env overrides
│   ├── io/
│   │   ├── schemas.py              # TABLE_SCHEMAS: D01–D25 column dtypes (docs/04 §4)
│   │   ├── adapters.py             # SyntheticAdapter / ReplayAdapter / LiveAdapter(NotImplementedError)
│   │   └── tables.py               # load_table / save_table / check_schema / is_synthetic enforcement
│   ├── output/
│   │   ├── risk.py                 # risk_level_from_bands()
│   │   └── builder.py              # OutputBuilder
│   ├── scenarios/__init__.py       # load_scenarios, resolve_overrides, apply_adjustments
│   ├── upstream/__init__.py        # resolve() with the docs/02 §7 order; stub hook interface
│   ├── api/__init__.py             # create_app(ModelClass)
│   ├── testing/__init__.py         # assert_valid_output, assert_kpi_coverage, assert_deterministic, api_smoke
│   └── engines/__init__.py         # namespace only; engines land in phases 4–8
├── config/                         # 8 yaml files, transcribed from the docs
└── tests/                          # 8 test modules
```

## 2. Where `twin_common` finds its config and data
One rule, works both in-repo and vendored (docs/02 §10):
`config_dir = Path(twin_common.__file__).parent.parent / "config"`, `data_dir = .../ "data"`.
- in repo: `00_common/twin_common/..` → `00_common/config` ✅
- vendored: `Mxx/vendor/twin_common/..` → `Mxx/vendor/config` ✅ (packaging copies `00_common/config` alongside)
Overridable by `TWIN_COMMON_CONFIG_DIR` / `TWIN_COMMON_DATA_DIR`. `TWIN_OFFLINE=1` → `paths.is_offline()`.

## 3. Config files transcribed (no invented values)
| File | Source | Notes |
|---|---|---|
| `assumptions.yaml` | docs/04 §6 verbatim | every value a placeholder |
| `scenarios.yaml` | docs/05 §1.1 verbatim | + `affects` list per scenario from docs/05 §1.2 |
| `risk_bands.yaml` | docs/05 §3 verbatim | exactly the 4 tables; no invented 5th table |
| `resources.yaml` | docs/05 §5 (15 types) + `available` from docs/04 §6 | |
| `kpi_registry.yaml` | docs/05 §2 | 93 study KPIs + 16 extensions = 109 |
| `model_registry.yaml` | docs/02 §13 verbatim | |
| `reason_codes.yaml` | docs/05 §4 verbatim | grouped by domain |
| `world.yaml` | docs/04 §2–§3 | zones, adjacency, routing, entities, time grid, venue_center |

## 4. Two contract decisions that the docs left implicit
**D1 — risk bands for "↓bad" percentage KPIs.** docs/05 §3 defines only 4 band tables and none fits a KPI where
100 is good (availability, coverage, effectiveness). Rather than invent a 5th table, each such KPI carries
`band: probability_pct` + `band_input: complement_100` in the registry, so the band is applied to the *shortfall*
(100 − value). Example: `emergency_route_availability` 100 → shortfall 0 → green; 70 → 30 → amber; 20 → 80 → critical.

**D2 — `AS\d{2}` ambulance-staging IDs.** docs/04 §2.2 requires entities AS01–AS06 but the docs/02 §3 ID table has no
`AS` prefix. Added to the `facility` pattern and flagged in `ids.py`; this is the one documented extension to §3.
Also noted: shuttle routes `SH1/SH2` (route, 1 digit) vs shelters `SH01/SH02` (facility, 2 digits) — distinguished by
`entity_type`, so both regexes can coexist.

## 5. Validators implemented on `ResultRecord` (docs/02 §4)
1. `entity_id` matches the regex for its `entity_type`.
2. `kpi` exists in `kpi_registry.yaml`; `unit` equals the registry unit exactly.
3. `lower <= value <= upper` whenever bands are present; `quantile_level` in (0,1).
4. value inside the registry's numeric bounds for its unit (`%` → 0–100 unless `allow_overflow`).
5. timestamps tz-aware; normalized to Asia/Kolkata (+05:30) on serialization.
6. `state: current` ⇒ `horizon_min == 0`.
7. `reason_codes` ⊂ `reason_codes.yaml`, UPPER_SNAKE_CASE, non-empty when `risk_level` ∈ {amber, red, critical}.
8. `recommendation.requires_approval` is always `True` (Literal[True]).
9. `delta == value - baseline_value` (±1e-6) when both present.
10. `resource_type` ∈ the 15 types in `resources.yaml`.
`ModelOutput`: `model_id` matches `M\d{2}`; `is_synthetic` must be True when `data_source == synthetic`; `status:
degraded` requires ≥1 warning; `scenario_id` matches `S\d{2}`.

Pydantic gotcha handled: fields named `model_*` need `protected_namespaces=()` in `model_config`, otherwise pydantic v2
warns and may shadow. The contract mandates `model_id`/`model_name`/`model_version`, so this is set on every affected model.

## 6. Tests (`00_common/tests`)
| Module | Asserts |
|---|---|
| `test_kpi_registry.py` | exactly 93 non-extension KPIs; 16 extensions; every owner a valid registry model ID; every model owns ≥1 KPI; every unit in the docs/02 §5 allowed list; band names exist in risk_bands.yaml |
| `test_model_registry.py` | 25 models; ports 8001–8025 unique and in order; upstream graph acyclic (topological sort); no model depends on itself; M24 has no M25 |
| `test_contracts.py` | valid output accepts; each of the 10 validator rules rejects a crafted bad record; tz normalization; round-trip JSON |
| `test_ids.py` | every documented ID example matches its type; cross-type mismatches rejected |
| `test_config.py` | shared configs load; deep_merge precedence (shared < model < env); assumptions keys present |
| `test_io.py` | save→load Parquet round-trip; dtype check failure raises; `is_synthetic` enforced; live adapter raises NotImplementedError |
| `test_output_builder.py` | risk band assignment incl. `complement_100`; reason-code requirement on amber+; delta computation |
| `test_scenarios.py` | all 15 load; override merge left→right; `apply_adjustments` multiplier behaviour |
| `test_api_factory.py` | dummy model → /health /metadata /predict /scenario valid; unknown scenario 404; unsupported scenario → degraded + warning |
| `test_schemas_export.py` | `export_schemas()` writes model_output/predict_request/scenario_request/metadata schemas |

## 7. Acceptance (docs/07 Phase 1)
- `uv pip install -e "00_common[dev]"` works
- `uv run pytest 00_common/tests -q` passes
- KPI count assertion 93 + acyclic graph assertion visible in output
- dummy model via `create_app` passes `api_smoke` and yields a valid `ModelOutput`
- JSON Schemas exported to `00_common/schemas/`

## 8. Open questions
1. D1 and D2 above — flagging, not blocking; both are reversible in config/`ids.py`.
2. `world.yaml` `venue_center` remains the docs/04 default until real coordinates/zones.geojson are supplied.
