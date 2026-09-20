"""Scaffold a model folder from the template and the registries (docs/07 Phase 3).

    uv run python scripts/new_model.py M05
    uv run python scripts/new_model.py M05 --force   # overwrite an existing folder

Everything the scaffold needs is already declared: `model_registry.yaml` gives the folder
name, port, engine and upstream list, and `kpi_registry.yaml` gives the KPIs this model
owns and their units. So the generated `config.yaml` is correct by construction rather than
copied and edited, which is what stops the 25 folders from drifting apart.

What it does NOT do is write the model. `src/model.py` arrives as a working skeleton with
the contract wiring in place and a TODO where the domain logic goes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "00_common"))

from twin_common.contracts import registry as reg  # noqa: E402
from twin_common.io.schemas import TABLE_SCHEMAS, tables_for_model  # noqa: E402

TEMPLATE_DIR = REPO_ROOT / "templates" / "model_folder"

#: Which twin_common extra each engine needs (from 00_common/pyproject.toml).
ENGINE_EXTRA = {
    "forecast": "forecast",
    "formula": "formula",
    "rules": "dev",
    "mlclf": "mlclf",
    "vision": "vision",
    "network": "network",
    "optimize": "opt",
    "location": "opt",
    "pedsim": "pedsim",
}

#: Default horizons by engine. A vision or simulation model answers about now; a forecast
#: model answers about the next few hours.
ENGINE_HORIZONS = {
    "vision": ([0, 15], 15),
    "pedsim": ([0, 15, 60], 60),
    "rules": ([0, 15, 60], 60),
    "location": ([0, 60], 60),
    "optimize": ([0, 60, 180], 180),
}
DEFAULT_HORIZONS = ([15, 60, 180], 180)

#: Directories every model folder has (docs/02 section 2).
SUBDIRECTORIES = (
    "src",
    "tests",
    "schemas",
    "data/synthetic",
    "data/sample_upstream",
    "data/derived",
    "outputs",
)


def scenarios_for(model_id: str) -> list[str]:
    """Scenarios this model must respond to, from the docs/05 section 1.2 `affects` lists."""
    from twin_common.scenarios import all_scenarios

    supported = ["S01"]
    supported.extend(
        scenario_id
        for scenario_id, scenario in sorted(all_scenarios().items())
        if model_id in scenario.affects
    )
    return supported


def _yaml_list(values: list[str] | list[int]) -> str:
    if not values:
        return "[]"
    if isinstance(values[0], int):
        return "[" + ", ".join(str(v) for v in values) + "]"
    return "[" + ", ".join(str(v) for v in values) + "]"


def _markdown_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "none"


def build_substitutions(model_id: str) -> dict[str, str]:
    """Every placeholder the templates use, derived from the registries."""
    info = reg.model_info(model_id)
    kpis = reg.kpis_owned_by(model_id)
    inputs = [name for name in tables_for_model(model_id) if name != "zones"] or ["zones"]
    horizons, default_horizon = ENGINE_HORIZONS.get(info.engine, DEFAULT_HORIZONS)
    scenarios = scenarios_for(model_id)

    kpi_table = "\n".join(
        f"| `{kpi}` | {reg.kpi_unit(kpi)} | {reg.kpi_info(kpi).definition} |" for kpi in kpis
    )
    datasets = sorted(
        {TABLE_SCHEMAS[name].dataset_id for name in inputs if name in TABLE_SCHEMAS} - {"-"}
    )

    return {
        "MODEL_ID": model_id,
        "MODEL_NAME": info.name,
        "FOLDER": info.folder,
        "PORT": str(info.port),
        "ENGINE": info.engine,
        "EXTRA": ENGINE_EXTRA.get(info.engine, "dev"),
        "QUESTION": f"TODO: the operational question {model_id} answers (docs/03 card).",
        "HORIZONS": _yaml_list(horizons),
        "DEFAULT_HORIZON": str(default_horizon),
        "UPSTREAM": _yaml_list(list(info.upstream)),
        "INPUTS": _yaml_list(inputs),
        "KPIS": _yaml_list(kpis),
        "SCENARIOS": _yaml_list(scenarios),
        "KPI_TABLE": kpi_table,
        "INPUTS_MD": _markdown_list(inputs),
        "UPSTREAM_MD": _markdown_list(list(info.upstream)),
        "KPIS_MD": _markdown_list(kpis),
        "SCENARIOS_MD": _markdown_list(scenarios),
        "DATASETS_MD": _markdown_list(datasets),
    }


def render(text: str, substitutions: dict[str, str]) -> str:
    for key, value in substitutions.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def scaffold(model_id: str, *, force: bool = False, dry_run: bool = False) -> Path:
    """Create the folder for ``model_id``. Returns its path."""
    info = reg.model_info(model_id)
    target = REPO_ROOT / info.folder
    if target.exists() and not force:
        raise SystemExit(
            f"{target.name} already exists. Pass --force to overwrite, or edit it by hand."
        )

    substitutions = build_substitutions(model_id)
    if dry_run:
        print(f"would create {target}")
        for key, value in sorted(substitutions.items()):
            if "\n" not in value:
                print(f"  {key:16} {value}")
        return target

    for directory in SUBDIRECTORIES:
        (target / directory).mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for source in sorted(TEMPLATE_DIR.rglob("*.tmpl")):
        relative = source.relative_to(TEMPLATE_DIR)
        destination = target / relative.with_suffix("")  # strip .tmpl
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            render(source.read_text(encoding="utf-8"), substitutions), encoding="utf-8"
        )
        written.append(str(destination.relative_to(target)))

    # `tests` is a package so `from .conftest import MODEL_ID` resolves.
    (target / "tests" / "__init__.py").write_text(
        f'"""Tests for {model_id}."""\n', encoding="utf-8"
    )

    # The contract schemas, so a delivered zip is self-describing (docs/02 section 2).
    from twin_common.contracts import export_schemas

    export_schemas(target / "schemas")
    written.append("schemas/*.json")

    # Keep the empty data directories in git.
    for directory in ("data/synthetic", "data/sample_upstream", "data/derived", "outputs"):
        keep = target / directory / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    print(f"created {target.name}/")
    for path in sorted(written):
        print(f"  {path}")
    print()
    print(f"  engine    {info.engine}  (twin_common extra: {substitutions['EXTRA']})")
    print(f"  port      {info.port}")
    print(f"  KPIs      {substitutions['KPIS']}")
    print(f"  upstream  {substitutions['UPSTREAM']}")
    print(f"  scenarios {substitutions['SCENARIOS']}")
    print()
    print("next:")
    print(f"  uv run python scripts/slice_world.py {model_id}")
    print(f"  uv run python scripts/refresh_samples.py --for {model_id}")
    print("  # then write src/model.py and the scenario assertions in tests/")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/new_model.py",
        description="Scaffold a model folder from templates/model_folder.",
    )
    parser.add_argument("model_id", help="e.g. M05")
    parser.add_argument("--force", action="store_true", help="overwrite an existing folder")
    parser.add_argument("--dry-run", action="store_true", help="show what would be written")
    args = parser.parse_args(argv)

    model_id = args.model_id.upper()
    if not reg.model_exists(model_id):
        raise SystemExit(f"unknown model ID {model_id!r}; see 00_common/config/model_registry.yaml")
    scaffold(model_id, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
