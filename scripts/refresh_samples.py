"""Regenerate the sample upstream outputs models read (docs/02 section 7 steps 3-4).

    uv run python scripts/refresh_samples.py                 # everything, all scenarios
    uv run python scripts/refresh_samples.py --for M03       # just what M03 needs
    uv run python scripts/refresh_samples.py --from M21      # M21 and everything downstream
    uv run python scripts/refresh_samples.py --scenarios S01 S04

Models are visited in **topological order**, so a model that has been built produces a real
output which the next model downstream then reads. A model that does not exist yet is
represented by a stub from `twin_common.synthetic.stubs`. That is what lets the 25 models be
built in any order without the dependency graph blocking progress.

Files land in ``<model folder>/data/sample_upstream/<ID>__<Sxx>.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "00_common"))

from twin_common.contracts import ModelOutput, PredictRequest, ScenarioRequest  # noqa: E402
from twin_common.contracts import registry as reg  # noqa: E402
from twin_common.contracts.models import to_ist  # noqa: E402
from twin_common.synthetic.stubs import build_stub  # noqa: E402

DEFAULT_AS_OF = "2027-08-02T06:00:00+05:30"

#: Every produced output is also written here, whether or not a consumer folder exists yet.
#: That gives a complete catalogue of all 25 model outputs for the frontend developer to
#: integrate against long before the models themselves are built (docs/07 Phase 3).
CATALOGUE_DIR = REPO_ROOT / "00_common" / "data" / "samples"


def model_folder(model_id: str) -> Path:
    return REPO_ROOT / reg.model_info(model_id).folder


def is_built(model_id: str) -> bool:
    """A model counts as built once it has a real ``src/model.py``."""
    source = model_folder(model_id) / "src" / "model.py"
    if not source.is_file():
        return False
    # The scaffold leaves a TODO marker where the domain logic goes.
    return "TODO: the model." not in source.read_text(encoding="utf-8")


def run_real_model(model_id: str, scenario_id: str, as_of: datetime) -> ModelOutput | None:
    """Import and run a built model. Returns None if it cannot be run."""
    folder = model_folder(model_id)
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    try:
        # Each model folder has its own `src` package, so the cached one must be dropped.
        for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
            del sys.modules[name]
        module = importlib.import_module("src.model")
        model = module.Model.from_folder(folder)
        if scenario_id == "S01":
            return model.predict(PredictRequest(as_of=as_of))
        return model.handle_scenario(ScenarioRequest(scenario_id=scenario_id, as_of=as_of))
    except Exception as exc:  # a broken model must not stop the refresh
        print(f"    {model_id} could not be run ({type(exc).__name__}: {exc}); using a stub")
        return None
    finally:
        if str(folder) in sys.path:
            sys.path.remove(str(folder))


def write_sample(output: ModelOutput, consumer_id: str, scenario_id: str, *, dry_run: bool) -> Path:
    """Write ``<consumer>/data/sample_upstream/<producer>__<Sxx>.json``."""
    target = model_folder(consumer_id) / "data" / "sample_upstream"
    path = target / f"{output.model_id}__{scenario_id}.json"
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def refresh(
    model_ids: list[str],
    scenarios: list[str],
    *,
    as_of: datetime,
    dry_run: bool = False,
) -> dict[str, int]:
    """Produce each model's output and write it wherever a downstream model reads it."""
    order = reg.topological_order(model_ids)
    written: dict[str, int] = {}
    produced: dict[tuple[str, str], ModelOutput] = {}

    for model_id in order:
        built = is_built(model_id)
        consumers = [c for c in reg.downstream_of(model_id) if model_folder(c).is_dir()]
        label = "model" if built else "stub"
        print(f"  {model_id} ({label}) -> {len(consumers)} consumer(s)")

        for scenario_id in scenarios:
            output = run_real_model(model_id, scenario_id, as_of) if built else None
            if output is None:
                output = build_stub(model_id, as_of=as_of, scenario_id=scenario_id)
            produced[(model_id, scenario_id)] = output

            # The shared catalogue: one file per (model, scenario), always.
            if not dry_run:
                catalogue = CATALOGUE_DIR / scenario_id
                catalogue.mkdir(parents=True, exist_ok=True)
                (catalogue / f"{model_id}.json").write_text(
                    json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8"
                )
            written[model_id] = written.get(model_id, 0) + 1

            for consumer_id in consumers:
                path = write_sample(output, consumer_id, scenario_id, dry_run=dry_run)
                written[model_id] = written.get(model_id, 0) + 1
                if dry_run:
                    print(f"      would write {path.relative_to(REPO_ROOT)}")
        if consumers and not dry_run:
            print(
                f"      wrote {len(consumers) * len(scenarios)} files "
                f"({len(scenarios)} scenarios x {len(consumers)} consumers)"
            )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/refresh_samples.py",
        description="Regenerate the sample upstream outputs models read.",
    )
    parser.add_argument("--for", dest="for_model", help="only what this model consumes")
    parser.add_argument("--from", dest="from_model", help="this model and everything downstream")
    parser.add_argument("--scenarios", nargs="+", default=["S01"], help="default: S01")
    parser.add_argument("--as-of", default=DEFAULT_AS_OF)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    as_of = to_ist(datetime.fromisoformat(args.as_of))
    scenarios = [s.upper() for s in args.scenarios]

    if args.for_model:
        consumer = args.for_model.upper()
        model_ids = list(reg.upstream_of(consumer))
        if not model_ids:
            print(f"{consumer} has no upstream models; nothing to refresh.")
            return 0
        print(f"refreshing what {consumer} consumes: {model_ids}")
    elif args.from_model:
        producer = args.from_model.upper()
        model_ids = [producer, *reg.downstream_of(producer)]
        print(f"refreshing {producer} and its consumers: {model_ids}")
    else:
        model_ids = list(reg.all_models())
        print(f"refreshing all {len(model_ids)} models")

    print(f"scenarios: {scenarios}, as_of {as_of.isoformat()}")
    written = refresh(model_ids, scenarios, as_of=as_of, dry_run=args.dry_run)
    total = sum(written.values())
    verb = "would write" if args.dry_run else "wrote"
    print(f"\n{verb} {total} sample files for {len(written)} producing model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
