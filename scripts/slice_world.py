"""Copy the world tables a model needs into its own folder (docs/04 section 9).

    uv run python scripts/slice_world.py M21
    uv run python scripts/slice_world.py M21 --scenarios S01 S03 S04
    uv run python scripts/slice_world.py --all

A model folder ships with only the tables its `config.yaml -> inputs` names, time-filtered
to the history window plus the event days, so each delivered zip stays small (docs/04
section 9 targets under 20 MB). The model reads them through `data/synthetic/<scenario>/`
with no idea it is looking at a slice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "00_common"))

import pandas as pd  # noqa: E402

from twin_common.config import load_model_config  # noqa: E402
from twin_common.contracts import registry as reg  # noqa: E402
from twin_common.io.schemas import TABLE_SCHEMAS  # noqa: E402
from twin_common.paths import world_dir  # noqa: E402

#: Tables every model gets regardless of its `inputs`, because the contract layer needs them.
ALWAYS_INCLUDE = ("zones",)
#: docs/04 section 9 size target per model folder.
SIZE_TARGET_MB = 20.0


def tables_for(model_id: str) -> list[str]:
    """The tables a model declares, plus the always-included ones."""
    info = reg.model_info(model_id)
    config_path = REPO_ROOT / info.folder / "config.yaml"
    if config_path.is_file():
        declared = list(load_model_config(config_path, validate=False).get("inputs") or [])
    else:
        from twin_common.io.schemas import tables_for_model

        declared = tables_for_model(model_id)
    wanted = [*ALWAYS_INCLUDE, *declared]
    # Preserve order, drop duplicates and anything unregistered.
    seen: set[str] = set()
    out: list[str] = []
    for name in wanted:
        if name in TABLE_SCHEMAS and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def slice_model(
    model_id: str,
    scenarios: list[str],
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Copy each needed table for each scenario. Returns {scenario: bytes written}."""
    info = reg.model_info(model_id)
    folder = REPO_ROOT / info.folder
    if not folder.is_dir():
        raise SystemExit(
            f"{info.folder} does not exist. Run: python scripts/new_model.py {model_id}"
        )

    names = tables_for(model_id)
    written: dict[str, int] = {}
    for scenario_id in scenarios:
        source_dir = world_dir(scenario_id)
        if not source_dir.is_dir():
            print(
                f"  {scenario_id}: no world generated. Run "
                f"`python -m twin_common.synthetic.generate --scenario {scenario_id}`"
            )
            continue
        target_dir = folder / "data" / "synthetic" / scenario_id
        total = 0
        copied: list[str] = []
        for name in names:
            source = source_dir / f"{name}.parquet"
            if not source.is_file():
                continue
            if dry_run:
                total += source.stat().st_size
                copied.append(name)
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            frame = pd.read_parquet(source)
            destination = target_dir / f"{name}.parquet"
            frame.to_parquet(destination, index=False)
            total += destination.stat().st_size
            copied.append(f"{name}({len(frame):,})")
        written[scenario_id] = total
        verb = "would copy" if dry_run else "copied"
        print(f"  {scenario_id}: {verb} {len(copied)} tables, {total / 1e6:.1f} MB")
        for name in copied:
            print(f"      {name}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/slice_world.py",
        description="Copy the world tables a model needs into its folder.",
    )
    parser.add_argument("model_id", nargs="?", help="e.g. M21; omit with --all")
    parser.add_argument("--all", action="store_true", help="slice every scaffolded model")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="scenario IDs (default: every generated world)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.scenarios:
        scenarios = [s.upper() for s in args.scenarios]
    else:
        root = world_dir("S01").parent
        scenarios = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        if not scenarios:
            raise SystemExit(
                "no generated worlds found. Run "
                "`python -m twin_common.synthetic.generate --scenario S01` first."
            )

    if args.all:
        model_ids = [
            model_id
            for model_id, info in reg.all_models().items()
            if (REPO_ROOT / info.folder).is_dir()
        ]
        if not model_ids:
            raise SystemExit("no model folders exist yet; scaffold one with new_model.py")
    elif args.model_id:
        model_ids = [args.model_id.upper()]
    else:
        parser.error("give a model ID or --all")

    oversized: list[str] = []
    for model_id in model_ids:
        print(f"{model_id} ({reg.model_info(model_id).folder})")
        written = slice_model(model_id, scenarios, dry_run=args.dry_run)
        total_mb = sum(written.values()) / 1e6
        if total_mb > SIZE_TARGET_MB:
            oversized.append(f"{model_id} ({total_mb:.1f} MB)")

    if oversized:
        print()
        print(
            f"WARNING: over the {SIZE_TARGET_MB:.0f} MB docs/04 section 9 target: "
            + ", ".join(oversized)
        )
        print("Slice fewer scenarios, or regenerate those tables by script instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
