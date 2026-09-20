"""World generator CLI (docs/04 section 5).

    python -m twin_common.synthetic.generate --scenario S01
    python -m twin_common.synthetic.generate --scenario S02 --seed 7
    TWIN_OFFLINE=1 python -m twin_common.synthetic.generate --scenario S01

Pipeline order matters: weather feeds the arrivals multiplier, the crowd model feeds every
derived domain, and the power table feeds the asset telemetry. The validator runs last and
raises, so a broken world never reaches a model.

Output goes to ``00_common/data/world/<scenario_id>/`` as Parquet, plus
``world_manifest.json`` recording the seed, the row counts, the data sources and a sha256 per
table - which is what makes the "same seed, identical output" check possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import assumptions, world
from ..contracts.models import now_ist
from ..io.tables import save_table
from ..logging import get_logger
from ..paths import ensure_dir, is_offline, world_dir
from ..real.weather import ATTRIBUTION, get_air_quality_hourly, get_weather_hourly
from ..scenarios import get_scenario, resolve_overrides
from .assets_telemetry import build_asset_telemetry
from .calendar import build_calendar
from .crowd_flow import simulate
from .derived import build_derived
from .grid import TimeGrid
from .layout import build_layout
from .report import build_report
from .roster import build_lessons_learned, build_roster, build_sop_logs
from .validate import ValidationReport, validate_world
from .weather import apply_weather_overrides, default_weather_params

log = get_logger(__name__)

#: Tables written to disk, in dependency order. `gate_entries` and `exit_flows` are not
#: numbered datasets in docs/04 but are needed by M02, M11 and the validator.
EXTRA_TABLES = ("gate_entries", "exit_flows", "camera_registry", "assets", "key_links")


@dataclass
class GenerationResult:
    """Everything one generation run produced."""

    scenario_id: str
    seed: int
    out_dir: Path
    tables: dict[str, pd.DataFrame]
    report: ValidationReport
    manifest: dict[str, Any]
    elapsed_s: float

    @property
    def total_rows(self) -> int:
        return sum(len(frame) for frame in self.tables.values())


def _sha256(frame: pd.DataFrame) -> str:
    """A content hash that ignores row order but not values.

    Used by the determinism check, so it must not depend on anything but the data.
    """
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    payload = pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def generate_world(
    scenario_id: str = "S01",
    *,
    seed: int | None = None,
    out_dir: Path | None = None,
    write: bool = True,
    validate: bool = True,
    baseline_daily_arrivals: float | None = None,
) -> GenerationResult:
    """Generate one scenario world end to end.

    Args:
        scenario_id: which scenario to build.
        seed: base seed; defaults to the ``seed`` in world.yaml time config or 42.
        out_dir: override the output directory.
        write: set False to build in memory only (used by tests).
        validate: set False to skip the invariants (not recommended).
        baseline_daily_arrivals: S01 arrivals total, so a scenario run can check its ratio.
    """
    started = time.perf_counter()
    scenario = get_scenario(scenario_id)
    overrides = resolve_overrides(scenario_id)
    base_seed = int(seed if seed is not None else 42)
    grid = TimeGrid.from_config()
    venue = world()["venue"]
    time_cfg = world()["time"]
    latitude = float(venue["venue_center"]["lat"])
    longitude = float(venue["venue_center"]["lon"])
    source_year = int(time_cfg["weather_source_year"])

    log.info(
        "generating %s (%s) seed=%d offline=%s: %d days, %d steps",
        scenario_id,
        scenario.name,
        base_seed,
        is_offline(),
        len(grid.days),
        len(grid),
    )

    # 1 ---------------------------------------------------------------- layout
    layout = build_layout(
        offline_cameras_share=float(overrides.get("camera_outage_share", 0.0)),
        seed=base_seed,
    )

    # 2 ---------------------------------------------------------------- weather
    weather_result = get_weather_hourly(
        grid.days,
        latitude=latitude,
        longitude=longitude,
        source_year=source_year,
        seed=base_seed,
    )
    air_result = get_air_quality_hourly(
        grid.days,
        latitude=latitude,
        longitude=longitude,
        source_year=source_year,
        noise_base_db=float(assumptions()["environment"]["noise_base_db"]),
        seed=base_seed,
    )
    weather_scenario = apply_weather_overrides(
        weather_result.frame, overrides=overrides, params=default_weather_params()
    )

    # 3 ---------------------------------------------------------------- calendar
    calendar = build_calendar(grid, scenario_id=scenario_id, overrides=overrides)

    # 4 ------------------------------------------------------------- crowd flow
    crowd = simulate(
        grid,
        layout.zones,
        calendar.daily_multiplier,
        weather_scenario.arrival_multiplier,
        scenario_id=scenario_id,
        overrides=overrides,
        seed=base_seed,
    )

    # 5 ---------------------------------------------------------- derived domains
    derived = build_derived(
        grid,
        layout.zones,
        layout.assets,
        layout.cameras,
        layout.key_links,
        crowd.footfall,
        crowd.gate_entries,
        weather_scenario.frame,
        air_result.frame,
        scenario_id=scenario_id,
        overrides=overrides,
        seed=base_seed,
    )

    # 6 --------------------------------------------------------- asset telemetry
    telemetry = build_asset_telemetry(
        grid,
        layout.assets,
        derived["power_15min"],
        weather_scenario.frame,
        scenario_id=scenario_id,
        overrides=overrides,
        seed=base_seed,
    )

    # 7 ------------------------------------------------------------------ roster
    roster = build_roster(
        grid,
        layout.zones,
        crowd.footfall,
        scenario_id=scenario_id,
        overrides=overrides,
        seed=base_seed,
    )

    # 8 --------------------------------------------------------- assemble tables
    tables: dict[str, pd.DataFrame] = {
        "zones": layout.zones,
        "assets": layout.assets,
        "camera_registry": layout.cameras,
        "key_links": layout.key_links,
        "event_calendar": calendar.frame,
        "weather_hourly": weather_scenario.frame,
        "footfall_15min": crowd.footfall,
        "gate_entries": crowd.gate_entries,
        "exit_flows": crowd.exit_flows,
        "asset_maintenance": telemetry,
        "resource_roster": roster,
        "sop_logs": build_sop_logs(
            derived["medical_incidents"],
            derived["security_incidents"],
            scenario_id=scenario_id,
            seed=base_seed,
        ),
        "lessons_learned": build_lessons_learned(),
        **dict(derived.items()),
    }

    # 9 --------------------------------------------------------------- validate
    report = ValidationReport(scenario_id=scenario_id)
    if validate:
        report = validate_world(
            tables,
            grid,
            scenario_id=scenario_id,
            overrides=overrides,
            baseline_daily_arrivals=baseline_daily_arrivals,
        )
        report.raise_if_failed()

    # 10 ----------------------------------------------------- write and manifest
    target = Path(out_dir) if out_dir is not None else world_dir(scenario_id)
    hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    if write:
        ensure_dir(target)
        for name, frame in sorted(tables.items()):
            # `gate_entries` and friends are not in the registered schema set, so they are
            # written without the schema check.
            from ..io.schemas import table_exists

            if table_exists(name):
                save_table(frame, name, target)
            else:
                frame.to_parquet(target / f"{name}.parquet", index=False)
            hashes[name] = _sha256(frame)
            row_counts[name] = len(frame)
        # The zone polygons, for any consumer that wants GeoJSON rather than WKT.
        (target / "zones.geojson").write_text(
            json.dumps(layout.zones_geojson, indent=2), encoding="utf-8"
        )
    else:
        for name, frame in sorted(tables.items()):
            hashes[name] = _sha256(frame)
            row_counts[name] = len(frame)

    elapsed = time.perf_counter() - started
    manifest = {
        "scenario_id": scenario_id,
        "scenario_name": scenario.name,
        "overrides": overrides,
        "seed": base_seed,
        "generated_at": now_ist().isoformat(),
        "elapsed_s": round(elapsed, 2),
        "offline": is_offline(),
        "time_range": {
            "start": grid.history_start.isoformat(),
            "end": grid.end.isoformat(),
            "days": len(grid.days),
            "grid_min": grid.grid_min,
            "steps": len(grid),
            "demo_now": grid.demo_now.isoformat(),
        },
        "venue_center": {"lat": latitude, "lon": longitude},
        "data_sources": {
            "weather": weather_result.source,
            "weather_is_real": weather_result.is_real,
            "air_quality": air_result.source,
            "air_quality_is_real": air_result.is_real,
            "weather_scenario_modified": weather_scenario.modified,
        },
        "attribution": [ATTRIBUTION] if weather_result.is_real else [],
        "arrivals": {
            "admitted_total": float(sum(crowd.admitted_per_day.values())),
            "demanded_total": float(sum(crowd.demanded_per_day.values())),
            "per_day": {
                day.isoformat(): float(value)
                for day, value in sorted(crowd.admitted_per_day.items())
            },
        },
        "tables": {
            name: {"rows": row_counts[name], "sha256": hashes[name]} for name in sorted(tables)
        },
        "validation": {
            "passed": len(report.passed),
            "failed": len(report.failures),
            "warnings": report.warnings,
        },
    }
    report_path: Path | None = None
    if write:
        (target / "world_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        report_path = build_report(tables, manifest, scenario_id=scenario_id)
        manifest["sanity_report"] = str(report_path)
        (target / "world_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        log.info(
            "%s written to %s: %d tables, %d rows, %.1fs; report %s",
            scenario_id,
            target,
            len(tables),
            sum(row_counts.values()),
            elapsed,
            report_path,
        )

    return GenerationResult(
        scenario_id=scenario_id,
        seed=base_seed,
        out_dir=target,
        tables=tables,
        report=report,
        manifest=manifest,
        elapsed_s=elapsed,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m twin_common.synthetic.generate",
        description="Generate the synthetic world for one or more scenarios.",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        action="append",
        default=None,
        help="scenario ID, repeatable (default: S01)",
    )
    parser.add_argument("--seed", type=int, default=42, help="base seed (default: 42)")
    parser.add_argument("--out", type=Path, default=None, help="output directory override")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="force offline mode (equivalent to TWIN_OFFLINE=1)",
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="skip the world invariants (not advised)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="build in memory without writing files"
    )
    args = parser.parse_args(argv)

    if args.offline:
        import os

        os.environ["TWIN_OFFLINE"] = "1"

    scenarios = args.scenario or ["S01"]
    # The S01 arrivals total is what a scenario ratio check compares against, so the
    # baseline is generated first whenever more than one scenario is requested.
    if len(scenarios) > 1 and "S01" not in scenarios:
        scenarios = ["S01", *scenarios]
    elif "S01" in scenarios:
        scenarios = ["S01", *[s for s in scenarios if s != "S01"]]

    baseline_arrivals: float | None = None
    failures = 0
    for scenario_id in scenarios:
        try:
            result = generate_world(
                scenario_id,
                seed=args.seed,
                out_dir=(args.out / scenario_id) if args.out else None,
                write=not args.dry_run,
                validate=not args.no_validate,
                baseline_daily_arrivals=baseline_arrivals,
            )
        except Exception as exc:  # a failed world must not stop the remaining scenarios
            failures += 1
            print(f"FAILED {scenario_id}: {exc}")
            continue
        if scenario_id == "S01":
            baseline_arrivals = float(result.manifest["arrivals"]["admitted_total"])
        print(
            f"{scenario_id}  {result.report.summary()}  "
            f"{result.total_rows:,} rows in {result.elapsed_s:.1f}s -> {result.out_dir}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
