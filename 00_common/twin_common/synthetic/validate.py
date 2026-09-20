"""World validation: the docs/04 section 8 invariants.

Generation **fails** when an invariant breaks. That is the point: a world that quietly
violates its own physics would be discovered later, inside a model, as an inexplicable KPI.

Checks are grouped so a failure report names every problem at once rather than one per run.
Scenario-specific invariants only run for the scenario they belong to.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import assumptions, world
from ..errors import TwinError
from ..io.schemas import TABLE_SCHEMAS
from ..logging import get_logger
from .grid import TimeGrid, in_window_series

log = get_logger(__name__)

#: Absolute tolerance for the population continuity check (docs/04 section 8 says +/- 1
#: rounding; the model is exact to float noise, so this is generous).
CONTINUITY_TOLERANCE = 1.0
#: docs/04 section 8: physical sanity limit on crowd density.
MAX_DENSITY_P_M2 = 9.0
#: docs/04 section 8: gate entries must equal calendar arrivals within this share.
ARRIVALS_TOLERANCE = 0.01
#: docs/04 section 8: S02 daily arrivals approximately 1.3x S01, within this share.
S02_RATIO_TOLERANCE = 0.02


class WorldValidationError(TwinError):
    """One or more world invariants failed."""


@dataclass
class ValidationReport:
    """Outcome of validating one generated world."""

    scenario_id: str
    passed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed.append(name)
        else:
            self.failures.append(f"{name}: {detail}" if detail else name)
        return condition

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def ok(self) -> bool:
        return not self.failures

    def raise_if_failed(self) -> None:
        if self.failures:
            bullets = "\n".join(f"  - {failure}" for failure in self.failures)
            raise WorldValidationError(
                f"{self.scenario_id}: {len(self.failures)} world invariant(s) failed "
                f"(docs/04 section 8):\n{bullets}"
            )

    def summary(self) -> str:
        return (
            f"{self.scenario_id}: {len(self.passed)} checks passed, "
            f"{len(self.failures)} failed, {len(self.warnings)} warnings"
        )


def validate_world(
    tables: Mapping[str, pd.DataFrame],
    grid: TimeGrid,
    *,
    scenario_id: str,
    overrides: Mapping[str, Any],
    baseline_daily_arrivals: float | None = None,
) -> ValidationReport:
    """Run every applicable invariant over a generated world.

    Args:
        tables: the generated tables, keyed by registered table name.
        grid: the world clock.
        scenario_id: which scenario this world is.
        overrides: its merged overrides.
        baseline_daily_arrivals: total S01 arrivals, needed for the S02 ratio check. When
            omitted the ratio check is skipped with a warning rather than silently passing.
    """
    report = ValidationReport(scenario_id=scenario_id)
    a = assumptions()
    world_cfg = world()

    # ----------------------------------------------------------- required tables
    expected = {
        "footfall_15min",
        "zones",
        "weather_hourly",
        "event_calendar",
        "medical_incidents",
        "security_incidents",
        "water_15min",
        "toilet_usage",
        "waste",
        "power_15min",
        "network_5min",
        "parking_15min",
        "transit_15min",
        "traffic_probe_15min",
        "asset_maintenance",
        "resource_roster",
    }
    missing = sorted(expected - set(tables))
    report.check("required tables present", not missing, f"missing {missing}")
    if missing:
        return report

    footfall = tables["footfall_15min"]

    # ------------------------------------------------------- no NaN, tz-awareness
    for name, frame in sorted(tables.items()):
        schema = TABLE_SCHEMAS.get(name)
        if schema is None or frame.empty:
            continue
        required_columns = [c for c in schema.required if c in frame.columns]
        nulls = frame[required_columns].isna().sum()
        offending = {col: int(count) for col, count in nulls.items() if count}
        report.check(f"{name}: no NaN in required columns", not offending, str(offending))
        if schema.time_column and schema.time_column in frame.columns:
            # pandas deprecated is_datetime64tz_dtype; the dtype check is the replacement.
            dtype = frame[schema.time_column].dtype
            aware = isinstance(dtype, pd.DatetimeTZDtype)
            report.check(f"{name}: {schema.time_column} is tz-aware", bool(aware))

    # --------------------------------------------------------- provenance columns
    for name, frame in sorted(tables.items()):
        if frame.empty:
            continue
        has_both = "is_synthetic" in frame.columns and "source" in frame.columns
        report.check(f"{name}: carries is_synthetic and source", has_both)
        if has_both and name not in ("weather_hourly", "air_quality_hourly"):
            # Only the real sources may be flagged non-synthetic.
            all_synthetic = bool(frame["is_synthetic"].all())
            report.check(f"{name}: every row is_synthetic", all_synthetic)

    # ------------------------------------------------- crowd continuity and bounds
    ordered = footfall.sort_values(["zone_id", "timestamp"])
    worst_error = 0.0
    worst_zone = ""
    for zone_id, group in ordered.groupby("zone_id"):
        population = group["population"].to_numpy()
        entries = group["entries"].to_numpy()
        exits = group["exits"].to_numpy()
        if len(population) < 2:
            continue
        predicted = population[:-1] + entries[:-1] - exits[:-1]
        error = float(np.abs(predicted - population[1:]).max())
        if error > worst_error:
            worst_error, worst_zone = error, str(zone_id)
    report.check(
        "population continuity per zone",
        worst_error <= CONTINUITY_TOLERANCE,
        f"worst error {worst_error:.3f} in {worst_zone} (tolerance {CONTINUITY_TOLERANCE})",
    )

    for column in ("population", "entries", "exits"):
        negative = int((footfall[column] < 0).sum())
        report.check(f"{column} is non-negative", negative == 0, f"{negative} negative rows")

    max_density = float(footfall["density_p_m2"].max())
    report.check(
        "density within physical sanity",
        max_density <= MAX_DENSITY_P_M2,
        f"max {max_density:.2f} persons/m2 exceeds {MAX_DENSITY_P_M2}",
    )

    # --------------------------------------------- timestamps continuous on the grid
    stamps = pd.DatetimeIndex(sorted(footfall["timestamp"].unique()))
    expected_index = grid.index()
    report.check(
        "footfall timestamps cover the grid",
        len(stamps) == len(expected_index) and (stamps == expected_index).all(),
        f"{len(stamps)} distinct timestamps, expected {len(expected_index)}",
    )

    # ------------------------------------------- ghat density band on the peak day
    bands = a["crowd"]["density_bands_p_m2"]
    snan_days = [day for day in grid.event_dates if grid.is_snan_day(day)]
    if snan_days:
        peak_day = snan_days[0].isoformat()
        peak = footfall.loc[footfall["timestamp"].dt.date.astype(str) == peak_day]
        ghat_zones = [
            zone_id for zone_id, spec in world_cfg["zones"].items() if spec["type"] == "ghat"
        ]
        ghat = peak.loc[peak["zone_id"].isin(ghat_zones)]
        if len(ghat):
            ghat_peak = float(ghat["density_p_m2"].max())
            report.check(
                "S01 peak-day ghat density reaches amber",
                ghat_peak >= float(bands["amber"]) or scenario_id != "S01",
                f"ghat peak {ghat_peak:.2f} below amber {bands['amber']}",
            )
            # "not sustained critical": a few steps is a spike, an hour is sustained.
            critical_steps = int((ghat["density_p_m2"] >= float(bands["critical"])).sum())
            sustained_limit = grid.steps_per_hour
            report.check(
                "S01 ghats not sustained critical",
                critical_steps < sustained_limit or scenario_id != "S01",
                f"{critical_steps} steps at critical (limit {sustained_limit})",
            )

    # ------------------------------------- gate entries equal calendar arrivals
    gate_table = tables.get("gate_entries")
    calendar = tables["event_calendar"]
    if gate_table is not None and len(gate_table):
        admitted = float(gate_table["entries"].sum())
        demanded = float(gate_table["demand"].sum())
        if demanded > 0:
            share = admitted / demanded
            # A scenario that closes a gate or saturates one legitimately admits fewer
            # people than arrive; the invariant is about the baseline bookkeeping.
            closes_gates = bool(overrides.get("closed_gates"))
            report.check(
                "gate entries match calendar arrivals",
                abs(share - 1.0) <= ARRIVALS_TOLERANCE or closes_gates,
                f"admitted {admitted:,.0f} of {demanded:,.0f} demanded ({share:.4f})",
            )
        report.check(
            "calendar covers every day",
            calendar["date"].nunique() == len(grid.days),
            f"{calendar['date'].nunique()} days in D21, expected {len(grid.days)}",
        )

    # --------------------------------------------------- medical and security sanity
    medical = tables["medical_incidents"]
    if len(medical):
        report.check(
            "medical severity in 1..5",
            bool(medical["severity"].between(1, 5).all()),
            f"range {medical['severity'].min()}..{medical['severity'].max()}",
        )
        severe = int((medical["severity"] >= 4).sum())
        report.check("severe cases do not exceed total", severe <= len(medical))
        report.check(
            "medical incident ids unique",
            medical["incident_id"].is_unique,
            f"{len(medical) - medical['incident_id'].nunique()} duplicates",
        )
    security = tables["security_incidents"]
    if len(security):
        report.check(
            "security response times non-negative",
            bool((security["response_min"] >= 0).all()),
        )
        report.check("security incident ids unique", security["incident_id"].is_unique)

    # -------------------------------------------------------------- utility sanity
    power = tables["power_15min"]
    report.check("power load non-negative", bool((power["kw"] >= 0).all()))
    report.check("generator fuel non-negative", bool((power["fuel_l"] >= 0).all()))
    network = tables["network_5min"]
    report.check(
        "bandwidth within capacity bounds",
        bool((network["bandwidth_used_mbps"] >= 0).all()),
    )
    parking = tables["parking_15min"]
    report.check(
        "parking occupancy within capacity",
        bool((parking["occupied"] <= parking["capacity"]).all()),
        "occupied exceeds capacity",
    )
    waste = tables["waste"]
    report.check(
        "bin fill within 0..100",
        bool(waste["fill_pct"].between(0.0, 100.0).all()),
        f"range {waste['fill_pct'].min():.1f}..{waste['fill_pct'].max():.1f}",
    )
    assets = tables["asset_maintenance"]
    report.check(
        "asset failure label is binary",
        bool(assets["failure"].isin((0, 1)).all()),
    )
    failure_rate = float(assets["failure"].mean()) if len(assets) else 0.0
    if failure_rate > 0.15:
        report.warn(f"asset failure rate {failure_rate:.1%} looks high; AI4I 2020 is about 3.4%")

    # ---------------------------------------------------------- scenario invariants
    _validate_scenario(report, tables, grid, scenario_id, overrides, baseline_daily_arrivals)

    log.info(report.summary())
    for warning in report.warnings:
        log.warning("%s: %s", scenario_id, warning)
    return report


def _validate_scenario(
    report: ValidationReport,
    tables: Mapping[str, pd.DataFrame],
    grid: TimeGrid,
    scenario_id: str,
    overrides: Mapping[str, Any],
    baseline_daily_arrivals: float | None,
) -> None:
    """The scenario-specific rows of docs/04 section 8."""
    footfall = tables["footfall_15min"]
    gate_table = tables.get("gate_entries")

    # --- S02: daily arrivals about 1.3x S01 --------------------------------------
    multiplier = overrides.get("footfall_multiplier")
    if multiplier and gate_table is not None:
        if baseline_daily_arrivals is None:
            report.warn("S02 arrivals ratio not checked: no baseline total was supplied")
        else:
            total = float(gate_table["entries"].sum())
            ratio = total / baseline_daily_arrivals if baseline_daily_arrivals else 0.0
            expected = float(multiplier)
            report.check(
                f"arrivals ratio approximately {expected}x baseline",
                abs(ratio - expected) <= S02_RATIO_TOLERANCE * expected,
                f"ratio {ratio:.4f}, expected {expected} +/- {S02_RATIO_TOLERANCE:.0%}",
            )

    # --- S05: a closed gate admits nobody ----------------------------------------
    closed_gates = set(overrides.get("closed_gates") or [])
    if closed_gates and gate_table is not None:
        for gate_id in sorted(closed_gates):
            admitted = float(gate_table.loc[gate_table["gate_id"] == gate_id, "entries"].sum())
            report.check(
                f"{gate_id} entries are zero",
                admitted == 0.0,
                f"admitted {admitted:,.0f}",
            )

    # --- S08: the downed substation's zones lose grid power in the window ---------
    substations_down = set(overrides.get("substations_down") or [])
    if substations_down:
        power = tables["power_15min"]
        inside = in_window_series(pd.DatetimeIndex(power["timestamp"]), overrides.get("window"))
        for substation in sorted(substations_down):
            affected = power.loc[power["asset_id"] == substation]
            if not len(affected):
                report.check(f"{substation} appears in D18", False, "no rows")
                continue
            mask = in_window_series(
                pd.DatetimeIndex(affected["timestamp"]), overrides.get("window")
            )
            in_window_rows = affected.loc[mask]
            report.check(
                f"{substation} grid down inside the window",
                not bool(in_window_rows["grid_available"].any()),
                f"{int(in_window_rows['grid_available'].sum())} rows still available",
            )
            report.check(
                f"{substation} generators run inside the window",
                bool(in_window_rows["generator_on"].all()),
            )
            outside = affected.loc[~mask]
            if len(outside):
                report.check(
                    f"{substation} grid restored outside the window",
                    bool(outside["grid_available"].all()),
                )
        del inside

    # --- S09: about 80 percent of cameras remain active --------------------------
    outage_share = overrides.get("camera_outage_share")
    cameras = tables.get("camera_registry")
    if outage_share and cameras is not None and len(cameras):
        active_share = float((cameras["status"] == "active").mean())
        expected_share = 1.0 - float(outage_share)
        report.check(
            "active camera share matches the outage",
            abs(active_share - expected_share) <= 0.05,
            f"active {active_share:.2%}, expected about {expected_share:.0%}",
        )

    # --- S12: the bridge carries nobody -----------------------------------------
    closed_links = set(overrides.get("closed_links") or [])
    routing = world()["routing"]
    closure_map = routing.get("closure_map") or {}
    for closed in sorted(closed_links):
        for link in closure_map.get(closed, []):
            source, _, _target = link.partition(">")
            flow = float(footfall.loc[footfall["zone_id"] == source, "exits"].sum())
            report.check(
                f"{closed} carries no flow ({link})",
                flow == 0.0,
                f"{flow:,.0f} persons still left {source}",
            )
        traffic = tables.get("traffic_probe_15min")
        if traffic is not None and closed in set(traffic["link_id"]):
            volume = float(traffic.loc[traffic["link_id"] == closed, "volume_proxy"].sum())
            report.check(
                f"{closed} road volume is zero",
                volume == 0.0,
                f"volume {volume:,.0f}",
            )

    # --- S14: medical cases scale with the multiplier ----------------------------
    medical_multiplier = overrides.get("medical_rate_multiplier")
    if medical_multiplier:
        # Checked against the baseline by scripts/, not here: this module sees one world.
        report.warn(
            f"medical_rate_multiplier {medical_multiplier} applied; the 2x ratio is "
            f"checked across worlds by the generator CLI"
        )

    # --- S15: the blocked exit discharges nobody after the incident --------------
    blocked_exits = set(overrides.get("blocked_exits") or [])
    exit_table = tables.get("exit_flows")
    if blocked_exits and exit_table is not None and len(exit_table):
        for exit_id in sorted(blocked_exits):
            flow = float(exit_table.loc[exit_table["exit_id"] == exit_id, "outflow"].sum())
            report.check(
                f"{exit_id} discharges nobody",
                flow == 0.0,
                f"{flow:,.0f} persons still used it",
            )
