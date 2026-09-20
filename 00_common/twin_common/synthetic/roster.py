"""Resource roster (D22) and the small placeholder tables (D23, D25).

The roster is what M24 checks its allocations against, so the scenario availability
multipliers (S10 ambulance 0.8, S11 police 0.9) have to land here (docs/04 section 5.6).

Availability is split across zones in proportion to each zone's share of peak population,
because that is how a duty roster is actually written: more staff where more people are.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..config import resources_config
from ..contracts.enums import RESOURCE_TYPES
from ..logging import get_logger
from .grid import TimeGrid, parse_clock_on, rng_for

log = get_logger(__name__)

#: Roster status values.
STATUS_VALUES = ("on_duty", "standby", "off_duty")

#: Resource types held centrally rather than posted to a zone: they are dispatched, not
#: stationed, so the roster records them against the event rather than a zone.
CENTRAL_TYPES = frozenset({"network_capacity", "food_vehicle", "waste_vehicle", "water_tanker"})


def build_roster(
    grid: TimeGrid,
    zones: pd.DataFrame,
    footfall: pd.DataFrame,
    *,
    scenario_id: str,
    overrides: Mapping[str, Any],
    seed: int,
) -> pd.DataFrame:
    """D22: quantity of each resource type available, per shift and per zone."""
    resources = resources_config()
    shifts = resources["shifts"]
    resource_types = resources["resource_types"]
    multipliers = overrides.get("resource_availability_multiplier") or {}
    rng = rng_for(seed, scenario_id, "roster")

    # Zone weights from peak population: the busiest zones get the most staff.
    peak = footfall.groupby("zone_id")["population"].max()
    total_peak = float(peak.sum()) or 1.0
    weights = (peak / total_peak).to_dict()
    zone_ids = list(peak.index)

    rows: list[dict[str, Any]] = []
    for day in grid.days:
        for shift in shifts:
            shift_start = parse_clock_on(day, shift["start"])
            shift_end = parse_clock_on(day, shift["end"])
            if shift_end <= shift_start:
                # Night shift C runs past midnight.
                shift_end = shift_end + timedelta(days=1)
            for resource_type in RESOURCE_TYPES:
                spec = resource_types[resource_type]
                base = float(spec["available"]) * float(multipliers.get(resource_type, 1.0))
                # A small day-to-day variation, so the roster is not perfectly flat.
                available = base * float(np.clip(rng.normal(1.0, 0.03), 0.85, 1.1))
                if resource_type in CENTRAL_TYPES:
                    rows.append(
                        {
                            "shift_start": shift_start,
                            "shift_end": shift_end,
                            "resource_type": resource_type,
                            "zone_id": "EVENT",
                            "quantity_available": available,
                            "status": "on_duty",
                        }
                    )
                    continue
                for zone_id in zone_ids:
                    share = float(weights.get(zone_id, 0.0))
                    quantity = available * share
                    rows.append(
                        {
                            "shift_start": shift_start,
                            "shift_end": shift_end,
                            "resource_type": resource_type,
                            "zone_id": zone_id,
                            "quantity_available": quantity,
                            "status": "on_duty",
                        }
                    )

    frame = pd.DataFrame(rows)
    frame["is_synthetic"] = True
    frame["source"] = "roster"
    applied = {k: v for k, v in multipliers.items() if v != 1.0}
    log.info(
        "%s roster: %d rows over %d days x %d shifts x %d types%s",
        scenario_id,
        len(frame),
        len(grid.days),
        len(shifts),
        len(RESOURCE_TYPES),
        f"; availability multipliers {applied}" if applied else "",
    )
    return frame


def build_sop_logs(
    medical: pd.DataFrame,
    security: pd.DataFrame,
    *,
    scenario_id: str,
    seed: int,
) -> pd.DataFrame:
    """D23 ``sop_logs``: placeholder task timings for the future learning loop.

    docs/04 section 4 marks D23 as "(future learning loop)", so this is deliberately thin:
    one notify/complete pair per severe incident, enough to exercise the schema.
    """
    rng = rng_for(seed, scenario_id, "sop_logs")
    rows: list[dict[str, Any]] = []

    def add(frame: pd.DataFrame, sop_id: str, tasks: tuple[str, ...], severity_min: int) -> None:
        if not len(frame) or "severity" not in frame.columns:
            return
        severe = frame.loc[frame["severity"] >= severity_min]
        for row in severe.itertuples(index=False):
            notify = row.timestamp
            for task in tasks:
                duration = float(rng.gamma(shape=2.0, scale=3.0))
                rows.append(
                    {
                        "incident_id": row.incident_id,
                        "sop_id": sop_id,
                        "task": task,
                        "t_notify": notify,
                        "t_complete": notify + pd.Timedelta(minutes=duration),
                        "outcome": "completed" if duration < 12 else "delayed",
                    }
                )
                notify = notify + pd.Timedelta(minutes=duration)

    add(medical, "SOP-MED-01", ("dispatch_team", "on_site_triage", "handover"), 4)
    add(security, "SOP-SEC-01", ("dispatch_patrol", "secure_area", "report"), 3)

    frame = pd.DataFrame(
        rows,
        columns=["incident_id", "sop_id", "task", "t_notify", "t_complete", "outcome"],
    )
    frame["is_synthetic"] = True
    frame["source"] = "sop_logs"
    return frame


#: D25 placeholder rows. docs/04 section 4 marks the table "documentation only", so these are
#: qualitative notes rather than data any model reads.
LESSONS_LEARNED_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "previous_cycle",
        "crush_risk_score",
        "near_miss",
        "Ghat approach saturated for 20 minutes before the holding area was closed.",
    ),
    (
        "previous_cycle",
        "ambulance_response_time",
        "breach",
        "Response exceeded the SLA when the riverside road was blocked by parked vehicles.",
    ),
    (
        "previous_cycle",
        "expected_water_demand",
        "shortfall",
        "Water points ran dry in the market sector during the afternoon heat peak.",
    ),
    (
        "previous_cycle",
        "waste_bin_fill_level",
        "shortfall",
        "Bins overflowed on the peak day; collection rounds were planned for a normal day.",
    ),
    (
        "previous_cycle",
        "camera_availability",
        "degraded",
        "Two camera clusters shared one uplink; the tower outage created a blind sector.",
    ),
)


def build_lessons_learned() -> pd.DataFrame:
    """D25 ``lessons_learned``: placeholder rows, documentation only."""
    frame = pd.DataFrame(
        list(LESSONS_LEARNED_ROWS), columns=["event_name", "kpi", "outcome", "notes"]
    )
    frame["is_synthetic"] = True
    frame["source"] = "lessons_learned"
    return frame
