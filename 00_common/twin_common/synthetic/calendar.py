"""Event calendar: D21 ``event_calendar`` (docs/04 section 5.2).

The daily arrivals multiplier is
``base x weekday factor x event-day multiplier x scenario footfall_multiplier``,
and the crowd model multiplies it by ``normal_day_arrivals_sector``.

D21 has a (day, session) grain. ``expected_attendance_multiplier`` is a **per-day** figure
repeated on each session row of that day; consumers that want daily arrivals read one row
per date. Sessions exist so M09 and M13 can reason about bathing windows and VIP movement,
not to subdivide the attendance figure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from ..config import assumptions
from ..contracts.models import IST
from ..logging import get_logger
from .grid import TimeGrid, parse_clock_on

log = get_logger(__name__)

#: Operational sessions on an event day. A normal day is one `full_day` row.
EVENT_SESSIONS: tuple[tuple[str, str, str], ...] = (
    ("brahma_muhurta", "03:00", "06:00"),
    ("morning", "06:00", "11:00"),
    ("afternoon", "11:00", "17:00"),
    ("evening", "17:00", "22:00"),
)
FULL_DAY_SESSION = ("full_day", "00:00", "23:59")

#: Saturday and Sunday get the weekend multiplier.
WEEKEND_WEEKDAYS = frozenset({5, 6})


@dataclass(frozen=True, slots=True)
class CalendarResult:
    """D21 plus the per-day multipliers the crowd model needs."""

    frame: pd.DataFrame
    daily_multiplier: dict[date, float]

    def multiplier_for(self, day: date) -> float:
        return self.daily_multiplier.get(day, 0.0)


def daily_multiplier_for(
    day: date,
    grid: TimeGrid,
    crowd: Mapping[str, Any],
    footfall_multiplier: float,
) -> float:
    """The arrivals multiplier for one day.

    History days get 1.0, or the weekend factor at the weekend. An event day gets the
    multiplier its ``multiplier_key`` names in ``assumptions.crowd``. The scenario
    ``footfall_multiplier`` (S02) applies on top of everything.
    """
    event_day = grid.event_day_for(day)
    if event_day is not None:
        base = float(crowd[event_day.multiplier_key])
    elif day.weekday() in WEEKEND_WEEKDAYS:
        base = float(crowd["weekend_multiplier"])
    else:
        base = 1.0
    return base * float(footfall_multiplier)


def _small_festival_days(grid: TimeGrid) -> set[date]:
    """Two small festival spikes in the history window (docs/04 section 3).

    Placed one and three weeks before the first event day, so the forecasting models see a
    repeating-but-irregular pattern rather than a pure weekly cycle.
    """
    first_event = min(grid.event_dates)
    return {first_event - timedelta(days=7), first_event - timedelta(days=21)}


def build_calendar(
    grid: TimeGrid,
    *,
    scenario_id: str,
    overrides: Mapping[str, Any],
) -> CalendarResult:
    """Build D21 and the per-day multipliers."""
    crowd = assumptions()["crowd"]
    footfall_multiplier = float(overrides.get("footfall_multiplier", 1.0))
    festival_days = _small_festival_days(grid)
    festival_boost = float(crowd["weekend_multiplier"])

    vip_convoy = overrides.get("vip_convoy") or {}
    vip_start_clock_present = bool(vip_convoy)
    vip_start_clock = vip_convoy.get("start") if vip_start_clock_present else None
    vip_duration_min = float(vip_convoy.get("duration_min", 0.0))
    fire_zone = overrides.get("fire_zone")

    rows: list[dict[str, Any]] = []
    daily: dict[date, float] = {}

    for day in grid.days:
        multiplier = daily_multiplier_for(day, grid, crowd, footfall_multiplier)
        if day in festival_days:
            multiplier *= festival_boost
        daily[day] = multiplier
        is_snan = grid.is_snan_day(day)
        is_event = grid.is_event_day(day)

        sessions = EVENT_SESSIONS if is_event else (FULL_DAY_SESSION,)
        for session, start_clock, end_clock in sessions:
            start = parse_clock_on(day, start_clock)
            end = parse_clock_on(day, end_clock)
            if end <= start:
                end = datetime(day.year, day.month, day.day, tzinfo=IST) + timedelta(days=1)

            # A VIP convoy flags the session it falls inside, on the day it is scheduled.
            vip_flag = False
            if vip_start_clock and is_snan:
                vip_start = parse_clock_on(day, vip_start_clock)
                vip_end = vip_start + timedelta(minutes=vip_duration_min)
                vip_flag = vip_start < end and vip_end > start

            # The bathing sessions are anchored to the ghats; other sessions are venue-wide.
            zone_id = None
            if is_event and session in ("brahma_muhurta", "morning"):
                zone_id = "Z01"
            if fire_zone and is_snan and session == "morning":
                zone_id = fire_zone

            rows.append(
                {
                    "date": day.isoformat(),
                    "session": session,
                    "start": start,
                    "end": end,
                    "zone_id": zone_id,
                    "expected_attendance_multiplier": multiplier,
                    "is_snan_day": is_snan,
                    "vip_flag": vip_flag,
                    "is_synthetic": True,
                    "source": "calendar",
                }
            )

    frame = pd.DataFrame(rows)
    log.info(
        "%s calendar: %d rows over %d days; multipliers %.2f..%.2f (snan %.2f)",
        scenario_id,
        len(frame),
        len(grid.days),
        min(daily.values()),
        max(daily.values()),
        daily[next(d for d in grid.event_dates if grid.is_snan_day(d))],
    )
    return CalendarResult(frame=frame, daily_multiplier=daily)
