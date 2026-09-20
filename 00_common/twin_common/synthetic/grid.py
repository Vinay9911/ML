"""Time grid and deterministic seeding for the synthetic world.

Two concerns that every generator step shares:

**The grid.** docs/04 section 3: 15-minute steps in Asia/Kolkata, 28 days of history from
``history_start`` plus the three event days. Network telemetry (D19) uses 5 minutes and
weather (D12/D13) uses 1 hour, so the helpers take the step size as an argument.

**Seeding.** Every step draws from its own generator, seeded from
``(base_seed, scenario_id, step_name)`` through a stable hash. That way adding a step later
does not shift the random stream of an earlier one, which is what keeps
"same seed -> identical output" true across code changes (CLAUDE.md hard rule).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import cached_property

import numpy as np
import pandas as pd

from ..config import world as world_config
from ..contracts.models import IST, to_ist

#: Steps per hour on the main 15-minute grid.
STEPS_PER_HOUR = 4
#: Main grid step, minutes (docs/04 section 3).
GRID_MIN = 15
#: Network telemetry grid, minutes (D19).
NETWORK_GRID_MIN = 5
#: Days of normal context before the event days (docs/04 section 3).
HISTORY_DAYS = 28


def stable_seed(base_seed: int, *parts: str) -> int:
    """A reproducible 63-bit seed from a base seed and any number of labels.

    ``hash()`` is salted per process in Python, so it cannot be used here.
    """
    payload = "|".join((str(base_seed), *parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def rng_for(base_seed: int, scenario_id: str, step_name: str) -> np.random.Generator:
    """The generator one pipeline step should use."""
    return np.random.default_rng(stable_seed(base_seed, scenario_id, step_name))


def parse_ts(value: str | datetime) -> datetime:
    """Parse an ISO timestamp from config into an IST-aware datetime."""
    if isinstance(value, datetime):
        return to_ist(value)
    return to_ist(datetime.fromisoformat(value))


def _parse_clock(clock: str) -> time:
    """Parse an "HH:MM" string from config into a naive time."""
    hour, _, minute = clock.partition(":")
    return time(int(hour), int(minute or 0))


def parse_clock_on(day: date, clock: str) -> datetime:
    """Combine a date with an ``"HH:MM"`` string from config into an IST datetime."""
    hour, _, minute = clock.partition(":")
    return datetime(day.year, day.month, day.day, int(hour), int(minute or 0), tzinfo=IST)


@dataclass(frozen=True, slots=True)
class EventDay:
    """One of the three event days (docs/04 section 3)."""

    day: date
    kind: str
    multiplier_key: str

    @property
    def is_snan(self) -> bool:
        return self.kind == "snan_day"


class TimeGrid:
    """The world clock: history window, event days and the 15-minute index.

    Built from ``world.yaml -> time`` so no module hard-codes a date.
    """

    def __init__(
        self,
        history_start: datetime,
        event_days: list[EventDay],
        demo_now: datetime,
        *,
        grid_min: int = GRID_MIN,
        network_grid_min: int = NETWORK_GRID_MIN,
    ) -> None:
        self.history_start = history_start
        self.event_days = event_days
        self.demo_now = demo_now
        self.grid_min = grid_min
        self.network_grid_min = network_grid_min

    @classmethod
    def from_config(cls) -> TimeGrid:
        time_cfg = world_config()["time"]
        return cls(
            history_start=parse_ts(time_cfg["history_start"]),
            event_days=[
                EventDay(
                    day=date.fromisoformat(row["date"]),
                    kind=row["kind"],
                    multiplier_key=row["multiplier_key"],
                )
                for row in time_cfg["event_days"]
            ],
            demo_now=parse_ts(time_cfg["demo_now"]),
            grid_min=int(time_cfg["grid_min"]),
            network_grid_min=int(time_cfg["network_grid_min"]),
        )

    # ------------------------------------------------------------------ days
    @cached_property
    def event_dates(self) -> list[date]:
        return [day.day for day in self.event_days]

    @cached_property
    def last_day(self) -> date:
        """The final day covered by the world: the last event day."""
        return max(self.event_dates)

    @cached_property
    def days(self) -> list[date]:
        """Every day from ``history_start`` through the last event day, inclusive."""
        start = self.history_start.date()
        span = (self.last_day - start).days
        return [start + timedelta(days=offset) for offset in range(span + 1)]

    @cached_property
    def history_days(self) -> list[date]:
        """Days before the first event day."""
        first_event = min(self.event_dates)
        return [day for day in self.days if day < first_event]

    def event_day_for(self, day: date) -> EventDay | None:
        for event_day in self.event_days:
            if event_day.day == day:
                return event_day
        return None

    def is_event_day(self, day: date) -> bool:
        return day in set(self.event_dates)

    def is_snan_day(self, day: date) -> bool:
        event_day = self.event_day_for(day)
        return event_day is not None and event_day.is_snan

    # ------------------------------------------------------------------ index
    @property
    def end(self) -> datetime:
        """Exclusive end of the world: midnight after the last day."""
        last = self.last_day + timedelta(days=1)
        return datetime(last.year, last.month, last.day, tzinfo=IST)

    def index(self, step_min: int | None = None) -> pd.DatetimeIndex:
        """The full timestamp index at ``step_min`` resolution (default: the main grid)."""
        minutes = step_min or self.grid_min
        return pd.date_range(
            start=self.history_start,
            end=self.end - timedelta(minutes=minutes),
            freq=f"{minutes}min",
            tz=IST,
        )

    def day_index(self, day: date, step_min: int | None = None) -> pd.DatetimeIndex:
        """The timestamps that fall on one day."""
        minutes = step_min or self.grid_min
        start = datetime(day.year, day.month, day.day, tzinfo=IST)
        return pd.date_range(
            start=start,
            end=start + timedelta(days=1) - timedelta(minutes=minutes),
            freq=f"{minutes}min",
            tz=IST,
        )

    def hourly_index(self) -> pd.DatetimeIndex:
        return self.index(step_min=60)

    def network_index(self) -> pd.DatetimeIndex:
        return self.index(step_min=self.network_grid_min)

    @property
    def steps_per_day(self) -> int:
        return (24 * 60) // self.grid_min

    @property
    def steps_per_hour(self) -> int:
        return 60 // self.grid_min

    def __len__(self) -> int:
        return len(self.index())

    def __repr__(self) -> str:
        return (
            f"TimeGrid({self.history_start.date()} .. {self.last_day}, "
            f"{len(self.days)} days, {self.grid_min}min, {len(self)} steps)"
        )


def expand_hourly_profile(profile: list[float], steps_per_hour: int) -> np.ndarray:
    """Spread a 24-value hourly profile over a day of finer steps.

    The profile is a share of the day, so each hour's weight is divided evenly across its
    steps and the result is normalised to sum to 1. This is what docs/04 section 5.4 means
    by "normalized hourly profile / 4".
    """
    if len(profile) != 24:
        raise ValueError(f"an hourly profile must have 24 entries, got {len(profile)}")
    weights = np.repeat(np.asarray(profile, dtype=float), steps_per_hour) / steps_per_hour
    total = weights.sum()
    if total <= 0:
        raise ValueError("hourly profile sums to zero")
    return weights / total


def lognormal_noise(
    rng: np.random.Generator,
    size: int | tuple[int, ...],
    sigma: float,
) -> np.ndarray:
    """Multiplicative noise with mean 1 (docs/04 section 5.4 "lognormal noise (sigma param)").

    A raw lognormal has mean ``exp(sigma^2/2)``, which would inflate totals and break the
    "gate entries equal calendar arrivals" invariant, so it is divided out.
    """
    if sigma <= 0:
        return np.ones(size)
    raw = rng.lognormal(mean=0.0, sigma=sigma, size=size)
    return raw / np.exp(sigma**2 / 2)


def in_window_series(
    index: pd.DatetimeIndex,
    window: list[str] | tuple[str, str] | None,
) -> np.ndarray:
    """Boolean mask of which timestamps fall inside a scenario ``["HH:MM", "HH:MM"]`` window.

    Used by the S03 rain, S08 power and S09 camera scenarios, which all apply only inside a
    window. With no window every timestamp is inside, so an override with no window applies
    all day - the same convention as :func:`twin_common.scenarios.in_window`.
    """
    if not window:
        return np.ones(len(index), dtype=bool)
    start, end = _parse_clock(str(window[0])), _parse_clock(str(window[1]))
    minutes = index.hour * 60 + index.minute
    start_min = start.hour * 60 + start.minute
    end_min = end.hour * 60 + end.minute
    if start_min <= end_min:
        return np.asarray((minutes >= start_min) & (minutes <= end_min))
    # A window that wraps past midnight is the union of the two ends.
    return np.asarray((minutes >= start_min) | (minutes <= end_min))


def dwell_steps(mean_dwell_min: float, grid_min: int = GRID_MIN) -> float:
    """Mean dwell expressed in grid steps, floored at one step.

    A zone whose mean dwell is shorter than one step would otherwise drain instantly and
    produce an outflow demand above its own population.
    """
    return max(1.0, float(mean_dwell_min) / grid_min)
