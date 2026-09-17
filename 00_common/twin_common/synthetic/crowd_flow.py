"""The crowd compartment model: D01 ``footfall_15min`` (docs/04 section 5.5).

Every other domain table is derived from the population this module produces, so this is
where scenario effects have to be right.

**Three compartments per zone.**

``queue``
    People who have reached the venue but not yet passed a gate. They are physically
    standing in the gate's zone - Z05 is literally the "Gate plaza / holding area" - so they
    count toward that zone's population and density. This is what makes a gate closure
    visible: when S05 hands G02's share to the narrow G03, the plaza fills with people
    waiting, which is where crowd risk builds at a real event.
``pop_in``
    Admitted, heading toward a ghat.
``pop_out``
    Bathed, heading back to an exit.

A single tank per zone could not do this: the route graph sends arrivals and departures
opposite ways, and a queue is neither. A zone with no ``inbound`` route - Z01 and Z02, the
ghats - is a destination: its ``pop_in`` dwells, then turns around into ``pop_out``.

**Why the continuity invariant holds exactly.** ``entries`` counts arrivals reaching the zone
and ``exits`` counts the flows that actually left after every cap, so
``population(t+1) = population(t) + entries(t) - exits(t)`` is true by construction rather
than by luck (docs/04 section 8). Movement between compartments of the same zone - queue to
pop_in at a gate, pop_in to pop_out at a ghat - moves nobody in or out and so appears in
neither column.

**Where accumulation comes from.** Every flow is ``min(demand, link capacity, space in the
receiving zone)``. Whatever a cap removes stays where it was: a queue at a saturated gate, or
density in the zone upstream of a closed link. That is the ACCUMULATION signal M03 detects,
and the back-pressure that stops a sealed zone filling past jam density.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..config import assumptions, world
from ..logging import get_logger
from .grid import TimeGrid, dwell_steps, expand_hourly_profile, lognormal_noise, rng_for

log = get_logger(__name__)

#: Separator used by the ``link_width_m`` and ``closure_map`` keys in world.yaml.
LINK_SEP = ">"

#: Rounds of the space-allocation fixed point (see pass 3 in :func:`simulate`).
#: Acceptance only decreases, so this converges quickly; 6 is comfortably more than the
#: longest chain in the routing graph (Z08 - Z05 - Z04 - Z03 - Z01).
SPACE_ITERATIONS = 6


def link_key(source: str, target: str) -> str:
    return f"{source}{LINK_SEP}{target}"


@dataclass
class CrowdFlowResult:
    """The generated crowd state."""

    #: D01: one row per (timestamp, zone_id).
    footfall: pd.DataFrame
    #: One row per (timestamp, gate_id): demand, admissions and the waiting queue.
    gate_entries: pd.DataFrame
    #: One row per (timestamp, exit_id): the departures actually discharged.
    exit_flows: pd.DataFrame
    #: Per-day arrivals admitted through the gates.
    admitted_per_day: dict[date, float] = field(default_factory=dict)
    #: Per-day arrivals the calendar asked for.
    demanded_per_day: dict[date, float] = field(default_factory=dict)

    @property
    def zone_ids(self) -> list[str]:
        return sorted(self.footfall["zone_id"].unique())


def effective_gate_split(
    base_split: Mapping[str, float],
    reassignment: Mapping[str, Mapping[str, float]],
    closed_gates: set[str],
) -> dict[str, float]:
    """Redistribute the share of any closed gate (S05).

    ``gate_reassignment`` says where a closed gate sends its crowd, e.g.
    ``G02 -> {G03: 0.7, G01: 0.3}``. A closed gate with no mapping has its share spread over
    the remaining open gates in proportion to their own shares, so the day's arrivals stay
    the same instead of quietly vanishing.
    """
    split = {gate: float(share) for gate, share in base_split.items()}
    for gate in closed_gates:
        share = split.pop(gate, 0.0)
        if share <= 0:
            continue
        targets = {
            target: float(weight)
            for target, weight in (reassignment.get(gate) or {}).items()
            if target in split
        }
        if targets:
            total = sum(targets.values())
            for target, weight in targets.items():
                split[target] += share * weight / total
        elif split:
            total = sum(split.values())
            if total > 0:
                for target in list(split):
                    split[target] += share * split[target] / total
            else:
                for target in list(split):
                    split[target] += share / len(split)
        else:
            log.warning("every gate is closed; %.0f%% of arrivals cannot enter", share * 100)
    return split


def flow_capacity_p_min(width_m: float, specific_flow: float) -> float:
    """Pedestrian throughput of an opening from its width (docs/05 section 6)."""
    return specific_flow * float(width_m) * 60.0


def simulate(
    grid: TimeGrid,
    zones: pd.DataFrame,
    daily_multiplier: Mapping[date, float],
    weather_arrival_multiplier: Mapping[pd.Timestamp, float],
    *,
    scenario_id: str,
    overrides: Mapping[str, Any],
    seed: int,
) -> CrowdFlowResult:
    """Run the compartment model over the whole world.

    Args:
        grid: the world clock.
        zones: the layout zone table (needs ``zone_id``, ``type``, ``usable_area_m2``).
        daily_multiplier: per-day arrivals multiplier from the calendar, already including
            the scenario ``footfall_multiplier``.
        weather_arrival_multiplier: per-hour multiplier; rain suppresses arrivals.
        scenario_id: for seeding and logging.
        overrides: merged scenario overrides.
        seed: base seed.
    """
    crowd = assumptions()["crowd"]
    world_cfg = world()
    routing = world_cfg["routing"]
    rng = rng_for(seed, scenario_id, "crowd_flow")

    zone_ids: list[str] = zones["zone_id"].tolist()
    zone_type = dict(zip(zones["zone_id"], zones["type"], strict=True))
    usable_area = dict(zip(zones["zone_id"], zones["usable_area_m2"].astype(float), strict=True))

    base_arrivals = float(crowd["normal_day_arrivals_sector"])
    specific_flow = float(crowd["exit_specific_flow_p_per_m_s"])
    noise_sigma = float(crowd["arrival_noise_sigma"])
    max_holding_density = float(crowd["max_holding_density_p_m2"])
    default_gate_throughput = float(crowd["gate_throughput_p_min"])
    dwell_by_type = crowd["mean_dwell_min"]

    steps_per_hour = grid.steps_per_hour
    profile_normal = expand_hourly_profile(crowd["hourly_profile_normal"], steps_per_hour)
    profile_snan = expand_hourly_profile(crowd["hourly_profile_snan"], steps_per_hour)

    # ------------------------------------------------------------ scenario closures
    closed_gates = set(overrides.get("closed_gates") or [])
    closed_links_raw = set(overrides.get("closed_links") or [])
    blocked_exits = set(overrides.get("blocked_exits") or [])
    closure_map = routing.get("closure_map") or {}
    closed_pedestrian_links: set[str] = set()
    for closed in closed_links_raw:
        closed_pedestrian_links.update(closure_map.get(closed, []))
    if closed_gates:
        log.info("%s: gates closed %s", scenario_id, sorted(closed_gates))
    if closed_pedestrian_links:
        log.info("%s: pedestrian links closed %s", scenario_id, sorted(closed_pedestrian_links))
    if blocked_exits:
        log.info("%s: exits blocked %s", scenario_id, sorted(blocked_exits))

    gate_split = effective_gate_split(
        crowd["gate_split"], crowd.get("gate_reassignment") or {}, closed_gates
    )
    gates_cfg = world_cfg["gates"]
    gate_zone = {gate_id: spec["zone_id"] for gate_id, spec in gates_cfg.items()}
    # Per-gate throughput: the gates differ in width, and that difference is what drives S05.
    gate_throughput = {
        gate_id: float(spec.get("throughput_p_min", default_gate_throughput))
        for gate_id, spec in gates_cfg.items()
    }
    all_gates = list(gates_cfg)

    exits_cfg = world_cfg["exits"]
    exit_zone = {exit_id: spec["zone_id"] for exit_id, spec in exits_cfg.items()}
    exit_capacity = {
        exit_id: flow_capacity_p_min(spec["width_m"], specific_flow)
        for exit_id, spec in exits_cfg.items()
    }

    link_widths = routing.get("link_width_m") or {}
    zone_link_capacity = {
        key: flow_capacity_p_min(width, specific_flow) for key, width in link_widths.items()
    }

    def capacity_per_step(source: str, target: str, minutes: float) -> float:
        """Throughput of one link over one step, honouring scenario closures."""
        key = link_key(source, target)
        if key in closed_pedestrian_links:
            return 0.0
        if target.startswith("E"):
            if target in blocked_exits:
                return 0.0
            per_min = exit_capacity.get(target, 0.0)
        else:
            per_min = zone_link_capacity.get(key)
            if per_min is None:
                raise KeyError(
                    f"no link_width_m entry for {key!r} in world.yaml routing; every routed "
                    f"link needs a width so its capacity stays config-driven"
                )
        return per_min * minutes

    # ------------------------------------------------------------ state
    pop_queue = dict.fromkeys(zone_ids, 0.0)
    pop_in = dict.fromkeys(zone_ids, 0.0)
    pop_out = dict.fromkeys(zone_ids, 0.0)
    gate_queue = dict.fromkeys(all_gates, 0.0)
    dwell = {z: dwell_steps(dwell_by_type[zone_type[z]], grid.grid_min) for z in zone_ids}
    holding_capacity = {z: usable_area[z] * max_holding_density for z in zone_ids}

    inbound = routing["inbound"]
    outbound = routing["outbound"]

    index = grid.index()
    minutes = float(grid.grid_min)

    # Per-step arrival demand for the whole world, drawn in one vectorised call so the
    # random stream is stable for a given seed.
    step_weight = np.zeros(len(index))
    day_of_step = np.empty(len(index), dtype=object)
    for position, timestamp in enumerate(index):
        day = timestamp.date()
        day_of_step[position] = day
        profile = profile_snan if grid.is_snan_day(day) else profile_normal
        slot = timestamp.hour * steps_per_hour + timestamp.minute // grid.grid_min
        step_weight[position] = profile[slot]
    multiplier_of_step = np.array([float(daily_multiplier.get(day, 0.0)) for day in day_of_step])
    weather_of_step = np.array(
        [float(weather_arrival_multiplier.get(ts.floor("h"), 1.0)) for ts in index]
    )
    noise = lognormal_noise(rng, len(index), noise_sigma)
    arrival_demand = base_arrivals * multiplier_of_step * step_weight * weather_of_step * noise

    footfall_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    exit_rows: list[dict[str, Any]] = []
    admitted_per_day: dict[date, float] = {}
    demanded_per_day: dict[date, float] = {}

    for position, timestamp in enumerate(index):
        day = day_of_step[position]
        demand = float(arrival_demand[position])
        demanded_per_day[day] = demanded_per_day.get(day, 0.0) + demand

        entries = dict.fromkeys(zone_ids, 0.0)
        exits = dict.fromkeys(zone_ids, 0.0)
        delta_queue = dict.fromkeys(zone_ids, 0.0)
        delta_in = dict.fromkeys(zone_ids, 0.0)
        delta_out = dict.fromkeys(zone_ids, 0.0)
        exit_discharge: dict[str, float] = dict.fromkeys(exits_cfg, 0.0)

        # ---------------------------------------------------- arrivals join the queue
        # Reaching the venue is an ENTRY into the gate zone: these people are standing in
        # the plaza. Whether the gate admits them is a separate, internal transfer.
        arrivals_by_gate: dict[str, float] = {}
        for gate_id in all_gates:
            share = gate_split.get(gate_id, 0.0)
            arriving = demand * share
            arrivals_by_gate[gate_id] = arriving
            if arriving > 0:
                zone_id = gate_zone[gate_id]
                gate_queue[gate_id] += arriving
                delta_queue[zone_id] += arriving
                entries[zone_id] += arriving

        # ---------------------------------------------------- movement, three passes
        # 1. desired flows from the start-of-step population
        # 2. cap by link capacity
        # 3. cap by the space the receiving zone actually has
        desired: list[tuple[str, str, str, float]] = []

        for zone_id in zone_ids:
            available_in = pop_in[zone_id]
            if available_in > 0:
                spec = inbound.get(zone_id)
                if spec is None:
                    # A destination zone (a ghat): dwell, then turn around. Internal, so it
                    # touches neither entries nor exits.
                    converted = min(available_in, available_in / dwell[zone_id])
                    delta_in[zone_id] -= converted
                    delta_out[zone_id] += converted
                else:
                    demand_out = min(available_in, available_in / dwell[zone_id])
                    for target, share in spec["to"].items():
                        amount = demand_out * float(share)
                        if amount > 0:
                            desired.append(("in", zone_id, target, amount))

            available_out = pop_out[zone_id]
            if available_out > 0:
                spec = outbound.get(zone_id)
                if spec is not None:
                    demand_out = min(available_out, available_out / dwell[zone_id])
                    for target, share in spec["to"].items():
                        amount = demand_out * float(share)
                        if amount > 0:
                            desired.append(("out", zone_id, target, amount))

        # pass 2: link capacity. Widths in world.yaml are per direction, so the two
        # directions of a corridor each get their own allowance.
        link_budget: dict[str, float] = {}
        capped: list[tuple[str, str, str, float]] = []
        for phase, source, target, amount in desired:
            key = link_key(source, target)
            if key not in link_budget:
                link_budget[key] = capacity_per_step(source, target, minutes)
            allowed = min(amount, max(0.0, link_budget[key]))
            if allowed > 0:
                link_budget[key] -= allowed
                capped.append((phase, source, target, allowed))

        # pass 3: receiving-zone space, up to jam density.
        #
        # A zone has room for the people replacing those leaving it, so its own outflow is
        # credited against its occupancy. But how much it can actually shed depends on
        # whether ITS downstream neighbours have room, which depends on their outflow, and
        # so on. Solving that in one shot either ignores the coupling - which let a zone
        # accept inflow it could not shed and overshoot jam density - or ignores the
        # outflow entirely, which under-accepts and can deadlock a full corridor.
        #
        # So the allocation is iterated to a fixed point. Acceptance only ever decreases
        # (less accepted downstream means less outflow credited upstream, means less
        # accepted upstream), so the iteration is monotone and settles in a few rounds.
        wanted_into = dict.fromkeys(zone_ids, 0.0)
        for _phase, _source, target, amount in capped:
            if target in wanted_into:
                wanted_into[target] += amount

        accept_factor = dict.fromkeys(zone_ids, 1.0)
        for _ in range(SPACE_ITERATIONS):
            # Outflow actually achievable under the current acceptance estimate.
            leaving = dict.fromkeys(zone_ids, 0.0)
            for _phase, source, target, amount in capped:
                leaving[source] += amount * accept_factor.get(target, 1.0)
            changed = False
            for zone_id in zone_ids:
                wanted = wanted_into[zone_id]
                if wanted <= 0:
                    continue
                occupied = (
                    pop_queue[zone_id] + pop_in[zone_id] + pop_out[zone_id] - leaving[zone_id]
                )
                space = max(0.0, holding_capacity[zone_id] - occupied)
                updated = 1.0 if wanted <= space else space / wanted
                if updated < accept_factor[zone_id] - 1e-12:
                    accept_factor[zone_id] = updated
                    changed = True
            if not changed:
                break

        for phase, source, target, amount in capped:
            accepted = amount * accept_factor.get(target, 1.0)
            if accepted <= 0:
                continue
            if phase == "in":
                delta_in[source] -= accepted
            else:
                delta_out[source] -= accepted
            exits[source] += accepted
            if target.startswith("E"):
                exit_discharge[target] += accepted
            else:
                entries[target] += accepted
                if phase == "in":
                    delta_in[target] += accepted
                else:
                    delta_out[target] += accepted

        # ------------------------------------------- gates admit from the queue
        # An internal transfer inside the gate zone: queue -> pop_in, capped by the gate
        # throughput. A closed gate admits nobody, so its reassigned share is what matters.
        admitted_total = 0.0
        for gate_id in all_gates:
            waiting = gate_queue[gate_id]
            zone_id = gate_zone[gate_id]
            capacity = 0.0 if gate_id in closed_gates else gate_throughput[gate_id] * minutes
            admitted = min(waiting, capacity)
            if admitted > 0:
                gate_queue[gate_id] -= admitted
                delta_queue[zone_id] -= admitted
                delta_in[zone_id] += admitted
                admitted_total += admitted
            gate_rows.append(
                {
                    "timestamp": timestamp,
                    "gate_id": gate_id,
                    "zone_id": zone_id,
                    "entries": admitted,
                    "demand": arrivals_by_gate[gate_id],
                    "queue_persons": gate_queue[gate_id],
                    "capacity_p_min": gate_throughput[gate_id],
                    "closed": gate_id in closed_gates,
                }
            )
        admitted_per_day[day] = admitted_per_day.get(day, 0.0) + admitted_total

        # ------------------------------------------------- record and advance
        for zone_id in zone_ids:
            population = pop_queue[zone_id] + pop_in[zone_id] + pop_out[zone_id]
            footfall_rows.append(
                {
                    "timestamp": timestamp,
                    "zone_id": zone_id,
                    "gate_id": None,
                    "entries": entries[zone_id],
                    "exits": exits[zone_id],
                    "population": population,
                    "density_p_m2": population / usable_area[zone_id],
                }
            )
        for exit_id, amount in exit_discharge.items():
            exit_rows.append(
                {
                    "timestamp": timestamp,
                    "exit_id": exit_id,
                    "zone_id": exit_zone[exit_id],
                    "outflow": amount,
                    "capacity_p_min": exit_capacity[exit_id],
                    "blocked": exit_id in blocked_exits,
                }
            )

        for zone_id in zone_ids:
            pop_queue[zone_id] = max(0.0, pop_queue[zone_id] + delta_queue[zone_id])
            pop_in[zone_id] = max(0.0, pop_in[zone_id] + delta_in[zone_id])
            pop_out[zone_id] = max(0.0, pop_out[zone_id] + delta_out[zone_id])

    footfall = pd.DataFrame(footfall_rows)
    # `population` is the value at the START of the step, so the entries and exits on the
    # same row are the flows that produce the next row. The continuity invariant is written
    # against that convention.
    footfall["is_synthetic"] = True
    footfall["source"] = "crowd_flow"

    gate_frame = pd.DataFrame(gate_rows)
    gate_frame["is_synthetic"] = True
    gate_frame["source"] = "crowd_flow"
    exit_frame = pd.DataFrame(exit_rows)
    exit_frame["is_synthetic"] = True
    exit_frame["source"] = "crowd_flow"

    peak_density = footfall.groupby("zone_id")["density_p_m2"].max()
    peak_queue = gate_frame.groupby("gate_id")["queue_persons"].max()
    log.info(
        "%s crowd flow: %d rows, admitted %.0f of %.0f demanded; peak density %s; peak queue %s",
        scenario_id,
        len(footfall),
        sum(admitted_per_day.values()),
        sum(demanded_per_day.values()),
        peak_density.round(2).to_dict(),
        peak_queue.round(0).to_dict(),
    )

    return CrowdFlowResult(
        footfall=footfall,
        gate_entries=gate_frame,
        exit_flows=exit_frame,
        admitted_per_day=admitted_per_day,
        demanded_per_day=demanded_per_day,
    )
