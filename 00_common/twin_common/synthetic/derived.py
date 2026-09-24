"""Derived domain tables (docs/04 section 5.6).

Everything here is a function of the crowd state D01 plus the weather, so a scenario that
changes the crowd changes every domain consistently - which is the whole point of the
"one world, many views" principle in docs/04 section 1.

All rates come from ``assumptions.yaml``; this module contains no numbers of its own.

Tables produced:

======  =========================  =============================================
D04     traffic_probe_15min        a quick BPR pass on the key links
D05     signal_cycles              junction timings
D06     parking_15min              car arrivals, dwell and occupancy
D07     transit_15min              shuttle boardings and waits
D08     medical_incidents          Poisson draws from the M07 rate formula
D09     ambulance_dispatch         travel times from the incident zone
D10     security_incidents         Poisson draws by density and hour
D11     access_control             gate credential checks and violations
D13     air_quality_hourly         background AQ plus crowd and traffic increments
D14     water_15min                consumption, supply, storage
D15     toilet_usage               uses, queues, cleaning status
D16     waste                      accumulation and collection
D17     food_inventory_hourly      sales, stock and deliveries
D18     power_15min                load, grid availability, generators, fuel
D19     network_5min               bandwidth, latency, tower uptime
======  =========================  =============================================
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import assumptions, world
from ..engines.formula import bpr_travel_time
from ..logging import get_logger
from .grid import TimeGrid, in_window_series, rng_for

log = get_logger(__name__)

#: Incident severity 1-5 (D08). The severe share from assumptions decides how often the
#: top two bands are drawn; within a band the choice is uniform.
SEVERITY_LEVELS = (1, 2, 3, 4, 5)
SEVERE_LEVELS = (4, 5)

#: Medical categories, weighted toward heat and minor injury at a summer river event.
MEDICAL_CATEGORIES: tuple[tuple[str, float], ...] = (
    ("heat_exhaustion", 0.30),
    ("minor_injury", 0.25),
    ("fainting", 0.15),
    ("respiratory", 0.10),
    ("cardiac", 0.05),
    ("gastrointestinal", 0.10),
    ("other", 0.05),
)
MEDICAL_OUTCOMES: tuple[str, ...] = ("treated_on_site", "referred", "transferred")

#: Security incident types (D10).
SECURITY_TYPES: tuple[tuple[str, float], ...] = (
    ("lost_person", 0.45),
    ("theft", 0.20),
    ("altercation", 0.15),
    ("unauthorized_entry", 0.10),
    ("suspicious_object", 0.05),
    ("other", 0.05),
)

#: Access-control credential classes (D11).
CREDENTIAL_CLASSES: tuple[tuple[str, float], ...] = (
    ("public", 0.86),
    ("staff", 0.08),
    ("vip", 0.02),
    ("emergency", 0.04),
)

CLEAN_STATUSES = ("clean", "due", "overdue")


@dataclass
class DerivedTables:
    """Every derived table, keyed by its registered table name."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tables

    def items(self):
        return self.tables.items()


def _weighted_choice(
    rng: np.random.Generator, options: tuple[tuple[str, float], ...], size: int
) -> np.ndarray:
    labels = [label for label, _ in options]
    weights = np.array([weight for _, weight in options], dtype=float)
    return rng.choice(labels, size=size, p=weights / weights.sum())


def _provenance(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame.copy()
    out["is_synthetic"] = True
    out["source"] = source
    return out


def build_derived(
    grid: TimeGrid,
    zones: pd.DataFrame,
    assets: pd.DataFrame,
    cameras: pd.DataFrame,
    key_links: pd.DataFrame,
    footfall: pd.DataFrame,
    gate_entries: pd.DataFrame,
    weather: pd.DataFrame,
    air_quality: pd.DataFrame,
    *,
    scenario_id: str,
    overrides: Mapping[str, Any],
    seed: int,
) -> DerivedTables:
    """Derive every domain table from the crowd state and the weather."""
    a = assumptions()
    world_cfg = world()
    out = DerivedTables()

    # ---------------------------------------------------------------- shared frames
    # Population and density per (timestamp, zone), and venue-wide totals per step.
    pop = footfall.pivot_table(
        index="timestamp", columns="zone_id", values="population", aggfunc="sum"
    ).fillna(0.0)
    density = footfall.pivot_table(
        index="timestamp", columns="zone_id", values="density_p_m2", aggfunc="sum"
    ).fillna(0.0)
    entries = footfall.pivot_table(
        index="timestamp", columns="zone_id", values="entries", aggfunc="sum"
    ).fillna(0.0)
    zone_ids = list(pop.columns)
    steps = pop.index
    hours_per_step = grid.grid_min / 60.0

    # Weather aligned to the 15-min grid (it arrives hourly).
    weather_idx = weather.set_index("timestamp").sort_index()
    weather_15 = weather_idx.reindex(steps, method="ffill").ffill().bfill()
    heat_index = weather_15["heat_index_c"].to_numpy()
    temperature = weather_15["temperature_c"].to_numpy()
    rain = weather_15["rain_mm"].to_numpy()

    # ------------------------------------------------------- D08 medical incidents
    med = a["medical"]
    rng = rng_for(seed, scenario_id, "medical")
    rate_per_1000_day = float(med["presentations_per_1000_attendees_per_day"])
    # docs/03 M07 wants lambda = population x base_rate x multipliers, so base_rate has to be
    # per person PRESENT per hour. The assumption, though, is per ATTENDEE per day - a person
    # who comes, not a person-hour of presence. Converting by dividing the daily rate by 24
    # was wrong: an attendee is inside for a couple of hours, not 24, which under-counted the
    # peak day by about 6x (59 cases against the ~336 the rate implies for 336,000 attendees).
    #
    # The correct divisor is the mean time an attendee actually spends in the venue, which the
    # simulation already knows exactly: total person-hours of presence divided by the number
    # of attendees admitted. Deriving it rather than adding another assumption keeps D08
    # consistent with the documented rate however the layout or dwell times change.
    #
    # The attendee count comes from the GATE admissions, not from summing zone `entries`.
    # A zone entry is a movement, and one visitor crossing Z08 - Z05 - Z04 - Z03 - Z01 and
    # back generates about seven of them, so summing them over-counts attendees sevenfold and
    # would inflate the case count by the same factor.
    #
    # f_heat and f_density then lift the count above that baseline, which is the intended
    # behaviour - the documented rate is a normal-conditions figure.
    person_hours = float(pop.to_numpy().sum()) * hours_per_step
    attendees = float(gate_entries["entries"].sum())
    mean_dwell_hours = person_hours / attendees if attendees > 0 else 1.0
    base_rate_per_person_hour = rate_per_1000_day / 1000.0 / max(mean_dwell_hours, 1e-9)
    log.debug(
        "medical baseline: %.0f person-hours over %.0f attendees -> mean dwell %.2f h",
        person_hours,
        attendees,
        mean_dwell_hours,
    )
    heat_threshold = float(med["heat_index_threshold_c"])
    heat_increase = float(med["heat_rate_increase_per_c"])
    severe_share = float(med["severe_share"])
    transfer_share = float(med["hospital_transfer_share"])
    medical_multiplier = float(overrides.get("medical_rate_multiplier", 1.0))

    bands = a["crowd"]["density_bands_p_m2"]
    fire_zone = overrides.get("fire_zone")

    def density_factor(values: np.ndarray) -> np.ndarray:
        """Piecewise factor by density band (docs/03 M07 ``f_density``)."""
        factor = np.ones_like(values)
        factor = np.where(values >= float(bands["amber"]), 1.25, factor)
        factor = np.where(values >= float(bands["red"]), 1.6, factor)
        factor = np.where(values >= float(bands["critical"]), 2.0, factor)
        return factor

    f_heat = 1.0 + heat_increase * np.clip(heat_index - heat_threshold, 0.0, None)

    incident_rows: list[dict[str, Any]] = []
    zone_centroid = dict(
        zip(
            zones["zone_id"],
            zip(zones["centroid_lat"], zones["centroid_lon"], strict=True),
            strict=True,
        )
    )
    incident_counter = 0
    for zone_id in zone_ids:
        zone_pop = pop[zone_id].to_numpy()
        f_dens = density_factor(density[zone_id].to_numpy())
        zone_boost = 1.0 if zone_id != fire_zone else 1.0 + severe_share * 10.0
        lam = (
            zone_pop
            * base_rate_per_person_hour
            * hours_per_step
            * f_heat
            * f_dens
            * medical_multiplier
            * zone_boost
        )
        counts = rng.poisson(np.clip(lam, 0.0, None))
        lat, lon = zone_centroid.get(zone_id, (np.nan, np.nan))
        for position, count in enumerate(counts):
            if count <= 0:
                continue
            timestamp = steps[position]
            categories = _weighted_choice(rng, MEDICAL_CATEGORIES, int(count))
            is_severe = rng.random(int(count)) < severe_share
            severities = np.where(
                is_severe,
                rng.choice(SEVERE_LEVELS, size=int(count)),
                rng.choice((1, 2, 3), size=int(count)),
            )
            transferred = rng.random(int(count)) < transfer_share
            for index in range(int(count)):
                incident_counter += 1
                hospital = None
                outcome = "treated_on_site"
                if transferred[index]:
                    hospital = "H01" if rng.random() < 0.6 else "H02"
                    outcome = "transferred"
                elif severities[index] >= 3:
                    outcome = "referred"
                incident_rows.append(
                    {
                        "incident_id": f"MED{incident_counter:06d}",
                        "timestamp": timestamp,
                        "zone_id": zone_id,
                        "lat": float(lat),
                        "lon": float(lon),
                        "category": str(categories[index]),
                        "severity": int(severities[index]),
                        "outcome": outcome,
                        "transferred_to": hospital,
                    }
                )
    medical = pd.DataFrame(
        incident_rows,
        columns=[
            "incident_id",
            "timestamp",
            "zone_id",
            "lat",
            "lon",
            "category",
            "severity",
            "outcome",
            "transferred_to",
        ],
    )
    out.tables["medical_incidents"] = _provenance(medical, "derived_medical")

    # ------------------------------------------------------- D09 ambulance dispatch
    rng = rng_for(seed, scenario_id, "ambulance")
    dispatch_delay = float(med["dispatch_delay_min"])
    cycle_time = float(med["ambulance_cycle_time_min"])
    fleet_multiplier = float(
        (overrides.get("resource_availability_multiplier") or {}).get("ambulance", 1.0)
    )
    staging = list(world_cfg["ambulance_staging"])
    dispatch_rows: list[dict[str, Any]] = []
    transfers = medical.loc[medical["outcome"] == "transferred"] if len(medical) else medical
    for position, row in enumerate(transfers.itertuples(index=False)):
        # A smaller fleet means longer waits before a vehicle is free.
        congestion = 1.0 + 0.5 * rng.random()
        queue_penalty = (1.0 / max(fleet_multiplier, 0.1) - 1.0) * dispatch_delay
        travel = (cycle_time / 4.0) * congestion
        dispatch_ts = row.timestamp + pd.Timedelta(minutes=dispatch_delay + queue_penalty)
        arrival_ts = dispatch_ts + pd.Timedelta(minutes=travel)
        hospital_ts = arrival_ts + pd.Timedelta(minutes=travel * 1.2)
        dispatch_rows.append(
            {
                "incident_id": row.incident_id,
                "vehicle_id": f"AMB{(position % max(1, int(20 * fleet_multiplier))) + 1:02d}",
                "dispatch_ts": dispatch_ts,
                "arrival_ts": arrival_ts,
                "hospital_ts": hospital_ts,
                "from_site": staging[position % len(staging)],
            }
        )
    out.tables["ambulance_dispatch"] = _provenance(
        pd.DataFrame(
            dispatch_rows,
            columns=[
                "incident_id",
                "vehicle_id",
                "dispatch_ts",
                "arrival_ts",
                "hospital_ts",
                "from_site",
            ],
        ),
        "derived_ambulance",
    )

    # ------------------------------------------------------- D10 security incidents
    sec = a["security"]
    rng = rng_for(seed, scenario_id, "security")
    base_incidents = float(sec["base_incidents_per_100k_per_hr"])
    lost_rate = float(sec["lost_persons_per_100k_per_hr"])
    response_sla = float(sec["response_sla_min"])
    police_multiplier = float(
        (overrides.get("resource_availability_multiplier") or {}).get("police", 1.0)
    )
    sec_rows: list[dict[str, Any]] = []
    sec_counter = 0
    for zone_id in zone_ids:
        zone_pop = pop[zone_id].to_numpy()
        f_dens = density_factor(density[zone_id].to_numpy())
        per_100k = (base_incidents + lost_rate) * hours_per_step
        lam = zone_pop / 100_000.0 * per_100k * f_dens
        counts = rng.poisson(np.clip(lam, 0.0, None))
        for position, count in enumerate(counts):
            if count <= 0:
                continue
            types = _weighted_choice(rng, SECURITY_TYPES, int(count))
            for index in range(int(count)):
                sec_counter += 1
                # Fewer police means slower response.
                response = float(sec["dispatch_delay_min"]) + response_sla * (
                    0.5 + 0.5 * rng.random()
                ) / max(police_multiplier, 0.1)
                sec_rows.append(
                    {
                        "incident_id": f"SEC{sec_counter:06d}",
                        "timestamp": steps[position],
                        "zone_id": zone_id,
                        "type": str(types[index]),
                        "severity": int(rng.choice((1, 2, 3, 4), p=(0.5, 0.3, 0.15, 0.05))),
                        "response_min": float(response),
                    }
                )
    out.tables["security_incidents"] = _provenance(
        pd.DataFrame(
            sec_rows,
            columns=["incident_id", "timestamp", "zone_id", "type", "severity", "response_min"],
        ),
        "derived_security",
    )

    # ---------------------------------------------------------- D11 access control
    rng = rng_for(seed, scenario_id, "access")
    violation_rate = float(sec["unauthorized_entries_per_secure_gate_per_hr"])
    secure_gates = [
        gate_id for gate_id, spec in world_cfg["gates"].items() if bool(spec.get("secure"))
    ]
    access_rows: list[dict[str, Any]] = []
    gate_pivot = gate_entries.pivot_table(
        index="timestamp", columns="gate_id", values="entries", aggfunc="sum"
    ).fillna(0.0)
    for gate_id in gate_pivot.columns:
        admitted = gate_pivot[gate_id].to_numpy()
        is_secure = gate_id in secure_gates
        # One sampled check per step per gate, plus violations at the secure gates.
        for position, count in enumerate(admitted):
            if count <= 0:
                continue
            timestamp = steps[position]
            credential = str(_weighted_choice(rng, CREDENTIAL_CLASSES, 1)[0])
            access_rows.append(
                {
                    "timestamp": timestamp,
                    "gate_id": gate_id,
                    "credential_class": credential,
                    "result": "granted",
                }
            )
            if is_secure:
                violations = rng.poisson(violation_rate * hours_per_step)
                for _ in range(int(violations)):
                    access_rows.append(
                        {
                            "timestamp": timestamp,
                            "gate_id": gate_id,
                            "credential_class": "public",
                            "result": "forced" if rng.random() < 0.3 else "denied",
                        }
                    )
    out.tables["access_control"] = _provenance(
        pd.DataFrame(access_rows, columns=["timestamp", "gate_id", "credential_class", "result"]),
        "derived_access",
    )

    # ---------------------------------------------------------------- D14 water
    water = a["water"]
    rng = rng_for(seed, scenario_id, "water")
    litres_per_person_hour = float(water["litres_per_person_per_hour_present"])
    heat_bonus = float(water["heat_multiplier_per_c_above_30"])
    pumps = {pump_id: float(spec["design_lph"]) for pump_id, spec in world_cfg["pumps"].items()}
    outage_supply_factor = float(water["pump_outage_supply_factor"])
    substations_down = set(overrides.get("substations_down") or [])
    outage = in_window_series(steps, overrides.get("window"))
    water_rows: list[dict[str, Any]] = []
    heat_factor = 1.0 + heat_bonus * np.clip(temperature - 30.0, 0.0, None)
    supply_per_zone = sum(pumps.values()) * hours_per_step / max(1, len(zone_ids))
    for zone_id in zone_ids:
        consumption = (
            pop[zone_id].to_numpy() * litres_per_person_hour * hours_per_step * heat_factor
        )
        # A pump on a downed substation delivers nothing during the outage window.
        zone_pumps = [
            pump_id for pump_id, spec in world_cfg["pumps"].items() if spec["zone_id"] == zone_id
        ]
        pump_down = any(
            world_cfg["pumps"][pump_id]["substation"] in substations_down for pump_id in zone_pumps
        )
        supply = np.full(len(steps), supply_per_zone)
        if pump_down:
            supply = np.where(outage, supply * outage_supply_factor, supply)
        storage = np.clip(np.cumsum(supply - consumption) + supply_per_zone * 4.0, 0.0, None)
        water_rows.append(
            pd.DataFrame(
                {
                    "timestamp": steps,
                    "zone_id": zone_id,
                    "asset_id": zone_pumps[0] if zone_pumps else None,
                    "consumption_l": consumption,
                    "supply_l": supply,
                    "storage_l": storage,
                }
            )
        )
    out.tables["water_15min"] = _provenance(
        pd.concat(water_rows, ignore_index=True), "derived_water"
    )

    # ------------------------------------------------------------- D15 toilet usage
    san = a["sanitation"]
    rng = rng_for(seed, scenario_id, "sanitation")
    uses_per_person_hour = float(san["uses_per_person_per_hour_present"])
    users_per_toilet_hour = float(san["users_per_toilet_per_hour"])
    uses_per_cleaning = float(san["uses_per_cleaning"])
    flood_rain_mm_hr = float(san["flood_rain_mm_hr"])
    flooded_unit_share = float(san["flooded_unit_share"])
    clusters = world_cfg["toilet_clusters"]
    toilet_rows: list[pd.DataFrame] = []
    for cluster_id, spec in clusters.items():
        zone_id = spec["zone_id"]
        if zone_id not in pop.columns:
            continue
        units = float(spec["max_units"])
        # A low-lying cluster loses units while it is actually raining hard, taken from the
        # weather series rather than the scenario flag, so a wet day in the real Open-Meteo
        # history affects sanitation too - not only the S03 override.
        units_active = np.full(len(steps), units)
        if bool(spec.get("low_lying")):
            heavy = rain >= flood_rain_mm_hr
            units_active = np.where(heavy, units * flooded_unit_share, units)
        uses = (
            pop[zone_id].to_numpy()
            * uses_per_person_hour
            * hours_per_step
            / max(1, len([c for c in clusters.values() if c["zone_id"] == zone_id]))
        )
        capacity = units_active * users_per_toilet_hour * hours_per_step
        queue = np.clip(uses - capacity, 0.0, None) * 0.5
        cumulative = np.cumsum(uses)
        cycles = np.floor(cumulative / uses_per_cleaning)
        status_index = np.clip((uses / np.maximum(capacity, 1e-9) * 2).astype(int), 0, 2)
        toilet_rows.append(
            pd.DataFrame(
                {
                    "timestamp": steps,
                    "cluster_id": cluster_id,
                    "uses": np.round(uses).astype("int64"),
                    "queue_persons": np.round(queue).astype("int64"),
                    "clean_status": [CLEAN_STATUSES[i] for i in status_index],
                    "units_active": np.round(units_active).astype("int64"),
                    "cleaning_cycles": cycles,
                }
            )
        )
    out.tables["toilet_usage"] = _provenance(
        pd.concat(toilet_rows, ignore_index=True), "derived_sanitation"
    )

    # ------------------------------------------------------------------- D16 waste
    waste_cfg = a["waste"]
    kg_per_person_hour = float(waste_cfg["kg_per_person_per_hour_present"])
    bin_capacity = float(waste_cfg["bin_capacity_kg"])
    bin_groups = world_cfg["waste_bin_groups"]
    waste_rows: list[pd.DataFrame] = []
    for group_id, spec in bin_groups.items():
        zone_id = spec["zone_id"]
        if zone_id not in pop.columns:
            continue
        added = pop[zone_id].to_numpy() * kg_per_person_hour * hours_per_step
        # Collected on a fixed round; the fill level resets when a vehicle calls.
        collection_every = grid.steps_per_hour * 4
        fill = np.zeros(len(steps))
        collected = np.zeros(len(steps))
        level = 0.0
        for position, amount in enumerate(added):
            level += amount
            if position > 0 and position % collection_every == 0:
                collected[position] = level
                level = 0.0
            fill[position] = min(100.0, 100.0 * level / bin_capacity)
        waste_rows.append(
            pd.DataFrame(
                {
                    "timestamp": steps,
                    "bin_group_id": group_id,
                    "zone_id": zone_id,
                    "kg_added": added,
                    "fill_pct": fill,
                    "collected_kg": collected,
                }
            )
        )
    out.tables["waste"] = _provenance(pd.concat(waste_rows, ignore_index=True), "derived_waste")

    # ------------------------------------------------------------- D17 food inventory
    food = a["food"]
    rng = rng_for(seed, scenario_id, "food")
    meals_per_person_hour = float(food["meals_per_person_per_hour_market"])
    lead_time_hours = float(food["lead_time_hours"]) * float(
        overrides.get("food_lead_time_multiplier", 1.0)
    )
    min_cover = float(food["min_stock_cover_hours"])
    hourly = grid.hourly_index()
    pop_hourly = pop.reindex(hourly, method="ffill").ffill().bfill()
    market_zones = [
        zone_id
        for zone_id in zone_ids
        if str(zones.loc[zones["zone_id"] == zone_id, "type"].iloc[0]) in ("market", "hub")
    ]
    food_rows: list[pd.DataFrame] = []
    for index, zone_id in enumerate(market_zones):
        outlet_id = f"FO{index + 1:02d}"
        sold = pop_hourly[zone_id].to_numpy() * meals_per_person_hour
        stock = np.zeros(len(hourly))
        deliveries = np.zeros(len(hourly))
        level = float(sold[: int(min_cover)].sum()) if len(sold) else 0.0
        for position, amount in enumerate(sold):
            level -= amount
            # Reorder when cover drops below the minimum; the delivery lands after the lead time.
            if level < amount * min_cover:
                delivery = amount * (min_cover + lead_time_hours)
                deliveries[position] = delivery
                level += delivery
            stock[position] = max(0.0, level)
        food_rows.append(
            pd.DataFrame(
                {
                    "timestamp": hourly,
                    "outlet_id": outlet_id,
                    "zone_id": zone_id,
                    "meals_sold": sold,
                    "stock_meals": stock,
                    "deliveries": deliveries,
                    "lead_time_h": lead_time_hours,
                }
            )
        )
    out.tables["food_inventory_hourly"] = _provenance(
        pd.concat(food_rows, ignore_index=True), "derived_food"
    )

    # ------------------------------------------------------------------- D18 power
    power = a["power"]
    rng = rng_for(seed, scenario_id, "power")
    base_kw = float(power["base_kw_per_zone"])
    watts_per_person = float(power["watts_per_person"])
    cooling_kw = float(power["cooling_kw_per_c_above_30_per_zone"])
    generator_fuel = float(power["generator_fuel_l"])
    generator_lph = float(power["generator_l_per_hr_at_full_load"])
    zone_substation: dict[str, str] = {}
    for substation_id, spec in world_cfg["substations"].items():
        for zone_id in spec["feeds_zones"]:
            zone_substation[zone_id] = substation_id
    power_rows: list[pd.DataFrame] = []
    cooling_term = cooling_kw * np.clip(temperature - 30.0, 0.0, None)
    for zone_id in zone_ids:
        substation = zone_substation.get(zone_id, "SS02")
        kw = base_kw + pop[zone_id].to_numpy() * watts_per_person / 1000.0 + cooling_term
        grid_available = np.ones(len(steps), dtype=bool)
        if substation in substations_down:
            grid_available = ~outage
        generator_on = ~grid_available
        # Fuel drains only while a generator is running.
        load_fraction = np.clip(kw / max(base_kw * 4.0, 1e-9), 0.05, 1.0)
        burn = np.where(generator_on, generator_lph * load_fraction * hours_per_step, 0.0)
        fuel = np.clip(generator_fuel - np.cumsum(burn), 0.0, None)
        power_rows.append(
            pd.DataFrame(
                {
                    "timestamp": steps,
                    "asset_id": substation,
                    "zone_id": zone_id,
                    "kw": kw,
                    "kwh": kw * hours_per_step,
                    "voltage": np.where(grid_available, 415.0, 400.0)
                    + rng.normal(0.0, 1.5, len(steps)),
                    "grid_available": grid_available,
                    "generator_on": generator_on,
                    "fuel_l": fuel,
                }
            )
        )
    out.tables["power_15min"] = _provenance(
        pd.concat(power_rows, ignore_index=True), "derived_power"
    )

    # ----------------------------------------------------------------- D19 network
    net = a["network"]
    rng = rng_for(seed, scenario_id, "network")
    camera_bitrate = float(net["camera_bitrate_mbps"])
    active_share = float(net["active_user_share"])
    kbps_per_user = float(net["kbps_per_active_user"])
    tower_capacity = net["tower_capacity_mbps"]
    network_steps = grid.network_index()
    pop_net = pop.reindex(network_steps, method="ffill").ffill().bfill()
    tower_zone = {
        tower_id: spec["zone_id"] for tower_id, spec in world_cfg["network_towers"].items()
    }
    tower_substation = {
        tower_id: spec["substation"] for tower_id, spec in world_cfg["network_towers"].items()
    }
    active_cameras = cameras.loc[cameras["status"] == "active"]
    cameras_per_tower = active_cameras.groupby("tower_id").size().to_dict()
    net_rows: list[pd.DataFrame] = []
    outage_net = in_window_series(network_steps, overrides.get("window"))
    for tower_id, capacity in tower_capacity.items():
        zone_id = tower_zone.get(tower_id)
        served = (
            pop_net[zone_id].to_numpy()
            if zone_id in pop_net.columns
            else np.zeros(len(network_steps))
        )
        camera_load = cameras_per_tower.get(tower_id, 0) * camera_bitrate
        user_load = served * active_share * kbps_per_user / 1000.0
        used = camera_load + user_load
        up = np.ones(len(network_steps), dtype=bool)
        if tower_substation.get(tower_id) in substations_down:
            up = ~outage_net
        used = np.where(up, used, 0.0)
        utilisation = used / float(capacity)
        net_rows.append(
            pd.DataFrame(
                {
                    "timestamp": network_steps,
                    "tower_id": tower_id,
                    "bandwidth_used_mbps": used,
                    "capacity_mbps": float(capacity),
                    # Latency climbs steeply as a link saturates.
                    "latency_ms": 12.0 + 90.0 * np.clip(utilisation - 0.7, 0.0, None) ** 2 * 10,
                    "packet_loss_pct": np.clip((utilisation - 0.85) * 20.0, 0.0, 100.0),
                    "up": up,
                }
            )
        )
    out.tables["network_5min"] = _provenance(
        pd.concat(net_rows, ignore_index=True), "derived_network"
    )

    # ----------------------------------------------------------------- D06 parking
    transport = a["transport"]
    rng = rng_for(seed, scenario_id, "parking")
    car_share = float(transport["car_mode_share"])
    persons_per_car = float(transport["persons_per_car"])
    capacities = transport["parking_capacity"]
    total_capacity = sum(float(v) for v in capacities.values())
    arrivals_total = entries.sum(axis=1).to_numpy()
    car_arrivals = arrivals_total * car_share / persons_per_car
    # Cars stay for roughly twice the ghat dwell: the visit plus walking at both ends.
    ghat_dwell_min = float(a["crowd"]["mean_dwell_min"]["ghat"])
    dwell_steps_cars = max(1, round(ghat_dwell_min * 2 / grid.grid_min))
    parking_rows: list[pd.DataFrame] = []
    for parking_id, capacity in capacities.items():
        share = float(capacity) / total_capacity
        site_entries = car_arrivals * share
        site_exits = np.zeros(len(steps))
        site_exits[dwell_steps_cars:] = site_entries[:-dwell_steps_cars]
        occupied = np.clip(np.cumsum(site_entries - site_exits), 0.0, float(capacity))
        parking_rows.append(
            pd.DataFrame(
                {
                    "timestamp": steps,
                    "parking_id": parking_id,
                    "entries": np.round(site_entries).astype("int64"),
                    "exits": np.round(site_exits).astype("int64"),
                    "occupied": np.round(occupied).astype("int64"),
                    "capacity": int(capacity),
                }
            )
        )
    out.tables["parking_15min"] = _provenance(
        pd.concat(parking_rows, ignore_index=True), "derived_parking"
    )

    # ----------------------------------------------------------------- D07 transit
    rng = rng_for(seed, scenario_id, "transit")
    bus_share = float(transport["bus_mode_share"])
    bus_capacity = float(transport["bus_capacity"])
    load_factor = float(transport["bus_load_factor"])
    round_trip = float(transport["round_trip_min"])
    routes = world_cfg["shuttle_routes"]
    transit_rows: list[pd.DataFrame] = []
    for index, (route_id, _spec) in enumerate(routes.items()):
        share = 0.6 if index == 0 else 0.4
        boardings = arrivals_total * bus_share * share
        vehicles = np.ceil(
            boardings * (round_trip / grid.grid_min) / max(bus_capacity * load_factor, 1e-9)
        )
        headway = np.where(vehicles > 0, round_trip / np.maximum(vehicles, 1), round_trip)
        transit_rows.append(
            pd.DataFrame(
                {
                    "timestamp": steps,
                    "route_id": route_id,
                    "boardings": np.round(boardings).astype("int64"),
                    "alightings": np.round(boardings * 0.95).astype("int64"),
                    "vehicles_active": vehicles.astype("int64"),
                    "avg_wait_min": headway / 2.0,
                }
            )
        )
    out.tables["transit_15min"] = _provenance(
        pd.concat(transit_rows, ignore_index=True), "derived_transit"
    )

    # ----------------------------------------------------------- D04 traffic probes
    rng = rng_for(seed, scenario_id, "traffic")
    alpha = float(transport["bpr_alpha"])
    beta = float(transport["bpr_beta"])
    closed_links = set(overrides.get("closed_links") or [])
    vehicles_total = car_arrivals + arrivals_total * bus_share / bus_capacity
    probe_rows: list[pd.DataFrame] = []
    link_count = len(key_links)
    for row in key_links.itertuples(index=False):
        capacity = float(row.capacity_veh_hr)
        length_km = float(row.length_m) / 1000.0
        free_flow_min = length_km / max(float(row.maxspeed_kmh), 1e-9) * 60.0
        # A share of event traffic plus a background profile.
        share = 1.0 / link_count + 0.08 * math.sin(hash(row.link_id) % 7)
        volume = vehicles_total * max(share, 0.02) * grid.steps_per_hour
        if row.link_id in closed_links:
            volume = np.zeros(len(steps))
            travel_min = np.full(len(steps), np.inf)
            speed = np.zeros(len(steps))
        else:
            travel_min = bpr_travel_time(free_flow_min, volume, capacity, alpha=alpha, beta=beta)
            speed = np.where(travel_min > 0, length_km / (travel_min / 60.0), 0.0)
        probe_rows.append(
            pd.DataFrame(
                {
                    "timestamp": steps,
                    "link_id": row.link_id,
                    "speed_kmh": np.where(np.isfinite(speed), speed, 0.0),
                    "travel_time_s": np.where(np.isfinite(travel_min), travel_min * 60.0, 0.0),
                    "volume_proxy": volume,
                }
            )
        )
    out.tables["traffic_probe_15min"] = _provenance(
        pd.concat(probe_rows, ignore_index=True), "derived_traffic"
    )

    # ------------------------------------------------------------ D05 signal cycles
    rng = rng_for(seed, scenario_id, "signals")
    junctions = world_cfg["key_intersections"]
    signal_rows: list[pd.DataFrame] = []
    for junction_id, spec in junctions.items():
        approaches = spec["approaches"]
        approach_volume = np.zeros(len(steps))
        for link_id in approaches:
            match = out.tables["traffic_probe_15min"]
            selected = match.loc[match["link_id"] == link_id, "volume_proxy"]
            if len(selected) == len(steps):
                approach_volume = approach_volume + selected.to_numpy()
        cycle = 90.0 + 30.0 * np.clip(approach_volume / max(approach_volume.max(), 1e-9), 0, 1)
        signal_rows.append(
            pd.DataFrame(
                {
                    "timestamp": steps,
                    "junction_id": junction_id,
                    "cycle_s": cycle,
                    "green_s": cycle * 0.45,
                    "queue_veh": approach_volume / grid.steps_per_hour * 0.1,
                }
            )
        )
    out.tables["signal_cycles"] = _provenance(
        pd.concat(signal_rows, ignore_index=True), "derived_signals"
    )

    # ------------------------------------------------- D13 air quality increments
    env = a["environment"]
    aq = air_quality.set_index("timestamp").sort_index()
    aq_hourly = aq.reindex(hourly, method="ffill").ffill().bfill()
    total_pop_hourly = pop_hourly.sum(axis=1).to_numpy()
    crowd_increment = total_pop_hourly / 100_000.0 * 6.0
    traffic_increment = (
        out.tables["traffic_probe_15min"]
        .groupby("timestamp")["volume_proxy"]
        .sum()
        .reindex(hourly, method="ffill")
        .ffill()
        .bfill()
        .to_numpy()
        / 10_000.0
        * 4.0
    )
    noise_base = float(env["noise_base_db"])
    density_hourly = density.reindex(hourly, method="ffill").ffill().bfill()
    mean_density = density_hourly.mean(axis=1).to_numpy()
    enriched = pd.DataFrame(
        {
            "timestamp": hourly,
            "pm2_5": aq_hourly["pm2_5"].to_numpy() + crowd_increment + traffic_increment,
            "pm10": aq_hourly["pm10"].to_numpy() + (crowd_increment + traffic_increment) * 1.8,
            "no2": aq_hourly["no2"].to_numpy() + traffic_increment * 0.8,
            "co": aq_hourly["co"].to_numpy() + traffic_increment * 12.0,
            "noise_db": noise_base + 10.0 * np.log10(1.0 + mean_density / 1.0),
        }
    )
    enriched["is_synthetic"] = air_quality["is_synthetic"].iloc[0] if len(air_quality) else True
    enriched["source"] = (
        f"{air_quality['source'].iloc[0]}+crowd" if len(air_quality) else "derived_air_quality"
    )
    out.tables["air_quality_hourly"] = enriched

    log.info(
        "%s derived %d tables: %s",
        scenario_id,
        len(out.tables),
        ", ".join(f"{name}({len(frame)})" for name, frame in sorted(out.tables.items())),
    )
    return out
