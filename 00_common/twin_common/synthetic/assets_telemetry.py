"""Asset telemetry: D20 ``asset_maintenance`` (docs/04 section 5.6).

M23 trains on the AI4I 2020 dataset, so the event assets need telemetry with the **same
feature semantics** - ambient temperature, process temperature, rotational speed, torque and
tool wear. That mapping is a documented stand-in, not a claim that a generator behaves like a
milling machine (docs/03 M23: "clearly documented as a stand-in").

The failure labels follow the five AI4I failure modes, translated to event equipment:

=====  ==========================  ======================================================
TWF    tool wear failure           cumulative running hours past a wear limit
HDF    heat dissipation failure    small temperature difference at high rotational speed
PWF    power failure               mechanical power outside a usable band
OSF    overstrain failure          wear times torque past a strain limit
RNF    random failure              a small constant background rate
=====  ==========================  ======================================================

Scenario sensitivity (docs/03 M23): S04 heat raises ambient temperature, so heat-dissipation
failures rise; S08 puts generators on full load, so power and overstrain failures rise.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..config import assumptions
from ..logging import get_logger
from .grid import TimeGrid, in_window_series, rng_for

log = get_logger(__name__)

#: Asset kinds that get telemetry, and how hard each is driven relative to a generator.
#: A pump runs steadily, a generator only when the grid is down, a tower is electronic.
TELEMETRY_KINDS: dict[str, float] = {
    "generator": 1.0,
    "pump": 0.75,
    "substation": 0.55,
    "network_tower": 0.35,
}

#: The AI4I failure modes, in the order the dataset lists them.
FAILURE_MODES = ("TWF", "HDF", "PWF", "OSF", "RNF")

#: Modes whose repair renews the worn part, so accumulated wear resets. A power or
#: heat-dissipation fault is transient and leaves the wear where it was.
WEAR_RESETTING_MODES = frozenset({"TWF", "OSF"})

#: Kelvin offset, so the AI4I temperature columns keep their original units.
KELVIN = 273.15


def build_asset_telemetry(
    grid: TimeGrid,
    assets: pd.DataFrame,
    power: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    scenario_id: str,
    overrides: Mapping[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Hourly telemetry and failure labels for every monitored asset.

    Args:
        grid: the world clock (telemetry is hourly, matching AI4I).
        assets: the layout asset registry.
        power: D18, used to know when a generator is actually running.
        weather: D12, for ambient temperature.
        scenario_id, overrides, seed: scenario context.
    """
    a = assumptions()
    rules = a["asset_failure"]
    rng = rng_for(seed, scenario_id, "asset_telemetry")
    hourly = grid.hourly_index()

    weather_hourly = (
        weather.set_index("timestamp").sort_index().reindex(hourly, method="ffill").ffill().bfill()
    )
    ambient_c = weather_hourly["temperature_c"].to_numpy()

    # Which hours each generator is actually carrying load (S08).
    generator_on: dict[str, np.ndarray] = {}
    if len(power):
        running = (
            power.assign(hour=power["timestamp"].dt.floor("h"))
            .groupby(["hour", "asset_id"])["generator_on"]
            .max()
            .unstack(fill_value=False)
            .reindex(hourly, fill_value=False)
        )
        for substation in running.columns:
            generator_on[str(substation)] = running[substation].to_numpy().astype(bool)

    substations_down = set(overrides.get("substations_down") or [])
    outage = in_window_series(hourly, overrides.get("window"))

    monitored = assets.loc[assets["entity_type"] == "asset"].copy()
    monitored["kind"] = monitored["attributes"].map(
        lambda raw: __import__("json").loads(raw).get("kind", "")
    )
    monitored = monitored.loc[monitored["kind"].isin(TELEMETRY_KINDS)]

    frames: list[pd.DataFrame] = []
    for row in monitored.itertuples(index=False):
        kind = str(row.kind)
        duty = TELEMETRY_KINDS[kind]
        asset_id = str(row.entity_id)
        attributes = __import__("json").loads(row.attributes)
        substation = str(attributes.get("substation", ""))

        # Load fraction: a generator only works during an outage; everything else follows a
        # steady duty cycle with noise.
        if kind == "generator":
            on = generator_on.get(substation)
            if on is None:
                on = outage if substation in substations_down else np.zeros(len(hourly), bool)
            load = np.where(on, duty, 0.05)
        elif substation in substations_down:
            # A substation that is down carries nothing; its pumps idle.
            load = np.where(outage, 0.05, duty)
        else:
            load = np.full(len(hourly), duty)
        load = np.clip(load * rng.normal(1.0, 0.06, len(hourly)), 0.02, 1.2)

        # --- AI4I-shaped features -------------------------------------------------
        ambient_k = ambient_c + KELVIN + rng.normal(0.0, 0.4, len(hourly))
        # Process temperature sits above ambient by a load-driven margin.
        process_k = (
            ambient_k
            + float(rules["process_temp_rise_k"]) * load
            + rng.normal(0.0, 0.6, len(hourly))
        )
        speed_rpm = float(rules["nominal_speed_rpm"]) * (
            1.0 + float(rules["speed_load_sensitivity"]) * (1.0 - load)
        ) + rng.normal(0.0, 40.0, len(hourly))
        # Noise is proportional to the signal: an asset at 35 percent duty does not wobble
        # by the same absolute torque as one at full load. An absolute sigma made the
        # coefficient of variation 25 percent for lightly loaded assets and 9 percent for
        # heavy ones, which is what pushed every network tower outside the power band.
        nominal_torque = float(rules["nominal_torque_nm"])
        torque_noise = rng.normal(0.0, float(rules["torque_noise_frac"]), len(hourly))
        torque_nm = nominal_torque * load * (1.0 + torque_noise)
        torque_nm = np.clip(torque_nm, 0.5, None)

        # --- AI4I failure rules ---------------------------------------------------
        # Evaluated in one sequential pass, because wear both drives two of the rules and is
        # reset by a failure (the asset is serviced). Computing the rules vectorised over a
        # cumulative wear series instead made TWF latch on: once wear passed the limit every
        # later hour was a fresh failure, which is how 12 assets produced 7,000 events.
        #
        # The rules also only apply while the asset is RUNNING. An idle generator draws almost
        # no torque, so the AI4I power band would otherwise read every idle hour as a power
        # failure - that alone accounted for 4,981 spurious events.
        temp_difference = process_k - ambient_k
        # Mechanical power: torque (Nm) x angular velocity (rad/s).
        power_w = torque_nm * speed_rpm * 2.0 * np.pi / 60.0

        failure = np.zeros(len(hourly), dtype=int)
        mode = np.full(len(hourly), "", dtype=object)
        wear_series = np.zeros(len(hourly))

        wear_rate = float(rules["wear_min_per_hour"])
        wear_limit = float(rules["wear_failure_min"])
        strain_limit = float(rules["overstrain_min_nm"])
        random_rate = float(rules["random_failure_rate_per_hour"])
        running_load = float(rules["running_load_threshold"])
        hdf_diff = float(rules["heat_dissipation_min_diff_k"])
        hdf_speed = float(rules["heat_dissipation_speed_rpm"])
        power_tolerance = float(rules["power_tolerance"])
        # Expected mechanical power at this duty, i.e. the asset's own operating point.
        expected_power_w = nominal_torque * load * speed_rpm * 2.0 * np.pi / 60.0
        random_draws = rng.random(len(hourly))

        wear = 0.0
        for position in range(len(hourly)):
            is_running = load[position] >= running_load
            if is_running:
                wear += load[position] * wear_rate
            wear_series[position] = wear
            if not is_running:
                continue
            label = ""
            if wear >= wear_limit:
                label = "TWF"
            elif temp_difference[position] < hdf_diff and speed_rpm[position] < hdf_speed:
                label = "HDF"
            elif (
                abs(power_w[position] - expected_power_w[position])
                > power_tolerance * expected_power_w[position]
            ):
                label = "PWF"
            elif wear * torque_nm[position] >= strain_limit:
                label = "OSF"
            elif random_draws[position] < random_rate:
                label = "RNF"
            if label:
                failure[position] = 1
                mode[position] = label
                # Only a wear-related service renews the wear part. A transient power or
                # heat-dissipation fault does not: letting PWF reset wear meant wear never
                # reached its limit and TWF could never fire at all.
                if label in WEAR_RESETTING_MODES:
                    wear = 0.0
        wear_min = wear_series

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": hourly,
                    "asset_id": asset_id,
                    "asset_type": kind,
                    "ambient_temp_k": ambient_k,
                    "process_temp_k": process_k,
                    "speed_rpm": speed_rpm,
                    "torque_nm": torque_nm,
                    "wear_min": wear_min,
                    "failure": failure,
                    "failure_mode": mode,
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    out["is_synthetic"] = True
    out["source"] = "asset_telemetry"
    failures = int(out["failure"].sum())
    log.info(
        "%s asset telemetry: %d rows over %d assets, %d failure events (%s)",
        scenario_id,
        len(out),
        out["asset_id"].nunique(),
        failures,
        dict(out.loc[out["failure"] == 1, "failure_mode"].value_counts()) if failures else {},
    )
    return out
