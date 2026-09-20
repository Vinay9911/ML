"""Phase 2 acceptance: the synthetic world and its invariants (docs/04, docs/07 Phase 2).

Generating a world takes about three seconds, so the shared scenarios are built once per
session and the assertions read from the cached result.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from twin_common.config import assumptions, world
from twin_common.io.schemas import TABLE_SCHEMAS
from twin_common.io.tables import check_schema
from twin_common.synthetic.crowd_flow import effective_gate_split
from twin_common.synthetic.generate import generate_world
from twin_common.synthetic.grid import (
    TimeGrid,
    expand_hourly_profile,
    in_window_series,
    lognormal_noise,
    rng_for,
    stable_seed,
)
from twin_common.synthetic.layout import build_layout

PEAK_DAY = "2027-08-02"
#: Scenarios the Phase 2 acceptance list needs. Generated once and shared.
ACCEPTANCE_SCENARIOS = ("S01", "S02", "S05", "S08", "S09", "S12")


@pytest.fixture(scope="module")
def worlds() -> dict[str, object]:
    """Generate every acceptance scenario once, in memory."""
    built: dict[str, object] = {}
    baseline = None
    for scenario_id in ACCEPTANCE_SCENARIOS:
        result = generate_world(scenario_id, seed=42, write=False, baseline_daily_arrivals=baseline)
        built[scenario_id] = result
        if scenario_id == "S01":
            baseline = float(result.manifest["arrivals"]["admitted_total"])
    return built


@pytest.fixture(scope="module")
def s01(worlds):
    return worlds["S01"]


def _peak_density(result) -> pd.Series:
    footfall = result.tables["footfall_15min"]
    peak = footfall.loc[footfall["timestamp"].dt.date.astype(str) == PEAK_DAY]
    return peak.groupby("zone_id")["density_p_m2"].max()


# ------------------------------------------------------------------------ grid
def test_stable_seed_is_reproducible_across_processes() -> None:
    """Python salts hash(), so the seed helper must not use it."""
    assert stable_seed(42, "S01", "crowd") == stable_seed(42, "S01", "crowd")
    assert stable_seed(42, "S01", "crowd") != stable_seed(42, "S01", "medical")
    assert stable_seed(42, "S01", "crowd") != stable_seed(7, "S01", "crowd")
    assert 0 <= stable_seed(42, "a") < 2**63


def test_adding_a_step_does_not_shift_an_existing_stream() -> None:
    """The point of per-step seeding: a new pipeline step must not move an older one."""
    before = rng_for(42, "S01", "crowd_flow").normal(size=5)
    rng_for(42, "S01", "a_new_step_added_later").normal(size=100)
    after = rng_for(42, "S01", "crowd_flow").normal(size=5)
    assert np.allclose(before, after)


def test_time_grid_matches_the_config() -> None:
    grid = TimeGrid.from_config()
    assert grid.grid_min == 15
    assert len(grid.days) == 31
    assert grid.event_dates == [date(2027, 8, 1), date(2027, 8, 2), date(2027, 8, 3)]
    assert grid.is_snan_day(date(2027, 8, 2))
    assert not grid.is_snan_day(date(2027, 8, 1))
    assert len(grid) == 31 * 96
    assert grid.demo_now.isoformat() == "2027-08-02T06:00:00+05:30"


def test_hourly_profile_expands_and_normalises() -> None:
    profile = assumptions()["crowd"]["hourly_profile_snan"]
    expanded = expand_hourly_profile(profile, 4)
    assert len(expanded) == 96
    assert expanded.sum() == pytest.approx(1.0)
    # The busiest hour of the profile must still be the busiest set of steps.
    busiest_hour = int(np.argmax(profile))
    assert int(np.argmax(expanded)) // 4 == busiest_hour


def test_hourly_profile_rejects_a_wrong_length() -> None:
    with pytest.raises(ValueError, match="24 entries"):
        expand_hourly_profile([1, 2, 3], 4)


def test_lognormal_noise_has_mean_one() -> None:
    """Biased noise would break the 'gate entries equal calendar arrivals' invariant."""
    noise = lognormal_noise(np.random.default_rng(0), 200_000, 0.08)
    assert noise.mean() == pytest.approx(1.0, abs=0.01)
    assert (noise > 0).all()


def test_zero_sigma_noise_is_exactly_one() -> None:
    assert (lognormal_noise(np.random.default_rng(0), 10, 0.0) == 1.0).all()


def test_in_window_series_handles_a_wrapping_window() -> None:
    index = pd.date_range("2027-08-02", periods=24, freq="h", tz="Asia/Kolkata")
    inside = in_window_series(index, ["22:00", "02:00"])
    assert inside[23] and inside[0] and inside[1]
    assert not inside[12]
    assert in_window_series(index, None).all()


# ---------------------------------------------------------------------- layout
def test_zone_polygon_areas_match_the_config() -> None:
    layout = build_layout()
    for zone_id, polygon in layout.zone_polygons_m.items():
        configured = float(world()["zones"][zone_id]["area_m2"])
        assert polygon.area == pytest.approx(configured, rel=1e-6)


def test_corridors_run_north_south() -> None:
    """The adjacency chain runs north-south, so a corridor must be taller than it is wide."""
    layout = build_layout()
    for zone_id, spec in world()["zones"].items():
        if spec["type"] != "corridor":
            continue
        min_x, min_y, max_x, max_y = layout.zone_polygons_m[zone_id].bounds
        assert (max_y - min_y) > (max_x - min_x), f"{zone_id} is not portrait"


def test_camera_coverage_is_plausible_in_every_zone() -> None:
    """A camera aimed across a thin zone covers nothing; this catches that regression."""
    layout = build_layout()
    coverage = layout.cameras.groupby("zone_id")["roi_ground_area_m2"].sum()
    for zone_id, covered in coverage.items():
        area = layout.zone_polygons_m[str(zone_id)].area
        share = covered / area
        assert share > 0.15, f"{zone_id} camera coverage is only {share:.1%}"


def test_safe_capacity_follows_the_design_density() -> None:
    layout = build_layout()
    design = float(assumptions()["crowd"]["design_density_p_m2"])
    for row in layout.zones.itertuples(index=False):
        assert row.safe_capacity == pytest.approx(row.usable_area_m2 * design)


def test_camera_outage_is_deterministic_and_correctly_sized() -> None:
    first = build_layout(offline_cameras_share=0.2, seed=42)
    second = build_layout(offline_cameras_share=0.2, seed=42)
    assert (first.cameras["status"].to_numpy() == second.cameras["status"].to_numpy()).all()
    offline = int((first.cameras["status"] == "offline").sum())
    assert offline == round(len(first.cameras) * 0.2)


# ------------------------------------------------------------- gate reassignment
def test_closed_gate_share_is_redistributed_not_lost() -> None:
    crowd = assumptions()["crowd"]
    split = effective_gate_split(crowd["gate_split"], crowd["gate_reassignment"], {"G02"})
    assert "G02" not in split
    assert sum(split.values()) == pytest.approx(1.0)
    # G02 -> G03 0.7, G01 0.3
    assert split["G03"] == pytest.approx(0.20 + 0.30 * 0.7)
    assert split["G01"] == pytest.approx(0.40 + 0.30 * 0.3)


def test_closed_gate_without_a_mapping_spreads_proportionally() -> None:
    split = effective_gate_split({"A": 0.5, "B": 0.3, "C": 0.2}, {}, {"A"})
    assert sum(split.values()) == pytest.approx(1.0)
    assert split["B"] / split["C"] == pytest.approx(0.3 / 0.2)


# ------------------------------------------- docs/04 section 8 core invariants
@pytest.mark.parametrize("scenario_id", ACCEPTANCE_SCENARIOS)
def test_world_validation_passes(worlds, scenario_id: str) -> None:
    report = worlds[scenario_id].report
    assert report.ok, f"{scenario_id} failed: {report.failures}"
    assert len(report.passed) > 100


@pytest.mark.parametrize("scenario_id", ACCEPTANCE_SCENARIOS)
def test_population_continuity(worlds, scenario_id: str) -> None:
    """population(t+1) = population(t) + entries - exits, per zone."""
    footfall = worlds[scenario_id].tables["footfall_15min"].sort_values(["zone_id", "timestamp"])
    for zone_id, group in footfall.groupby("zone_id"):
        population = group["population"].to_numpy()
        predicted = (
            population[:-1] + group["entries"].to_numpy()[:-1] - group["exits"].to_numpy()[:-1]
        )
        error = float(np.abs(predicted - population[1:]).max())
        assert error <= 1.0, f"{scenario_id} {zone_id} continuity error {error}"


@pytest.mark.parametrize("scenario_id", ACCEPTANCE_SCENARIOS)
def test_density_within_physical_sanity(worlds, scenario_id: str) -> None:
    footfall = worlds[scenario_id].tables["footfall_15min"]
    assert float(footfall["density_p_m2"].max()) <= 9.0


@pytest.mark.parametrize("scenario_id", ACCEPTANCE_SCENARIOS)
def test_no_negative_flows(worlds, scenario_id: str) -> None:
    footfall = worlds[scenario_id].tables["footfall_15min"]
    for column in ("population", "entries", "exits"):
        assert (footfall[column] >= 0).all()


@pytest.mark.parametrize("scenario_id", ACCEPTANCE_SCENARIOS)
def test_every_table_validates_against_its_schema(worlds, scenario_id: str) -> None:
    for name, frame in worlds[scenario_id].tables.items():
        if name in TABLE_SCHEMAS:
            check_schema(frame, name)


@pytest.mark.parametrize("scenario_id", ACCEPTANCE_SCENARIOS)
def test_every_row_carries_provenance(worlds, scenario_id: str) -> None:
    """CLAUDE.md hard rule: every data row carries is_synthetic."""
    for name, frame in worlds[scenario_id].tables.items():
        if frame.empty:
            continue
        assert "is_synthetic" in frame.columns, name
        assert "source" in frame.columns, name
        assert not frame["is_synthetic"].isna().any(), name


def test_real_source_rows_are_not_flagged_synthetic(s01) -> None:
    """Weather from Open-Meteo is real data; only the fallback is synthetic."""
    weather = s01.tables["weather_hourly"]
    is_real = s01.manifest["data_sources"]["weather_is_real"]
    assert bool(weather["is_synthetic"].iloc[0]) is (not is_real)


# ---------------------------------------------------- docs/07 Phase 2 acceptance
def test_s01_ghats_reach_amber_but_not_sustained_critical(s01) -> None:
    bands = assumptions()["crowd"]["density_bands_p_m2"]
    peak = _peak_density(s01)
    ghats = [zone_id for zone_id, spec in world()["zones"].items() if spec["type"] == "ghat"]
    assert max(peak[zone] for zone in ghats) >= float(bands["amber"])

    footfall = s01.tables["footfall_15min"]
    ghat_rows = footfall.loc[footfall["zone_id"].isin(ghats)]
    critical_steps = int((ghat_rows["density_p_m2"] >= float(bands["critical"])).sum())
    assert critical_steps < 4, f"{critical_steps} steps at critical is sustained"


def test_demo_story_ghat_a_is_amber_and_rising_at_demo_now(s01) -> None:
    """docs/01 section 5: at 06:00 Ghat A is amber and forecast red within ~40 minutes."""
    bands = assumptions()["crowd"]["density_bands_p_m2"]
    footfall = s01.tables["footfall_15min"]
    ghat = footfall.loc[footfall["zone_id"] == "Z01"].set_index("timestamp")
    at_six = float(ghat.loc[f"{PEAK_DAY} 06:00:00+05:30", "density_p_m2"])
    at_six_45 = float(ghat.loc[f"{PEAK_DAY} 06:45:00+05:30", "density_p_m2"])
    assert float(bands["amber"]) <= at_six < float(bands["red"]), f"Z01 is {at_six} at 06:00"
    assert at_six_45 > at_six, "Z01 density must be rising through demo_now"
    assert at_six_45 >= float(bands["red"]), "Z01 must reach red within the demo window"


def test_s02_arrivals_ratio(worlds) -> None:
    """docs/07 Phase 2: the S02/S01 daily arrivals ratio sits in [1.28, 1.32]."""
    baseline = float(worlds["S01"].manifest["arrivals"]["admitted_total"])
    scenario = float(worlds["S02"].manifest["arrivals"]["admitted_total"])
    ratio = scenario / baseline
    assert 1.28 <= ratio <= 1.32, f"ratio {ratio:.4f}"


def test_s05_closed_gate_admits_nobody_and_fills_the_plaza(worlds) -> None:
    """docs/07 Phase 2: S05 G02 entries = 0 and Z05 peak density > S01."""
    gates = worlds["S05"].tables["gate_entries"]
    assert float(gates.loc[gates["gate_id"] == "G02", "entries"].sum()) == 0.0
    assert _peak_density(worlds["S05"])["Z05"] > _peak_density(worlds["S01"])["Z05"]


def test_s05_queue_builds_at_the_remaining_plaza_gate(worlds) -> None:
    """The mechanism behind the density rise: G03 saturates and people wait in Z05."""
    gates = worlds["S05"].tables["gate_entries"]
    queue = float(gates.loc[gates["gate_id"] == "G03", "queue_persons"].max())
    assert queue > 0, "G03 must saturate under S05"
    baseline = worlds["S01"].tables["gate_entries"]
    assert float(baseline.loc[baseline["gate_id"] == "G03", "queue_persons"].max()) == 0.0


def test_s08_substation_down_in_the_window(worlds) -> None:
    power = worlds["S08"].tables["power_15min"]
    affected = power.loc[power["asset_id"] == "SS01"]
    inside = in_window_series(pd.DatetimeIndex(affected["timestamp"]), ["05:30", "08:30"])
    assert not affected.loc[inside, "grid_available"].any()
    assert affected.loc[inside, "generator_on"].all()
    assert affected.loc[~inside, "grid_available"].all()


def test_s08_generator_fuel_falls_while_running(worlds) -> None:
    power = worlds["S08"].tables["power_15min"]
    affected = power.loc[power["asset_id"] == "SS01"].sort_values("timestamp")
    fuel = affected["fuel_l"].to_numpy()
    assert fuel[-1] < fuel[0], "fuel must drain while the generators run"
    assert (fuel >= 0).all()


def test_s09_camera_availability_drops_by_the_outage_share(worlds) -> None:
    cameras = worlds["S09"].tables["camera_registry"]
    active = float((cameras["status"] == "active").mean())
    assert abs(active - 0.8) <= 0.05, f"active share {active:.2%}"


def test_s12_bridge_carries_nobody(worlds) -> None:
    """docs/04 section 8: S12 B01 flow = 0, and the crowd backs up behind it."""
    footfall = worlds["S12"].tables["footfall_15min"]
    assert float(footfall.loc[footfall["zone_id"] == "Z06", "exits"].sum()) == 0.0
    baseline = worlds["S01"].tables["footfall_15min"]
    assert float(baseline.loc[baseline["zone_id"] == "Z06", "exits"].sum()) > 0.0


def test_s12_back_pressure_holds_the_sealed_zone_at_jam_density(worlds) -> None:
    """Without back-pressure Z06 reached 302 persons/m2; the cap must hold."""
    jam = float(assumptions()["crowd"]["max_holding_density_p_m2"])
    footfall = worlds["S12"].tables["footfall_15min"]
    z06 = float(footfall.loc[footfall["zone_id"] == "Z06", "density_p_m2"].max())
    assert z06 <= jam + 0.01, f"Z06 reached {z06}"
    # and the crowd it cannot shed backs into Z02
    assert _peak_density(worlds["S12"])["Z02"] > _peak_density(worlds["S01"])["Z02"]


# -------------------------------------------------------------- determinism
def test_same_seed_produces_identical_tables() -> None:
    """docs/07 Phase 2: two runs with the same seed give identical file hashes."""
    first = generate_world("S01", seed=42, write=False)
    second = generate_world("S01", seed=42, write=False)
    for name, info in first.manifest["tables"].items():
        assert info["sha256"] == second.manifest["tables"][name]["sha256"], name


def test_a_different_seed_changes_the_stochastic_tables() -> None:
    first = generate_world("S01", seed=42, write=False)
    other = generate_world("S01", seed=7, write=False)
    changed = [
        name
        for name, info in first.manifest["tables"].items()
        if info["sha256"] != other.manifest["tables"][name]["sha256"]
    ]
    assert "footfall_15min" in changed
    assert "medical_incidents" in changed
    # The layout is deliberately seed-independent.
    assert "zones" not in changed


# ---------------------------------------------------------------- performance
def test_generation_is_well_under_the_two_minute_budget(s01) -> None:
    """docs/07 Phase 2: one scenario world in under 2 minutes on CPU."""
    assert s01.elapsed_s < 120.0, f"took {s01.elapsed_s:.1f}s"


def test_manifest_records_what_a_reviewer_needs(s01) -> None:
    manifest = s01.manifest
    assert manifest["seed"] == 42
    assert manifest["scenario_id"] == "S01"
    assert manifest["time_range"]["days"] == 31
    assert manifest["tables"]
    assert all("sha256" in info and "rows" in info for info in manifest["tables"].values())
    assert "weather" in manifest["data_sources"]


def test_medical_rate_matches_the_documented_assumption(s01) -> None:
    """1 presentation per 1000 attendees per day, before the heat and density uplift.

    The uplift sits near 1.0 on the baseline because the real July weather is cool: the heat
    index peaks at 31.2 C, just under the 32 C threshold at which `f_heat` starts to bite, so
    only the density factor contributes. Poisson sampling then puts the realised ratio a
    fraction either side of 1. S04 is where the heat uplift actually shows, and
    `test_s03_rain_suppresses_arrivals_and_s04_heat_raises_cases` covers that.
    """
    rate = float(assumptions()["medical"]["presentations_per_1000_attendees_per_day"])
    attendees = float(s01.tables["gate_entries"]["entries"].sum())
    cases = len(s01.tables["medical_incidents"])
    baseline = attendees / 1000.0 * rate
    uplift = cases / baseline
    assert 0.9 <= uplift <= 1.6, f"uplift {uplift:.2f}x is outside the plausible range"


def test_asset_failure_rate_is_plausible(s01) -> None:
    """AI4I 2020 runs at 3.4 percent; a synthetic fleet should be in the same order."""
    assets = s01.tables["asset_maintenance"]
    rate = float(assets["failure"].mean())
    assert 0.0 < rate < 0.10, f"failure rate {rate:.2%}"
    assert assets.loc[assets["failure"] == 1, "failure_mode"].nunique() >= 2


def test_roster_covers_every_resource_type(s01) -> None:
    from twin_common.contracts.enums import RESOURCE_TYPES

    roster = s01.tables["resource_roster"]
    assert set(roster["resource_type"]) == set(RESOURCE_TYPES)
    assert (roster["quantity_available"] >= 0).all()


def test_scenario_availability_multiplier_reaches_the_roster() -> None:
    """S10 cuts the ambulance fleet by 20 percent; M24 reads that from D22."""
    baseline = generate_world("S01", seed=42, write=False)
    reduced = generate_world("S10", seed=42, write=False)

    def total(result, resource_type: str) -> float:
        roster = result.tables["resource_roster"]
        return float(
            roster.loc[roster["resource_type"] == resource_type, "quantity_available"].sum()
        )

    ratio = total(reduced, "ambulance") / total(baseline, "ambulance")
    assert ratio == pytest.approx(0.8, abs=0.02)
    # An untouched resource type is unchanged.
    assert total(reduced, "police") / total(baseline, "police") == pytest.approx(1.0, abs=0.02)


def test_s14_doubles_medical_cases() -> None:
    """docs/05 section 1.2: S14 cases approximately 2x, within 5 percent."""
    baseline = generate_world("S01", seed=42, write=False)
    surge = generate_world("S14", seed=42, write=False)
    ratio = len(surge.tables["medical_incidents"]) / len(baseline.tables["medical_incidents"])
    assert 1.90 <= ratio <= 2.10, f"ratio {ratio:.4f}"


def test_s03_rain_suppresses_arrivals_and_s04_heat_raises_cases() -> None:
    """docs/05 section 1.2 directions for the two weather scenarios."""
    baseline = generate_world("S01", seed=42, write=False)
    rain = generate_world("S03", seed=42, write=False)
    heat = generate_world("S04", seed=42, write=False)

    assert float(rain.manifest["arrivals"]["admitted_total"]) < float(
        baseline.manifest["arrivals"]["admitted_total"]
    ), "S03 must reduce arrivals"
    assert float(rain.tables["weather_hourly"]["rain_mm"].max()) >= 50.0

    baseline_heat = float(baseline.tables["weather_hourly"]["heat_index_c"].max())
    scenario_heat = float(heat.tables["weather_hourly"]["heat_index_c"].max())
    assert scenario_heat > baseline_heat, "S04 must raise the heat index"
    assert len(heat.tables["medical_incidents"]) > len(baseline.tables["medical_incidents"])


def test_scenario_weather_is_internally_consistent() -> None:
    """The heat index is recomputed, so it can never disagree with its inputs."""
    from twin_common.engines.formula import heat_index_c

    result = generate_world("S04", seed=42, write=False)
    weather = result.tables["weather_hourly"]
    recomputed = heat_index_c(weather["temperature_c"], weather["humidity_pct"])
    assert np.allclose(weather["heat_index_c"].to_numpy(), recomputed)


def test_offline_generation_succeeds(monkeypatch) -> None:
    """docs/07 Phase 2: TWIN_OFFLINE=1 generation works from cached or bundled data."""
    monkeypatch.setenv("TWIN_OFFLINE", "1")
    result = generate_world("S01", seed=42, write=False)
    assert result.report.ok
    assert result.tables["footfall_15min"].shape[0] > 0
