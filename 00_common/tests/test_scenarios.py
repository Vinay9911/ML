"""Scenario loading, override merging and the generic adjustment layer (docs/05 section 1)."""

from __future__ import annotations

from datetime import datetime

import pytest

from twin_common.contracts import IST
from twin_common.errors import UnknownScenarioError
from twin_common.scenarios import (
    affects_model,
    all_scenarios,
    apply_adjustments,
    closed_entities,
    get_scenario,
    in_window,
    resolve_overrides,
    resource_multiplier,
    scenario_exists,
)

SCENARIO_COUNT = 15

#: docs/05 section 1.1, verbatim.
EXPECTED_OVERRIDES: dict[str, dict] = {
    "S01": {},
    "S02": {"footfall_multiplier": 1.3},
    "S03": {"rain_mm_hr": 50, "window": ["06:00", "09:00"]},
    "S04": {"temperature_offset_c": 6, "humidity_pct_min": 60},
    "S05": {"closed_gates": ["G02"]},
    "S06": {"closed_links": ["R07"]},
    "S07": {"vip_convoy": {"route_id": "VIP1", "start": "07:00", "duration_min": 30}},
    "S08": {"substations_down": ["SS01"], "window": ["05:30", "08:30"]},
    "S09": {"camera_outage_share": 0.2},
    "S10": {"resource_availability_multiplier": {"ambulance": 0.8}},
    "S11": {"resource_availability_multiplier": {"police": 0.9}},
    "S12": {"closed_links": ["B01"]},
    "S13": {"food_lead_time_multiplier": 1.5},
    "S14": {"medical_rate_multiplier": 2.0},
    "S15": {"fire_zone": "Z03", "blocked_exits": ["E03"], "start": "07:30"},
}


# --------------------------------------------------------------------- loading
def test_fifteen_scenarios() -> None:
    assert len(all_scenarios()) == SCENARIO_COUNT
    assert set(all_scenarios()) == {f"S{i:02d}" for i in range(1, SCENARIO_COUNT + 1)}


@pytest.mark.parametrize(("scenario_id", "overrides"), list(EXPECTED_OVERRIDES.items()))
def test_overrides_match_the_docs(scenario_id: str, overrides: dict) -> None:
    assert get_scenario(scenario_id).overrides == overrides


def test_every_scenario_has_a_name() -> None:
    for scenario_id, scenario in all_scenarios().items():
        assert scenario.name and scenario.name != scenario_id


def test_s01_is_the_baseline_with_no_overrides() -> None:
    baseline = get_scenario("S01")
    assert baseline.is_baseline
    assert baseline.overrides == {}
    assert baseline.affects == ()


def test_affects_lists_reference_real_models() -> None:
    from twin_common.contracts import registry as reg

    known = set(reg.all_models())
    for scenario_id, scenario in all_scenarios().items():
        unknown = [m for m in scenario.affects if m not in known]
        assert not unknown, f"{scenario_id} lists unknown models {unknown}"


def test_affects_matches_the_docs_expectations() -> None:
    """Spot-check the docs/05 section 1.2 table."""
    assert affects_model("S14", "M07")
    assert affects_model("S02", "M01")
    assert affects_model("S15", "M11")
    assert not affects_model("S13", "M01")


def test_unknown_scenario_raises_for_a_404() -> None:
    assert not scenario_exists("S99")
    with pytest.raises(UnknownScenarioError, match="unknown scenario_id"):
        get_scenario("S99")


# ------------------------------------------------------------------- merging
def test_resolve_overrides_of_one_scenario() -> None:
    assert resolve_overrides("S02") == {"footfall_multiplier": 1.3}


def test_demo_combination_merges_s02_and_s04() -> None:
    """The docs/01 section 5 demo runs S02 + S04 together."""
    merged = resolve_overrides(["S02", "S04"])
    assert merged == {
        "footfall_multiplier": 1.3,
        "temperature_offset_c": 6,
        "humidity_pct_min": 60,
    }


def test_request_overrides_win() -> None:
    merged = resolve_overrides("S02", {"footfall_multiplier": 2.0, "extra": True})
    assert merged["footfall_multiplier"] == 2.0
    assert merged["extra"] is True


def test_later_scenario_wins_on_a_shared_key() -> None:
    merged = resolve_overrides(["S06", "S12"])
    assert merged["closed_links"] == ["B01"]


def test_merging_does_not_mutate_the_loaded_scenario() -> None:
    resolve_overrides("S02", {"footfall_multiplier": 9.0})
    assert get_scenario("S02").overrides == {"footfall_multiplier": 1.3}


# ---------------------------------------------------------------- adjustments
def test_footfall_multiplier_is_applied() -> None:
    assert apply_adjustments(1000.0, "footfall", resolve_overrides("S02")) == pytest.approx(1300.0)


def test_medical_rate_multiplier_doubles_for_s14() -> None:
    """docs/05 section 1.2: S14 cases approximately 2x."""
    assert apply_adjustments(5.0, "medical_rate", resolve_overrides("S14")) == pytest.approx(10.0)


def test_temperature_offset_is_additive_for_s04() -> None:
    assert apply_adjustments(31.0, "temperature_c", resolve_overrides("S04")) == pytest.approx(37.0)


def test_rain_is_absolute_for_s03() -> None:
    assert apply_adjustments(0.0, "rain_mm_hr", resolve_overrides("S03")) == pytest.approx(50.0)


def test_humidity_minimum_acts_as_a_floor() -> None:
    overrides = resolve_overrides("S04")
    assert apply_adjustments(40.0, "humidity_pct", overrides, floor=True) == pytest.approx(60.0)
    assert apply_adjustments(80.0, "humidity_pct", overrides, floor=True) == pytest.approx(80.0)


def test_combined_scenario_applies_both_effects() -> None:
    overrides = resolve_overrides(["S02", "S04"])
    assert apply_adjustments(1000.0, "footfall", overrides) == pytest.approx(1300.0)
    assert apply_adjustments(31.0, "temperature_c", overrides) == pytest.approx(37.0)


def test_unknown_quantity_is_left_alone() -> None:
    """This is what makes a model insensitive rather than wrong."""
    assert apply_adjustments(42.0, "not_a_quantity", resolve_overrides("S02")) == 42.0


def test_baseline_changes_nothing() -> None:
    for quantity in ("footfall", "temperature_c", "medical_rate", "rain_mm_hr"):
        assert apply_adjustments(7.0, quantity, resolve_overrides("S01")) == 7.0


# --------------------------------------------------------------------- windows
@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [(5, 59, False), (6, 0, True), (7, 30, True), (9, 0, True), (9, 1, False)],
)
def test_s03_rain_window(hour: int, minute: int, expected: bool) -> None:
    window = resolve_overrides("S03")["window"]
    ts = datetime(2027, 8, 2, hour, minute, tzinfo=IST)
    assert in_window(ts, window) is expected


def test_no_window_means_all_day() -> None:
    assert in_window(datetime(2027, 8, 2, 3, 0, tzinfo=IST), None) is True


def test_window_wrapping_midnight() -> None:
    window = ["22:00", "02:00"]
    assert in_window(datetime(2027, 8, 2, 23, 0, tzinfo=IST), window) is True
    assert in_window(datetime(2027, 8, 2, 1, 0, tzinfo=IST), window) is True
    assert in_window(datetime(2027, 8, 2, 12, 0, tzinfo=IST), window) is False


def test_adjustment_respects_the_window_when_a_timestamp_is_given() -> None:
    overrides = resolve_overrides("S03")
    inside = datetime(2027, 8, 2, 7, 0, tzinfo=IST)
    outside = datetime(2027, 8, 2, 12, 0, tzinfo=IST)
    assert apply_adjustments(0.0, "rain_mm_hr", overrides, timestamp=inside) == pytest.approx(50.0)
    assert apply_adjustments(0.0, "rain_mm_hr", overrides, timestamp=outside) == pytest.approx(0.0)


def test_s08_power_window() -> None:
    window = resolve_overrides("S08")["window"]
    assert in_window(datetime(2027, 8, 2, 6, 0, tzinfo=IST), window) is True
    assert in_window(datetime(2027, 8, 2, 9, 0, tzinfo=IST), window) is False


# ------------------------------------------------------------------ structural
def test_closed_entities_groups_the_structural_overrides() -> None:
    assert closed_entities(resolve_overrides("S05"))["gates"] == {"G02"}
    assert closed_entities(resolve_overrides("S06"))["links"] == {"R07"}
    assert closed_entities(resolve_overrides("S12"))["links"] == {"B01"}
    assert closed_entities(resolve_overrides("S15"))["exits"] == {"E03"}
    assert closed_entities(resolve_overrides("S08"))["substations"] == {"SS01"}


def test_closed_entities_of_the_baseline_is_empty() -> None:
    groups = closed_entities(resolve_overrides("S01"))
    assert all(not value for value in groups.values())


def test_resource_multipliers_for_s10_and_s11() -> None:
    assert resource_multiplier(resolve_overrides("S10"), "ambulance") == pytest.approx(0.8)
    assert resource_multiplier(resolve_overrides("S10"), "police") == pytest.approx(1.0)
    assert resource_multiplier(resolve_overrides("S11"), "police") == pytest.approx(0.9)


def test_structural_override_ids_are_valid_entity_ids() -> None:
    """A typo in scenarios.yaml must not reach a model."""
    from twin_common.contracts import validate_entity_id

    for scenario_id in all_scenarios():
        groups = closed_entities(resolve_overrides(scenario_id))
        for gate in groups["gates"]:
            validate_entity_id("gate", gate)
        for exit_id in groups["exits"]:
            validate_entity_id("exit", exit_id)
        for substation in groups["substations"]:
            validate_entity_id("asset", substation)
        for link in groups["links"]:
            # A closed link may be a road segment or the bridge.
            from twin_common.contracts import is_valid_entity_id

            assert is_valid_entity_id("road_segment", link) or is_valid_entity_id("bridge", link), (
                f"{scenario_id} closes unknown link {link}"
            )


def test_resource_multiplier_keys_are_real_resource_types() -> None:
    from twin_common.contracts import registry as reg

    for scenario_id in all_scenarios():
        mapping = resolve_overrides(scenario_id).get("resource_availability_multiplier") or {}
        for resource_type in mapping:
            assert reg.resource_type_exists(resource_type), (
                f"{scenario_id} adjusts unknown resource type {resource_type}"
            )


def test_fire_zone_and_vip_route_reference_the_world() -> None:
    from twin_common.config import world

    w = world()
    assert resolve_overrides("S15")["fire_zone"] in w["zones"]
    assert resolve_overrides("S07")["vip_convoy"]["route_id"] in w["vip_routes"]
