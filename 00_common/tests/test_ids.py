"""Entity-ID patterns from docs/02 section 3, plus the documented AS extension (D2)."""

from __future__ import annotations

import pytest

from twin_common.contracts import (
    ENTITY_ID_PATTERNS,
    EntityType,
    infer_entity_types,
    is_valid_entity_id,
    is_valid_model_id,
    is_valid_scenario_id,
    validate_entity_id,
)
from twin_common.errors import RegistryError

#: Every example in the docs/02 section 3 ID table.
DOC_EXAMPLES: list[tuple[str, str]] = [
    ("event", "EVENT"),
    ("sector", "SA"),
    ("zone", "Z01"),
    ("gate", "G02"),
    ("exit", "E03"),
    ("bridge", "B01"),
    ("road_segment", "R07"),
    ("intersection", "J04"),
    ("route", "VIP1"),
    ("route", "EVAC2"),
    ("parking_site", "P2"),
    ("facility", "H01"),
    ("facility", "MP03"),
    ("service_point", "TC04"),
    ("asset", "GEN02"),
    ("camera", "CAM07"),
    ("resource_pool", "police"),
]

#: Entities docs/04 section 2.2 requires that the docs/02 table does not spell out.
WORLD_EXAMPLES: list[tuple[str, str]] = [
    ("sector", "SB"),
    ("zone", "Z08"),
    ("gate", "G04"),
    ("exit", "E06"),
    ("facility", "PP04"),
    ("facility", "FS01"),
    ("facility", "SH02"),  # shelter
    ("facility", "AS06"),  # ambulance staging - decision D2
    ("facility", "H02"),
    ("service_point", "WP10"),
    ("service_point", "WB08"),
    ("asset", "SS01"),
    ("asset", "NT03"),
    ("asset", "PMP03"),
    ("camera", "CAM24"),
    ("route", "SH1"),  # shuttle route
    ("route", "VIP1"),
    ("parking_site", "P3"),
    ("resource_pool", "network_capacity"),
]


@pytest.mark.parametrize(("entity_type", "entity_id"), DOC_EXAMPLES + WORLD_EXAMPLES)
def test_documented_ids_are_valid(entity_type: str, entity_id: str) -> None:
    assert validate_entity_id(entity_type, entity_id) == entity_id


@pytest.mark.parametrize(
    ("entity_type", "entity_id"),
    [
        ("zone", "G02"),  # gate ID as a zone
        ("zone", "Z1"),  # one digit
        ("zone", "Z001"),  # three digits
        ("gate", "Z01"),
        ("camera", "CAM7"),  # one digit
        ("parking_site", "P12"),  # two digits
        ("sector", "S01"),  # sectors are S + letter
        ("event", "EVENTS"),
        ("resource_pool", "water_cannon"),
        ("asset", "XX01"),
        ("service_point", "TD01"),
    ],
)
def test_mismatched_ids_are_rejected(entity_type: str, entity_id: str) -> None:
    assert not is_valid_entity_id(entity_type, entity_id)
    with pytest.raises(RegistryError, match="does not match the pattern"):
        validate_entity_id(entity_type, entity_id)


def test_every_entity_type_has_a_pattern() -> None:
    assert set(ENTITY_ID_PATTERNS) == set(EntityType)


def test_patterns_are_anchored_by_fullmatch() -> None:
    """A prefix must not be accepted as a whole ID."""
    assert not is_valid_entity_id("zone", "Z01X")
    assert not is_valid_entity_id("camera", "XCAM07")


def test_shuttle_route_and_shelter_are_told_apart_by_type() -> None:
    """SH1 is a route, SH01 is a shelter (facility). Digit count plus type resolves it."""
    assert is_valid_entity_id("route", "SH1")
    assert is_valid_entity_id("facility", "SH01")
    assert not is_valid_entity_id("facility", "SH1")


def test_infer_entity_types_reports_overlaps() -> None:
    """The generic route pattern overlaps others; validation uses the declared type."""
    assert EntityType.ZONE in infer_entity_types("Z01")
    assert EntityType.ROUTE in infer_entity_types("Z01")
    assert infer_entity_types("police") == [EntityType.RESOURCE_POOL]
    assert infer_entity_types("nonsense id") == []


def test_unknown_entity_type_raises() -> None:
    with pytest.raises(RegistryError, match="unknown entity_type"):
        validate_entity_id("spaceship", "X01")


@pytest.mark.parametrize("model_id", ["M01", "M09", "M10", "M21", "M25"])
def test_valid_model_ids(model_id: str) -> None:
    assert is_valid_model_id(model_id)


@pytest.mark.parametrize("model_id", ["M00", "M26", "M1", "M001", "m01", "X01"])
def test_invalid_model_ids(model_id: str) -> None:
    assert not is_valid_model_id(model_id)


@pytest.mark.parametrize("scenario_id", ["S01", "S09", "S10", "S15"])
def test_valid_scenario_ids(scenario_id: str) -> None:
    assert is_valid_scenario_id(scenario_id)


@pytest.mark.parametrize("scenario_id", ["S00", "S16", "S1", "s01", "S99"])
def test_invalid_scenario_ids(scenario_id: str) -> None:
    assert not is_valid_scenario_id(scenario_id)


def test_world_config_entities_all_validate() -> None:
    """Every entity ID in world.yaml must match its type pattern.

    This is the check that catches a typo in the layout config before the generator runs.
    """
    from twin_common.config import world

    w = world()
    groups: list[tuple[str, dict]] = [
        ("zone", w["zones"]),
        ("gate", w["gates"]),
        ("exit", w["exits"]),
        ("bridge", w["bridges"]),
        ("road_segment", w["key_links"]),
        ("intersection", w["key_intersections"]),
        ("parking_site", w["parking_sites"]),
        ("route", w["shuttle_routes"]),
        ("facility", w["hospitals"]),
        ("facility", w["medical_posts"]),
        ("facility", w["police_posts"]),
        ("facility", w["fire_stations"]),
        ("facility", w["shelters"]),
        ("facility", w["ambulance_staging"]),
        ("service_point", w["toilet_clusters"]),
        ("service_point", w["water_points"]),
        ("service_point", w["waste_bin_groups"]),
        ("asset", w["substations"]),
        ("asset", w["generators"]),
        ("asset", w["pumps"]),
        ("asset", w["network_towers"]),
        ("route", w["vip_routes"]),
        ("route", w["emergency_corridors"]),
        ("route", w["evacuation_routes"]),
        ("sector", w["venue"]["sectors"]),
    ]
    for entity_type, mapping in groups:
        for entity_id in mapping:
            validate_entity_id(entity_type, entity_id)


def test_world_camera_ids_validate() -> None:
    from twin_common.config import world

    placement = world()["cameras"]["placement"]
    camera_ids = [row[0] for rows in placement.values() for row in rows]
    assert len(camera_ids) == world()["cameras"]["count"]
    assert len(set(camera_ids)) == len(camera_ids), "duplicate camera IDs"
    for camera_id in camera_ids:
        validate_entity_id("camera", camera_id)
