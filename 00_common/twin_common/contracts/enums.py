"""Closed vocabularies from docs/02 sections 3-4 and docs/05 section 5.

All are str-valued so they serialize to plain JSON strings.
"""

from __future__ import annotations

from enum import StrEnum


class State(StrEnum):
    """What kind of value a record carries (docs/02 section 4)."""

    CURRENT = "current"
    FORECAST = "forecast"
    SCENARIO = "scenario"


class Status(StrEnum):
    """Overall outcome of a model run. `degraded` requires at least one warning."""

    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"


class DataSource(StrEnum):
    """Where the input rows came from. `live` is never faked (docs/02 section 8)."""

    SYNTHETIC = "synthetic"
    REPLAY = "replay"
    LIVE = "live"


class RiskLevel(StrEnum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    CRITICAL = "critical"


class EntityType(StrEnum):
    """Entity types from the docs/02 section 3 ID table."""

    EVENT = "event"
    SECTOR = "sector"
    ZONE = "zone"
    GATE = "gate"
    EXIT = "exit"
    BRIDGE = "bridge"
    ROAD_SEGMENT = "road_segment"
    INTERSECTION = "intersection"
    ROUTE = "route"
    PARKING_SITE = "parking_site"
    FACILITY = "facility"
    SERVICE_POINT = "service_point"
    ASSET = "asset"
    CAMERA = "camera"
    RESOURCE_POOL = "resource_pool"


class UpstreamSource(StrEnum):
    """How an upstream output was obtained (docs/02 section 7 resolution order)."""

    INLINE = "inline"
    URL = "url"
    SAMPLE = "sample"
    STUB = "stub"


class Engine(StrEnum):
    """Shared engine that powers a model (docs/01 section 4)."""

    FORECAST = "forecast"
    VISION = "vision"
    RULES = "rules"
    NETWORK = "network"
    FORMULA = "formula"
    LOCATION = "location"
    PEDSIM = "pedsim"
    OPTIMIZE = "optimize"
    MLCLF = "mlclf"


#: The 15 resource types of docs/05 section 5, in the order the study lists them.
RESOURCE_TYPES: tuple[str, ...] = (
    "police",
    "crpf",
    "ambulance",
    "medical_team",
    "fire_vehicle",
    "mobile_toilet",
    "cleaning_staff",
    "water_point",
    "water_tanker",
    "shuttle_bus",
    "traffic_police",
    "waste_vehicle",
    "food_vehicle",
    "backup_generator",
    "network_capacity",
)

#: Risk levels that make reason codes mandatory (docs/02 section 4).
REASON_REQUIRED_LEVELS: frozenset[RiskLevel] = frozenset(
    {RiskLevel.AMBER, RiskLevel.RED, RiskLevel.CRITICAL}
)
