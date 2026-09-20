r"""Entity-ID patterns from docs/02 section 3, plus model and scenario ID patterns.

Each entity_type owns a regex. `validate_entity_id` checks an ID against the type it was
declared as; it never guesses the type, because some patterns overlap on purpose
(for example the shuttle route SH1 and the shelter SH01 share a prefix and are told apart
by entity_type and digit count).

Documented extension (plans/PHASE_1.md decision D2): `AS\d{2}` ambulance-staging sites are
accepted as `facility`. docs/04 section 2.2 requires AS01-AS06 but the docs/02 section 3 ID
table omits the prefix.
"""

from __future__ import annotations

import re
from typing import Final

from ..errors import RegistryError
from .enums import RESOURCE_TYPES, EntityType

MODEL_ID_RE: Final = re.compile(r"^M(?:0[1-9]|1\d|2[0-5])$")
SCENARIO_ID_RE: Final = re.compile(r"^S(?:0[1-9]|1[0-5])$")
KPI_NAME_RE: Final = re.compile(r"^[a-z][a-z0-9_]*$")
REASON_CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]*$")
SEMVER_RE: Final = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$")

#: One regex per entity_type. Anchored; `fullmatch` is used for checks.
ENTITY_ID_PATTERNS: Final[dict[EntityType, re.Pattern[str]]] = {
    EntityType.EVENT: re.compile(r"EVENT"),
    EntityType.SECTOR: re.compile(r"S[A-Z]"),
    EntityType.ZONE: re.compile(r"Z\d{2}"),
    EntityType.GATE: re.compile(r"G\d{2}"),
    EntityType.EXIT: re.compile(r"E\d{2}"),
    EntityType.BRIDGE: re.compile(r"B\d{2}"),
    EntityType.ROAD_SEGMENT: re.compile(r"R\d{2}"),
    EntityType.INTERSECTION: re.compile(r"J\d{2}"),
    EntityType.ROUTE: re.compile(r"[A-Z]+\d+"),
    EntityType.PARKING_SITE: re.compile(r"P\d"),
    # hospital | medical post | police post | fire station | shelter | ambulance staging (D2)
    EntityType.FACILITY: re.compile(r"(?:H|MP|PP|FS|SH|AS)\d{2}"),
    # toilet cluster | water point | waste bin group
    EntityType.SERVICE_POINT: re.compile(r"(?:TC|WP|WB)\d{2}"),
    # substation | generator | network tower | pump
    EntityType.ASSET: re.compile(r"(?:SS|GEN|NT|PMP)\d{2}"),
    EntityType.CAMERA: re.compile(r"CAM\d{2}"),
    # resource_pool IDs are resource type names, not coded IDs.
    EntityType.RESOURCE_POOL: re.compile("|".join(re.escape(r) for r in RESOURCE_TYPES)),
}

#: The one zone that is not inside another zone.
EVENT_ID: Final = "EVENT"


def is_valid_entity_id(entity_type: EntityType | str, entity_id: str) -> bool:
    """True when `entity_id` matches the pattern registered for `entity_type`."""
    try:
        etype = EntityType(entity_type)
    except ValueError:
        return False
    pattern = ENTITY_ID_PATTERNS[etype]
    return pattern.fullmatch(entity_id) is not None


def validate_entity_id(entity_type: EntityType | str, entity_id: str) -> str:
    """Return `entity_id` unchanged, or raise RegistryError describing the mismatch."""
    try:
        etype = EntityType(entity_type)
    except ValueError as exc:
        raise RegistryError(f"unknown entity_type: {entity_type!r}") from exc
    if not is_valid_entity_id(etype, entity_id):
        raise RegistryError(
            f"entity_id {entity_id!r} does not match the pattern for entity_type "
            f"{etype.value!r} ({ENTITY_ID_PATTERNS[etype].pattern})"
        )
    return entity_id


def infer_entity_types(entity_id: str) -> list[EntityType]:
    """Every entity_type whose pattern accepts `entity_id`.

    Used by tooling and error messages only. Contract validation always uses the declared
    type, because patterns overlap (Z01 also matches the generic route pattern).
    """
    return [etype for etype, pat in ENTITY_ID_PATTERNS.items() if pat.fullmatch(entity_id)]


def is_valid_model_id(model_id: str) -> bool:
    return MODEL_ID_RE.fullmatch(model_id) is not None


def is_valid_scenario_id(scenario_id: str) -> bool:
    return SCENARIO_ID_RE.fullmatch(scenario_id) is not None
