"""The KPI registry must match docs/05 section 2 exactly.

The headline assertion is the one docs/07 Phase 1 names: exactly 93 non-extension KPIs.
"""

from __future__ import annotations

import pytest

from twin_common.config import kpi_registry
from twin_common.contracts import registry as reg

#: docs/05 section 2: 17+14+9+8+9+4+12+12+8 = 93.
STUDY_KPI_COUNT = 93
EXTENSION_KPI_COUNT = 16

#: The domain totals of the docs/05 section 2 headings. `utilities`, `weather` and
#: `environment` are one 12-KPI heading in the docs; `resources` and `overall` are one
#: 8-KPI heading. The registry splits them for clarity, so they are grouped back here.
DOC_SECTION_TOTALS: dict[tuple[str, ...], int] = {
    ("crowd",): 17,
    ("traffic",): 14,
    ("medical",): 9,
    ("security",): 8,
    ("emergency",): 9,
    ("vip",): 4,
    ("supply",): 12,
    ("utilities", "weather", "environment"): 12,
    ("resources", "overall"): 8,
}


def test_study_kpi_count_is_93() -> None:
    study = [name for name, info in reg.all_kpis().items() if not info.extension]
    assert len(study) == STUDY_KPI_COUNT, (
        f"expected {STUDY_KPI_COUNT} study KPIs from docs/05 section 2, got {len(study)}"
    )


def test_extension_kpi_count() -> None:
    extensions = [name for name, info in reg.all_kpis().items() if info.extension]
    assert len(extensions) == EXTENSION_KPI_COUNT


def test_registry_declares_its_own_counts() -> None:
    """The meta block must agree with the rows, so a stale header cannot hide a mistake."""
    meta = kpi_registry()["meta"]
    assert meta["study_kpi_count"] == STUDY_KPI_COUNT
    assert meta["extension_count"] == EXTENSION_KPI_COUNT


@pytest.mark.parametrize(("domains", "expected"), list(DOC_SECTION_TOTALS.items()))
def test_domain_totals_match_docs(domains: tuple[str, ...], expected: int) -> None:
    count = sum(
        1 for info in reg.all_kpis().values() if info.domain in domains and not info.extension
    )
    assert count == expected, f"docs/05 section 2 heading {domains} should hold {expected} KPIs"


def test_every_owner_is_a_registered_model() -> None:
    models = set(reg.all_models())
    bad = {name: info.owner for name, info in reg.all_kpis().items() if info.owner not in models}
    assert not bad, f"KPIs owned by unknown models: {bad}"


def test_every_model_owns_at_least_one_kpi() -> None:
    owners = {info.owner for info in reg.all_kpis().values()}
    missing = sorted(set(reg.all_models()) - owners)
    assert not missing, f"models with no KPI of their own: {missing}"


def test_every_unit_is_allowed() -> None:
    allowed = reg.allowed_units()
    bad = {name: info.unit for name, info in reg.all_kpis().items() if info.unit not in allowed}
    assert not bad, f"KPIs with units outside the docs/02 section 5 list: {bad}"


def test_every_band_reference_resolves() -> None:
    tables = set(reg.all_bands())
    bad = {
        name: info.band
        for name, info in reg.all_kpis().items()
        if info.band is not None and info.band not in tables
    }
    assert not bad, f"KPIs referencing undefined band tables: {bad}"


def test_band_input_values_are_supported() -> None:
    bad = {
        name: info.band_input
        for name, info in reg.all_kpis().items()
        if info.band_input not in ("value", "complement_100")
    }
    assert not bad, f"unsupported band_input values: {bad}"


def test_complement_100_only_on_percentage_kpis() -> None:
    """The shortfall transform only makes sense on a 0-100 percentage (decision D1)."""
    bad = {
        name: info.unit
        for name, info in reg.all_kpis().items()
        if info.band_input == "complement_100" and info.unit != "%"
    }
    assert not bad, f"complement_100 used on non-percentage KPIs: {bad}"


def test_directions_are_valid() -> None:
    bad = {
        name: info.direction
        for name, info in reg.all_kpis().items()
        if info.direction not in ("higher_is_worse", "higher_is_better")
    }
    assert not bad, f"invalid direction values: {bad}"


def test_kpi_names_are_snake_case() -> None:
    from twin_common.contracts import KPI_NAME_RE

    bad = [name for name in reg.all_kpis() if not KPI_NAME_RE.fullmatch(name)]
    assert not bad, f"KPI names must be snake_case (CLAUDE.md): {bad}"


def test_every_unit_has_bounds() -> None:
    raw = kpi_registry()
    missing = [u for u in raw["allowed_units"] if u not in raw["unit_bounds"]]
    assert not missing, f"units without sanity bounds: {missing}"


def test_bounds_are_ordered() -> None:
    bad = {
        name: info.bounds
        for name, info in reg.all_kpis().items()
        if info.bounds[0] is not None
        and info.bounds[1] is not None
        and info.bounds[0] > info.bounds[1]
    }
    assert not bad, f"KPIs with min > max: {bad}"


def test_percentage_kpis_that_allow_overflow_are_utilizations() -> None:
    """A `%` KPI may exceed 100 only when it is a utilization-style ratio."""
    overflowing = {
        name for name, info in reg.all_kpis().items() if info.unit == "%" and info.bounds[1] is None
    }
    # utilization_pct has a critical band of [100, 9999], so these must be the overflowers.
    banded_utilization = {
        name for name, info in reg.all_kpis().items() if info.band == "utilization_pct"
    }
    unexpected = overflowing - banded_utilization - {"resource_coverage", "safe_shelter_capacity"}
    assert not unexpected, (
        f"percentage KPIs allowing overflow without a utilization band: {unexpected}"
    )


def test_kpis_owned_by_matches_registry() -> None:
    for model_id in reg.all_models():
        owned = reg.kpis_owned_by(model_id)
        assert owned, f"{model_id} owns no KPI"
        assert all(reg.kpi_info(k).owner == model_id for k in owned)


def test_unknown_kpi_raises_with_a_helpful_message() -> None:
    from twin_common.errors import RegistryError

    with pytest.raises(RegistryError, match=r"kpi_registry\.yaml"):
        reg.kpi_info("definitely_not_a_kpi")
