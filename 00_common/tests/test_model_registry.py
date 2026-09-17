"""The model registry must match docs/02 section 13 and stay acyclic."""

from __future__ import annotations

import pytest

from twin_common.contracts import registry as reg
from twin_common.errors import RegistryError

MODEL_COUNT = 25
FIRST_PORT = 8001
LAST_PORT = 8025

#: docs/01 section 4 / docs/02 section 13.
EXPECTED_ENGINES = {
    "forecast",
    "vision",
    "rules",
    "network",
    "formula",
    "location",
    "pedsim",
    "optimize",
    "mlclf",
}


def test_twenty_five_models() -> None:
    assert len(reg.all_models()) == MODEL_COUNT


def test_model_ids_are_m01_to_m25() -> None:
    expected = {f"M{i:02d}" for i in range(1, MODEL_COUNT + 1)}
    assert set(reg.all_models()) == expected


def test_ports_are_8001_to_8025_and_match_the_id() -> None:
    """docs/02 section 3: Mxx serves on port 80xx."""
    for model_id, info in reg.all_models().items():
        expected = FIRST_PORT + int(model_id[1:]) - 1
        assert info.port == expected, f"{model_id} should use port {expected}, got {info.port}"


def test_ports_are_unique_and_in_range() -> None:
    ports = [info.port for info in reg.all_models().values()]
    assert len(set(ports)) == len(ports), "ports must be unique"
    assert min(ports) == FIRST_PORT
    assert max(ports) == LAST_PORT


def test_folders_are_unique_and_named_after_the_id() -> None:
    folders = [info.folder for info in reg.all_models().values()]
    assert len(set(folders)) == len(folders)
    for model_id, info in reg.all_models().items():
        assert info.folder.startswith(f"{model_id}_"), (
            f"{model_id} folder {info.folder!r} must start with the model ID"
        )


def test_engines_are_known() -> None:
    bad = {
        mid: info.engine
        for mid, info in reg.all_models().items()
        if info.engine not in EXPECTED_ENGINES
    }
    assert not bad, f"unknown engines: {bad}"


def test_phases_are_in_range() -> None:
    for model_id, info in reg.all_models().items():
        assert 3 <= info.phase <= 9, f"{model_id} has phase {info.phase}, expected 3-9"


def test_upstream_graph_is_acyclic() -> None:
    """The acyclicity assertion named in docs/07 Phase 1 acceptance."""
    order = reg.topological_order()
    assert len(order) == MODEL_COUNT
    position = {mid: i for i, mid in enumerate(order)}
    for model_id, info in reg.all_models().items():
        for dep in info.upstream:
            assert position[dep] < position[model_id], (
                f"{model_id} appears before its upstream {dep} in the topological order"
            )


def test_no_model_depends_on_itself() -> None:
    self_deps = [mid for mid, info in reg.all_models().items() if mid in info.upstream]
    assert not self_deps, f"self-dependencies: {self_deps}"


def test_all_upstream_ids_are_registered() -> None:
    known = set(reg.all_models())
    for model_id, info in reg.all_models().items():
        unknown = [dep for dep in info.upstream if dep not in known]
        assert not unknown, f"{model_id} declares unknown upstreams {unknown}"


def test_m24_does_not_depend_on_m25() -> None:
    """docs/03 M24 card: M24 must not consume M25, which would create a cycle."""
    assert "M25" not in reg.upstream_of("M24")
    assert "M24" in reg.upstream_of("M25")


def test_m21_is_the_root() -> None:
    """M21 is built first (phase 3) and has no upstream."""
    assert reg.upstream_of("M21") == ()
    assert reg.topological_order()[0] == "M21"


def test_downstream_of_m01_is_wide() -> None:
    """M01 feeds most of the forecasting and resource chain."""
    downstream = set(reg.downstream_of("M01"))
    for expected in ("M03", "M04", "M05", "M07", "M11", "M15", "M24"):
        assert expected in downstream


def test_topological_order_of_a_subset_includes_dependencies() -> None:
    order = reg.topological_order(["M25"])
    assert order[-1] == "M25"
    assert "M24" in order and "M21" in order


def test_unknown_model_raises() -> None:
    with pytest.raises(RegistryError, match="unknown model_id"):
        reg.model_info("M99")


def test_resource_types_are_the_fifteen_of_the_study() -> None:
    from twin_common.contracts.enums import RESOURCE_TYPES

    assert len(RESOURCE_TYPES) == 15
    assert set(reg.all_resource_types()) == set(RESOURCE_TYPES)


def test_every_resource_type_has_availability() -> None:
    for resource_type in reg.all_resource_types():
        assert reg.resource_available(resource_type) > 0
