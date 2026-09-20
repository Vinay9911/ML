"""Config loading, merging, environment overrides and the shared yaml files."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import yaml

from twin_common import config as cfg
from twin_common import paths
from twin_common.errors import ConfigError


# ------------------------------------------------------------------- primitives
def test_deep_merge_merges_mappings_and_replaces_scalars() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "lst": [1, 2, 3]}
    override = {"a": 2, "nested": {"y": 20, "z": 30}, "lst": [9]}
    merged = cfg.deep_merge(base, override)
    assert merged == {"a": 2, "nested": {"x": 1, "y": 20, "z": 30}, "lst": [9]}


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"nested": {"x": 1}}
    cfg.deep_merge(base, {"nested": {"y": 2}})
    assert base == {"nested": {"x": 1}}


def test_load_yaml_rejects_a_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        cfg.load_yaml(path)


def test_load_yaml_rejects_broken_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("key: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse yaml"):
        cfg.load_yaml(path)


def test_load_yaml_of_an_empty_file_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert cfg.load_yaml(path) == {}


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        cfg.load_yaml("does/not/exist.yaml")


# ------------------------------------------------------------- env overrides
def test_env_prefix_override_parses_yaml_scalars() -> None:
    env = {
        "TWIN_CFG_params__horizon_steps": "12",
        "TWIN_CFG_params__alpha": "0.15",
        "TWIN_CFG_params__enabled": "true",
        "TWIN_CFG_params__weights": "[0.5, 0.5]",
        "IGNORED": "x",
    }
    out = cfg.env_overrides(env)
    assert out == {
        "params": {
            "horizon_steps": 12,
            "alpha": 0.15,
            "enabled": True,
            "weights": [0.5, 0.5],
        }
    }


def test_env_shortcuts() -> None:
    out = cfg.env_overrides({"TWIN_SEED": "7", "TWIN_DATA_SOURCE": "replay"})
    assert out == {"seed": 7, "data_source": "replay"}


def test_env_override_keys_are_case_insensitive() -> None:
    """Windows upper-cases environment variable names, so both spellings must work.

    Without this, `TWIN_CFG_params__alpha` silently becomes `PARAMS.ALPHA` on Windows and
    the override is dropped.
    """
    lower = cfg.env_overrides({"TWIN_CFG_params__alpha": "0.5"})
    upper = cfg.env_overrides({"TWIN_CFG_PARAMS__ALPHA": "0.5"})
    assert lower == upper == {"params": {"alpha": 0.5}}
    assert cfg.env_overrides({"twin_seed": "7"}) == {"seed": 7}


def test_env_override_precedence_over_file(tmp_path: Path, monkeypatch) -> None:
    path = _write_model_config(tmp_path, {"seed": 42, "params": {"alpha": 0.1}})
    monkeypatch.setenv("TWIN_SEED", "99")
    monkeypatch.setenv("TWIN_CFG_params__alpha", "0.5")
    config = cfg.load_model_config(path)
    assert config.require_int("seed") == 99
    assert config.require_float("params.alpha") == pytest.approx(0.5)


# ------------------------------------------------------------------- Config object
def _write_model_config(folder: Path, extra: dict) -> Path:
    payload = {
        "model_id": "M21",
        "model_name": "Weather Impact",
        "model_version": "0.1.0",
        "port": 8021,
        "engine": "formula",
        "data_source": "synthetic",
        "seed": 42,
        "demo_now": "2027-08-02T06:00:00+05:30",
        "horizons_min": [15, 60, 180],
        "default_horizon_min": 180,
        "upstream": [],
        "inputs": ["weather_hourly"],
        "kpis": ["heat_index"],
        "scenarios_supported": ["S01", "S03", "S04"],
        "upstream_timeout_s": 5,
    }
    payload.update(extra)
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_dotted_access_and_require(tmp_path: Path) -> None:
    path = _write_model_config(tmp_path, {"params": {"nested": {"k": 3}}})
    config = cfg.load_model_config(path, apply_env=False)
    assert config["params.nested.k"] == 3
    assert config.get("params.nested.missing", "fallback") == "fallback"
    assert config.require("model_id") == "M21"
    assert config["params"] == {"nested": {"k": 3}}
    assert config.require_list("horizons_min") == [15, 60, 180]


def test_require_names_the_source_file(tmp_path: Path) -> None:
    path = _write_model_config(tmp_path, {})
    config = cfg.load_model_config(path, apply_env=False)
    with pytest.raises(ConfigError, match=r"config\.yaml"):
        config.require("params.nope")


def test_require_list_rejects_a_scalar(tmp_path: Path) -> None:
    path = _write_model_config(tmp_path, {"params": {"weights": 0.5}})
    config = cfg.load_model_config(path, apply_env=False)
    with pytest.raises(ConfigError, match="must be a list"):
        config.require_list("params.weights")


def test_mapping_protocol(tmp_path: Path) -> None:
    path = _write_model_config(tmp_path, {})
    config = cfg.load_model_config(path, apply_env=False)
    assert "model_id" in config
    assert len(config) > 0
    assert "model_id" in list(config)


def test_all_standard_keys_are_required(tmp_path: Path) -> None:
    """docs/02 section 8 keys must all be present."""
    payload = yaml.safe_load(_write_model_config(tmp_path, {}).read_text(encoding="utf-8"))
    for key in cfg.REQUIRED_MODEL_KEYS:
        reduced = {k: v for k, v in payload.items() if k != key}
        path = tmp_path / "reduced.yaml"
        path.write_text(yaml.safe_dump(reduced), encoding="utf-8")
        with pytest.raises(ConfigError, match="missing required config keys"):
            cfg.load_model_config(path, apply_env=False)


def test_empty_upstream_list_is_accepted(tmp_path: Path) -> None:
    """M21 and M23 legitimately have upstream: [] - truthiness must not reject it."""
    path = _write_model_config(tmp_path, {"upstream": []})
    config = cfg.load_model_config(path, apply_env=False)
    assert config.require("upstream") == []


def test_params_defaults_to_empty(tmp_path: Path) -> None:
    path = _write_model_config(tmp_path, {})
    config = cfg.load_model_config(path, apply_env=False)
    assert config.get("params") == {}


# ------------------------------------------------------------- shared config files
def test_all_eight_shared_files_load() -> None:
    loaders = [
        cfg.assumptions,
        cfg.world,
        cfg.scenarios_config,
        cfg.risk_bands,
        cfg.kpi_registry,
        cfg.model_registry,
        cfg.reason_codes_config,
        cfg.resources_config,
    ]
    for loader in loaders:
        data = loader()
        assert isinstance(data, dict) and data, f"{loader.__name__} returned nothing"


@pytest.mark.parametrize(
    "dotted",
    [
        "crowd.design_density_p_m2",
        "crowd.density_bands_p_m2",
        "crowd.sustained_exposure",
        "crowd.gate_split",
        "crowd.hourly_profile_snan",
        "medical.presentations_per_1000_attendees_per_day",
        "medical.hospital_beds",
        "water.litres_per_person_per_hour_present",
        "sanitation.users_per_toilet_per_hour",
        "waste.kg_per_person_per_hour_present",
        "food.vehicle_payload_meals",
        "transport.bpr_alpha",
        "transport.parking_capacity",
        "security.base_incidents_per_100k_per_hr",
        "power.generator_unit_kw",
        "network.tower_capacity_mbps",
        "environment.grid_ef_kgco2_per_kwh",
        "resources.available",
        "resources.risk_multiplier",
    ],
)
def test_assumptions_keys_present(dotted: str) -> None:
    """Every docs/04 section 6 group a model reads must exist."""
    node = cfg.assumptions()
    for part in dotted.split("."):
        assert part in node, f"assumptions.yaml is missing {dotted}"
        node = node[part]


def test_hourly_profiles_have_24_entries() -> None:
    crowd = cfg.assumptions()["crowd"]
    assert len(crowd["hourly_profile_normal"]) == 24
    assert len(crowd["hourly_profile_snan"]) == 24


def test_gate_split_sums_to_one() -> None:
    total = sum(cfg.assumptions()["crowd"]["gate_split"].values())
    assert total == pytest.approx(1.0)


def test_gate_reassignment_shares_sum_to_one() -> None:
    """S05 must redistribute the whole closed-gate share (docs/04 section 5.4)."""
    for gate, shares in cfg.assumptions()["crowd"]["gate_reassignment"].items():
        assert sum(shares.values()) == pytest.approx(1.0), f"{gate} reassignment must sum to 1"


def test_density_bands_are_ordered() -> None:
    bands = cfg.assumptions()["crowd"]["density_bands_p_m2"]
    assert bands["amber"] < bands["red"] < bands["critical"]


def test_resource_availability_covers_all_fifteen_types() -> None:
    from twin_common.contracts.enums import RESOURCE_TYPES

    available = cfg.assumptions()["resources"]["available"]
    assert set(available) == set(RESOURCE_TYPES)


def test_assumptions_and_resources_agree_on_availability() -> None:
    """resources.yaml mirrors assumptions.resources.available; they must not drift."""
    from twin_common.contracts import registry as reg

    available = cfg.assumptions()["resources"]["available"]
    for resource_type, quantity in available.items():
        assert reg.resource_available(resource_type) == pytest.approx(float(quantity)), (
            f"{resource_type} availability differs between assumptions.yaml and resources.yaml"
        )


def test_risk_bands_are_the_four_documented_tables() -> None:
    assert set(cfg.risk_bands()["bands"]) == {
        "score_0_100",
        "crowd_density_p_m2",
        "probability_pct",
        "utilization_pct",
    }


def test_band_intervals_are_contiguous_and_ordered() -> None:
    for name, table in cfg.risk_bands()["bands"].items():
        levels = ["green", "amber", "red", "critical"]
        bounds = [table[level] for level in levels]
        for level, (lo, hi) in zip(levels, bounds, strict=True):
            assert lo < hi, f"{name}.{level} has lower >= upper"
        for (_, prev_hi), (next_lo, _) in itertools.pairwise(bounds):
            assert prev_hi == next_lo, f"{name} bands are not contiguous"


def test_reason_codes_are_unique_across_groups() -> None:
    groups = cfg.reason_codes_config()["groups"]
    flat = [code for codes in groups.values() for code in codes]
    duplicates = {c for c in flat if flat.count(c) > 1}
    assert not duplicates, f"reason codes appear in more than one group: {duplicates}"


def test_world_time_grid_is_15_minutes() -> None:
    time_cfg = cfg.world()["time"]
    assert time_cfg["grid_min"] == 15
    assert time_cfg["timezone"] == "Asia/Kolkata"
    assert time_cfg["utc_offset"] == "+05:30"


def test_world_has_eight_zones_with_required_attributes() -> None:
    zones = cfg.world()["zones"]
    assert len(zones) == 8
    for zone_id, row in zones.items():
        for key in ("name", "type", "area_m2", "usable_share", "low_lying"):
            assert key in row, f"{zone_id} is missing {key}"
        assert 0 < row["usable_share"] <= 1


def test_world_adjacency_only_references_known_zones() -> None:
    world = cfg.world()
    zones = set(world["zones"])
    for left, right in world["adjacency"]:
        assert left in zones and right in zones


#: The movement phases in world.yaml routing. The other keys there (link_width_m,
#: closure_map) are capacity data, not splits.
ROUTING_PHASES = ("inbound", "outbound")


@pytest.mark.parametrize("phase", ROUTING_PHASES)
def test_routing_splits_sum_to_one(phase: str) -> None:
    """The compartment model needs each split to be a probability distribution."""
    table = cfg.world()["routing"][phase]
    for zone, spec in table.items():
        total = sum(spec["to"].values())
        assert total == pytest.approx(1.0), f"routing.{phase}.{zone} sums to {total}, not 1"


def test_every_routed_link_has_a_width() -> None:
    """A routed zone-to-zone link with no width would make its capacity un-configurable."""
    routing = cfg.world()["routing"]
    widths = set(routing["link_width_m"])
    exits = set(cfg.world()["exits"])
    missing: list[str] = []
    for phase in ROUTING_PHASES:
        for source, spec in routing[phase].items():
            for target in spec["to"]:
                if target in exits:
                    continue  # exits carry their own width
                key = f"{source}>{target}"
                if key not in widths:
                    missing.append(key)
    assert not missing, f"routing links with no link_width_m entry: {missing}"


def test_closure_map_targets_are_real_links() -> None:
    routing = cfg.world()["routing"]
    widths = set(routing["link_width_m"])
    exits = set(cfg.world()["exits"])
    for closed, links in (routing.get("closure_map") or {}).items():
        for link in links:
            _source, _, target = link.partition(">")
            assert link in widths or target in exits, (
                f"closure_map[{closed}] names {link!r}, which is not a routed link"
            )


# ------------------------------------------------------------------------- paths
def test_config_dir_resolves_next_to_the_package() -> None:
    assert paths.config_dir().is_dir()
    assert (paths.config_dir() / "kpi_registry.yaml").is_file()
    assert paths.config_dir().parent == paths.common_root()


def test_env_override_of_config_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TWIN_COMMON_CONFIG_DIR", str(tmp_path))
    assert paths.config_dir() == tmp_path.resolve()


def test_require_config_file_raises_for_a_missing_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TWIN_COMMON_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ConfigError, match="shared config file not found"):
        paths.require_config_file("nope.yaml")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
    ],
)
def test_is_offline_reads_the_env_flag(monkeypatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("TWIN_OFFLINE", value)
    assert paths.is_offline() is expected


def test_is_offline_defaults_to_false(monkeypatch) -> None:
    monkeypatch.delenv("TWIN_OFFLINE", raising=False)
    assert paths.is_offline() is False


def test_world_dir_is_scenario_scoped() -> None:
    assert paths.world_dir("S02").name == "S02"
    assert paths.world_dir("S02").parent.name == "world"


def test_clear_cache_picks_up_a_changed_file(tmp_path: Path, monkeypatch) -> None:
    """A test that writes fixture config must be able to invalidate the cache."""
    from twin_common.contracts import registry as reg

    real = paths.config_dir()
    for name in (
        "kpi_registry.yaml",
        "risk_bands.yaml",
        "model_registry.yaml",
        "reason_codes.yaml",
        "resources.yaml",
    ):
        (tmp_path / name).write_text((real / name).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("TWIN_COMMON_CONFIG_DIR", str(tmp_path))
    reg.clear_cache()
    try:
        assert len(reg.all_kpis()) == 109
        doc = yaml.safe_load((tmp_path / "kpi_registry.yaml").read_text(encoding="utf-8"))
        doc["kpis"].pop("pm25")
        (tmp_path / "kpi_registry.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
        reg.clear_cache()
        assert len(reg.all_kpis()) == 108
    finally:
        monkeypatch.delenv("TWIN_COMMON_CONFIG_DIR", raising=False)
        reg.clear_cache()
