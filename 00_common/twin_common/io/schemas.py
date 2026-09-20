"""Column schemas for the data tables D01-D25 (docs/04 section 4).

The point of this module is that a table has the same columns whether it came from the
synthetic generator, a replay file or a future live connector, so swapping data sources is
a config change rather than a code change (docs/04 section 10).

Checks are deliberately simple: a dict of column -> dtype family, no pandera dependency.
`required` columns must exist and be coercible; `optional` may be absent or all-null.

Every table carries `is_synthetic` (bool) and `source` (str). Nothing may drop them
(CLAUDE.md hard rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

#: Coarse dtype families. Kept coarse on purpose: a real feed may hand us int32 vs int64.
Dtype = Literal["timestamp", "str", "int", "float", "bool", "json"]

#: Columns every table must have (docs/04 section 1 principle 5).
PROVENANCE_COLUMNS: dict[str, Dtype] = {
    "is_synthetic": "bool",
    "source": "str",
}


@dataclass(frozen=True, slots=True)
class TableSchema:
    """Schema for one table."""

    name: str
    dataset_id: str
    grain: str
    required: dict[str, Dtype]
    optional: dict[str, Dtype] = field(default_factory=dict)
    time_column: str | None = "timestamp"
    used_by: tuple[str, ...] = ()

    @property
    def columns(self) -> dict[str, Dtype]:
        """Required + optional + provenance, in a stable order."""
        return {**self.required, **self.optional, **PROVENANCE_COLUMNS}

    @property
    def required_with_provenance(self) -> dict[str, Dtype]:
        return {**self.required, **PROVENANCE_COLUMNS}


def _t(
    name: str,
    dataset_id: str,
    grain: str,
    required: dict[str, Dtype],
    optional: dict[str, Dtype] | None = None,
    time_column: str | None = "timestamp",
    used_by: tuple[str, ...] = (),
) -> TableSchema:
    return TableSchema(
        name=name,
        dataset_id=dataset_id,
        grain=grain,
        required=required,
        optional=optional or {},
        time_column=time_column,
        used_by=used_by,
    )


#: The full table registry, keyed by table name. Order follows docs/04 section 4.
TABLE_SCHEMAS: dict[str, TableSchema] = {
    "footfall_15min": _t(
        "footfall_15min",
        "D01",
        "zone/gate x 15 min",
        {
            "timestamp": "timestamp",
            "zone_id": "str",
            "entries": "int",
            "exits": "int",
            "population": "int",
        },
        {"gate_id": "str", "density_p_m2": "float"},
        used_by=("M01", "M03", "M24"),
    ),
    "vision_timeseries": _t(
        "vision_timeseries",
        "D02",
        "camera x 15 s / 15 min",
        {
            "timestamp": "timestamp",
            "camera_id": "str",
            "zone_id": "str",
            "count": "float",
            "density": "float",
            "in_count": "float",
            "out_count": "float",
            "speed_m_s": "float",
            "direction_deg": "float",
        },
        {"window_s": "int", "roi_area_m2": "float"},
        used_by=("M02", "M10"),
    ),
    "zones": _t(
        "zones",
        "D03",
        "zone",
        {
            "zone_id": "str",
            "name": "str",
            "type": "str",
            "area_m2": "float",
            "usable_area_m2": "float",
            "safe_capacity": "float",
            "low_lying": "bool",
            "drainage_capacity_mm_hr": "float",
        },
        {"adjacency": "json", "attributes": "json", "geometry": "str"},
        time_column=None,
        used_by=("all",),
    ),
    "traffic_probe_15min": _t(
        "traffic_probe_15min",
        "D04",
        "road segment x 15 min",
        {
            "timestamp": "timestamp",
            "link_id": "str",
            "speed_kmh": "float",
            "travel_time_s": "float",
            "volume_proxy": "float",
        },
        used_by=("M04",),
    ),
    "signal_cycles": _t(
        "signal_cycles",
        "D05",
        "intersection x cycle",
        {
            "timestamp": "timestamp",
            "junction_id": "str",
            "cycle_s": "float",
            "green_s": "float",
            "queue_veh": "float",
        },
        used_by=("M04",),
    ),
    "parking_15min": _t(
        "parking_15min",
        "D06",
        "site x 15 min",
        {
            "timestamp": "timestamp",
            "parking_id": "str",
            "entries": "int",
            "exits": "int",
            "occupied": "int",
            "capacity": "int",
        },
        used_by=("M05",),
    ),
    "transit_15min": _t(
        "transit_15min",
        "D07",
        "route x 15 min",
        {
            "timestamp": "timestamp",
            "route_id": "str",
            "boardings": "int",
            "alightings": "int",
            "vehicles_active": "int",
            "avg_wait_min": "float",
        },
        used_by=("M06",),
    ),
    "medical_incidents": _t(
        "medical_incidents",
        "D08",
        "event",
        {
            "incident_id": "str",
            "timestamp": "timestamp",
            "zone_id": "str",
            "lat": "float",
            "lon": "float",
            "category": "str",
            "severity": "int",
            "outcome": "str",
        },
        {"transferred_to": "str"},
        used_by=("M07",),
    ),
    "ambulance_dispatch": _t(
        "ambulance_dispatch",
        "D09",
        "incident x dispatch",
        {
            "incident_id": "str",
            "vehicle_id": "str",
            "dispatch_ts": "timestamp",
            "arrival_ts": "timestamp",
            "from_site": "str",
        },
        {"hospital_ts": "timestamp"},
        time_column="dispatch_ts",
        used_by=("M08",),
    ),
    "security_incidents": _t(
        "security_incidents",
        "D10",
        "event",
        {
            "incident_id": "str",
            "timestamp": "timestamp",
            "zone_id": "str",
            "type": "str",
            "severity": "int",
            "response_min": "float",
        },
        used_by=("M09",),
    ),
    "access_control": _t(
        "access_control",
        "D11",
        "gate x event",
        {"timestamp": "timestamp", "gate_id": "str", "credential_class": "str", "result": "str"},
        used_by=("M09",),
    ),
    "weather_hourly": _t(
        "weather_hourly",
        "D12",
        "venue x hour",
        {
            "timestamp": "timestamp",
            "temperature_c": "float",
            "humidity_pct": "float",
            "rain_mm": "float",
            "wind_kmh": "float",
            "heat_index_c": "float",
        },
        {"visibility_m": "float"},
        used_by=("M01", "M07", "M15", "M19", "M21"),
    ),
    "air_quality_hourly": _t(
        "air_quality_hourly",
        "D13",
        "venue x hour",
        {
            "timestamp": "timestamp",
            "pm2_5": "float",
            "pm10": "float",
            "no2": "float",
            "co": "float",
            "noise_db": "float",
        },
        used_by=("M22",),
    ),
    "water_15min": _t(
        "water_15min",
        "D14",
        "zone/asset x 15 min",
        {
            "timestamp": "timestamp",
            "zone_id": "str",
            "consumption_l": "float",
            "supply_l": "float",
            "storage_l": "float",
        },
        {"asset_id": "str"},
        used_by=("M15",),
    ),
    "toilet_usage": _t(
        "toilet_usage",
        "D15",
        "cluster x 15 min",
        {
            "timestamp": "timestamp",
            "cluster_id": "str",
            "uses": "int",
            "queue_persons": "int",
            "clean_status": "str",
            "units_active": "int",
        },
        used_by=("M16",),
    ),
    "waste": _t(
        "waste",
        "D16",
        "bin group x 15 min",
        {
            "timestamp": "timestamp",
            "bin_group_id": "str",
            "zone_id": "str",
            "kg_added": "float",
            "fill_pct": "float",
            "collected_kg": "float",
        },
        used_by=("M18",),
    ),
    "food_inventory_hourly": _t(
        "food_inventory_hourly",
        "D17",
        "outlet x hour",
        {
            "timestamp": "timestamp",
            "outlet_id": "str",
            "zone_id": "str",
            "meals_sold": "float",
            "stock_meals": "float",
            "deliveries": "float",
            "lead_time_h": "float",
        },
        used_by=("M17",),
    ),
    "power_15min": _t(
        "power_15min",
        "D18",
        "asset/zone x 15 min",
        {
            "timestamp": "timestamp",
            "asset_id": "str",
            "zone_id": "str",
            "kw": "float",
            "kwh": "float",
            "voltage": "float",
            "grid_available": "bool",
            "generator_on": "bool",
            "fuel_l": "float",
        },
        used_by=("M19",),
    ),
    "network_5min": _t(
        "network_5min",
        "D19",
        "tower x 5 min",
        {
            "timestamp": "timestamp",
            "tower_id": "str",
            "bandwidth_used_mbps": "float",
            "capacity_mbps": "float",
            "latency_ms": "float",
            "packet_loss_pct": "float",
            "up": "bool",
        },
        used_by=("M20",),
    ),
    "asset_maintenance": _t(
        "asset_maintenance",
        "D20",
        "asset x hour",
        {
            "timestamp": "timestamp",
            "asset_id": "str",
            "asset_type": "str",
            "ambient_temp_k": "float",
            "process_temp_k": "float",
            "speed_rpm": "float",
            "torque_nm": "float",
            "wear_min": "float",
            "failure": "int",
        },
        {"failure_mode": "str"},
        used_by=("M23",),
    ),
    "event_calendar": _t(
        "event_calendar",
        "D21",
        "day/session",
        {
            "date": "str",
            "session": "str",
            "start": "timestamp",
            "end": "timestamp",
            "expected_attendance_multiplier": "float",
            "is_snan_day": "bool",
            "vip_flag": "bool",
        },
        {"zone_id": "str"},
        time_column="start",
        used_by=("M01", "M04", "M09", "M13"),
    ),
    "resource_roster": _t(
        "resource_roster",
        "D22",
        "resource x shift x zone",
        {
            "shift_start": "timestamp",
            "shift_end": "timestamp",
            "resource_type": "str",
            "zone_id": "str",
            "quantity_available": "float",
            "status": "str",
        },
        time_column="shift_start",
        used_by=("M24",),
    ),
    "sop_logs": _t(
        "sop_logs",
        "D23",
        "incident x task",
        {
            "incident_id": "str",
            "sop_id": "str",
            "task": "str",
            "t_notify": "timestamp",
            "t_complete": "timestamp",
            "outcome": "str",
        },
        time_column="t_notify",
        used_by=(),
    ),
    "key_links": _t(
        "key_links",
        "D24",
        "graph key links",
        {
            "link_id": "str",
            "name": "str",
            "lanes": "int",
            "length_m": "float",
            "maxspeed_kmh": "float",
            "capacity_veh_hr": "float",
        },
        {"osm_u": "str", "osm_v": "str", "osm_key": "int", "geometry": "str"},
        time_column=None,
        used_by=("M04", "M08", "M12", "M13", "M14"),
    ),
    "lessons_learned": _t(
        "lessons_learned",
        "D25",
        "event x KPI",
        {"event_name": "str", "kpi": "str", "outcome": "str", "notes": "str"},
        time_column=None,
        used_by=(),
    ),
    # --- written by the crowd model; not numbered in docs/04 section 4, but M01 needs the
    # --- per-gate series ("one series per zone and per gate", docs/03 M01) and M11 needs
    # --- the per-exit discharge.
    "gate_entries": _t(
        "gate_entries",
        "-",
        "gate x 15 min",
        {
            "timestamp": "timestamp",
            "gate_id": "str",
            "zone_id": "str",
            "entries": "float",
            "demand": "float",
            "queue_persons": "float",
            "capacity_p_min": "float",
            "closed": "bool",
        },
        used_by=("M01", "M02", "M09"),
    ),
    "exit_flows": _t(
        "exit_flows",
        "-",
        "exit x 15 min",
        {
            "timestamp": "timestamp",
            "exit_id": "str",
            "zone_id": "str",
            "outflow": "float",
            "capacity_p_min": "float",
            "blocked": "bool",
        },
        used_by=("M11",),
    ),
    # --- registries written by the layout step, not numbered in docs/04 section 4 ---
    "assets": _t(
        "assets",
        "-",
        "entity",
        {
            "entity_id": "str",
            "entity_type": "str",
            "zone_id": "str",
            "lat": "float",
            "lon": "float",
            "capacity": "float",
            "status": "str",
        },
        {"attributes": "json"},
        time_column=None,
        used_by=("all",),
    ),
    "camera_registry": _t(
        "camera_registry",
        "-",
        "camera",
        {
            "camera_id": "str",
            "zone_id": "str",
            "mode": "str",
            "bitrate_mbps": "float",
            "tower_id": "str",
            "status": "str",
        },
        {
            "fov_wkt": "str",
            "roi_ground_area_m2": "float",
            "metres_per_pixel": "float",
            "source_uri": "str",
        },
        time_column=None,
        used_by=("M02", "M10", "M20", "M23"),
    ),
}


def table_schema(name: str) -> TableSchema:
    """Look up a table schema by name."""
    try:
        return TABLE_SCHEMAS[name]
    except KeyError as exc:
        from ..errors import TableError

        raise TableError(
            f"unknown table {name!r}: registered tables are {sorted(TABLE_SCHEMAS)}"
        ) from exc


def table_exists(name: str) -> bool:
    return name in TABLE_SCHEMAS


def tables_for_model(model_id: str) -> list[str]:
    """Tables whose `used_by` names this model (or `all`)."""
    return [
        name
        for name, schema in TABLE_SCHEMAS.items()
        if model_id in schema.used_by or "all" in schema.used_by
    ]


def dataset_id_of(name: str) -> str:
    """The D-number from docs/04 section 4, for model-card traceability."""
    return table_schema(name).dataset_id
