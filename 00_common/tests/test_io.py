"""Table IO: schema checks, provenance enforcement and the three adapters."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from twin_common.contracts import IST
from twin_common.errors import TableError
from twin_common.io import (
    TABLE_SCHEMAS,
    LiveAdapter,
    check_schema,
    dataset_id_of,
    describe_table,
    load_table,
    save_table,
    table_exists,
    table_is_available,
    table_schema,
    tables_for_model,
)

START = datetime(2027, 8, 2, 0, 0, tzinfo=IST)


def _footfall_frame(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [START + timedelta(minutes=15 * i) for i in range(rows)],
            "zone_id": ["Z01"] * rows,
            "gate_id": [None] * rows,
            "entries": list(range(rows)),
            "exits": [0] * rows,
            "population": [100 + i for i in range(rows)],
            "is_synthetic": [True] * rows,
            "source": ["generator"] * rows,
        }
    )


# --------------------------------------------------------------------- registry
def test_all_25_datasets_are_registered() -> None:
    """D01-D25 from docs/04 section 4, plus the two unnumbered registries."""
    dataset_ids = {s.dataset_id for s in TABLE_SCHEMAS.values() if s.dataset_id != "-"}
    expected = {f"D{i:02d}" for i in range(1, 26)}
    assert dataset_ids == expected, f"missing datasets: {sorted(expected - dataset_ids)}"


def test_every_schema_requires_provenance() -> None:
    for name, schema in TABLE_SCHEMAS.items():
        assert "is_synthetic" in schema.required_with_provenance, name
        assert "source" in schema.required_with_provenance, name


def test_time_column_is_declared_and_present() -> None:
    for name, schema in TABLE_SCHEMAS.items():
        if schema.time_column is None:
            continue
        assert schema.time_column in schema.required, (
            f"{name}: time column {schema.time_column} is not a required column"
        )


def test_dataset_id_lookup() -> None:
    assert dataset_id_of("footfall_15min") == "D01"
    assert dataset_id_of("weather_hourly") == "D12"
    assert dataset_id_of("asset_maintenance") == "D20"


def test_tables_for_model() -> None:
    m01 = tables_for_model("M01")
    assert "footfall_15min" in m01
    assert "weather_hourly" in m01
    assert "event_calendar" in m01
    # `zones` is marked used_by all
    assert "zones" in m01


def test_unknown_table_raises() -> None:
    assert not table_exists("not_a_table")
    with pytest.raises(TableError, match="unknown table"):
        table_schema("not_a_table")


def test_describe_table_shape() -> None:
    described = describe_table("weather_hourly")
    assert described["dataset_id"] == "D12"
    assert "temperature_c" in described["required"]
    assert "visibility_m" in described["optional"]


# ------------------------------------------------------------------ round trip
def test_save_then_load_round_trip(tmp_path: Path) -> None:
    save_table(_footfall_frame(), "footfall_15min", tmp_path)
    loaded = load_table("footfall_15min", base_dir=tmp_path)
    assert len(loaded) == 8
    assert loaded["population"].tolist() == [100 + i for i in range(8)]
    assert loaded["is_synthetic"].all()


def test_saved_timestamps_stay_tz_aware(tmp_path: Path) -> None:
    save_table(_footfall_frame(), "footfall_15min", tmp_path)
    loaded = load_table("footfall_15min", base_dir=tmp_path)
    assert loaded["timestamp"].dt.tz is not None
    assert loaded["timestamp"].iloc[0].utcoffset() == timedelta(hours=5, minutes=30)


def test_load_filters_by_time(tmp_path: Path) -> None:
    save_table(_footfall_frame(8), "footfall_15min", tmp_path)
    loaded = load_table(
        "footfall_15min",
        base_dir=tmp_path,
        start=START + timedelta(minutes=30),
        end=START + timedelta(minutes=75),
    )
    assert len(loaded) == 4


def test_scenario_scoped_directory_is_preferred(tmp_path: Path) -> None:
    """A model folder slice lives under data/synthetic/<scenario>/."""
    scoped = tmp_path / "S02"
    save_table(_footfall_frame(3), "footfall_15min", scoped)
    save_table(_footfall_frame(8), "footfall_15min", tmp_path)
    assert len(load_table("footfall_15min", base_dir=tmp_path, scenario_id="S02")) == 3
    assert len(load_table("footfall_15min", base_dir=tmp_path, scenario_id="S01")) == 8


def test_table_is_available(tmp_path: Path) -> None:
    assert not table_is_available("footfall_15min", base_dir=tmp_path)
    save_table(_footfall_frame(), "footfall_15min", tmp_path)
    assert table_is_available("footfall_15min", base_dir=tmp_path)


def test_missing_table_error_explains_how_to_generate(tmp_path: Path) -> None:
    with pytest.raises(TableError, match=r"synthetic\.generate"):
        load_table("footfall_15min", base_dir=tmp_path)


# ------------------------------------------------------------------ provenance
def test_missing_is_synthetic_is_an_error(tmp_path: Path) -> None:
    """CLAUDE.md hard rule: is_synthetic is never dropped or defaulted."""
    frame = _footfall_frame().drop(columns=["is_synthetic"])
    frame.to_parquet(tmp_path / "footfall_15min.parquet", index=False)
    with pytest.raises(TableError, match="is_synthetic"):
        load_table("footfall_15min", base_dir=tmp_path)


def test_missing_source_is_an_error(tmp_path: Path) -> None:
    frame = _footfall_frame().drop(columns=["source"])
    frame.to_parquet(tmp_path / "footfall_15min.parquet", index=False)
    with pytest.raises(TableError, match=r"'source'"):
        load_table("footfall_15min", base_dir=tmp_path)


def test_null_is_synthetic_is_an_error(tmp_path: Path) -> None:
    frame = _footfall_frame()
    frame["is_synthetic"] = [True, None, True, True, True, True, True, True]
    frame.to_parquet(tmp_path / "footfall_15min.parquet", index=False)
    with pytest.raises(TableError, match="null values in is_synthetic"):
        load_table("footfall_15min", base_dir=tmp_path)


def test_save_requires_provenance_when_absent(tmp_path: Path) -> None:
    frame = _footfall_frame().drop(columns=["is_synthetic", "source"])
    with pytest.raises(TableError, match="is_synthetic must be given"):
        save_table(frame, "footfall_15min", tmp_path)


def test_save_fills_provenance_when_asked(tmp_path: Path) -> None:
    frame = _footfall_frame().drop(columns=["is_synthetic", "source"])
    save_table(frame, "footfall_15min", tmp_path, is_synthetic=True, source="test")
    loaded = load_table("footfall_15min", base_dir=tmp_path)
    assert loaded["source"].unique().tolist() == ["test"]


def test_real_source_rows_keep_is_synthetic_false(tmp_path: Path) -> None:
    """Weather from Open-Meteo is real data and must stay flagged as such."""
    frame = pd.DataFrame(
        {
            "timestamp": [START],
            "temperature_c": [31.0],
            "humidity_pct": [58.0],
            "rain_mm": [0.0],
            "wind_kmh": [8.0],
            "heat_index_c": [34.2],
            "is_synthetic": [False],
            "source": ["open-meteo"],
        }
    )
    save_table(frame, "weather_hourly", tmp_path)
    loaded = load_table("weather_hourly", base_dir=tmp_path)
    assert not loaded["is_synthetic"].any()


# ------------------------------------------------------------------ schema checks
def test_missing_required_column_is_reported(tmp_path: Path) -> None:
    frame = _footfall_frame().drop(columns=["population"])
    with pytest.raises(TableError, match="population"):
        check_schema(frame, "footfall_15min")


def test_all_missing_columns_are_reported_at_once() -> None:
    frame = _footfall_frame().drop(columns=["population", "entries"])
    with pytest.raises(TableError) as excinfo:
        check_schema(frame, "footfall_15min")
    message = str(excinfo.value)
    assert "population" in message and "entries" in message


def test_optional_column_may_be_absent() -> None:
    frame = _footfall_frame().drop(columns=["gate_id"])
    checked = check_schema(frame, "footfall_15min")
    assert "gate_id" not in checked.columns


def test_dtype_coercion_of_numeric_strings() -> None:
    frame = _footfall_frame()
    frame["population"] = frame["population"].astype(str)
    checked = check_schema(frame, "footfall_15min")
    assert checked["population"].dtype == "int64"


def test_uncoercible_value_raises_with_column_context() -> None:
    frame = _footfall_frame()
    frame["population"] = "not a number"
    with pytest.raises(TableError, match=r"footfall_15min\.population"):
        check_schema(frame, "footfall_15min")


def test_naive_timestamps_are_localised_to_ist() -> None:
    frame = _footfall_frame()
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    checked = check_schema(frame, "footfall_15min")
    assert checked["timestamp"].dt.tz is not None
    assert checked["timestamp"].iloc[0].utcoffset() == timedelta(hours=5, minutes=30)


def test_utc_timestamps_are_converted_to_ist() -> None:
    frame = _footfall_frame()
    frame["timestamp"] = frame["timestamp"].dt.tz_convert("UTC")
    checked = check_schema(frame, "footfall_15min")
    assert checked["timestamp"].iloc[0].utcoffset() == timedelta(hours=5, minutes=30)


def test_bool_coercion_from_strings() -> None:
    frame = pd.DataFrame(
        {
            "zone_id": ["Z01"],
            "name": ["Ghat A"],
            "type": ["ghat"],
            "area_m2": [12000.0],
            "usable_area_m2": [8400.0],
            "safe_capacity": [16800.0],
            "low_lying": ["true"],
            "drainage_capacity_mm_hr": [20.0],
            "is_synthetic": ["true"],
            "source": ["layout"],
        }
    )
    checked = check_schema(frame, "zones")
    assert checked["low_lying"].iloc[0] is True or bool(checked["low_lying"].iloc[0])
    assert checked["low_lying"].dtype == bool


# ------------------------------------------------------------------ live adapter
def test_live_adapter_raises_not_implemented() -> None:
    """docs/02 section 8: live must raise clearly, never fabricate data."""
    LiveAdapter.unregister_all()
    with pytest.raises(NotImplementedError, match="live connector for footfall_15min"):
        load_table("footfall_15min", data_source="live")


def test_live_adapter_uses_a_registered_connector() -> None:
    from twin_common.io import LiveConnector

    class FakeConnector(LiveConnector):
        def read(self, name, start, end):
            return _footfall_frame(2)

    LiveAdapter.register("footfall_15min", FakeConnector())
    try:
        loaded = load_table("footfall_15min", data_source="live")
        assert len(loaded) == 2
    finally:
        LiveAdapter.unregister_all()


def test_live_adapter_has_no_filesystem_path() -> None:
    assert not table_is_available("footfall_15min", data_source="live")


def test_replay_adapter_reads_the_same_schema(tmp_path: Path) -> None:
    save_table(_footfall_frame(4), "footfall_15min", tmp_path)
    loaded = load_table("footfall_15min", data_source="replay", base_dir=tmp_path)
    assert len(loaded) == 4


def test_replay_missing_file_mentions_the_shared_schema(tmp_path: Path) -> None:
    with pytest.raises(TableError, match="same columns"):
        load_table("footfall_15min", data_source="replay", base_dir=tmp_path)
