"""JSON Schema export - the artifact the frontend developer integrates against."""

from __future__ import annotations

import json
from pathlib import Path

from twin_common.contracts import SCHEMA_TARGETS, export_schemas, schema_for
from twin_common.contracts.models import ModelOutput
from twin_common.paths import schemas_dir


def test_export_writes_the_four_contract_schemas(tmp_path: Path) -> None:
    written = export_schemas(tmp_path)
    assert set(written) == {
        "model_output.schema.json",
        "predict_request.schema.json",
        "scenario_request.schema.json",
        "metadata.schema.json",
    }
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0


def test_exported_files_are_valid_json(tmp_path: Path) -> None:
    for path in export_schemas(tmp_path).values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "properties" in payload or "$ref" in payload


def test_export_is_idempotent(tmp_path: Path) -> None:
    """Re-running the export must produce no diff, so it is safe in CI."""
    first = {p.name: p.read_text(encoding="utf-8") for p in export_schemas(tmp_path).values()}
    second = {p.name: p.read_text(encoding="utf-8") for p in export_schemas(tmp_path).values()}
    assert first == second


def test_model_output_schema_documents_the_contract_fields() -> None:
    schema = schema_for(ModelOutput)
    required = set(schema.get("required", []))
    for field in (
        "model_id",
        "model_name",
        "model_version",
        "as_of",
        "data_source",
        "is_synthetic",
    ):
        assert field in schema["properties"], f"{field} missing from the schema"
    assert {"model_id", "as_of", "data_source", "is_synthetic"} <= required


def test_model_output_schema_pins_the_schema_version() -> None:
    schema = schema_for(ModelOutput)
    assert schema["properties"]["schema_version"]["const"] == "1.0"


def test_result_record_is_reachable_from_the_output_schema() -> None:
    schema = schema_for(ModelOutput)
    definitions = schema.get("$defs", {})
    assert "ResultRecord" in definitions
    record = definitions["ResultRecord"]["properties"]
    for field in (
        "entity_type",
        "entity_id",
        "kpi",
        "timestamp",
        "horizon_min",
        "state",
        "value",
        "unit",
        "risk_level",
        "reason_codes",
        "recommendation",
    ):
        assert field in record, f"{field} missing from the ResultRecord schema"


def test_recommendation_schema_pins_requires_approval() -> None:
    """requires_approval must be visibly constant in the published schema."""
    definitions = schema_for(ModelOutput).get("$defs", {})
    approval = definitions["Recommendation"]["properties"]["requires_approval"]
    assert approval["const"] is True


def test_enums_are_published_as_closed_vocabularies() -> None:
    definitions = schema_for(ModelOutput).get("$defs", {})
    assert set(definitions["State"]["enum"]) == {"current", "forecast", "scenario"}
    assert set(definitions["Status"]["enum"]) == {"ok", "degraded", "error"}
    assert set(definitions["DataSource"]["enum"]) == {"synthetic", "replay", "live"}
    assert set(definitions["RiskLevel"]["enum"]) == {"green", "amber", "red", "critical"}
    assert set(definitions["UpstreamSource"]["enum"]) == {"inline", "url", "sample", "stub"}


def test_entity_type_enum_has_all_fifteen_types() -> None:
    definitions = schema_for(ModelOutput).get("$defs", {})
    assert len(definitions["EntityType"]["enum"]) == 15


def test_schema_targets_are_all_exported() -> None:
    assert len(SCHEMA_TARGETS) == 4


def test_repo_schemas_directory_is_up_to_date() -> None:
    """The committed schemas under 00_common/schemas/ must match the current contracts."""
    target = schemas_dir()
    if not target.is_dir():
        export_schemas()
    for filename, model in SCHEMA_TARGETS.items():
        path = target / filename
        assert path.is_file(), f"{filename} has not been exported; run export_schemas()"
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk == schema_for(model), (
            f"{filename} is stale; re-run twin_common.contracts.export_schemas()"
        )
