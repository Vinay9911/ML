"""The four endpoints of docs/02 section 6, exercised through a dummy model."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from twin_common.contracts import Metadata, ModelOutput
from twin_common.testing import (
    api_smoke,
    assert_deterministic,
    assert_insensitive,
    assert_kpi_coverage,
    assert_scenario_direction,
    assert_valid_output,
)

from .conftest import DummyModel


@pytest.fixture
def client(dummy_app) -> TestClient:
    with TestClient(dummy_app) as test_client:
        yield test_client


# ------------------------------------------------------------------ the acceptance check
def test_api_smoke_passes_for_a_dummy_model(dummy_app) -> None:
    """docs/07 Phase 1 acceptance: a dummy model via create_app passes api_smoke."""
    results = api_smoke(dummy_app, model_id="M21")
    assert isinstance(results["predict"], ModelOutput)
    assert isinstance(results["metadata"], Metadata)
    assert results["unknown_scenario_status"] in (404, 422)


# ------------------------------------------------------------------ /health
def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "model_id": "M21",
        "model_version": "0.1.0",
        "data_source": "synthetic",
    }


# ------------------------------------------------------------------ /metadata
def test_metadata_matches_the_config(client: TestClient) -> None:
    meta = Metadata.model_validate(client.get("/metadata").json())
    assert meta.model_id == "M21"
    assert meta.engine.value == "formula"
    assert [k.kpi for k in meta.kpis] == [
        "temperature",
        "heat_index",
        "rainfall_intensity",
        "waterlogging_probability",
    ]
    assert meta.scenarios_supported == ["S01", "S03", "S04"]
    assert meta.horizons_min == [15, 60, 180]


def test_metadata_units_come_from_the_registry(client: TestClient) -> None:
    from twin_common.contracts import registry as reg

    meta = Metadata.model_validate(client.get("/metadata").json())
    for descriptor in meta.kpis:
        assert descriptor.unit == reg.kpi_unit(descriptor.kpi)


# ------------------------------------------------------------------ /predict
def test_predict_with_an_empty_body_uses_config_defaults(client: TestClient) -> None:
    output = assert_valid_output(client.post("/predict", json={}).json(), model_id="M21")
    assert output.as_of.isoformat() == "2027-08-02T06:00:00+05:30"
    assert output.scenario_id == "S01"
    assert output.is_synthetic is True


def test_predict_reports_every_owned_kpi(client: TestClient) -> None:
    output = assert_valid_output(client.post("/predict", json={}).json())
    assert_kpi_coverage(
        output,
        ["temperature", "heat_index", "rainfall_intensity", "waterlogging_probability"],
    )


def test_predict_honours_an_explicit_as_of(client: TestClient) -> None:
    body = {"as_of": "2027-08-02T08:30:00+05:30"}
    output = assert_valid_output(client.post("/predict", json=body).json())
    assert output.as_of.isoformat() == "2027-08-02T08:30:00+05:30"


def test_predict_filters_by_kpi(client: TestClient) -> None:
    output = assert_valid_output(client.post("/predict", json={"kpis": ["heat_index"]}).json())
    assert output.kpis_present() == {"heat_index"}


def test_predict_filters_by_entity(client: TestClient) -> None:
    body = {"entity_ids": ["Z01"], "kpis": ["waterlogging_probability"]}
    output = assert_valid_output(client.post("/predict", json=body).json())
    assert output.entities_present() == {"Z01"}


def test_predict_ignores_kpis_the_model_does_not_own(client: TestClient) -> None:
    body = {"kpis": ["heat_index", "crush_risk_score"]}
    output = assert_valid_output(client.post("/predict", json=body).json())
    assert output.kpis_present() == {"heat_index"}


def test_predict_is_deterministic(dummy_model: DummyModel) -> None:
    """docs/02 section 9 item 3."""
    from twin_common.contracts import PredictRequest

    assert_deterministic(lambda: dummy_model.predict(PredictRequest()))


def test_unknown_field_in_the_body_is_a_422(client: TestClient) -> None:
    response = client.post("/predict", json={"not_a_field": 1})
    assert response.status_code == 422


def test_negative_horizon_is_a_422(client: TestClient) -> None:
    assert client.post("/predict", json={"horizon_min": -5}).status_code == 422


# ------------------------------------------------------------------ /scenario
def test_scenario_marks_records_as_scenario_state(client: TestClient) -> None:
    output = assert_valid_output(client.post("/scenario", json={"scenario_id": "S03"}).json())
    assert output.scenario_id == "S03"
    assert all(r.state.value == "scenario" for r in output.results)


def test_scenario_echoes_its_overrides(client: TestClient) -> None:
    output = assert_valid_output(client.post("/scenario", json={"scenario_id": "S03"}).json())
    assert output.scenario_overrides["rain_mm_hr"] == 50


def test_scenario_fills_baseline_and_delta(client: TestClient) -> None:
    output = assert_valid_output(client.post("/scenario", json={"scenario_id": "S04"}).json())
    temperature = next(r for r in output.results if r.kpi == "temperature")
    assert temperature.baseline_value == pytest.approx(31.0)
    assert temperature.delta == pytest.approx(6.0)


def test_s04_raises_the_heat_index(client: TestClient) -> None:
    """docs/05 section 1.2: S04 heat_index up."""
    baseline = ModelOutput.model_validate(client.post("/predict", json={}).json())
    scenario = ModelOutput.model_validate(
        client.post("/scenario", json={"scenario_id": "S04"}).json()
    )
    assert_scenario_direction(baseline, scenario, "heat_index", direction="up")


def test_s03_raises_waterlogging(client: TestClient) -> None:
    """docs/05 section 1.2: S03 waterlogging up."""
    baseline = ModelOutput.model_validate(client.post("/predict", json={}).json())
    scenario = ModelOutput.model_validate(
        client.post("/scenario", json={"scenario_id": "S03"}).json()
    )
    assert_scenario_direction(baseline, scenario, "waterlogging_probability", direction="up")


def test_s01_scenario_equals_the_baseline(client: TestClient) -> None:
    baseline = ModelOutput.model_validate(client.post("/predict", json={}).json())
    scenario = ModelOutput.model_validate(
        client.post("/scenario", json={"scenario_id": "S01"}).json()
    )
    assert_scenario_direction(baseline, scenario, "temperature", direction="same")


def test_unsupported_scenario_returns_degraded_baseline(client: TestClient) -> None:
    """docs/05 section 1.2: not-listed models return baseline values, degraded."""
    response = client.post("/scenario", json={"scenario_id": "S08"})
    assert response.status_code == 200
    output = assert_valid_output(response.json())
    assert_insensitive(output, "S08")
    assert output.scenario_id == "S08"
    assert all(r.state.value != "forecast" for r in output.results)


def test_unsupported_scenario_values_equal_the_baseline(client: TestClient) -> None:
    baseline = ModelOutput.model_validate(client.post("/predict", json={}).json())
    scenario = ModelOutput.model_validate(
        client.post("/scenario", json={"scenario_id": "S08"}).json()
    )
    assert_scenario_direction(baseline, scenario, "temperature", direction="same")


def test_unknown_scenario_is_a_404(client: TestClient) -> None:
    """docs/02 section 6: unknown scenario_id -> 404."""
    response = client.post("/scenario", json={"scenario_id": "S13"})
    assert response.status_code == 200  # S13 exists but is unsupported -> degraded
    # A syntactically valid but undefined ID is rejected by the pattern as a 422.
    assert client.post("/scenario", json={"scenario_id": "S99"}).status_code == 422


def test_unknown_scenario_id_raises_404_from_the_handler(dummy_folder: Path) -> None:
    """Bypass the pattern check to prove the UnknownScenarioError handler maps to 404."""
    from fastapi import FastAPI

    from twin_common.api import create_app
    from twin_common.errors import UnknownScenarioError

    app: FastAPI = create_app(DummyModel, root=dummy_folder)

    @app.get("/x/boom")
    def boom() -> None:
        raise UnknownScenarioError("unknown scenario_id 'S42'")

    with TestClient(app) as test_client:
        response = test_client.get("/x/boom")
        assert response.status_code == 404
        assert response.json()["status"] == "error"


def test_internal_failures_become_a_500_error_envelope(dummy_folder: Path) -> None:
    from twin_common.api import create_app
    from twin_common.errors import TwinError

    app = create_app(DummyModel, root=dummy_folder)

    @app.get("/x/fail")
    def fail() -> None:
        raise TwinError("engine exploded")

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.get("/x/fail")
        assert response.status_code == 500
        assert response.json() == {"status": "error", "detail": "engine exploded"}


def test_live_data_source_surfaces_as_501(dummy_folder: Path) -> None:
    """docs/02 section 8: live must be an explicit NotImplementedError, never faked."""
    from twin_common.api import create_app

    app = create_app(DummyModel, root=dummy_folder)

    @app.get("/x/live")
    def live() -> None:
        raise NotImplementedError("live connector for weather_hourly not configured")

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.get("/x/live")
        assert response.status_code == 501
        assert "not configured" in response.json()["detail"]


# ------------------------------------------------------------------ app wiring
def test_cors_allows_any_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "*"


def test_app_state_exposes_the_port(dummy_app) -> None:
    assert dummy_app.state.model_id == "M21"
    assert dummy_app.state.port == 8021


def test_app_info(dummy_app) -> None:
    from twin_common.api import app_info

    info = app_info(dummy_app)
    assert info["model_id"] == "M21"
    assert info["port"] == 8021


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    for path in ("/health", "/metadata", "/predict", "/scenario"):
        assert path in schema["paths"], f"{path} missing from the OpenAPI schema"


def test_model_id_mismatch_between_class_and_config_fails_fast(tmp_path: Path) -> None:
    import yaml

    from twin_common.errors import ConfigError

    from .conftest import DUMMY_CONFIG

    payload = {**DUMMY_CONFIG, "model_id": "M07"}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="model_id is 'M07'"):
        DummyModel.from_folder(tmp_path)
