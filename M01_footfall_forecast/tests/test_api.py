"""M01 API tests: the four endpoints of docs/02 section 6."""

from __future__ import annotations

from twin_common.testing import api_smoke

MODEL_ID = "M01"


def test_all_four_endpoints_answer(app) -> None:
    results = api_smoke(app, model_id=MODEL_ID)
    assert results["health"]["model_id"] == MODEL_ID
    assert results["metadata"].kpis
    assert results["predict"].results


def test_metadata_matches_the_config(app, model) -> None:
    from fastapi.testclient import TestClient

    from twin_common.contracts import Metadata

    with TestClient(app) as client:
        meta = Metadata.model_validate(client.get("/metadata").json())
    assert meta.model_id == MODEL_ID
    assert [k.kpi for k in meta.kpis] == model.owned_kpis
    assert meta.scenarios_supported == model.scenarios_supported


def test_unknown_scenario_is_rejected(app) -> None:
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        assert client.post("/scenario", json={"scenario_id": "S99"}).status_code in (404, 422)


def test_response_is_fast_enough(app) -> None:
    """docs/02 section 6: under 10 s on CPU for the default request."""
    import time

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        started = time.perf_counter()
        response = client.post("/predict", json={})
        elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed < 10.0, f"/predict took {elapsed:.1f}s"
