"""M21 API tests: the four endpoints of docs/02 section 6."""

from __future__ import annotations

from twin_common.testing import api_smoke

MODEL_ID = "M21"


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


# ---------------------------------------------------------------- /inputs on real data
def test_inputs_reports_real_row_counts_and_a_sample(app) -> None:
    """`/inputs` against a folder that HAS its world slice: the sample must be real rows."""
    from fastapi.testclient import TestClient

    body = TestClient(app).get("/inputs?rows=5").json()
    available = [t for t in body["tables"] if t.get("available")]
    assert available, "M21 should have its world slice"
    for table in available:
        assert table["total_rows"] > 0
        assert 0 < table["sampled_rows"] <= 5
        assert len(table["sample"]) == table["sampled_rows"]
        assert table["dataset_id"]


def test_inputs_samples_around_as_of_not_the_head(app) -> None:
    """The head of a 31-day table is four weeks of history nothing reads."""
    from fastapi.testclient import TestClient

    body = TestClient(app).get("/inputs?rows=8").json()
    dated = [
        t
        for t in body["tables"]
        if t.get("available") and t["sample"] and "timestamp" in t["sample"][0]
    ]
    assert dated, "no time-indexed table to check"
    for table in dated:
        assert "around" in table["sample_label"]
        stamps = [row["timestamp"] for row in table["sample"]]
        assert stamps == sorted(stamps), "the window should be in time order"


def test_inputs_caps_the_sample_size(app) -> None:
    """A browser showing 24,000 rows helps nobody, and this is called on every page load."""
    from fastapi.testclient import TestClient

    from twin_common.api.inputs import MAX_SAMPLE_ROWS

    body = TestClient(app).get(f"/inputs?rows={MAX_SAMPLE_ROWS * 10}").json()
    for table in body["tables"]:
        if table.get("available"):
            assert table["sampled_rows"] <= MAX_SAMPLE_ROWS


def test_inputs_reports_total_rows_across_every_table(app) -> None:
    from fastapi.testclient import TestClient

    body = TestClient(app).get("/inputs?rows=3").json()
    expected = sum(t.get("total_rows", 0) for t in body["tables"])
    assert body["total_input_rows"] == expected
    assert body["total_input_rows"] > 0
