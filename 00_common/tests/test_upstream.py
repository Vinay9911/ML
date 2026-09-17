"""Upstream resolution must follow the docs/02 section 7 order exactly."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from twin_common.contracts import IST, ModelOutput
from twin_common.contracts.enums import UpstreamSource
from twin_common.errors import UpstreamError
from twin_common.upstream import (
    URL_ENV_TEMPLATE,
    resolve_all,
    resolve_one,
    sample_path,
    series_for_entity,
    set_stub_provider,
    values_by_entity,
)

AS_OF = datetime(2027, 8, 2, 6, 0, tzinfo=IST)


def _m01_output(value: float = 1000.0, *, scenario_id: str = "S01", run_id: str = "r1") -> dict:
    return {
        "model_id": "M01",
        "model_name": "Footfall Forecast",
        "model_version": "0.1.0",
        "run_id": run_id,
        "as_of": AS_OF.isoformat(),
        "scenario_id": scenario_id,
        "data_source": "synthetic",
        "is_synthetic": True,
        "results": [
            {
                "entity_type": "zone",
                "entity_id": zone,
                "zone_id": zone,
                "kpi": "expected_footfall",
                "timestamp": AS_OF.isoformat(),
                "horizon_min": horizon,
                "state": "scenario" if scenario_id != "S01" else "forecast",
                "value": value + offset,
                "unit": "persons/hr",
            }
            for zone, base in (("Z01", 0.0), ("Z02", 500.0))
            for horizon, offset in ((15, base), (60, base + 100.0), (180, base + 200.0))
        ],
    }


@pytest.fixture(autouse=True)
def _no_stub_provider():
    """Each test opts in to a stub provider explicitly."""
    set_stub_provider(None)
    yield
    set_stub_provider(None)


@pytest.fixture(autouse=True)
def _no_upstream_urls(monkeypatch):
    monkeypatch.delenv(URL_ENV_TEMPLATE.format(model_id="M01"), raising=False)


# ------------------------------------------------------------------ step 1: inline
def test_inline_wins_over_everything(tmp_path: Path) -> None:
    (tmp_path / "M01__S01.json").write_text(json.dumps(_m01_output(9.0)), encoding="utf-8")
    inline = {"M01": ModelOutput.model_validate(_m01_output(1000.0))}
    resolution = resolve_one("M01", as_of=AS_OF, inline=inline, sample_dir=tmp_path)
    assert resolution.source is UpstreamSource.INLINE
    assert resolution.warning is None
    assert values_by_entity(resolution, "expected_footfall", horizon_min=15)["Z01"] == 1000.0


# ---------------------------------------------------------- steps 3 and 4: samples
def test_exact_scenario_sample_is_used(tmp_path: Path) -> None:
    (tmp_path / "M01__S02.json").write_text(
        json.dumps(_m01_output(1300.0, scenario_id="S02")), encoding="utf-8"
    )
    resolution = resolve_one("M01", as_of=AS_OF, scenario_id="S02", sample_dir=tmp_path)
    assert resolution.source is UpstreamSource.SAMPLE
    assert resolution.warning is None
    assert resolution.output.scenario_id == "S02"


def test_baseline_sample_is_the_documented_fallback(tmp_path: Path) -> None:
    """Step 4: S01 sample plus the 'baseline upstream used for scenario' warning."""
    (tmp_path / "M01__S01.json").write_text(json.dumps(_m01_output()), encoding="utf-8")
    resolution = resolve_one("M01", as_of=AS_OF, scenario_id="S06", sample_dir=tmp_path)
    assert resolution.source is UpstreamSource.SAMPLE
    assert resolution.warning is not None
    assert "baseline upstream used for scenario" in resolution.warning


def test_baseline_sample_is_not_used_for_the_baseline_request(tmp_path: Path) -> None:
    """A missing S01 sample must fall through to the stub, not loop on itself."""
    calls: list[str] = []

    def provider(model_id, *, as_of, scenario_id, entity_ids=None):
        calls.append(model_id)
        return ModelOutput.model_validate(_m01_output(1.0))

    set_stub_provider(provider)
    resolution = resolve_one("M01", as_of=AS_OF, sample_dir=tmp_path)
    assert resolution.source is UpstreamSource.STUB
    assert calls == ["M01"]


# -------------------------------------------------------------------- step 5: stub
def test_stub_is_the_last_resort_and_warns(tmp_path: Path) -> None:
    set_stub_provider(
        lambda model_id, *, as_of, scenario_id, entity_ids=None: ModelOutput.model_validate(
            _m01_output(1.0)
        )
    )
    resolution = resolve_one("M01", as_of=AS_OF, sample_dir=tmp_path)
    assert resolution.source is UpstreamSource.STUB
    assert resolution.warning is not None and "stub" in resolution.warning


def test_no_source_at_all_raises_with_guidance(tmp_path: Path) -> None:
    with pytest.raises(UpstreamError, match="UPSTREAM_M01_URL"):
        resolve_one("M01", as_of=AS_OF, sample_dir=tmp_path)


def test_allow_stub_false_skips_the_stub(tmp_path: Path) -> None:
    set_stub_provider(
        lambda model_id, *, as_of, scenario_id, entity_ids=None: ModelOutput.model_validate(
            _m01_output(1.0)
        )
    )
    with pytest.raises(UpstreamError):
        resolve_one("M01", as_of=AS_OF, sample_dir=tmp_path, allow_stub=False)


# ------------------------------------------------------------------ step 2: url
def test_url_failure_falls_through_to_the_sample(tmp_path: Path, monkeypatch) -> None:
    """A dead upstream service must degrade, not fail the request."""
    monkeypatch.setenv(URL_ENV_TEMPLATE.format(model_id="M01"), "http://127.0.0.1:9/unreachable")
    (tmp_path / "M01__S01.json").write_text(json.dumps(_m01_output()), encoding="utf-8")
    resolution = resolve_one("M01", as_of=AS_OF, sample_dir=tmp_path, timeout_s=0.05)
    assert resolution.source is UpstreamSource.SAMPLE


def test_url_is_called_when_reachable(tmp_path: Path, monkeypatch) -> None:
    """Step 2 uses /predict for the baseline and /scenario otherwise."""
    called: dict[str, str] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return _m01_output(4242.0, run_id="from-url")

    def fake_post(url, json=None, timeout=None):
        called["url"] = url
        called["body"] = json
        return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv(URL_ENV_TEMPLATE.format(model_id="M01"), "http://localhost:8001")

    resolution = resolve_one("M01", as_of=AS_OF, sample_dir=tmp_path)
    assert resolution.source is UpstreamSource.URL
    assert resolution.output.run_id == "from-url"
    assert called["url"] == "http://localhost:8001/predict"

    resolve_one("M01", as_of=AS_OF, scenario_id="S02", sample_dir=tmp_path)
    assert called["url"] == "http://localhost:8001/scenario"
    assert called["body"]["scenario_id"] == "S02"


# ------------------------------------------------------------------ helpers
def test_sample_path_naming() -> None:
    """docs/02 section 7: data/sample_upstream/<ID>__<Sxx>.json"""
    assert sample_path(Path("/d"), "M01", "S02").name == "M01__S02.json"


def test_resolve_all_preserves_order(tmp_path: Path) -> None:
    for model_id in ("M01", "M21"):
        payload = _m01_output()
        payload["model_id"] = model_id
        payload["model_name"] = model_id
        if model_id == "M21":
            payload["results"] = [
                {
                    "entity_type": "event",
                    "entity_id": "EVENT",
                    "kpi": "heat_index",
                    "timestamp": AS_OF.isoformat(),
                    "horizon_min": 0,
                    "state": "current",
                    "value": 34.0,
                    "unit": "degC",
                }
            ]
        (tmp_path / f"{model_id}__S01.json").write_text(json.dumps(payload), encoding="utf-8")
    resolutions = resolve_all(["M21", "M01"], as_of=AS_OF, sample_dir=tmp_path)
    assert list(resolutions) == ["M21", "M01"]
    assert all(r.source is UpstreamSource.SAMPLE for r in resolutions.values())


def test_values_by_entity_picks_the_closest_horizon() -> None:
    output = ModelOutput.model_validate(_m01_output(1000.0))
    at_15 = values_by_entity(output, "expected_footfall", horizon_min=15)
    at_180 = values_by_entity(output, "expected_footfall", horizon_min=180)
    assert at_15 == {"Z01": 1000.0, "Z02": 1500.0}
    assert at_180 == {"Z01": 1200.0, "Z02": 1700.0}


def test_values_by_entity_ignores_other_kpis() -> None:
    output = ModelOutput.model_validate(_m01_output())
    assert values_by_entity(output, "peak_footfall") == {}


def test_series_for_entity_is_sorted() -> None:
    output = ModelOutput.model_validate(_m01_output())
    series = series_for_entity(output, "expected_footfall", "Z01")
    assert len(series) == 3
    assert series == sorted(series, key=lambda item: item[0])


def test_a_corrupt_sample_file_raises_rather_than_passing_bad_data(tmp_path: Path) -> None:
    """A sample that violates the contract must not slip through to a model."""
    bad = _m01_output()
    bad["results"][0]["unit"] = "persons"  # wrong unit for expected_footfall
    (tmp_path / "M01__S01.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(Exception, match="unit mismatch"):
        resolve_one("M01", as_of=AS_OF, sample_dir=tmp_path)
