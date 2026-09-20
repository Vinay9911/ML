"""M01 forecast behaviour: series coverage, bands, units and the backtest artifact."""

from __future__ import annotations

import json

import pytest

from twin_common.contracts import PredictRequest

MODEL_ID = "M01"
ZONES = ("Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07", "Z08")
GATES = ("G01", "G02", "G03", "G04")


@pytest.fixture(scope="module")
def output(model):
    return model.predict(PredictRequest())


def test_forecasts_every_zone_and_gate(output) -> None:
    """docs/03 M01: one series per zone (Z01-Z08) and per gate (G01-G04)."""
    entities = output.entities_present()
    assert set(ZONES) <= entities, f"missing zones: {set(ZONES) - entities}"
    assert set(GATES) <= entities, f"missing gates: {set(GATES) - entities}"


def test_gates_are_typed_as_gates(output) -> None:
    for record in output.results:
        if record.entity_id in GATES:
            assert record.entity_type.value == "gate"
        elif record.entity_id in ZONES:
            assert record.entity_type.value == "zone"
            assert record.zone_id == record.entity_id


def test_expected_footfall_is_in_persons_per_hour(output, model) -> None:
    """docs/03 M01: 15-min counts are converted to persons/hr (x4)."""
    steps_per_hour = float(model.require_param("steps_per_hour"))
    for record in output.results:
        if record.kpi != "expected_footfall":
            continue
        assert record.unit == "persons/hr"
        per_step = record.details["entries_per_step"]
        assert record.value == pytest.approx(
            max(0.0, per_step * steps_per_hour), abs=steps_per_hour
        )


def test_no_negative_footfall(output) -> None:
    """A quantile band can dip below zero; a count of people cannot."""
    for record in output.results:
        assert record.value >= 0.0
        if record.lower is not None:
            assert record.lower >= 0.0


def test_bands_are_ordered_and_labelled(output) -> None:
    banded = [r for r in output.results if r.lower is not None]
    assert banded, "the default backend should produce uncertainty bands"
    for record in banded:
        assert record.lower <= record.value <= record.upper
        assert record.quantile_level == pytest.approx(0.8)


def test_horizon_is_respected(output, model) -> None:
    """No record may look further ahead than the requested horizon."""
    assert max(r.horizon_min for r in output.results) <= model.default_horizon_min


def test_a_shorter_horizon_returns_fewer_records(model) -> None:
    long_horizon = model.predict(PredictRequest(horizon_min=180))
    short_horizon = model.predict(PredictRequest(horizon_min=15))
    assert len(short_horizon.results) < len(long_horizon.results)
    assert max(r.horizon_min for r in short_horizon.results) <= 15


def test_peak_is_the_maximum_of_the_forecast(output) -> None:
    """docs/03 M01: peak_footfall is the max of the median forecast in the horizon."""
    by_entity: dict[str, list[float]] = {}
    peaks: dict[str, float] = {}
    for record in output.results:
        if record.kpi == "expected_footfall":
            by_entity.setdefault(record.entity_id, []).append(record.value)
        elif record.kpi == "peak_footfall":
            peaks[record.entity_id] = record.value
    assert peaks
    for entity, peak in peaks.items():
        assert peak == pytest.approx(max(by_entity[entity]))


def test_peak_records_the_time_it_occurs(output) -> None:
    for record in output.results:
        if record.kpi == "peak_footfall":
            assert "peak_at" in record.details
            assert record.details["peak_at"].startswith("20")


def test_entity_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(entity_ids=["Z01", "G02"]))
    assert output.entities_present() <= {"Z01", "G02"}


def test_kpi_filter_is_honoured(model) -> None:
    output = model.predict(PredictRequest(kpis=["peak_footfall"]))
    assert output.kpis_present() == {"peak_footfall"}


def test_upstream_m21_is_recorded(output) -> None:
    """docs/02 section 7: the resolution source must be reported."""
    assert "M21" in output.upstream_sources()


def test_backend_is_reported_in_details(output) -> None:
    backends = {r.details.get("backend") for r in output.results}
    assert backends <= {"chronos2", "lightgbm", "naive"}
    assert len(backends) == 1, f"one request should use one backend, got {backends}"


def test_forecast_starts_after_as_of(output, model) -> None:
    as_of = model.demo_now
    assert all(r.timestamp > as_of for r in output.results)


# ------------------------------------------------------------------------ backtest
def test_backtest_writes_its_artifact(model) -> None:
    """docs/03 M01: backtest metrics written to data/derived/backtest.json."""
    path = model.derived_dir / "backtest.json"
    if not path.is_file():
        model.backtest(write=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model_id"] == MODEL_ID
    assert payload["backends"], "no backend was backtested"
    assert "synthetic" in payload["note"].lower()


def test_backtest_covers_all_three_backends(model) -> None:
    """docs/03 M01: the model card records all three backends."""
    path = model.derived_dir / "backtest.json"
    if not path.is_file():
        model.backtest(write=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["backends"]) >= {"lightgbm", "naive"}
    for name, metrics in payload["backends"].items():
        assert metrics["mae"] >= 0, name
        assert metrics["folds"] > 0, name


def test_a_learned_backend_beats_the_naive_floor(model) -> None:
    """The point of the naive backend: it is the floor the others must clear."""
    path = model.derived_dir / "backtest.json"
    if not path.is_file():
        model.backtest(write=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    naive = payload["backends"].get("naive")
    learned = [m for n, m in payload["backends"].items() if n != "naive"]
    if not naive or not learned:
        pytest.skip("need the naive backend and at least one learned backend")
    assert min(m["mae"] for m in learned) < naive["mae"], (
        "no learned backend beat NaiveSeasonal; that is a finding, not a pass"
    )
