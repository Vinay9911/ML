"""M19 contract tests (docs/02 section 9 item 3)."""

from __future__ import annotations

from twin_common.contracts import PredictRequest
from twin_common.contracts import registry as reg
from twin_common.testing import (
    assert_deterministic,
    assert_kpi_coverage,
    assert_units_match_registry,
    assert_valid_output,
)

MODEL_ID = "M19"


def test_output_satisfies_the_contract(model) -> None:
    assert_valid_output(model.predict(PredictRequest()), model_id=MODEL_ID)


def test_every_owned_kpi_is_reported(model) -> None:
    """The model card and the registry must agree with what the model actually emits."""
    output = model.predict(PredictRequest())
    assert_kpi_coverage(output, reg.kpis_owned_by(MODEL_ID))


def test_units_come_from_the_registry(model) -> None:
    assert_units_match_registry(model.predict(PredictRequest()))


def test_bounds_hold(model) -> None:
    output = model.predict(PredictRequest())
    for record in output.results:
        low, high = reg.kpi_info(record.kpi).bounds
        if low is not None:
            assert record.value >= low - 1e-9, f"{record.kpi} below {low}"
        if high is not None:
            assert record.value <= high + 1e-9, f"{record.kpi} above {high}"
        if record.lower is not None:
            assert record.lower <= record.value + 1e-9
        if record.upper is not None:
            assert record.upper >= record.value - 1e-9


def test_is_deterministic(model) -> None:
    """Same seed and inputs give identical JSON apart from run_id and generated_at."""
    assert_deterministic(lambda: model.predict(PredictRequest()))


def test_provenance_is_correct(model) -> None:
    output = model.predict(PredictRequest())
    assert output.data_source.value == model.data_source.value
    assert output.is_synthetic is (output.data_source.value == "synthetic")


def test_upstream_provenance_is_recorded(model) -> None:
    """Every declared upstream must appear in the output with its resolution source."""
    output = model.predict(PredictRequest())
    assert set(output.upstream_sources()) >= set(model.upstream_ids)


def test_no_hard_coded_thresholds_in_src() -> None:
    """CLAUDE.md: thresholds, rates and weights live in config.yaml, not in src/."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src"
    # Bare floats in comparisons are the tell-tale sign of an inline threshold.
    suspicious = re.compile(r"[<>]=?\s*\d+\.\d+")
    offences: list[str] = []
    for path in src.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "noqa: threshold" in stripped:
                continue
            if suspicious.search(stripped):
                offences.append(f"{path.name}:{number}: {stripped}")
    assert not offences, "inline thresholds found:\n" + "\n".join(offences)
