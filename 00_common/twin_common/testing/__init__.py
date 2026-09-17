"""Shared assertions for the per-model test suites (docs/02 section 9 item 3).

Every model folder has the same four test concerns, so they live here once:

- :func:`assert_valid_output` - the output satisfies docs/02 section 4
- :func:`assert_kpi_coverage` - every KPI the model claims to own is actually present
- :func:`assert_deterministic` - same seed and inputs produce identical JSON
- :func:`api_smoke` - the four endpoints answer and validate

These are plain functions that raise ``AssertionError``, so they work with pytest without
importing it here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from ..contracts import registry as reg
from ..contracts.enums import REASON_REQUIRED_LEVELS, DataSource, State, Status
from ..contracts.models import (
    Metadata,
    ModelOutput,
    PredictRequest,
    ScenarioRequest,
)

#: Fields that legitimately differ between two runs of the same request.
NON_DETERMINISTIC_FIELDS: tuple[str, ...] = ("run_id", "generated_at")


# ----------------------------------------------------------------------- validity
def assert_valid_output(
    output: ModelOutput | dict[str, Any],
    *,
    model_id: str | None = None,
    expect_state: State | str | None = None,
) -> ModelOutput:
    """Validate an output against the contract and the cross-field rules.

    Re-validating a ModelOutput instance is deliberate: it catches an object that was
    mutated after construction, which is exactly what a sloppy scenario path does.
    """
    validated = (
        ModelOutput.model_validate(output)
        if isinstance(output, dict)
        else ModelOutput.model_validate(output.model_dump(mode="json"))
    )
    if model_id is not None:
        assert validated.model_id == model_id, (
            f"expected model_id {model_id!r}, got {validated.model_id!r}"
        )
    assert validated.results, f"{validated.model_id} returned no results"

    if validated.data_source is DataSource.SYNTHETIC:
        assert validated.is_synthetic, "synthetic data_source requires is_synthetic true"
    if validated.status is Status.DEGRADED:
        assert validated.warnings, "degraded status requires a warning"

    for record in validated.results:
        info = reg.kpi_info(record.kpi)
        assert record.unit == info.unit, (
            f"{record.kpi} unit {record.unit!r} does not match registry {info.unit!r}"
        )
        lo, hi = info.bounds
        if lo is not None:
            assert record.value >= lo - 1e-9, f"{record.kpi}={record.value} below bound {lo}"
        if hi is not None:
            assert record.value <= hi + 1e-9, f"{record.kpi}={record.value} above bound {hi}"
        if record.lower is not None:
            assert record.lower <= record.value + 1e-9, (
                f"{record.kpi} lower {record.lower} > value {record.value}"
            )
        if record.upper is not None:
            assert record.upper >= record.value - 1e-9, (
                f"{record.kpi} upper {record.upper} < value {record.value}"
            )
        if record.risk_level in REASON_REQUIRED_LEVELS:
            assert record.reason_codes, (
                f"{record.kpi} on {record.entity_id} is {record.risk_level.value} "
                f"but carries no reason codes"
            )
        if record.recommendation is not None:
            assert record.recommendation.requires_approval is True, (
                "requires_approval must always be true"
            )
        assert record.timestamp.utcoffset() is not None, "timestamps must be tz-aware"
        if expect_state is not None:
            assert record.state == State(expect_state), (
                f"expected state {expect_state}, got {record.state} for {record.kpi}"
            )
    return validated


def assert_kpi_coverage(
    output: ModelOutput,
    expected_kpis: Iterable[str],
    *,
    allow_extra: bool = True,
) -> None:
    """Every KPI in ``expected_kpis`` appears in the output."""
    present = output.kpis_present()
    expected = set(expected_kpis)
    missing = sorted(expected - present)
    assert not missing, (
        f"{output.model_id} does not report its own KPIs {missing}; present: {sorted(present)}"
    )
    if not allow_extra:
        extra = sorted(present - expected)
        assert not extra, f"{output.model_id} reports unexpected KPIs {extra}"


def assert_owned_kpis_reported(output: ModelOutput) -> None:
    """Every KPI whose registry owner is this model is present in the output."""
    assert_kpi_coverage(output, reg.kpis_owned_by(output.model_id))


def assert_units_match_registry(output: ModelOutput) -> None:
    for record in output.results:
        expected = reg.kpi_unit(record.kpi)
        assert record.unit == expected, (
            f"{record.kpi}: unit {record.unit!r} != registry {expected!r}"
        )


# ----------------------------------------------------------------------- determinism
def canonical_json(output: ModelOutput, *, drop: Iterable[str] = NON_DETERMINISTIC_FIELDS) -> str:
    """Stable JSON for two outputs to be compared, minus run_id and generated_at."""
    payload = output.model_dump(mode="json")
    for field in drop:
        payload.pop(field, None)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def assert_deterministic(
    run: Callable[[], ModelOutput],
    *,
    times: int = 2,
) -> ModelOutput:
    """Call ``run`` repeatedly and assert the outputs are byte-identical.

    docs/02 section 9 item 3: same seed and inputs must give identical JSON apart from
    ``run_id`` and ``generated_at``.
    """
    first = run()
    reference = canonical_json(first)
    for attempt in range(2, times + 1):
        other = canonical_json(run())
        if other != reference:
            diff = _first_difference(reference, other)
            raise AssertionError(
                f"run {attempt} differs from run 1 - the model is not deterministic. "
                f"First difference near: {diff}"
            )
    return first


def _first_difference(left: str, right: str, context: int = 60) -> str:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            start = max(0, index - context)
            return f"...{left[start : index + context]!r} vs ...{right[start : index + context]!r}"
    return f"length {len(left)} vs {len(right)}"


# ----------------------------------------------------------------------- API smoke
def api_smoke(
    app: Any,
    *,
    model_id: str | None = None,
    scenario_id: str | None = None,
    predict_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Exercise /health, /metadata, /predict and /scenario through a TestClient.

    Returns the four parsed responses so a test can assert on them further.
    """
    from fastapi.testclient import TestClient

    results: dict[str, Any] = {}
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, f"/health returned {health.status_code}"
        health_body = health.json()
        assert health_body["status"] == "ok"
        if model_id is not None:
            assert health_body["model_id"] == model_id
        results["health"] = health_body

        metadata = client.get("/metadata")
        assert metadata.status_code == 200, f"/metadata returned {metadata.status_code}"
        meta = Metadata.model_validate(metadata.json())
        assert meta.kpis, "metadata lists no KPIs"
        results["metadata"] = meta

        body = predict_body if predict_body is not None else {}
        PredictRequest.model_validate(body)
        predict = client.post("/predict", json=body)
        assert predict.status_code == 200, (
            f"/predict returned {predict.status_code}: {predict.text[:400]}"
        )
        predict_out = assert_valid_output(predict.json(), model_id=model_id)
        assert_kpi_coverage(predict_out, [k.kpi for k in meta.kpis])
        results["predict"] = predict_out

        sid = scenario_id or (meta.scenarios_supported[0] if meta.scenarios_supported else "S01")
        scenario_body = {**body, "scenario_id": sid}
        ScenarioRequest.model_validate(scenario_body)
        scenario = client.post("/scenario", json=scenario_body)
        assert scenario.status_code == 200, (
            f"/scenario {sid} returned {scenario.status_code}: {scenario.text[:400]}"
        )
        scenario_out = assert_valid_output(scenario.json(), model_id=model_id)
        assert scenario_out.scenario_id == sid
        results["scenario"] = scenario_out

        unknown = client.post("/scenario", json={**body, "scenario_id": "S99"})
        assert unknown.status_code in (404, 422), (
            f"an unknown scenario_id must give 404 or 422, got {unknown.status_code}"
        )
        results["unknown_scenario_status"] = unknown.status_code
    return results


# ----------------------------------------------------------------------- scenarios
def assert_scenario_direction(
    baseline: ModelOutput,
    scenario: ModelOutput,
    kpi: str,
    *,
    direction: str,
    entity_ids: Iterable[str] | None = None,
    ratio_range: tuple[float, float] | None = None,
    tolerance: float = 1e-9,
) -> tuple[float, float]:
    """Compare a KPI total between a baseline and a scenario run (docs/05 section 1.2).

    Args:
        direction: ``up``, ``down``, ``same``, or ``ratio`` to only check ``ratio_range``.
        entity_ids: restrict the comparison to these entities.
        ratio_range: inclusive (low, high) bounds on scenario/baseline.

    Returns:
        (baseline_total, scenario_total)
    """
    wanted = set(entity_ids) if entity_ids else None

    def total(output: ModelOutput) -> float:
        return sum(
            r.value
            for r in output.results
            if r.kpi == kpi and (wanted is None or r.entity_id in wanted)
        )

    base_total, scen_total = total(baseline), total(scenario)
    label = f"{kpi}{'' if wanted is None else f' for {sorted(wanted)}'}"

    if direction == "up":
        assert scen_total > base_total + tolerance, (
            f"expected {label} to rise: baseline {base_total:.4f}, scenario {scen_total:.4f}"
        )
    elif direction == "down":
        assert scen_total < base_total - tolerance, (
            f"expected {label} to fall: baseline {base_total:.4f}, scenario {scen_total:.4f}"
        )
    elif direction == "same":
        assert abs(scen_total - base_total) <= max(tolerance, abs(base_total) * 1e-6), (
            f"expected {label} to stay flat: baseline {base_total:.4f}, scenario {scen_total:.4f}"
        )
    elif direction != "ratio":
        raise ValueError(f"unknown direction {direction!r}")

    if ratio_range is not None:
        assert base_total != 0, f"cannot check a ratio for {label}: baseline total is 0"
        ratio = scen_total / base_total
        low, high = ratio_range
        assert low <= ratio <= high, (
            f"{label} ratio {ratio:.4f} outside the expected range [{low}, {high}] "
            f"(baseline {base_total:.4f}, scenario {scen_total:.4f})"
        )
    return base_total, scen_total


def assert_insensitive(output: ModelOutput, scenario_id: str) -> None:
    """The documented response for an unsupported scenario (docs/05 section 1.2)."""
    assert output.status is Status.DEGRADED, (
        f"an unsupported scenario must return status degraded, got {output.status}"
    )
    assert any(f"insensitive to {scenario_id}" in w for w in output.warnings), (
        f"expected an 'insensitive to {scenario_id}' warning, got {output.warnings}"
    )
