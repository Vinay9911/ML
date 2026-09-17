"""Every field rule of docs/02 section 4 must be enforced, not just documented.

Each validator gets a positive case and a crafted negative case, so a future refactor that
drops a check fails here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from twin_common.contracts import (
    IST,
    Metadata,
    ModelOutput,
    PredictRequest,
    Recommendation,
    ResultRecord,
    ScenarioRequest,
    UpstreamRef,
)
from twin_common.errors import RegistryError

TS = datetime(2027, 8, 2, 6, 45, tzinfo=IST)
AS_OF = datetime(2027, 8, 2, 6, 0, tzinfo=IST)

#: The exact record from the docs/02 section 4 example payload.
DOC_RECORD: dict[str, Any] = {
    "entity_type": "zone",
    "entity_id": "Z01",
    "zone_id": "Z01",
    "kpi": "crush_risk_score",
    "timestamp": "2027-08-02T06:45:00+05:30",
    "horizon_min": 45,
    "state": "forecast",
    "value": 72.4,
    "unit": "score_0_100",
    "lower": 61.0,
    "upper": 81.5,
    "quantile_level": 0.8,
    "baseline_value": None,
    "delta": None,
    "risk_level": "red",
    "reason_codes": ["DENSITY_RISING_FAST", "ACCUMULATION", "HEAT_STRESS"],
    "resource_type": None,
    "recommendation": None,
    "confidence": 0.7,
    "details": {"density_p_m2": 4.3},
}


def _base(**overrides: Any) -> dict[str, Any]:
    record = {
        "entity_type": "zone",
        "entity_id": "Z01",
        "kpi": "crush_risk_score",
        "timestamp": TS,
        "horizon_min": 45,
        "state": "forecast",
        "value": 50.0,
        "unit": "score_0_100",
    }
    record.update(overrides)
    return record


def _output(**overrides: Any) -> dict[str, Any]:
    payload = {
        "model_id": "M03",
        "model_name": "Hotspot / Crush Risk",
        "model_version": "0.1.0",
        "as_of": AS_OF,
        "data_source": "synthetic",
        "is_synthetic": True,
        "results": [_base()],
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------------------ positive cases
def test_docs_example_record_validates() -> None:
    record = ResultRecord.model_validate(DOC_RECORD)
    assert record.value == pytest.approx(72.4)
    assert record.risk_level is not None and record.risk_level.value == "red"


def test_docs_example_output_validates_and_round_trips() -> None:
    output = ModelOutput.model_validate(
        {
            "schema_version": "1.0",
            "model_id": "M03",
            "model_name": "Hotspot / Crush Risk",
            "model_version": "0.1.0",
            "run_id": "2f1c7b9e-6b4a-4d7e-9a57-0d6c5a3e9e11",
            "generated_at": "2027-08-02T06:00:04+05:30",
            "as_of": "2027-08-02T06:00:00+05:30",
            "scenario_id": "S01",
            "scenario_overrides": {},
            "data_source": "synthetic",
            "is_synthetic": True,
            "status": "ok",
            "warnings": [],
            "upstream": [
                {
                    "model_id": "M01",
                    "run_id": "abc",
                    "source": "sample",
                    "generated_at": "2027-08-02T05:59:50+05:30",
                }
            ],
            "results": [DOC_RECORD],
        }
    )
    again = ModelOutput.model_validate(json.loads(output.model_dump_json()))
    assert again.model_dump(mode="json") == output.model_dump(mode="json")


def test_timestamps_normalise_to_ist() -> None:
    """A UTC input must come back as +05:30 (docs/02 section 4)."""
    utc = datetime(2027, 8, 2, 1, 15, tzinfo=UTC)
    record = ResultRecord.model_validate(_base(timestamp=utc))
    assert record.timestamp.utcoffset() == timedelta(hours=5, minutes=30)
    assert record.timestamp.isoformat() == "2027-08-02T06:45:00+05:30"


def test_current_state_with_zero_horizon_is_accepted() -> None:
    record = ResultRecord.model_validate(_base(state="current", horizon_min=0))
    assert record.horizon_min == 0


def test_delta_consistent_with_baseline_is_accepted() -> None:
    record = ResultRecord.model_validate(_base(value=55.0, baseline_value=50.0, delta=5.0))
    assert record.delta == pytest.approx(5.0)


def test_recommendation_defaults_to_requires_approval() -> None:
    rec = Recommendation(action="stage ambulances", rationale="medical surge forecast")
    assert rec.requires_approval is True


def test_utilization_may_exceed_100() -> None:
    """zone_capacity_utilization has an explicit open upper bound.

    ResultRecord does not derive risk_level - that is OutputBuilder's job, because an
    auto-derived amber would then demand reason codes the record has no way to supply.
    Band derivation itself is covered in test_output_builder.py.
    """
    record = ResultRecord.model_validate(
        _base(kpi="zone_capacity_utilization", unit="%", value=134.0)
    )
    assert record.value == pytest.approx(134.0)
    assert record.risk_level is None


def test_negative_gap_is_accepted() -> None:
    """resource_gap is required - available and may be negative (a surplus)."""
    record = ResultRecord.model_validate(
        _base(
            entity_type="resource_pool",
            entity_id="police",
            kpi="resource_gap",
            unit="units",
            value=-12.0,
        )
    )
    assert record.value == pytest.approx(-12.0)


# ------------------------------------------------------------------ negative cases
def test_entity_id_must_match_entity_type() -> None:
    with pytest.raises(RegistryError, match="does not match the pattern"):
        ResultRecord.model_validate(_base(entity_id="G02"))


def test_zone_id_must_be_a_zone() -> None:
    with pytest.raises(RegistryError, match="does not match the pattern"):
        ResultRecord.model_validate(_base(zone_id="G02"))


def test_unknown_kpi_is_rejected() -> None:
    with pytest.raises(RegistryError, match="unknown KPI"):
        ResultRecord.model_validate(_base(kpi="invented_kpi"))


def test_unit_must_equal_the_registry_unit() -> None:
    with pytest.raises(ValidationError, match="unit mismatch"):
        ResultRecord.model_validate(_base(unit="persons"))


def test_value_above_registry_bound_is_rejected() -> None:
    with pytest.raises(ValidationError, match="above the registry maximum"):
        ResultRecord.model_validate(_base(value=140.0))


def test_value_below_registry_bound_is_rejected() -> None:
    with pytest.raises(ValidationError, match="below the registry minimum"):
        ResultRecord.model_validate(_base(value=-5.0))


def test_density_above_physical_sanity_is_rejected() -> None:
    """docs/04 section 8: density <= 9 persons/m2."""
    with pytest.raises(ValidationError, match="above the registry maximum"):
        ResultRecord.model_validate(
            _base(entity_type="zone", kpi="crowd_density", unit="persons/m2", value=12.0)
        )


def test_lower_above_value_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must be <= value"):
        ResultRecord.model_validate(_base(value=50.0, lower=60.0, upper=70.0, quantile_level=0.8))


def test_upper_below_value_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must be >= value"):
        ResultRecord.model_validate(_base(value=50.0, lower=10.0, upper=20.0, quantile_level=0.8))


def test_interval_without_quantile_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="quantile_level is required"):
        ResultRecord.model_validate(_base(value=50.0, lower=40.0, upper=60.0))


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResultRecord.model_validate(_base(timestamp=datetime(2027, 8, 2, 6, 45)))


def test_nan_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResultRecord.model_validate(_base(value=float("nan")))


def test_current_state_requires_zero_horizon() -> None:
    with pytest.raises(ValidationError, match="requires horizon_min == 0"):
        ResultRecord.model_validate(_base(state="current", horizon_min=45))


def test_inconsistent_delta_is_rejected() -> None:
    with pytest.raises(ValidationError, match="delta must equal"):
        ResultRecord.model_validate(_base(value=55.0, baseline_value=50.0, delta=1.0))


def test_delta_without_baseline_is_rejected() -> None:
    with pytest.raises(ValidationError, match="delta requires baseline_value"):
        ResultRecord.model_validate(_base(delta=5.0))


@pytest.mark.parametrize("level", ["amber", "red", "critical"])
def test_reason_codes_are_mandatory_at_amber_and_worse(level: str) -> None:
    with pytest.raises(ValidationError, match="reason_codes are mandatory"):
        ResultRecord.model_validate(_base(risk_level=level, reason_codes=[]))


def test_green_needs_no_reason_codes() -> None:
    record = ResultRecord.model_validate(_base(value=10.0, risk_level="green"))
    assert record.reason_codes == []


def test_unknown_reason_code_is_rejected() -> None:
    with pytest.raises(RegistryError, match="unknown reason codes"):
        ResultRecord.model_validate(_base(risk_level="amber", reason_codes=["NOT_A_CODE"]))


def test_lowercase_reason_code_is_rejected() -> None:
    with pytest.raises(ValidationError, match="UPPER_SNAKE_CASE"):
        ResultRecord.model_validate(_base(risk_level="amber", reason_codes=["density_high"]))


def test_unknown_resource_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown resource_type"):
        ResultRecord.model_validate(_base(resource_type="water_cannon"))


def test_requires_approval_cannot_be_false() -> None:
    with pytest.raises(ValidationError):
        Recommendation(action="a", rationale="r", requires_approval=False)


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        ResultRecord.model_validate(_base(surprise="field"))


def test_confidence_outside_zero_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResultRecord.model_validate(_base(confidence=1.5))


# ------------------------------------------------------------------ ModelOutput rules
def test_synthetic_source_requires_the_synthetic_flag() -> None:
    with pytest.raises(ValidationError, match="is_synthetic must be true"):
        ModelOutput.model_validate(_output(is_synthetic=False))


def test_degraded_requires_a_warning() -> None:
    with pytest.raises(ValidationError, match="requires at least one warning"):
        ModelOutput.model_validate(_output(status="degraded"))


def test_degraded_with_a_warning_is_accepted() -> None:
    output = ModelOutput.model_validate(_output(status="degraded", warnings=["stub used"]))
    assert output.warnings == ["stub used"]


def test_unknown_model_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelOutput.model_validate(_output(model_id="M26"))


def test_non_semver_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelOutput.model_validate(_output(model_version="v1"))


def test_unknown_scenario_id_shape_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelOutput.model_validate(_output(scenario_id="S99"))


def test_scenario_response_may_not_hold_forecast_records() -> None:
    """A /scenario response uses state 'scenario' (docs/02 section 4)."""
    with pytest.raises(ValidationError, match="must use state 'scenario'"):
        ModelOutput.model_validate(_output(scenario_id="S02", results=[_base(state="forecast")]))


def test_scenario_response_accepts_scenario_and_current_records() -> None:
    output = ModelOutput.model_validate(
        _output(
            scenario_id="S02",
            results=[_base(state="scenario"), _base(state="current", horizon_min=0)],
        )
    )
    assert {r.state.value for r in output.results} == {"scenario", "current"}


def test_model_namespace_fields_do_not_warn() -> None:
    """protected_namespaces=() must be set, or pydantic warns on model_id/model_name."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ModelOutput.model_validate(_output())
        UpstreamRef(model_id="M01", source="stub")


def test_convenience_accessors() -> None:
    output = ModelOutput.model_validate(_output(upstream=[{"model_id": "M01", "source": "sample"}]))
    assert output.kpis_present() == {"crush_risk_score"}
    assert output.entities_present() == {"Z01"}
    assert len(output.records_for("crush_risk_score")) == 1
    assert output.upstream_sources() == {"M01": "sample"}


# ------------------------------------------------------------------ requests
def test_predict_request_defaults_are_all_optional() -> None:
    request = PredictRequest()
    assert request.as_of is None
    assert request.horizon_min is None
    assert request.include_current is True


def test_scenario_request_defaults() -> None:
    request = ScenarioRequest()
    assert request.scenario_id == "S01"
    assert request.compare_to_baseline is True


def test_inline_upstream_key_must_match_payload() -> None:
    payload = _output(model_id="M01", model_name="Footfall Forecast")
    payload["results"] = [
        _base(
            entity_type="zone",
            entity_id="Z01",
            kpi="expected_footfall",
            unit="persons/hr",
            value=1000.0,
        )
    ]
    with pytest.raises(ValidationError, match="does not match its key"):
        PredictRequest.model_validate({"upstream": {"M02": payload}})


def test_inline_upstream_with_matching_key_is_accepted() -> None:
    payload = _output(model_id="M01", model_name="Footfall Forecast")
    payload["results"] = [
        _base(
            entity_type="zone",
            entity_id="Z01",
            kpi="expected_footfall",
            unit="persons/hr",
            value=1000.0,
        )
    ]
    request = PredictRequest.model_validate({"upstream": {"M01": payload}})
    assert request.upstream is not None and "M01" in request.upstream


def test_non_model_id_upstream_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"upstream": {"footfall": _output()}})


def test_as_scenario_widens_a_predict_request() -> None:
    request = PredictRequest(horizon_min=60, entity_ids=["Z01"])
    widened = request.as_scenario("S02")
    assert widened.scenario_id == "S02"
    assert widened.horizon_min == 60
    assert widened.entity_ids == ["Z01"]


# ------------------------------------------------------------------ metadata
def test_metadata_unit_must_match_registry() -> None:
    with pytest.raises(ValidationError, match="metadata unit mismatch"):
        Metadata.model_validate(
            {
                "model_id": "M01",
                "model_name": "Footfall Forecast",
                "model_version": "0.1.0",
                "question": "How many people will arrive?",
                "engine": "forecast",
                "method_summary": "Chronos-2 zero-shot",
                "kpis": [{"kpi": "expected_footfall", "unit": "persons"}],
            }
        )


def test_metadata_example_from_docs_validates() -> None:
    meta = Metadata.model_validate(
        {
            "model_id": "M01",
            "model_name": "Footfall Forecast",
            "model_version": "0.1.0",
            "question": "How many people will arrive, by zone and time?",
            "engine": "forecast",
            "method_summary": "Chronos-2 zero-shot with covariates; LightGBM baseline",
            "kpis": [{"kpi": "expected_footfall", "unit": "persons/hr", "description": "arrivals"}],
            "upstream": ["M21"],
            "inputs": ["footfall_15min", "event_calendar", "weather_hourly"],
            "horizons_min": [15, 60, 180],
            "scenarios_supported": ["S01", "S02", "S03", "S04", "S05"],
            "limitations": ["Synthetic training context"],
            "license_notes": ["Chronos-2 weights: Apache-2.0"],
        }
    )
    assert meta.engine.value == "forecast"
    assert meta.scenarios_supported == ["S01", "S02", "S03", "S04", "S05"]
