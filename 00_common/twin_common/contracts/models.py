"""The response and request contracts from docs/02 sections 4 and 6.

Every model service in this repository must answer with :class:`ModelOutput` and nothing
else. The validators here enforce the field rules of docs/02 section 4 so a contract
violation fails at the boundary instead of reaching the frontend.

Pydantic note: fields named ``model_id`` / ``model_name`` / ``model_version`` collide with
the protected ``model_`` namespace of pydantic v2. The contract mandates those names, so
every affected class sets ``protected_namespaces=()``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from . import registry as reg
from .enums import (
    REASON_REQUIRED_LEVELS,
    DataSource,
    Engine,
    EntityType,
    RiskLevel,
    State,
    Status,
    UpstreamSource,
)
from .ids import (
    MODEL_ID_RE,
    REASON_CODE_RE,
    SCENARIO_ID_RE,
    SEMVER_RE,
    validate_entity_id,
)

#: docs/02 section 4 - all timestamps are tz-aware ISO 8601 in Asia/Kolkata.
IST = timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")

SCHEMA_VERSION = "1.0"

#: Tolerance for the delta == value - baseline_value check.
_DELTA_TOL = 1e-6
#: Tolerance applied to registry bounds so float noise does not fail a valid value.
_BOUND_TOL = 1e-9

ModelId = Annotated[str, Field(pattern=MODEL_ID_RE.pattern, examples=["M01"])]
ScenarioId = Annotated[str, Field(pattern=SCENARIO_ID_RE.pattern, examples=["S01"])]
SemVer = Annotated[str, Field(pattern=SEMVER_RE.pattern, examples=["0.1.0"])]


def to_ist(value: datetime) -> datetime:
    """Convert an aware datetime to the +05:30 offset used across the project."""
    return value.astimezone(IST)


def now_ist() -> datetime:
    return datetime.now(IST)


def new_run_id() -> str:
    return str(uuid.uuid4())


# --------------------------------------------------------------------------- sub-models
class UpstreamRef(BaseModel):
    """Provenance of one consumed upstream output (docs/02 sections 4 and 7)."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: ModelId
    run_id: str | None = None
    source: UpstreamSource
    generated_at: AwareDatetime | None = None

    @field_validator("generated_at")
    @classmethod
    def _normalize_tz(cls, v: datetime | None) -> datetime | None:
        return None if v is None else to_ist(v)


class Recommendation(BaseModel):
    """An action proposed to an operator.

    ``requires_approval`` is typed ``Literal[True]``: docs/02 section 4 states it is ALWAYS
    true in this project, so no model can emit an auto-executing recommendation.
    """

    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1, description="Imperative action, e.g. 'stage ambulances'")
    resource_type: str | None = None
    quantity: float | None = None
    target_entity_id: str | None = None
    from_entity_id: str | None = None
    rationale: str = Field(min_length=1, description="Why, in operator-readable words")
    requires_approval: Literal[True] = True

    @field_validator("resource_type")
    @classmethod
    def _known_resource_type(cls, v: str | None) -> str | None:
        if v is not None and not reg.resource_type_exists(v):
            raise ValueError(
                f"unknown resource_type {v!r}; must be one of the 15 types in "
                f"docs/05 section 5 / resources.yaml"
            )
        return v


class ResultRecord(BaseModel):
    """One KPI value for one entity at one timestamp (docs/02 section 4)."""

    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType
    entity_id: str = Field(min_length=1)
    zone_id: str | None = Field(
        default=None, description="Parent zone when the entity sits inside one, else null"
    )
    kpi: str
    timestamp: AwareDatetime
    horizon_min: int = Field(ge=0, description="0 for current state")
    state: State
    value: float
    unit: str
    lower: float | None = None
    upper: float | None = None
    quantile_level: float | None = Field(default=None, gt=0.0, lt=1.0)
    baseline_value: float | None = None
    delta: float | None = None
    risk_level: RiskLevel | None = None
    reason_codes: list[str] = Field(default_factory=list)
    resource_type: str | None = None
    recommendation: Recommendation | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)

    # -- field-level checks ------------------------------------------------
    @field_validator("timestamp")
    @classmethod
    def _normalize_tz(cls, v: datetime) -> datetime:
        return to_ist(v)

    @field_validator("value")
    @classmethod
    def _finite_value(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("value must be finite (no NaN or inf)")
        return v

    @field_validator("kpi")
    @classmethod
    def _known_kpi(cls, v: str) -> str:
        reg.kpi_info(v)  # raises RegistryError with a helpful message
        return v

    @field_validator("reason_codes")
    @classmethod
    def _known_reason_codes(cls, v: list[str]) -> list[str]:
        bad_case = [c for c in v if not REASON_CODE_RE.fullmatch(c)]
        if bad_case:
            raise ValueError(f"reason codes must be UPPER_SNAKE_CASE, got {bad_case}")
        reg.validate_reason_codes(v)
        return v

    @field_validator("resource_type")
    @classmethod
    def _known_resource_type(cls, v: str | None) -> str | None:
        if v is not None and not reg.resource_type_exists(v):
            raise ValueError(f"unknown resource_type {v!r} (docs/05 section 5)")
        return v

    # -- cross-field checks ------------------------------------------------
    @model_validator(mode="after")
    def _check_entity_id(self) -> ResultRecord:
        validate_entity_id(self.entity_type, self.entity_id)
        if self.zone_id is not None:
            validate_entity_id(EntityType.ZONE, self.zone_id)
        return self

    @model_validator(mode="after")
    def _check_unit_matches_registry(self) -> ResultRecord:
        expected = reg.kpi_unit(self.kpi)
        if self.unit != expected:
            raise ValueError(
                f"unit mismatch for KPI {self.kpi!r}: got {self.unit!r}, "
                f"kpi_registry.yaml says {expected!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_bounds(self) -> ResultRecord:
        lo, hi = reg.kpi_info(self.kpi).bounds
        if lo is not None and self.value < lo - _BOUND_TOL:
            raise ValueError(
                f"{self.kpi} value {self.value} below the registry minimum {lo} "
                f"for unit {self.unit!r}"
            )
        if hi is not None and self.value > hi + _BOUND_TOL:
            raise ValueError(
                f"{self.kpi} value {self.value} above the registry maximum {hi} "
                f"for unit {self.unit!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_interval(self) -> ResultRecord:
        if self.lower is not None and self.lower > self.value + _BOUND_TOL:
            raise ValueError(f"lower ({self.lower}) must be <= value ({self.value})")
        if self.upper is not None and self.upper < self.value - _BOUND_TOL:
            raise ValueError(f"upper ({self.upper}) must be >= value ({self.value})")
        if (self.lower is not None or self.upper is not None) and self.quantile_level is None:
            raise ValueError("quantile_level is required when lower or upper is present")
        return self

    @model_validator(mode="after")
    def _check_state_horizon(self) -> ResultRecord:
        if self.state is State.CURRENT and self.horizon_min != 0:
            raise ValueError(f"state 'current' requires horizon_min == 0, got {self.horizon_min}")
        return self

    @model_validator(mode="after")
    def _check_delta(self) -> ResultRecord:
        if self.baseline_value is not None and self.delta is not None:
            expected = self.value - self.baseline_value
            if abs(self.delta - expected) > _DELTA_TOL:
                raise ValueError(
                    f"delta must equal value - baseline_value "
                    f"({self.value} - {self.baseline_value} = {expected}), got {self.delta}"
                )
        if self.delta is not None and self.baseline_value is None:
            raise ValueError("delta requires baseline_value")
        return self

    @model_validator(mode="after")
    def _check_reason_codes_required(self) -> ResultRecord:
        if self.risk_level in REASON_REQUIRED_LEVELS and not self.reason_codes:
            raise ValueError(
                f"reason_codes are mandatory when risk_level is {self.risk_level.value!r} "
                f"(docs/02 section 4); KPI {self.kpi} on {self.entity_id}"
            )
        return self


# ------------------------------------------------------------------------ main response
class ModelOutput(BaseModel):
    """The response body of /predict and /scenario. The only shape any model may return."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    model_id: ModelId
    model_name: str = Field(min_length=1)
    model_version: SemVer
    run_id: str = Field(default_factory=new_run_id)
    generated_at: AwareDatetime = Field(default_factory=now_ist)
    as_of: AwareDatetime
    scenario_id: ScenarioId = "S01"
    scenario_overrides: dict[str, Any] = Field(default_factory=dict)
    data_source: DataSource
    is_synthetic: bool
    status: Status = Status.OK
    warnings: list[str] = Field(default_factory=list)
    upstream: list[UpstreamRef] = Field(default_factory=list)
    results: list[ResultRecord] = Field(default_factory=list)

    @field_validator("generated_at", "as_of")
    @classmethod
    def _normalize_tz(cls, v: datetime) -> datetime:
        return to_ist(v)

    @model_validator(mode="after")
    def _synthetic_flag_consistent(self) -> ModelOutput:
        if self.data_source is DataSource.SYNTHETIC and not self.is_synthetic:
            raise ValueError("is_synthetic must be true when data_source is 'synthetic'")
        return self

    @model_validator(mode="after")
    def _degraded_needs_warning(self) -> ModelOutput:
        if self.status is Status.DEGRADED and not self.warnings:
            raise ValueError(
                "status 'degraded' requires at least one warning explaining the fallback "
                "(docs/02 section 4)"
            )
        return self

    @model_validator(mode="after")
    def _scenario_state_consistent(self) -> ModelOutput:
        """A /scenario response marks its records ``state: scenario`` (docs/02 section 4).

        Records that describe observed current state are still allowed, so only forecast
        records are rejected in a non-baseline scenario response.
        """
        if self.scenario_id != "S01" and any(r.state is State.FORECAST for r in self.results):
            raise ValueError(
                "results of a non-baseline scenario response must use state 'scenario' "
                "(or 'current' for observed values), not 'forecast'"
            )
        return self

    # -- convenience -------------------------------------------------------
    def kpis_present(self) -> set[str]:
        return {r.kpi for r in self.results}

    def entities_present(self) -> set[str]:
        return {r.entity_id for r in self.results}

    def records_for(self, kpi: str) -> list[ResultRecord]:
        return [r for r in self.results if r.kpi == kpi]

    def upstream_sources(self) -> dict[str, UpstreamSource]:
        return {u.model_id: u.source for u in self.upstream}


# -------------------------------------------------------------------------- requests
class PredictRequest(BaseModel):
    """Body of POST /predict. Every field is optional; defaults come from config.yaml."""

    model_config = ConfigDict(extra="forbid")

    as_of: AwareDatetime | None = Field(default=None, description="Defaults to config demo_now")
    horizon_min: int | None = Field(
        default=None, ge=0, description="Defaults to config default_horizon_min"
    )
    entity_ids: list[str] | None = Field(default=None, description="Defaults to all entities")
    kpis: list[str] | None = Field(default=None, description="Defaults to all KPIs owned")
    include_current: bool = True
    upstream: dict[str, ModelOutput] | None = Field(
        default=None, description="Inline upstream outputs, keyed by model ID"
    )

    @field_validator("as_of")
    @classmethod
    def _normalize_tz(cls, v: datetime | None) -> datetime | None:
        return None if v is None else to_ist(v)

    @field_validator("upstream")
    @classmethod
    def _upstream_keys_are_model_ids(
        cls, v: dict[str, ModelOutput] | None
    ) -> dict[str, ModelOutput] | None:
        if v is None:
            return None
        bad = [k for k in v if not MODEL_ID_RE.fullmatch(k)]
        if bad:
            raise ValueError(f"upstream keys must be model IDs like 'M01', got {bad}")
        mismatched = [k for k, out in v.items() if out.model_id != k]
        if mismatched:
            raise ValueError(
                f"inline upstream payload does not match its key for {mismatched} "
                f"(the ModelOutput.model_id must equal the dict key)"
            )
        return v

    def as_scenario(self, scenario_id: str = "S01", **kwargs: Any) -> ScenarioRequest:
        """Widen this request into a ScenarioRequest, keeping every field."""
        return ScenarioRequest(**self.model_dump(), scenario_id=scenario_id, **kwargs)


class ScenarioRequest(PredictRequest):
    """Body of POST /scenario: a PredictRequest plus the what-if selection."""

    scenario_id: ScenarioId = "S01"
    overrides: dict[str, Any] = Field(
        default_factory=dict, description="Merged on top of the scenario overrides"
    )
    compare_to_baseline: bool = True


# -------------------------------------------------------------------------- metadata
class KpiDescriptor(BaseModel):
    """One entry of Metadata.kpis."""

    model_config = ConfigDict(extra="forbid")

    kpi: str
    unit: str
    description: str = ""

    @model_validator(mode="after")
    def _matches_registry(self) -> KpiDescriptor:
        expected = reg.kpi_unit(self.kpi)
        if self.unit != expected:
            raise ValueError(
                f"metadata unit mismatch for {self.kpi!r}: got {self.unit!r}, "
                f"registry says {expected!r}"
            )
        return self


class Metadata(BaseModel):
    """Response of GET /metadata (docs/02 section 6)."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: ModelId
    model_name: str = Field(min_length=1)
    model_version: SemVer
    question: str = Field(min_length=1, description="The operational question answered")
    engine: Engine
    method_summary: str = Field(min_length=1)
    kpis: list[KpiDescriptor]
    upstream: list[ModelId] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    horizons_min: list[int] = Field(default_factory=list)
    scenarios_supported: list[ScenarioId] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    license_notes: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response of GET /health (docs/02 section 6)."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    status: Literal["ok"] = "ok"
    model_id: ModelId
    model_version: SemVer
    data_source: DataSource


class ErrorResponse(BaseModel):
    """Body returned for a 500 (docs/02 section 6)."""

    status: Literal["error"] = "error"
    detail: str
