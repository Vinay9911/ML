"""OutputBuilder - the one way a model assembles its :class:`ModelOutput`.

Why a builder rather than constructing ModelOutput directly: it fills the unit from the KPI
registry, derives ``risk_level`` from the band tables, computes ``delta`` against a baseline,
and records upstream provenance. That keeps every model honest about the parts of
docs/02 section 4 that are easy to forget.

Typical use inside ``Model.predict``::

    b = OutputBuilder.from_config(self.config, as_of=as_of, scenario_id="S01")
    b.add_upstream(ref)
    b.add("zone", "Z01", "crush_risk_score", ts, 72.4,
          horizon_min=45, state=State.FORECAST,
          lower=61.0, upper=81.5, quantile_level=0.8,
          reason_codes=["DENSITY_RISING_FAST"], details={"density_p_m2": 4.3})
    return b.build()
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from ..config import Config
from ..contracts import registry as reg
from ..contracts.enums import DataSource, EntityType, RiskLevel, State, Status, UpstreamSource
from ..contracts.models import (
    ModelOutput,
    Recommendation,
    ResultRecord,
    UpstreamRef,
    new_run_id,
    now_ist,
    to_ist,
)
from ..errors import ContractError
from ..logging import get_logger
from .risk import risk_level_for_kpi

log = get_logger(__name__)

#: Reason code every synthetic-data output should carry at least once (docs/05 section 4).
SYNTHETIC_REASON = "SYNTHETIC_DATA"
#: Reason code to attach when an upstream fallback was used.
UPSTREAM_FALLBACK_REASON = "UPSTREAM_FALLBACK"


class OutputBuilder:
    """Accumulates result records and produces a validated ModelOutput."""

    def __init__(
        self,
        *,
        model_id: str,
        model_name: str,
        model_version: str,
        as_of: datetime,
        data_source: DataSource | str = DataSource.SYNTHETIC,
        scenario_id: str = "S01",
        scenario_overrides: dict[str, Any] | None = None,
        is_synthetic: bool | None = None,
        run_id: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.model_version = model_version
        self.as_of = to_ist(as_of)
        self.data_source = DataSource(data_source)
        self.scenario_id = scenario_id
        self.scenario_overrides: dict[str, Any] = dict(scenario_overrides or {})
        self.is_synthetic = (
            self.data_source is DataSource.SYNTHETIC if is_synthetic is None else is_synthetic
        )
        self.run_id = run_id or new_run_id()
        self._results: list[ResultRecord] = []
        self._warnings: list[str] = []
        self._upstream: list[UpstreamRef] = []
        self._status: Status = Status.OK

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        as_of: datetime,
        scenario_id: str = "S01",
        scenario_overrides: dict[str, Any] | None = None,
        is_synthetic: bool | None = None,
    ) -> Self:
        """Build from a model ``config.yaml`` (docs/02 section 8 keys)."""
        return cls(
            model_id=config.require_str("model_id"),
            model_name=config.require_str("model_name"),
            model_version=config.require_str("model_version"),
            as_of=as_of,
            data_source=config.require_str("data_source"),
            scenario_id=scenario_id,
            scenario_overrides=scenario_overrides,
            is_synthetic=is_synthetic,
        )

    # ------------------------------------------------------------------ status/warnings
    def warn(self, message: str, *, degrade: bool = True) -> Self:
        """Record a warning. By default this also flips the status to ``degraded``.

        docs/02 section 4: use ``degraded`` plus a warning whenever a fallback was used.
        """
        if message not in self._warnings:
            self._warnings.append(message)
        if degrade and self._status is Status.OK:
            self._status = Status.DEGRADED
        return self

    def mark_degraded(self, message: str) -> Self:
        return self.warn(message, degrade=True)

    def mark_insensitive(self, scenario_id: str) -> Self:
        """The docs/05 section 1.2 response for a scenario a model cannot react to."""
        return self.warn(f"insensitive to {scenario_id}")

    @property
    def status(self) -> Status:
        return self._status

    # ------------------------------------------------------------------ upstream
    def add_upstream(
        self,
        model_id: str,
        source: UpstreamSource | str,
        *,
        run_id: str | None = None,
        generated_at: datetime | None = None,
    ) -> Self:
        """Record where one upstream output came from (docs/02 section 7)."""
        ref = UpstreamRef(
            model_id=model_id,
            run_id=run_id,
            source=UpstreamSource(source),
            generated_at=generated_at,
        )
        self._upstream.append(ref)
        if ref.source in (UpstreamSource.SAMPLE, UpstreamSource.STUB):
            self.warn(
                f"upstream {model_id} resolved from {ref.source.value}; "
                f"values are not live model output"
            )
        return self

    def add_upstream_ref(self, ref: UpstreamRef) -> Self:
        self._upstream.append(ref)
        if ref.source in (UpstreamSource.SAMPLE, UpstreamSource.STUB):
            self.warn(
                f"upstream {ref.model_id} resolved from {ref.source.value}; "
                f"values are not live model output"
            )
        return self

    # ------------------------------------------------------------------ results
    def add(
        self,
        entity_type: EntityType | str,
        entity_id: str,
        kpi: str,
        timestamp: datetime,
        value: float,
        *,
        horizon_min: int = 0,
        state: State | str = State.CURRENT,
        zone_id: str | None = None,
        unit: str | None = None,
        lower: float | None = None,
        upper: float | None = None,
        quantile_level: float | None = None,
        baseline_value: float | None = None,
        risk_level: RiskLevel | str | None = None,
        reason_codes: list[str] | None = None,
        resource_type: str | None = None,
        recommendation: Recommendation | None = None,
        confidence: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> ResultRecord:
        """Append one result record and return it.

        ``unit`` defaults to the KPI registry unit, so a model cannot drift from the
        contract. ``risk_level`` defaults to the band derived from ``risk_bands.yaml``;
        pass an explicit level only when the model genuinely overrides it (for example the
        M03 sustained-exposure rule).

        ``delta`` is computed whenever ``baseline_value`` is given.
        """
        resolved_unit = unit if unit is not None else reg.kpi_unit(kpi)
        resolved_risk = (
            RiskLevel(risk_level) if risk_level is not None else risk_level_for_kpi(kpi, value)
        )
        delta = None if baseline_value is None else value - baseline_value
        codes = list(reason_codes or [])
        record = ResultRecord(
            entity_type=EntityType(entity_type),
            entity_id=entity_id,
            zone_id=zone_id,
            kpi=kpi,
            timestamp=timestamp,
            horizon_min=horizon_min,
            state=State(state),
            value=value,
            unit=resolved_unit,
            lower=lower,
            upper=upper,
            quantile_level=quantile_level,
            baseline_value=baseline_value,
            delta=delta,
            risk_level=resolved_risk,
            reason_codes=codes,
            resource_type=resource_type,
            recommendation=recommendation,
            confidence=confidence,
            details=dict(details or {}),
        )
        self._results.append(record)
        return record

    def extend(self, records: list[ResultRecord]) -> Self:
        self._results.extend(records)
        return self

    @property
    def results(self) -> list[ResultRecord]:
        return list(self._results)

    def __len__(self) -> int:
        return len(self._results)

    # ------------------------------------------------------------------ baseline compare
    def apply_baseline(self, baseline: ModelOutput) -> Self:
        """Fill ``baseline_value`` and ``delta`` on every record from a baseline run.

        Records are matched on (entity_id, kpi, timestamp, horizon_min). Unmatched records
        keep a null baseline, which is the honest answer when the baseline did not cover
        that entity.
        """
        index: dict[tuple[str, str, datetime, int], float] = {
            (r.entity_id, r.kpi, r.timestamp, r.horizon_min): r.value for r in baseline.results
        }
        matched = 0
        rebuilt: list[ResultRecord] = []
        for record in self._results:
            key = (record.entity_id, record.kpi, record.timestamp, record.horizon_min)
            base_value = index.get(key)
            if base_value is None:
                rebuilt.append(record)
                continue
            matched += 1
            rebuilt.append(
                record.model_copy(
                    update={"baseline_value": base_value, "delta": record.value - base_value}
                )
            )
        self._results = rebuilt
        if matched == 0 and self._results:
            self.warn("no baseline records matched; deltas are unavailable")
        log.debug("applied baseline to %d/%d records", matched, len(self._results))
        return self

    # ------------------------------------------------------------------ build
    def build(self, *, status: Status | str | None = None) -> ModelOutput:
        """Validate and return the ModelOutput.

        Raises ContractError when the accumulated state cannot make a valid output, so a
        model fails loudly at its own boundary rather than emitting a broken payload.
        """
        final_status = Status(status) if status is not None else self._status
        try:
            return ModelOutput(
                model_id=self.model_id,
                model_name=self.model_name,
                model_version=self.model_version,
                run_id=self.run_id,
                generated_at=now_ist(),
                as_of=self.as_of,
                scenario_id=self.scenario_id,
                scenario_overrides=self.scenario_overrides,
                data_source=self.data_source,
                is_synthetic=self.is_synthetic,
                status=final_status,
                warnings=list(self._warnings),
                upstream=list(self._upstream),
                results=list(self._results),
            )
        except Exception as exc:  # pydantic ValidationError and friends
            raise ContractError(
                f"{self.model_id} produced an output that violates docs/02 section 4: {exc}"
            ) from exc
