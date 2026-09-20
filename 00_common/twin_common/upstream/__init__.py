"""Upstream output resolution, in the exact order of docs/02 section 7.

For each upstream model ID in ``config.yaml -> upstream``:

1. ``request.upstream[ID]`` (inline) - must validate as ModelOutput
2. env var ``UPSTREAM_<ID>_URL`` - POST /scenario when a scenario is requested, else /predict
3. ``data/sample_upstream/<ID>__<scenario_id>.json``
4. ``data/sample_upstream/<ID>__S01.json`` with a "baseline upstream used for scenario" warning
5. a stub generated from the synthetic world (status degraded)

The chosen source is recorded in ``ModelOutput.upstream``. The stub generator lives in
``twin_common.synthetic.stubs`` and is injected through :func:`set_stub_provider`, so this
module stays importable before Phase 3 builds it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ..contracts.enums import UpstreamSource
from ..contracts.models import ModelOutput, UpstreamRef
from ..errors import UpstreamError
from ..logging import get_logger

log = get_logger(__name__)

#: Environment variable pattern for step 2.
URL_ENV_TEMPLATE = "UPSTREAM_{model_id}_URL"

BASELINE_SCENARIO = "S01"


class StubProvider(Protocol):
    """Builds a contract-valid placeholder output for a model (docs/02 section 7 step 5)."""

    def __call__(
        self,
        model_id: str,
        *,
        as_of: datetime,
        scenario_id: str,
        entity_ids: list[str] | None = None,
    ) -> ModelOutput: ...


_STUB_PROVIDER: StubProvider | None = None


def set_stub_provider(provider: StubProvider | None) -> None:
    """Register the stub generator. Phase 3 wires in ``twin_common.synthetic.stubs``."""
    global _STUB_PROVIDER
    _STUB_PROVIDER = provider


def get_stub_provider() -> StubProvider | None:
    return _STUB_PROVIDER


@dataclass(slots=True)
class Resolution:
    """One resolved upstream output plus the provenance to report."""

    model_id: str
    output: ModelOutput
    ref: UpstreamRef
    warning: str | None = None

    @property
    def source(self) -> UpstreamSource:
        return self.ref.source


def sample_path(sample_dir: Path, model_id: str, scenario_id: str) -> Path:
    """``<sample_dir>/<ID>__<Sxx>.json`` (docs/02 section 7 steps 3 and 4)."""
    return Path(sample_dir) / f"{model_id}__{scenario_id}.json"


def _load_sample(path: Path) -> ModelOutput:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelOutput.model_validate(payload)


def _fetch_url(
    model_id: str,
    base_url: str,
    *,
    as_of: datetime,
    scenario_id: str,
    horizon_min: int | None,
    entity_ids: list[str] | None,
    timeout_s: float,
) -> ModelOutput:
    """Call a running upstream service. Imported lazily so httpx stays optional at import."""
    import httpx

    is_scenario = scenario_id != BASELINE_SCENARIO
    endpoint = "/scenario" if is_scenario else "/predict"
    body: dict[str, Any] = {"as_of": as_of.isoformat()}
    if horizon_min is not None:
        body["horizon_min"] = horizon_min
    if entity_ids:
        body["entity_ids"] = entity_ids
    if is_scenario:
        body["scenario_id"] = scenario_id
    url = base_url.rstrip("/") + endpoint
    response = httpx.post(url, json=body, timeout=timeout_s)
    response.raise_for_status()
    return ModelOutput.model_validate(response.json())


def resolve_one(
    model_id: str,
    *,
    as_of: datetime,
    scenario_id: str = BASELINE_SCENARIO,
    inline: Mapping[str, ModelOutput] | None = None,
    sample_dir: Path | str | None = None,
    horizon_min: int | None = None,
    entity_ids: list[str] | None = None,
    timeout_s: float = 5.0,
    allow_stub: bool = True,
) -> Resolution:
    """Resolve one upstream model through the five-step order.

    Raises UpstreamError only when every step failed, which means the model genuinely
    cannot answer.
    """
    # 1 - inline
    if inline and model_id in inline:
        output = inline[model_id]
        log.debug("upstream %s resolved inline", model_id)
        return Resolution(
            model_id=model_id,
            output=output,
            ref=UpstreamRef(
                model_id=model_id,
                run_id=output.run_id,
                source=UpstreamSource.INLINE,
                generated_at=output.generated_at,
            ),
        )

    # 2 - running service
    env_key = URL_ENV_TEMPLATE.format(model_id=model_id)
    base_url = os.environ.get(env_key)
    if base_url:
        try:
            output = _fetch_url(
                model_id,
                base_url,
                as_of=as_of,
                scenario_id=scenario_id,
                horizon_min=horizon_min,
                entity_ids=entity_ids,
                timeout_s=timeout_s,
            )
            log.info("upstream %s resolved from %s", model_id, base_url)
            return Resolution(
                model_id=model_id,
                output=output,
                ref=UpstreamRef(
                    model_id=model_id,
                    run_id=output.run_id,
                    source=UpstreamSource.URL,
                    generated_at=output.generated_at,
                ),
            )
        except Exception as exc:
            log.warning("upstream %s at %s failed (%s); falling back", model_id, base_url, exc)

    # 3 and 4 - sample files
    if sample_dir is not None:
        exact = sample_path(Path(sample_dir), model_id, scenario_id)
        if exact.is_file():
            output = _load_sample(exact)
            log.debug("upstream %s resolved from sample %s", model_id, exact.name)
            return Resolution(
                model_id=model_id,
                output=output,
                ref=UpstreamRef(
                    model_id=model_id,
                    run_id=output.run_id,
                    source=UpstreamSource.SAMPLE,
                    generated_at=output.generated_at,
                ),
            )
        baseline = sample_path(Path(sample_dir), model_id, BASELINE_SCENARIO)
        if scenario_id != BASELINE_SCENARIO and baseline.is_file():
            output = _load_sample(baseline)
            return Resolution(
                model_id=model_id,
                output=output,
                ref=UpstreamRef(
                    model_id=model_id,
                    run_id=output.run_id,
                    source=UpstreamSource.SAMPLE,
                    generated_at=output.generated_at,
                ),
                warning=(
                    f"baseline upstream used for scenario: {model_id} has no {scenario_id} sample"
                ),
            )

    # 5 - stub. The default provider installs itself on first use, so no caller has to
    # remember to wire it up; docs/02 section 7 makes the stub the guaranteed last resort,
    # and a model that forgot to install it would fail instead of degrading.
    if allow_stub and _STUB_PROVIDER is None:
        try:
            from ..synthetic.stubs import install as install_default_stubs

            install_default_stubs()
        except Exception as exc:  # a missing world is not a reason to crash here
            log.debug("default stub provider unavailable: %s", exc)

    if allow_stub and _STUB_PROVIDER is not None:
        output = _STUB_PROVIDER(
            model_id, as_of=as_of, scenario_id=scenario_id, entity_ids=entity_ids
        )
        log.info("upstream %s resolved from a generated stub", model_id)
        return Resolution(
            model_id=model_id,
            output=output,
            ref=UpstreamRef(
                model_id=model_id,
                run_id=output.run_id,
                source=UpstreamSource.STUB,
                generated_at=output.generated_at,
            ),
            warning=f"stub output used for upstream {model_id}",
        )

    raise UpstreamError(
        f"could not resolve upstream {model_id}: no inline payload, no {env_key}, no sample "
        f"under {sample_dir}, and no stub provider registered"
    )


def resolve_all(
    model_ids: Iterable[str],
    *,
    as_of: datetime,
    scenario_id: str = BASELINE_SCENARIO,
    inline: Mapping[str, ModelOutput] | None = None,
    sample_dir: Path | str | None = None,
    horizon_min: int | None = None,
    entity_ids: list[str] | None = None,
    timeout_s: float = 5.0,
    allow_stub: bool = True,
) -> dict[str, Resolution]:
    """Resolve every declared upstream. Keys are model IDs, in the order given."""
    out: dict[str, Resolution] = {}
    for model_id in model_ids:
        out[model_id] = resolve_one(
            model_id,
            as_of=as_of,
            scenario_id=scenario_id,
            inline=inline,
            sample_dir=sample_dir,
            horizon_min=horizon_min,
            entity_ids=entity_ids,
            timeout_s=timeout_s,
            allow_stub=allow_stub,
        )
    return out


def values_by_entity(
    resolution: Resolution | ModelOutput,
    kpi: str,
    *,
    horizon_min: int | None = None,
) -> dict[str, float]:
    """Pull one KPI out of an upstream output as {entity_id: value}.

    When several records share an entity, the one closest to ``horizon_min`` wins; with
    ``horizon_min=None`` the last record in document order wins.
    """
    output = resolution.output if isinstance(resolution, Resolution) else resolution
    picked: dict[str, tuple[int, float]] = {}
    for record in output.results:
        if record.kpi != kpi:
            continue
        distance = 0 if horizon_min is None else abs(record.horizon_min - horizon_min)
        current = picked.get(record.entity_id)
        if current is None or distance <= current[0]:
            picked[record.entity_id] = (distance, record.value)
    return {entity: value for entity, (_, value) in picked.items()}


def series_for_entity(
    resolution: Resolution | ModelOutput,
    kpi: str,
    entity_id: str,
) -> list[tuple[datetime, float]]:
    """Time series of one KPI for one entity, sorted by timestamp."""
    output = resolution.output if isinstance(resolution, Resolution) else resolution
    points = [
        (r.timestamp, r.value) for r in output.results if r.kpi == kpi and r.entity_id == entity_id
    ]
    return sorted(points, key=lambda item: item[0])
