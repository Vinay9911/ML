"""``create_app(ModelClass)`` - the FastAPI factory behind every model service.

``api.py`` in a model folder holds exactly one statement (docs/02 section 2)::

    from twin_common.api import create_app
    from src.model import Model

    app = create_app(Model)

Endpoints, per docs/02 section 6:

===========  ================  ==================  ==============
Method       Path              Body                Returns
===========  ================  ==================  ==============
GET          /health           -                   HealthResponse
GET          /metadata         -                   Metadata
GET          /inputs           -                   InputsResponse
POST         /predict          PredictRequest      ModelOutput
POST         /scenario         ScenarioRequest     ModelOutput
===========  ================  ==================  ==============

``/inputs`` is an addition to docs/02 section 6, not part of the frontend contract. It exists
so a reader can see WHAT GOES IN to a model, not just what comes out: every declared table
with its real schema and a real sample of the rows that drive the answer, the full parameter
block, and the upstream models. Nothing else depends on it, so a change here cannot break a
consumer of the four contract endpoints.

Error mapping: 422 for validation (FastAPI default), 404 for an unknown entity or scenario,
500 with ``{"status": "error", "detail": ...}`` for anything else. CORS allows all origins
because this is a demo.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..contracts.models import (
    HealthResponse,
    Metadata,
    ModelOutput,
    PredictRequest,
    ScenarioRequest,
)
from ..errors import (
    ContractError,
    EngineError,
    TwinError,
    UnknownEntityError,
    UnknownScenarioError,
    UpstreamError,
)
from ..logging import get_logger
from ..model import TwinModel
from .inputs import describe_inputs

log = get_logger(__name__)

#: Response-time target from docs/02 section 6, logged as a warning when exceeded.
SLOW_RESPONSE_S = 10.0


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"status": "error", "detail": detail})


def create_app(
    model_class: type[TwinModel],
    *,
    root: str | Path | None = None,
    title: str | None = None,
) -> FastAPI:
    """Build the FastAPI app for one model service.

    The model is instantiated eagerly so a broken config or a missing table fails at
    startup rather than on the first request.
    """
    model = model_class.from_folder(root)
    meta = model.metadata()

    app = FastAPI(
        title=title or f"{meta.model_id} - {meta.model_name}",
        version=meta.model_version,
        description=meta.question,
        summary=meta.method_summary,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Reachable from tests and from scripts/run_all.py.
    app.state.model = model
    app.state.model_id = meta.model_id
    app.state.port = model.port

    # ------------------------------------------------------------------ error handlers
    @app.exception_handler(UnknownScenarioError)
    async def _unknown_scenario(_: Request, exc: UnknownScenarioError) -> JSONResponse:
        return _error_response(404, str(exc))

    @app.exception_handler(UnknownEntityError)
    async def _unknown_entity(_: Request, exc: UnknownEntityError) -> JSONResponse:
        return _error_response(404, str(exc))

    @app.exception_handler(ContractError)
    async def _contract_error(_: Request, exc: ContractError) -> JSONResponse:
        log.error("contract violation in %s: %s", meta.model_id, exc)
        return _error_response(500, str(exc))

    @app.exception_handler(UpstreamError)
    async def _upstream_error(_: Request, exc: UpstreamError) -> JSONResponse:
        log.error("upstream resolution failed in %s: %s", meta.model_id, exc)
        return _error_response(500, str(exc))

    @app.exception_handler(EngineError)
    async def _engine_error(_: Request, exc: EngineError) -> JSONResponse:
        log.error("engine failure in %s: %s", meta.model_id, exc)
        return _error_response(500, str(exc))

    @app.exception_handler(TwinError)
    async def _twin_error(_: Request, exc: TwinError) -> JSONResponse:
        log.error("%s failed: %s", meta.model_id, exc)
        return _error_response(500, str(exc))

    @app.exception_handler(NotImplementedError)
    async def _not_implemented(_: Request, exc: NotImplementedError) -> JSONResponse:
        # data_source: live with no connector (docs/02 section 8).
        return _error_response(501, str(exc))

    # ------------------------------------------------------------------ endpoints
    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        return HealthResponse(
            model_id=meta.model_id,
            model_version=meta.model_version,
            data_source=model.data_source,
        )

    @app.get("/metadata", response_model=Metadata, tags=["service"])
    def metadata() -> Metadata:
        return model.metadata()

    @app.get("/inputs", tags=["service"])
    def inputs(scenario_id: str = "S01", rows: int = 40) -> dict[str, Any]:
        """What this model reads, with a real sample of it.

        Deliberately shows the rows AROUND ``as_of`` rather than the head of the table: those
        are the ones that actually drive the answer, and the first rows of a 31-day table are
        four weeks of irrelevant history.
        """
        return describe_inputs(model, scenario_id=scenario_id, sample_rows=rows)

    @app.post("/predict", response_model=ModelOutput, tags=["model"])
    def predict(request: PredictRequest) -> ModelOutput:
        started = time.perf_counter()
        result = model.predict(request)
        _log_timing("predict", started)
        return result

    @app.post("/scenario", response_model=ModelOutput, tags=["model"])
    def scenario(request: ScenarioRequest) -> ModelOutput:
        started = time.perf_counter()
        # handle_scenario raises UnknownScenarioError (-> 404) for an unknown ID and returns
        # the documented degraded baseline for a scenario this model does not respond to.
        result = model.handle_scenario(request)
        _log_timing(f"scenario {request.scenario_id}", started)
        return result

    def _log_timing(label: str, started: float) -> None:
        elapsed = time.perf_counter() - started
        if elapsed > SLOW_RESPONSE_S:
            log.warning(
                "%s %s took %.1fs, above the %.0fs target in docs/02 section 6",
                meta.model_id,
                label,
                elapsed,
                SLOW_RESPONSE_S,
            )
        else:
            log.info("%s %s in %.2fs", meta.model_id, label, elapsed)

    return app


def run_app(app: FastAPI, *, host: str = "127.0.0.1", port: int | None = None) -> None:
    """Serve an app with uvicorn. Used by ``scripts/run_all.py`` and Dockerfiles."""
    import uvicorn

    uvicorn.run(app, host=host, port=port or app.state.port, log_level="info")


def app_info(app: FastAPI) -> dict[str, Any]:
    """Small summary used by ``scripts/run_all.py`` health reporting."""
    return {
        "model_id": app.state.model_id,
        "port": app.state.port,
        "title": app.title,
        "version": app.version,
    }
