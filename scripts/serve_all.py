"""Serve every built model from ONE process, for the live simulation page.

    uv run python scripts/serve_all.py
    # then open docs/SIMULATION.html

Why one process rather than the documented one-port-per-model layout (docs/02 section 3):
running ten uvicorns means ten terminals, ten warm-ups and ten things to go wrong in front of
an audience. Each model still has its own ``api.py`` on its own port for the real deployment -
this is a demo harness beside that, not a replacement for it.

Routes::

    GET  /models                    which models exist, which are built, which are ready
    GET  /{model_id}/metadata       the model card, as data
    GET  /{model_id}/inputs         what goes IN - real tables, real rows, real params
    POST /{model_id}/predict        the forecast
    POST /{model_id}/scenario       a what-if

**Models load in a background thread.** A forecasting model takes 10-25 s to start because it
reads a month of history, fits once and calibrates its uncertainty bands. Loading all ten
before answering anything would mean a blank page for three minutes, so ``/models`` reports
``ready`` per model and the page enables each one as it arrives.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "00_common"))

from twin_common.api.inputs import describe_inputs  # noqa: E402
from twin_common.config import model_registry  # noqa: E402
from twin_common.contracts import PredictRequest, ScenarioRequest  # noqa: E402
from twin_common.logging import get_logger  # noqa: E402

log = get_logger("twin.serve_all")


class Registry:
    """Holds the loaded models and the state the page needs to show progress."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._state: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._build_catalogue()

    def _build_catalogue(self) -> None:
        """Every model in the registry, built or not, so the page can show the roadmap."""
        for model_id, spec in model_registry()["models"].items():
            folder = ROOT / spec["folder"]
            built = (folder / "src" / "model.py").is_file() and (folder / "config.yaml").is_file()
            self._state[model_id] = {
                "model_id": model_id,
                "name": spec["name"],
                "folder": spec["folder"],
                "port": spec["port"],
                "engine": spec["engine"],
                "phase": spec["phase"],
                "upstream": list(spec.get("upstream") or []),
                "built": built,
                "ready": False,
                "loading": False,
                "error": None,
                "load_seconds": None,
            }

    @property
    def catalogue(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(v) for v in self._state.values()]

    def _import_model_class(self, model_id: str) -> type:
        """Import ``src.model.Model`` from one folder.

        Every model folder has its own ``src`` package, and Python caches the first one it
        imports under that name - so a second folder would silently receive the first one's
        class. The cached ``src`` modules are purged before each import, which is the same
        trick every model folder's conftest uses.
        """
        folder = ROOT / self._state[model_id]["folder"]
        for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
            del sys.modules[name]
        sys.path.insert(0, str(folder))
        try:
            return importlib.import_module("src.model").Model
        finally:
            if str(folder) in sys.path:
                sys.path.remove(str(folder))

    def load(self, model_id: str) -> Any:
        """Load one model, or return it if already loaded. Raises on failure."""
        with self._lock:
            if model_id in self._models:
                return self._models[model_id]
            if not self._state.get(model_id, {}).get("built"):
                raise KeyError(f"{model_id} is not built yet")
            self._state[model_id]["loading"] = True

        started = time.perf_counter()
        try:
            model_class = self._import_model_class(model_id)
            model = model_class.from_folder(ROOT / self._state[model_id]["folder"])
        except Exception as exc:
            with self._lock:
                self._state[model_id].update(loading=False, error=f"{type(exc).__name__}: {exc}")
            log.warning("%s failed to load: %s", model_id, exc)
            raise
        elapsed = time.perf_counter() - started
        with self._lock:
            self._models[model_id] = model
            self._state[model_id].update(
                loading=False, ready=True, error=None, load_seconds=round(elapsed, 1)
            )
        log.info("%s ready in %.1fs", model_id, elapsed)
        return model

    def get(self, model_id: str) -> Any:
        with self._lock:
            model = self._models.get(model_id)
        return model if model is not None else self.load(model_id)

    def warm_all(self) -> None:
        """Load every built model, one at a time, in the background.

        Sequentially on purpose: each model fits a forecaster and several want the same CPU,
        so loading them in parallel makes every one slower and the page stays useless longer.
        """
        for entry in self.catalogue:
            if entry["built"] and not entry["ready"]:
                try:
                    self.load(entry["model_id"])
                except Exception:
                    continue
        log.info("warm-up finished")


def build_app(registry: Registry) -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Event Twin - all models",
        description="Demo harness: every built model on one port. See scripts/serve_all.py.",
    )
    # The simulation page may be opened from a file:// URL or a published page, so the origin
    # cannot be predicted. This is a local demo harness, not a deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/models", tags=["discovery"])
    def models() -> dict[str, Any]:
        catalogue = registry.catalogue
        return {
            "models": catalogue,
            "built": sum(1 for m in catalogue if m["built"]),
            "ready": sum(1 for m in catalogue if m["ready"]),
            "total": len(catalogue),
        }

    def _model_or_404(model_id: str) -> Any:
        model_id = model_id.upper()
        state = {m["model_id"]: m for m in registry.catalogue}.get(model_id)
        if state is None:
            raise HTTPException(404, f"unknown model {model_id}")
        if not state["built"]:
            raise HTTPException(409, f"{model_id} is not built yet (phase {state['phase']})")
        try:
            return registry.get(model_id)
        except Exception as exc:
            raise HTTPException(500, f"{model_id} failed to load: {exc}") from exc

    @app.get("/{model_id}/metadata", tags=["model"])
    def metadata(model_id: str) -> Any:
        return _model_or_404(model_id).metadata()

    @app.get("/{model_id}/inputs", tags=["model"])
    def inputs(model_id: str, scenario_id: str = "S01", rows: int = 40) -> dict[str, Any]:
        return describe_inputs(_model_or_404(model_id), scenario_id=scenario_id, sample_rows=rows)

    @app.post("/{model_id}/predict", tags=["model"])
    def predict(model_id: str, request: PredictRequest) -> Any:
        model = _model_or_404(model_id)
        started = time.perf_counter()
        result = model.predict(request)
        return _with_timing(result, started)

    @app.post("/{model_id}/scenario", tags=["model"])
    def scenario(model_id: str, request: ScenarioRequest) -> Any:
        model = _model_or_404(model_id)
        started = time.perf_counter()
        result = model.handle_scenario(request)
        return _with_timing(result, started)

    def _with_timing(result: Any, started: float) -> dict[str, Any]:
        """The contract response plus how long it took, which the page displays.

        Added beside the payload rather than inside it: ModelOutput forbids extra fields, and
        a demo harness must not be able to change what the contract looks like.
        """
        payload = result.model_dump(mode="json")
        payload["_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return payload

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve every built model on one port.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--no-warm",
        action="store_true",
        help="do not preload models; each loads on its first request instead",
    )
    args = parser.parse_args(argv)

    import uvicorn

    registry = Registry()
    built = [m for m in registry.catalogue if m["built"]]
    print(f"Event Twin - serving {len(built)} built models on http://{args.host}:{args.port}")
    print("  " + ", ".join(m["model_id"] for m in built))
    print(f"  discovery:  http://{args.host}:{args.port}/models")
    print(f"  docs:       http://{args.host}:{args.port}/docs")
    if not args.no_warm:
        print("  warming up in the background; /models reports progress")
        threading.Thread(target=registry.warm_all, daemon=True).start()

    uvicorn.run(build_app(registry), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
