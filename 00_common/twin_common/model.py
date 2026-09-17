"""The TwinModel base class every model service implements.

``api.py`` in a model folder is one line - ``app = create_app(Model)`` - so everything the
API needs comes from this interface (docs/02 section 2).

A subclass must provide :meth:`predict`, :meth:`scenario` and :meth:`metadata`. The base
class already handles config loading, request defaults, upstream resolution and the
"insensitive to Sxx" degraded response, so a model folder contains domain logic only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from .config import Config, load_model_config
from .contracts.enums import DataSource, State, Status
from .contracts.models import Metadata, ModelOutput, PredictRequest, ScenarioRequest, to_ist
from .errors import ConfigError
from .logging import get_logger
from .output.builder import OutputBuilder
from .scenarios import get_scenario, resolve_overrides
from .upstream import Resolution, resolve_all

log = get_logger(__name__)

#: Standard sub-directories of a model folder (docs/02 section 2).
SAMPLE_UPSTREAM_DIR = "data/sample_upstream"
SYNTHETIC_DIR = "data/synthetic"
DERIVED_DIR = "data/derived"
OUTPUTS_DIR = "outputs"


class TwinModel(ABC):
    """Base class for the 25 model services.

    Subclasses set :attr:`model_id` and implement the three abstract methods. The default
    :meth:`load` is a no-op; override it to read tables or fitted artifacts once at startup.
    """

    #: Model ID, e.g. "M21". Used to locate the registry row and to sanity-check config.
    model_id: str = ""

    def __init__(self, config: Config, root: Path) -> None:
        self.config = config
        self.root = root
        self._loaded = False

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_folder(cls, root: str | Path | None = None, *, validate: bool = True) -> Self:
        """Build the model from a folder containing ``config.yaml``.

        With ``root=None`` the folder is inferred from the subclass module location, which
        is what ``create_app`` uses so ``api.py`` needs no paths.
        """
        folder = Path(root) if root is not None else cls.default_root()
        config_path = folder / "config.yaml"
        config = load_model_config(config_path, validate=validate)
        declared = config.require_str("model_id")
        if cls.model_id and declared != cls.model_id:
            raise ConfigError(
                f"{config_path}: model_id is {declared!r} but {cls.__name__}.model_id is "
                f"{cls.model_id!r}"
            )
        instance = cls(config, folder)
        instance.load()
        instance._loaded = True
        return instance

    @classmethod
    def default_root(cls) -> Path:
        """The model folder, inferred as the parent of the ``src/`` package."""
        import inspect

        module_file = inspect.getfile(cls)
        return Path(module_file).resolve().parent.parent

    def load(self) -> None:
        """Read tables and artifacts once at startup. Override as needed."""
        return None

    # ------------------------------------------------------------------ abstract API
    @abstractmethod
    def predict(self, request: PredictRequest) -> ModelOutput:
        """Answer POST /predict."""

    @abstractmethod
    def scenario(self, request: ScenarioRequest) -> ModelOutput:
        """Answer POST /scenario."""

    @abstractmethod
    def metadata(self) -> Metadata:
        """Answer GET /metadata."""

    # ------------------------------------------------------------- config shortcuts
    @property
    def model_name(self) -> str:
        return self.config.require_str("model_name")

    @property
    def model_version(self) -> str:
        return self.config.require_str("model_version")

    @property
    def port(self) -> int:
        return self.config.require_int("port")

    @property
    def engine(self) -> str:
        return self.config.require_str("engine")

    @property
    def data_source(self) -> DataSource:
        return DataSource(self.config.require_str("data_source"))

    @property
    def seed(self) -> int:
        return self.config.require_int("seed")

    @property
    def demo_now(self) -> datetime:
        return to_ist(datetime.fromisoformat(self.config.require_str("demo_now")))

    @property
    def upstream_ids(self) -> list[str]:
        return list(self.config.require("upstream"))

    @property
    def input_tables(self) -> list[str]:
        return list(self.config.require("inputs"))

    @property
    def owned_kpis(self) -> list[str]:
        return list(self.config.require("kpis"))

    @property
    def scenarios_supported(self) -> list[str]:
        return list(self.config.require("scenarios_supported"))

    @property
    def horizons_min(self) -> list[int]:
        return [int(h) for h in self.config.require("horizons_min")]

    @property
    def default_horizon_min(self) -> int:
        return self.config.require_int("default_horizon_min")

    @property
    def upstream_timeout_s(self) -> float:
        return float(self.config.require("upstream_timeout_s"))

    def params(self) -> dict[str, Any]:
        """The model-specific numbers. All thresholds/rates/weights live here."""
        return dict(self.config.get("params") or {})

    def param(self, key: str, default: Any = None) -> Any:
        """One entry of ``params``, by dotted path."""
        return self.config.get(f"params.{key}", default)

    def require_param(self, key: str) -> Any:
        return self.config.require(f"params.{key}")

    # ------------------------------------------------------------------ paths
    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def sample_upstream_dir(self) -> Path:
        return self.path(SAMPLE_UPSTREAM_DIR)

    @property
    def synthetic_dir(self) -> Path:
        return self.path(SYNTHETIC_DIR)

    @property
    def derived_dir(self) -> Path:
        return self.path(DERIVED_DIR)

    @property
    def outputs_dir(self) -> Path:
        return self.path(OUTPUTS_DIR)

    # ------------------------------------------------------- request normalisation
    def resolve_as_of(self, request: PredictRequest) -> datetime:
        """Request ``as_of``, or ``demo_now`` from config."""
        return to_ist(request.as_of) if request.as_of is not None else self.demo_now

    def resolve_horizon(self, request: PredictRequest) -> int:
        return request.horizon_min if request.horizon_min is not None else self.default_horizon_min

    def resolve_kpis(self, request: PredictRequest) -> list[str]:
        """Requested KPIs restricted to the ones this model owns, or all of them."""
        if not request.kpis:
            return self.owned_kpis
        owned = set(self.owned_kpis)
        return [k for k in request.kpis if k in owned]

    def resolve_entities(self, request: PredictRequest, available: list[str]) -> list[str]:
        """Requested entities restricted to those this model knows about, or all of them."""
        if not request.entity_ids:
            return list(available)
        wanted = set(request.entity_ids)
        return [e for e in available if e in wanted]

    def scenario_overrides(self, request: ScenarioRequest) -> dict[str, Any]:
        """Scenario overrides merged with request overrides (raises 404 on unknown ID)."""
        get_scenario(request.scenario_id)
        return resolve_overrides(request.scenario_id, request.overrides)

    def supports_scenario(self, scenario_id: str) -> bool:
        return scenario_id in set(self.scenarios_supported)

    # ------------------------------------------------------------------ upstream
    def resolve_upstream(
        self,
        request: PredictRequest,
        *,
        scenario_id: str = "S01",
        entity_ids: list[str] | None = None,
    ) -> dict[str, Resolution]:
        """Resolve every declared upstream using the docs/02 section 7 order."""
        if not self.upstream_ids:
            return {}
        return resolve_all(
            self.upstream_ids,
            as_of=self.resolve_as_of(request),
            scenario_id=scenario_id,
            inline=request.upstream,
            sample_dir=self.sample_upstream_dir,
            horizon_min=self.resolve_horizon(request),
            entity_ids=entity_ids,
            timeout_s=self.upstream_timeout_s,
        )

    def record_upstream(
        self, builder: OutputBuilder, resolutions: dict[str, Resolution]
    ) -> OutputBuilder:
        """Copy upstream provenance and any fallback warnings into the builder."""
        for resolution in resolutions.values():
            builder.add_upstream_ref(resolution.ref)
            if resolution.warning:
                builder.warn(resolution.warning)
        return builder

    # ------------------------------------------------------------------ builders
    def new_builder(
        self,
        request: PredictRequest,
        *,
        scenario_id: str = "S01",
        overrides: dict[str, Any] | None = None,
    ) -> OutputBuilder:
        """An OutputBuilder pre-filled from config and the request."""
        return OutputBuilder.from_config(
            self.config,
            as_of=self.resolve_as_of(request),
            scenario_id=scenario_id,
            scenario_overrides=overrides,
        )

    def insensitive_response(self, request: ScenarioRequest) -> ModelOutput:
        """The docs/02 section 6 answer for a scenario this model does not support.

        Returns the baseline values with ``status: degraded`` and the warning that the
        model is insensitive to the scenario. Record states are rewritten to ``scenario``
        so the response still satisfies the contract.
        """
        baseline = self.predict(PredictRequest(**request.model_dump(include=_PREDICT_FIELDS)))
        # model_copy does not coerce, so the enum member is passed rather than a bare string;
        # a raw "scenario" str would leave the field untyped and break JSON serialization.
        rewritten = [
            r.model_copy(update={"state": State.SCENARIO}) if r.state is State.FORECAST else r
            for r in baseline.results
        ]
        return baseline.model_copy(
            update={
                "scenario_id": request.scenario_id,
                "scenario_overrides": self.scenario_overrides(request),
                "status": Status.DEGRADED,
                "warnings": [*baseline.warnings, f"insensitive to {request.scenario_id}"],
                "results": rewritten,
            }
        )


#: Fields shared by PredictRequest and ScenarioRequest, for narrowing a scenario request.
_PREDICT_FIELDS = {
    "as_of",
    "horizon_min",
    "entity_ids",
    "kpis",
    "include_current",
    "upstream",
}
