"""JSON Schema export.

``export_schemas()`` writes the schemas the frontend developer integrates against into
``00_common/schemas/``. Every model folder also keeps a copy under its own ``schemas/``
(written by ``scripts/new_model.py``) so a delivered zip is self-describing.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..logging import get_logger
from ..paths import ensure_dir, schemas_dir
from .models import (
    Metadata,
    ModelOutput,
    PredictRequest,
    ScenarioRequest,
)

log = get_logger(__name__)

#: filename -> pydantic model. These four are the public contract surface.
SCHEMA_TARGETS: dict[str, type] = {
    "model_output.schema.json": ModelOutput,
    "predict_request.schema.json": PredictRequest,
    "scenario_request.schema.json": ScenarioRequest,
    "metadata.schema.json": Metadata,
}


def schema_for(model: type) -> dict:
    """JSON Schema for one pydantic model, in serialization mode."""
    return model.model_json_schema(mode="serialization")


def export_schemas(out_dir: str | Path | None = None) -> dict[str, Path]:
    """Write every contract schema as JSON. Returns {filename: path}.

    Files are written with a trailing newline and sorted keys so re-running the export
    produces no git diff unless the contract actually changed.
    """
    target = ensure_dir(Path(out_dir) if out_dir is not None else schemas_dir())
    written: dict[str, Path] = {}
    for filename, model in SCHEMA_TARGETS.items():
        path = target / filename
        payload = json.dumps(schema_for(model), indent=2, sort_keys=True, ensure_ascii=False)
        path.write_text(payload + "\n", encoding="utf-8")
        written[filename] = path
        log.info("exported %s", path)
    return written
