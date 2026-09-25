"""Describe what goes INTO a model, for the ``GET /inputs`` endpoint.

The four contract endpoints (docs/02 section 6) say what a model *produces*. Nothing said what
it *consumes*, which makes a model hard to explain to anyone who did not write it: the output
is a list of numbers with no visible provenance.

This module answers "where did that number come from" with the real thing rather than a
description of it:

* every declared input table, with its registry schema, its true row count, and a real sample
  of the rows **around ``as_of``** - the ones that actually drive the answer, not the first
  rows of four weeks of irrelevant history;
* the complete ``params`` block, which is where every threshold, rate and weight in the project
  lives, so an assumption cannot hide;
* the upstream models and what each one contributed.

It is additive. Nothing in the frontend contract depends on it, so this file cannot break a
consumer of ``/predict``.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd

from ..io.schemas import table_exists, table_schema
from ..io.tables import load_table
from ..logging import get_logger
from ..model import TwinModel

log = get_logger(__name__)

#: Hard ceiling on sampled rows per table, whatever the request asks for. A browser showing a
#: 24,000-row table helps nobody, and the endpoint is meant to be cheap enough to call on every
#: page load.
MAX_SAMPLE_ROWS = 200


def _json_safe(value: Any) -> Any:
    """Make one cell safe for JSON: NaN and NaT are not valid JSON numbers."""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, pd.Timestamp | datetime):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, bool | int | str):
        return value
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _window(frame: pd.DataFrame, as_of: datetime, limit: int) -> tuple[pd.DataFrame, str]:
    """The rows around ``as_of``, and a label saying which rows they are.

    Most tables in this project are 31 days long and the interesting moment is one instant
    inside them, so the head of the table is useless. This takes a window centred a little
    before ``as_of`` - the recent history a forecast actually reads - and the steps just after.
    """
    schema = None
    if table_exists(str(frame.attrs.get("table_name", ""))):
        schema = table_schema(str(frame.attrs["table_name"]))
    time_column = (schema.time_column if schema else None) or "timestamp"

    if time_column not in frame.columns:
        return frame.head(limit), f"first {min(limit, len(frame))} rows"

    stamps = pd.to_datetime(frame[time_column], errors="coerce")
    cutoff = pd.Timestamp(as_of)
    if stamps.dt.tz is not None and cutoff.tz is None:
        cutoff = cutoff.tz_localize(stamps.dt.tz)
    elif stamps.dt.tz is None and cutoff.tz is not None:
        cutoff = cutoff.tz_localize(None)

    before = frame.loc[stamps <= cutoff]
    after = frame.loc[stamps > cutoff]
    # Two thirds history, one third the window being forecast: enough of each to see the shape.
    take_before = max(1, (limit * 2) // 3)
    head = before.tail(take_before)
    tail = after.head(limit - len(head))
    window = pd.concat([head, tail]) if len(tail) else head
    if window.empty:
        return frame.head(limit), f"first {min(limit, len(frame))} rows"
    return window, f"{len(window)} rows around {cutoff.isoformat()}"


def describe_table(
    name: str, model: TwinModel, *, scenario_id: str, sample_rows: int
) -> dict[str, Any]:
    """One input table: what it is, how big it really is, and a real slice of it."""
    schema = table_schema(name) if table_exists(name) else None
    entry: dict[str, Any] = {
        "table": name,
        "dataset_id": schema.dataset_id if schema else None,
        "grain": schema.grain if schema else None,
        "columns": (
            [{"name": c, "dtype": str(d)} for c, d in schema.columns.items()] if schema else []
        ),
    }
    try:
        frame = load_table(
            name, model.data_source, scenario_id=scenario_id, base_dir=model.synthetic_dir
        )
    except Exception as exc:
        entry["available"] = False
        entry["note"] = f"not sliced for {scenario_id}: {exc}"
        return entry

    frame.attrs["table_name"] = name
    limit = max(1, min(int(sample_rows), MAX_SAMPLE_ROWS))
    window, label = _window(frame, model.demo_now, limit)

    entry["available"] = True
    entry["total_rows"] = len(frame)
    entry["sampled_rows"] = len(window)
    entry["sample_label"] = label
    entry["is_synthetic"] = (
        bool(frame["is_synthetic"].all()) if "is_synthetic" in frame.columns else None
    )
    entry["source"] = (
        sorted({str(v) for v in frame["source"].dropna().unique()})[:3]
        if "source" in frame.columns
        else []
    )
    entry["sample"] = [
        {str(k): _json_safe(v) for k, v in row.items()} for row in window.to_dict(orient="records")
    ]
    return entry


def describe_inputs(
    model: TwinModel, *, scenario_id: str = "S01", sample_rows: int = 40
) -> dict[str, Any]:
    """Everything that feeds one model, with the real data rather than a description of it."""
    tables = [
        describe_table(name, model, scenario_id=scenario_id, sample_rows=sample_rows)
        for name in model.input_tables
    ]
    upstream = []
    for model_id in model.upstream_ids:
        path = model.sample_upstream_dir / f"{model_id}__{scenario_id}.json"
        fallback = model.sample_upstream_dir / f"{model_id}__S01.json"
        chosen = path if path.is_file() else (fallback if fallback.is_file() else None)
        upstream.append(
            {
                "model_id": model_id,
                "available": chosen is not None,
                "path": None if chosen is None else str(chosen.name),
                "size_kb": None if chosen is None else round(chosen.stat().st_size / 1024, 1),
                "exact_scenario": bool(path.is_file()),
            }
        )

    return {
        "model_id": model.model_id,
        "model_name": model.model_name,
        "engine": model.engine,
        "scenario_id": scenario_id,
        "as_of": model.demo_now.isoformat(),
        "data_source": model.data_source.value,
        "default_horizon_min": model.default_horizon_min,
        "horizons_min": model.horizons_min,
        "kpis": model.owned_kpis,
        "scenarios_supported": model.scenarios_supported,
        "tables": tables,
        "upstream": upstream,
        # Every threshold, rate and weight the model uses. CLAUDE.md forbids them in code, so
        # this block IS the model's assumptions, in full.
        "params": model.params(),
        "total_input_rows": sum(t.get("total_rows", 0) for t in tables),
    }
