"""Table load/save with schema checking and provenance enforcement.

``load_table`` is the only function model code should use to read data. It

1. resolves the adapter for the configured data source,
2. reads the table (time-filtered if asked),
3. checks the registered columns and coerces dtypes,
4. enforces that ``is_synthetic`` and ``source`` are present and usable.

Dropping ``is_synthetic`` is a CLAUDE.md hard rule violation, so a missing column is an
error rather than something silently defaulted.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts.enums import DataSource
from ..contracts.models import IST
from ..errors import TableError
from ..logging import get_logger
from ..paths import ensure_dir
from .adapters import get_adapter
from .schemas import PROVENANCE_COLUMNS, TableSchema, table_schema

log = get_logger(__name__)


def _coerce_column(series: pd.Series, dtype: str, label: str) -> pd.Series:
    """Coerce one column to its dtype family, raising TableError with context on failure."""
    try:
        if dtype == "timestamp":
            out = pd.to_datetime(series, errors="raise")
            return out.dt.tz_localize(IST) if out.dt.tz is None else out.dt.tz_convert(IST)
        if dtype == "int":
            return pd.to_numeric(series, errors="raise").astype("int64")
        if dtype == "float":
            return pd.to_numeric(series, errors="raise").astype("float64")
        if dtype == "bool":
            if series.dtype == bool:
                return series
            return series.map(
                lambda v: (
                    v
                    if isinstance(v, bool)
                    else str(v).strip().lower() in {"1", "true", "yes", "on"}
                )
            ).astype(bool)
        if dtype == "str":
            return series.astype("string").astype(object)
        if dtype == "json":
            return series
    except (ValueError, TypeError) as exc:
        raise TableError(f"{label}: could not coerce to {dtype}: {exc}") from exc
    raise TableError(f"{label}: unknown dtype family {dtype!r}")


def check_schema(df: pd.DataFrame, name: str, *, coerce: bool = True) -> pd.DataFrame:
    """Validate and optionally coerce a dataframe against its registered schema.

    Raises TableError naming every missing column at once, so a broken generator is fixed
    in one pass rather than one column per run.
    """
    schema = table_schema(name)
    missing = [c for c in schema.required_with_provenance if c not in df.columns]
    if missing:
        raise TableError(
            f"table {name!r} (dataset {schema.dataset_id}) is missing required columns "
            f"{missing}; got {sorted(df.columns)}"
        )
    if not coerce:
        return df
    out = df.copy()
    for column, dtype in schema.columns.items():
        if column in out.columns:
            out[column] = _coerce_column(out[column], dtype, f"{name}.{column}")
    return out


def enforce_provenance(df: pd.DataFrame, name: str, *, default_is_synthetic: bool) -> pd.DataFrame:
    """Fail when provenance columns are absent; normalise their dtypes when present.

    ``default_is_synthetic`` is only used in the error message, to tell the caller what the
    adapter would have expected. The columns are never invented.
    """
    for column in PROVENANCE_COLUMNS:
        if column not in df.columns:
            raise TableError(
                f"table {name!r} has no {column!r} column. Every row must carry "
                f"is_synthetic and source (CLAUDE.md hard rule); the "
                f"{'synthetic' if default_is_synthetic else 'real'} writer must set them."
            )
    if df["is_synthetic"].isna().any():
        raise TableError(f"table {name!r} has null values in is_synthetic")
    return df


def load_table(
    name: str,
    data_source: DataSource | str = DataSource.SYNTHETIC,
    *,
    scenario_id: str = "S01",
    base_dir: str | Path | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    coerce: bool = True,
) -> pd.DataFrame:
    """Read one table through the adapter for ``data_source``.

    Args:
        name: registered table name, e.g. ``footfall_15min``.
        data_source: ``synthetic``, ``replay`` or ``live``.
        scenario_id: which generated world to read (synthetic only).
        base_dir: override the root, e.g. a model folder ``data/synthetic`` slice.
        start, end: inclusive filter on the table time column.
        coerce: coerce dtypes to the registered families.
    """
    adapter = get_adapter(
        data_source,
        base_dir=Path(base_dir) if base_dir is not None else None,
        scenario_id=scenario_id,
    )
    df = adapter.read(name, start=start, end=end)
    df = enforce_provenance(df, name, default_is_synthetic=adapter.default_is_synthetic)
    df = check_schema(df, name, coerce=coerce)
    log.debug("loaded %s rows=%d source=%s", name, len(df), adapter.data_source.value)
    return df


def save_table(
    df: pd.DataFrame,
    name: str,
    out_dir: str | Path,
    *,
    is_synthetic: bool | None = None,
    source: str | None = None,
    check: bool = True,
) -> Path:
    """Write a table to Parquet, filling provenance columns when asked.

    ``is_synthetic`` / ``source`` are written only when the dataframe does not already have
    them, so a real fetcher can mark its own rows ``is_synthetic=False``.
    """
    out = df.copy()
    if "is_synthetic" not in out.columns:
        if is_synthetic is None:
            raise TableError(
                f"save_table({name!r}): is_synthetic must be given when the frame has no "
                f"is_synthetic column"
            )
        out["is_synthetic"] = bool(is_synthetic)
    if "source" not in out.columns:
        if source is None:
            raise TableError(
                f"save_table({name!r}): source must be given when the frame has no source column"
            )
        out["source"] = str(source)
    if check:
        out = check_schema(out, name, coerce=True)
    directory = ensure_dir(Path(out_dir))
    path = directory / f"{name}.parquet"
    out.to_parquet(path, index=False)
    log.info("wrote %s rows=%d -> %s", name, len(out), path)
    return path


def table_is_available(
    name: str,
    data_source: DataSource | str = DataSource.SYNTHETIC,
    *,
    scenario_id: str = "S01",
    base_dir: str | Path | None = None,
) -> bool:
    """True when the table file exists, without reading it."""
    adapter = get_adapter(
        data_source,
        base_dir=Path(base_dir) if base_dir is not None else None,
        scenario_id=scenario_id,
    )
    try:
        return adapter.path_for(name).is_file()
    except NotImplementedError:
        return False


def describe_table(name: str) -> dict[str, Any]:
    """Schema summary for README and model-card generation."""
    schema: TableSchema = table_schema(name)
    return {
        "name": schema.name,
        "dataset_id": schema.dataset_id,
        "grain": schema.grain,
        "required": dict(schema.required),
        "optional": dict(schema.optional),
        "time_column": schema.time_column,
        "used_by": list(schema.used_by),
    }
