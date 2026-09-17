"""Table IO: one entry point for reading data whatever its source (docs/04 section 10)."""

from __future__ import annotations

from .adapters import (
    ADAPTERS,
    DataAdapter,
    LiveAdapter,
    LiveConnector,
    ReplayAdapter,
    SyntheticAdapter,
    get_adapter,
)
from .schemas import (
    PROVENANCE_COLUMNS,
    TABLE_SCHEMAS,
    TableSchema,
    dataset_id_of,
    table_exists,
    table_schema,
    tables_for_model,
)
from .tables import (
    check_schema,
    describe_table,
    enforce_provenance,
    load_table,
    save_table,
    table_is_available,
)

__all__ = [
    "ADAPTERS",
    "PROVENANCE_COLUMNS",
    "TABLE_SCHEMAS",
    "DataAdapter",
    "LiveAdapter",
    "LiveConnector",
    "ReplayAdapter",
    "SyntheticAdapter",
    "TableSchema",
    "check_schema",
    "dataset_id_of",
    "describe_table",
    "enforce_provenance",
    "get_adapter",
    "load_table",
    "save_table",
    "table_exists",
    "table_is_available",
    "table_schema",
    "tables_for_model",
]
