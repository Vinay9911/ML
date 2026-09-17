"""Data-source adapters: synthetic, replay and live (docs/04 section 10).

Model code calls :func:`twin_common.io.load_table` and never learns which mode is active.
The ``live`` adapter deliberately raises: docs/02 section 8 requires a clear
``NotImplementedError`` rather than fabricated live data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import pandas as pd

from ..contracts.enums import DataSource
from ..errors import TableError
from ..logging import get_logger
from ..paths import data_dir, world_dir
from .schemas import TableSchema, table_schema

log = get_logger(__name__)


class DataAdapter(ABC):
    """Reads one named table for a given data source."""

    data_source: ClassVar[DataSource]
    #: Whether rows produced by this adapter are synthetic by construction.
    default_is_synthetic: ClassVar[bool]

    def __init__(self, base_dir: Path | None = None, scenario_id: str = "S01") -> None:
        self.base_dir = base_dir
        self.scenario_id = scenario_id

    @abstractmethod
    def path_for(self, name: str) -> Path:
        """Filesystem location of a table, whether or not it exists."""

    @abstractmethod
    def read(
        self,
        name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Return the table, optionally filtered to [start, end] on its time column."""

    # -- shared helpers ---------------------------------------------------
    @staticmethod
    def _filter_time(
        df: pd.DataFrame,
        schema: TableSchema,
        start: datetime | None,
        end: datetime | None,
    ) -> pd.DataFrame:
        col = schema.time_column
        if col is None or col not in df.columns or (start is None and end is None):
            return df
        series = df[col]
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= series >= start
        if end is not None:
            mask &= series <= end
        return df.loc[mask].reset_index(drop=True)


class SyntheticAdapter(DataAdapter):
    """Reads the generated world: ``<base>/world/<scenario_id>/<table>.parquet``.

    ``base_dir`` lets a model folder read its own sliced copy under ``data/synthetic/``
    (written by ``scripts/slice_world.py``) instead of the shared world.
    """

    data_source = DataSource.SYNTHETIC
    default_is_synthetic = True

    def path_for(self, name: str) -> Path:
        table_schema(name)  # validate the name early
        if self.base_dir is not None:
            scoped = self.base_dir / self.scenario_id / f"{name}.parquet"
            return scoped if scoped.is_file() else self.base_dir / f"{name}.parquet"
        return world_dir(self.scenario_id) / f"{name}.parquet"

    def read(
        self,
        name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        schema = table_schema(name)
        path = self.path_for(name)
        if not path.is_file():
            raise TableError(
                f"synthetic table {name!r} not found at {path}. Generate the world with "
                f"`python -m twin_common.synthetic.generate --scenario {self.scenario_id}` "
                f"or slice it with `scripts/slice_world.py`."
            )
        df = pd.read_parquet(path)
        return self._filter_time(df, schema, start, end)


class ReplayAdapter(DataAdapter):
    """Reads real recorded data in the same schema: ``<base>/replay/<table>.parquet``."""

    data_source = DataSource.REPLAY
    default_is_synthetic = False

    def path_for(self, name: str) -> Path:
        table_schema(name)
        root = self.base_dir if self.base_dir is not None else data_dir() / "replay"
        return root / f"{name}.parquet"

    def read(
        self,
        name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        schema = table_schema(name)
        path = self.path_for(name)
        if not path.is_file():
            raise TableError(
                f"replay table {name!r} not found at {path}. Recorded real data must use the "
                f"same columns as the synthetic table (docs/04 section 10)."
            )
        df = pd.read_parquet(path)
        return self._filter_time(df, schema, start, end)


class LiveConnector(ABC):
    """Interface a real deployment implements per table. Not implemented in the demo."""

    @abstractmethod
    def read(self, name: str, start: datetime | None, end: datetime | None) -> pd.DataFrame:
        """Fetch live rows for one table."""


class LiveAdapter(DataAdapter):
    """Raises unless a connector has been registered for the table.

    docs/02 section 8: ``data_source: live`` must raise a clear NotImplementedError through
    this adapter. Never fake live data.
    """

    data_source = DataSource.LIVE
    default_is_synthetic = False

    #: table name -> connector instance. Populated by a real deployment, empty in the demo.
    _connectors: ClassVar[dict[str, LiveConnector]] = {}

    @classmethod
    def register(cls, name: str, connector: LiveConnector) -> None:
        table_schema(name)
        cls._connectors[name] = connector
        log.info("registered live connector for table %s", name)

    @classmethod
    def unregister_all(cls) -> None:
        cls._connectors.clear()

    def path_for(self, name: str) -> Path:
        raise NotImplementedError(f"live connector for {name} has no filesystem path")

    def read(
        self,
        name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        table_schema(name)
        connector = self._connectors.get(name)
        if connector is None:
            raise NotImplementedError(f"live connector for {name} not configured")
        schema = table_schema(name)
        df = connector.read(name, start, end)
        return self._filter_time(df, schema, start, end)


ADAPTERS: dict[DataSource, type[DataAdapter]] = {
    DataSource.SYNTHETIC: SyntheticAdapter,
    DataSource.REPLAY: ReplayAdapter,
    DataSource.LIVE: LiveAdapter,
}


def get_adapter(
    data_source: DataSource | str,
    base_dir: Path | None = None,
    scenario_id: str = "S01",
) -> DataAdapter:
    """Build the adapter for a data source string from config."""
    ds = DataSource(data_source)
    return ADAPTERS[ds](base_dir=base_dir, scenario_id=scenario_id)
