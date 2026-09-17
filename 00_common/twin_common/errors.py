"""Exception hierarchy for twin_common.

Model code raises these instead of bare exceptions so `twin_common.api` can map them to
the right HTTP status (docs/02 section 6).
"""

from __future__ import annotations


class TwinError(Exception):
    """Base class for every error raised by twin_common."""


class ConfigError(TwinError):
    """A config file is missing, malformed, or a required key is absent."""


class ContractError(TwinError):
    """Data does not satisfy the ModelOutput contract in docs/02 section 4."""


class RegistryError(TwinError):
    """An unknown KPI, model ID, reason code, band or resource type was referenced."""


class TableError(TwinError):
    """A data table is missing or its columns do not match the registered schema."""


class UnknownEntityError(TwinError):
    """An entity ID is not part of the world. Mapped to HTTP 404."""


class UnknownScenarioError(TwinError):
    """A scenario ID is not in scenarios.yaml. Mapped to HTTP 404."""


class UpstreamError(TwinError):
    """Every upstream resolution step failed (docs/02 section 7)."""


class OfflineError(TwinError):
    """A network fetch was attempted while TWIN_OFFLINE=1 and no cache or fallback existed."""


class EngineError(TwinError):
    """A shared engine failed and no fallback was available."""
