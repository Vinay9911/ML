"""Output assembly: the OutputBuilder and risk-band derivation."""

from __future__ import annotations

from .builder import SYNTHETIC_REASON, UPSTREAM_FALLBACK_REASON, OutputBuilder
from .risk import (
    BAND_ORDER,
    band_input_value,
    band_thresholds,
    risk_level_for_kpi,
    risk_level_from_table,
    risk_rank,
    worst_risk,
)

__all__ = [
    "BAND_ORDER",
    "SYNTHETIC_REASON",
    "UPSTREAM_FALLBACK_REASON",
    "OutputBuilder",
    "band_input_value",
    "band_thresholds",
    "risk_level_for_kpi",
    "risk_level_from_table",
    "risk_rank",
    "worst_risk",
]
