"""Shared heavy engines. Each lands in the phase that needs it (docs/07).

=========  =====  ===========================================
Engine     Phase  Models
=========  =====  ===========================================
formula    3      M21, M07
forecast   4      M01, M05, M06, M15, M17, M18, M19, M20, M22
rules      5      M03, M09, M12, M25
mlclf      5      M23
vision     6      M02, M10
network    7      M04, M13, M14
optimize   7      M16, M24
location   7      M08
pedsim     8      M11
=========  =====  ===========================================

Model folders import engines from here; they never import from each other.
"""

from __future__ import annotations

__all__: list[str] = []
