"""
FastAPI Verification Service Re-Export for ARES 2.0 (Milestones M13, M14).

Exposes the production FastAPI application from ares.service.api.
"""

from ares.service.api import (
    ClaimRequest,
    ClaimResponse,
    EvidencePassage,
    KillSwitchRequest,
    app,
)

__all__ = ["app", "ClaimRequest", "ClaimResponse", "EvidencePassage", "KillSwitchRequest"]
