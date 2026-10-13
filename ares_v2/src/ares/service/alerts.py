"""
Operational Alert Prioritization & Escalation Engine (Milestone M14, Claims C082, C083, RF4).

Implements the multi-factor operational alert severity formulation:
    A = w1 * R_m + w2 * I + w3 * E

Where:
- R_m in [0, 1]: Misinformation Risk (calibrated probability of falsehood * domain hazard).
- I in [0, 1]: Impact potential (normalized reach, audience exposure, virality proxy).
- E in [0, 1]: Escalation velocity (re-sharing velocity, cross-platform propagation rate).

SCIENTIFIC INTEGRITY DECLARATION (Audit Issue C083, RF4):
The weights (w1=0.45, w2=0.30, w3=0.25) and priority thresholds (tau_low=0.40, tau_crit=0.70)
are HEURISTIC / PROXY-FITTED policy parameters, calibrated against domain severity benchmarks
in Milestone M10. They are NOT 'learned gradient weights' as improperly claimed in the early
manuscript draft.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    CRITICAL = "CRITICAL"


@dataclass
class AlertNotification:
    alert_id: str
    claim_id: str
    claim_text: str
    verdict: str
    calibrated_confidence: float
    threat_index_rho: float
    risk_score_rm: float
    impact_score_i: float
    escalation_score_e: float
    composite_score_a: float
    severity: AlertSeverity
    created_at: str
    detection_to_alert_ms: float
    suggested_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "verdict": self.verdict,
            "calibrated_confidence": round(float(self.calibrated_confidence), 4),
            "threat_index_rho": round(float(self.threat_index_rho), 4),
            "risk_score_rm": round(float(self.risk_score_rm), 4),
            "impact_score_i": round(float(self.impact_score_i), 4),
            "escalation_score_e": round(float(self.escalation_score_e), 4),
            "composite_score_a": round(float(self.composite_score_a), 4),
            "severity": self.severity.value,
            "created_at": self.created_at,
            "detection_to_alert_ms": round(float(self.detection_to_alert_ms), 2),
            "suggested_action": self.suggested_action,
            "metadata": self.metadata,
        }


class AlertEngine:
    """Production Alert Prioritization Engine."""

    def __init__(
        self,
        w1: float = 0.45,  # Weight on Misinformation Risk
        w2: float = 0.30,  # Weight on Impact / Reach
        w3: float = 0.25,  # Weight on Escalation / Velocity
        tau_low: float = 0.40,
        tau_crit: float = 0.70,
    ) -> None:
        # Enforce convex combination: w1 + w2 + w3 = 1.0
        total_w = w1 + w2 + w3
        self.w1 = w1 / total_w
        self.w2 = w2 / total_w
        self.w3 = w3 / total_w
        self.tau_low = tau_low
        self.tau_crit = tau_crit
        self._alerts_history: List[AlertNotification] = []

    def compute_composite_score(
        self,
        risk_rm: float,
        impact_i: float,
        escalation_e: float,
    ) -> float:
        """Calculate composite alert score A in [0, 1]."""
        rm = max(0.0, min(1.0, float(risk_rm)))
        imp = max(0.0, min(1.0, float(impact_i)))
        esc = max(0.0, min(1.0, float(escalation_e)))
        return (self.w1 * rm) + (self.w2 * imp) + (self.w3 * esc)

    def determine_severity(self, composite_score: float) -> AlertSeverity:
        """Map composite score to operational severity level."""
        if composite_score >= self.tau_crit:
            return AlertSeverity.CRITICAL
        elif composite_score >= self.tau_low:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW

    def evaluate_and_dispatch(
        self,
        claim_id: str,
        claim_text: str,
        verdict: str,
        calibrated_confidence: float,
        threat_index_rho: float = 0.0,
        reach_count: int = 1000,
        velocity_shares_per_min: float = 10.0,
        t_detect_start_ns: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AlertNotification:
        """Evaluate claim indicators and generate structured operational alert."""
        t_now_ns = time.perf_counter_ns()
        detect_ms = (t_now_ns - t_detect_start_ns) / 1e6 if t_detect_start_ns else 1.5

        # Compute sub-scores:
        # Misinformation risk R_m
        prob_false = calibrated_confidence if verdict in ["FALSE", "DISPUTED"] else (0.2 if verdict == "NEI" else 0.05)
        risk_rm = max(prob_false, threat_index_rho)

        # Impact score I (normalized logarithmic reach up to 1M users)
        import math
        impact_i = min(1.0, math.log10(max(10, reach_count)) / 6.0)

        # Escalation score E (normalized sharing velocity up to 100 shares/min)
        escalation_e = min(1.0, velocity_shares_per_min / 100.0)

        comp_score = self.compute_composite_score(risk_rm, impact_i, escalation_e)
        severity = self.determine_severity(comp_score)

        if severity == AlertSeverity.CRITICAL:
            action = "IMMEDIATE_ANALYST_ESCALATION"
        elif severity == AlertSeverity.MEDIUM:
            action = "QUEUE_FOR_VERIFICATION"
        else:
            action = "PASSIVE_MONITORING"

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        alert_id = f"alt-{claim_id}-{int(time.time()*1000)%1000000:06d}"

        notification = AlertNotification(
            alert_id=alert_id,
            claim_id=claim_id,
            claim_text=claim_text,
            verdict=verdict,
            calibrated_confidence=calibrated_confidence,
            threat_index_rho=threat_index_rho,
            risk_score_rm=risk_rm,
            impact_score_i=impact_i,
            escalation_score_e=escalation_e,
            composite_score_a=comp_score,
            severity=severity,
            created_at=ts,
            detection_to_alert_ms=detect_ms,
            suggested_action=action,
            metadata=metadata or {},
        )

        self._alerts_history.append(notification)
        if len(self._alerts_history) > 10000:
            self._alerts_history.pop(0)

        return notification

    def get_recent_alerts(self, limit: int = 50, min_severity: Optional[AlertSeverity] = None) -> List[AlertNotification]:
        """Query recent alerts matching severity filter."""
        filtered = self._alerts_history
        if min_severity == AlertSeverity.CRITICAL:
            filtered = [a for a in filtered if a.severity == AlertSeverity.CRITICAL]
        elif min_severity == AlertSeverity.MEDIUM:
            filtered = [a for a in filtered if a.severity in (AlertSeverity.MEDIUM, AlertSeverity.CRITICAL)]
        return filtered[-limit:]


# Global alert engine instance
default_alert_engine = AlertEngine()
