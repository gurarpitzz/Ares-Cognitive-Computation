"""
Unit and Integration Tests for Operational Service & Safeguards (Milestone M14).

Tests:
1. PII Sanitisation & zero-leak enforcement (RF2, Claim C084).
2. Cryptographic audit log hash chaining and single-byte tamper detection (RF3, Claim C084, H14.1).
3. Operational alert prioritization, heuristic weights, and monotonicity (RF4, Claims C082, C083, H14.3).
4. FastAPI service endpoints (/health, /models, /verify, /trace, /alerts, /transparency, /killswitch).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import httpx
import pytest

from ares.service.alerts import AlertEngine, AlertSeverity
from ares.service.api import app
from ares.service.auditlog import AuditLogger, verify_auditlog
from ares.service.pii import PIISanitizer


def test_pii_sanitizer_entity_redaction():
    """Verify PII engine redacts emails, SSNs, credit cards, phones, IPs, and handles."""
    sanitizer = PIISanitizer(redact_handles=True)
    raw_text = (
        "Contact investigator Jane Doe at jane.doe@forensics.org or call 555-234-5678. "
        "User SSN is 123-45-6789 and router IP is 192.168.1.1. Reported by @alert_agent."
    )

    clean_text, spans = sanitizer.sanitize_text(raw_text)

    # Verify no raw PII remains
    assert "jane.doe@forensics.org" not in clean_text
    assert "555-234-5678" not in clean_text
    assert "123-45-6789" not in clean_text
    assert "192.168.1.1" not in clean_text
    assert "@alert_agent" not in clean_text

    # Verify standardized redaction tokens exist
    assert "[EMAIL_REDACTED]" in clean_text
    assert "[PHONE_REDACTED]" in clean_text
    assert "[SSN_REDACTED]" in clean_text
    assert "[IP_REDACTED]" in clean_text
    assert "[HANDLE_REDACTED]" in clean_text

    assert len(spans) == 5
    types = {s.entity_type for s in spans}
    assert types == {"EMAIL", "PHONE", "SSN", "IP_ADDRESS", "HANDLE"}


def test_pii_sanitizer_nested_dict():
    """Verify recursive PII scrubbing on nested dictionary structures."""
    sanitizer = PIISanitizer()
    data = {
        "author": "Dr. Smith",
        "contacts": ["admin@hospital.org", "555-987-6543"],
        "metadata": {
            "server": "10.0.0.1",
            "notes": "Verify account with ssn 987-65-4321",
        },
    }

    clean = sanitizer.sanitize_record(data)
    assert clean["contacts"][0] == "[EMAIL_REDACTED]"
    assert clean["contacts"][1] == "[PHONE_REDACTED]"
    assert clean["metadata"]["server"] == "[IP_REDACTED]"
    assert "[SSN_REDACTED]" in clean["metadata"]["notes"]


def test_audit_log_cryptographic_chaining():
    """Verify audit log appends blocks with continuous SHA-256 hash chaining."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "audit.jsonl"
        logger = AuditLogger(log_path)

        # Append 5 decisions
        for i in range(5):
            rec = logger.append_decision(
                claim_id=f"claim-{i:03d}",
                verdict="FALSE" if i % 2 == 0 else "TRUE",
                confidence=0.85,
                threat_index=0.40,
                metadata={"test_idx": i},
            )
            assert rec.block_index == i
            assert len(rec.block_hash) == 64

        assert logger.block_count == 5

        # Verify clean log passes verification
        res = verify_auditlog(log_path)
        assert res.is_valid is True
        assert res.total_blocks == 5
        assert res.head_hash == logger.head_hash


def test_audit_log_tamper_detection(tmp_path: Path):
    """Verify that tampering a single byte in any past record fails verification (H14.1, RF3)."""
    log_path = tmp_path / "audit_tamper.jsonl"
    logger = AuditLogger(log_path)

    # Append 4 blocks
    for i in range(4):
        logger.append_decision(
            claim_id=f"audit-claim-{i}",
            verdict="TRUE",
            confidence=0.90,
            threat_index=0.20,
            metadata={"status": "clean"},
        )

    # Clean check
    clean_res = verify_auditlog(log_path)
    assert clean_res.is_valid is True

    # Tamper with block 1: modify verdict from TRUE to FALSE
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4

    block_1_dict = json.loads(lines[1])
    block_1_dict["verdict"] = "FALSE"  # 1-word modification
    lines[1] = json.dumps(block_1_dict)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Verify auditlog must detect the corruption at block 1
    tamper_res = verify_auditlog(log_path)
    assert tamper_res.is_valid is False
    assert tamper_res.corrupted_block_index == 1
    assert "Payload hash mismatch" in tamper_res.error_reason or "Previous hash broken" in tamper_res.error_reason


def test_alert_prioritization_engine():
    """Verify alert formula A = w1*Rm + w2*I + w3*E and priority tiers (C082, C083, RF4)."""
    engine = AlertEngine(w1=0.45, w2=0.30, w3=0.25, tau_low=0.40, tau_crit=0.70)

    # Case 1: Low risk, low reach, low velocity
    score_low = engine.compute_composite_score(risk_rm=0.10, impact_i=0.20, escalation_e=0.10)
    assert score_low < 0.40
    assert engine.determine_severity(score_low) == AlertSeverity.LOW

    # Case 2: High risk, high virality -> Critical
    score_crit = engine.compute_composite_score(risk_rm=0.90, impact_i=0.80, escalation_e=0.85)
    assert score_crit >= 0.70
    assert engine.determine_severity(score_crit) == AlertSeverity.CRITICAL

    # Case 3: Monotonicity check
    s1 = engine.compute_composite_score(0.3, 0.3, 0.3)
    s2 = engine.compute_composite_score(0.6, 0.3, 0.3)
    s3 = engine.compute_composite_score(0.6, 0.6, 0.6)
    assert s1 < s2 < s3


def test_fastapi_endpoints_full_lifecycle():
    """Integration test verifying full REST lifecycle across FastAPI endpoints."""
    import asyncio

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Health check
            resp_health = await client.get("/health")
            assert resp_health.status_code == 200
            health_data = resp_health.json()
            assert health_data["status"] == "healthy"
            assert health_data["service"] == "ares-verification-api"

            # 2. Models registry
            resp_models = await client.get("/models")
            assert resp_models.status_code == 200
            models_data = resp_models.json()
            assert "resident_models" in models_data
            assert len(models_data["resident_models"]) >= 3

            # 3. Post verification
            claim_payload = {
                "claim_id": "test-claim-integration-001",
                "claim_text": "Debunked fake report alleges official emergency evacuation order in district.",
                "has_media": False,
                "author_handle": "@test_user_42",
                "location": "Central District",
                "reach_estimate": 25000,
                "shares_per_minute": 35.0,
            }
            resp_verify = await client.post("/verify", json=claim_payload)
            assert resp_verify.status_code == 200
            ver_data = resp_verify.json()
            assert ver_data["claim_id"] == "test-claim-integration-001"
            assert ver_data["verdict"] in ["TRUE", "FALSE", "NEI", "DISPUTED"]
            assert 0.0 <= ver_data["calibrated_confidence"] <= 1.0
            assert ver_data["latency_ms"] > 0.0

            # 4. Get trace
            resp_trace = await client.get(f"/trace/{ver_data['claim_id']}")
            assert resp_trace.status_code == 200
            trace_data = resp_trace.json()
            assert trace_data["claim_id"] == "test-claim-integration-001"
            assert len(trace_data["evidence_passages"]) >= 2
            assert "[HANDLE_REDACTED]" in trace_data["claim_text"] or "@test_user_42" not in trace_data["claim_text"]

            # 5. Get alerts
            resp_alerts = await client.get("/alerts?limit=10")
            assert resp_alerts.status_code == 200
            alerts_list = resp_alerts.json()
            assert len(alerts_list) >= 1

            # 6. Transparency decisions
            resp_transp = await client.get("/transparency/decisions?limit=5")
            assert resp_transp.status_code == 200
            transp_list = resp_transp.json()
            assert len(transp_list) >= 1

            # 7. Kill switch admin authorization
            # Unauthorized attempt
            resp_unauth = await client.post(
                "/admin/killswitch",
                json={"active": True, "reason": "Test freeze"},
            )
            assert resp_unauth.status_code == 401

            # Authorized activate
            headers = {"authorization": "Bearer ares-admin-emergency-key-2026"}
            resp_ks_on = await client.post(
                "/admin/killswitch",
                json={"active": True, "reason": "Test freeze"},
                headers=headers,
            )
            assert resp_ks_on.status_code == 200
            assert resp_ks_on.json()["kill_switch_active"] is True

            # When active, /verify returns 503
            resp_blocked = await client.post("/verify", json=claim_payload)
            assert resp_blocked.status_code == 503

            # Authorized deactivate
            resp_ks_off = await client.post(
                "/admin/killswitch",
                json={"active": False, "reason": "Resume normal ops"},
                headers=headers,
            )
            assert resp_ks_off.status_code == 200
            assert resp_ks_off.json()["kill_switch_active"] is False

            # Verify works again
            resp_resumed = await client.post("/verify", json=claim_payload)
            assert resp_resumed.status_code == 200

    asyncio.run(_run())
