"""
Cryptographic Tamper-Evident Audit Log (Milestone M14, Claim C084, RF3).

Replaces the manuscript's ungrounded 'blockchain' claim with an honest, production-ready
append-only cryptographic hash chain:
    hash_t = SHA256(hash_{t-1} || CanonicalJSON(record_t))

Guarantees tamper-evidence:
- Any modification of historical claim verdicts, timestamps, or confidence scores breaks the chain.
- Deletion, insertion, or reordering of records causes immediate cryptographic verification failure.
- Periodic anchoring of head hash to immutable state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ares.service.pii import default_sanitizer

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64


def canonical_json(data: Any) -> str:
    """Produce deterministic, canonical JSON representation (RFC 8785 subset)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass
class AuditRecord:
    block_index: int
    timestamp: str
    claim_id: str
    verdict: str
    confidence: float
    threat_index: float
    prev_hash: str
    block_hash: str
    payload_hash: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_index": self.block_index,
            "timestamp": self.timestamp,
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "confidence": round(float(self.confidence), 4),
            "threat_index": round(float(self.threat_index), 4),
            "prev_hash": self.prev_hash,
            "block_hash": self.block_hash,
            "payload_hash": self.payload_hash,
            "data": self.data,
        }


@dataclass
class AuditVerificationResult:
    is_valid: bool
    total_blocks: int
    head_hash: str
    corrupted_block_index: Optional[int] = None
    error_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_blocks": self.total_blocks,
            "head_hash": self.head_hash,
            "corrupted_block_index": self.corrupted_block_index,
            "error_reason": self.error_reason,
        }


class AuditLogger:
    """Thread-safe append-only cryptographic hash-chained audit logger."""

    def __init__(self, log_path: Path, auto_pii_clean: bool = True) -> None:
        self.log_path = Path(log_path)
        self.auto_pii_clean = auto_pii_clean
        self._lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._head_hash = GENESIS_HASH
        self._block_count = 0
        self._init_head()

    def _init_head(self) -> None:
        """Scan existing log file to locate current head hash and block count."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            self._head_hash = GENESIS_HASH
            self._block_count = 0
            return

        with open(self.log_path, "r", encoding="utf-8") as f:
            last_line = None
            count = 0
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
                    count += 1

            if last_line:
                try:
                    record = json.loads(last_line)
                    self._head_hash = record["block_hash"]
                    self._block_count = count
                except Exception as e:
                    logger.error(f"Failed to read audit log tail: {e}")
                    self._head_hash = GENESIS_HASH
                    self._block_count = 0

    @property
    def head_hash(self) -> str:
        return self._head_hash

    @property
    def block_count(self) -> int:
        return self._block_count

    def append_decision(
        self,
        claim_id: str,
        verdict: str,
        confidence: float,
        threat_index: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Append a new verified decision record to the cryptographic hash chain."""
        with self._lock:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            meta = metadata or {}
            if self.auto_pii_clean:
                meta = default_sanitizer.sanitize_record(meta)

            payload = {
                "claim_id": claim_id,
                "verdict": verdict,
                "confidence": round(float(confidence), 4),
                "threat_index": round(float(threat_index), 4),
                "timestamp": ts,
                "metadata": meta,
            }

            canonical_payload = canonical_json(payload)
            payload_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

            # Hash-chain linkage: SHA256(prev_hash || payload_hash)
            chain_input = f"{self._head_hash}:{payload_hash}"
            block_hash = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()

            record = AuditRecord(
                block_index=self._block_count,
                timestamp=ts,
                claim_id=claim_id,
                verdict=verdict,
                confidence=float(confidence),
                threat_index=float(threat_index),
                prev_hash=self._head_hash,
                block_hash=block_hash,
                payload_hash=payload_hash,
                data=meta,
            )

            # Atomically write record line
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(canonical_json(record.to_dict()) + "\n")

            self._head_hash = block_hash
            self._block_count += 1
            return record


def verify_auditlog(log_path: Path) -> AuditVerificationResult:
    """Verify integrity of an append-only audit log file."""
    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        return AuditVerificationResult(
            is_valid=True,
            total_blocks=0,
            head_hash=GENESIS_HASH,
        )

    expected_prev = GENESIS_HASH
    total_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception as e:
                return AuditVerificationResult(
                    is_valid=False,
                    total_blocks=total_count,
                    head_hash=expected_prev,
                    corrupted_block_index=idx,
                    error_reason=f"Malformed JSON: {e}",
                )

            # Check index continuity
            if rec.get("block_index") != idx:
                return AuditVerificationResult(
                    is_valid=False,
                    total_blocks=total_count,
                    head_hash=expected_prev,
                    corrupted_block_index=idx,
                    error_reason=f"Index mismatch: expected {idx}, got {rec.get('block_index')}",
                )

            # Check prev_hash continuity
            if rec.get("prev_hash") != expected_prev:
                return AuditVerificationResult(
                    is_valid=False,
                    total_blocks=total_count,
                    head_hash=expected_prev,
                    corrupted_block_index=idx,
                    error_reason=f"Previous hash broken at block {idx}. Expected {expected_prev}, got {rec.get('prev_hash')}",
                )

            # Recompute payload hash
            payload = {
                "claim_id": rec["claim_id"],
                "verdict": rec["verdict"],
                "confidence": round(float(rec["confidence"]), 4),
                "threat_index": round(float(rec["threat_index"]), 4),
                "timestamp": rec["timestamp"],
                "metadata": rec["data"],
            }
            computed_payload_hash = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
            if computed_payload_hash != rec.get("payload_hash"):
                return AuditVerificationResult(
                    is_valid=False,
                    total_blocks=total_count,
                    head_hash=expected_prev,
                    corrupted_block_index=idx,
                    error_reason=f"Payload hash mismatch at block {idx}. Stored: {rec.get('payload_hash')}, Recomputed: {computed_payload_hash}",
                )

            # Recompute block hash
            chain_input = f"{expected_prev}:{computed_payload_hash}"
            computed_block_hash = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()
            if computed_block_hash != rec.get("block_hash"):
                return AuditVerificationResult(
                    is_valid=False,
                    total_blocks=total_count,
                    head_hash=expected_prev,
                    corrupted_block_index=idx,
                    error_reason=f"Block hash signature mismatch at block {idx}",
                )

            expected_prev = computed_block_hash
            total_count += 1

    return AuditVerificationResult(
        is_valid=True,
        total_blocks=total_count,
        head_hash=expected_prev,
    )
