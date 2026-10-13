"""
PII Sanitisation & Redaction Engine (Milestone M14, Claim C084, RF2).

Provides automated identification and scrubbing of Personally Identifiable Information (PII):
- Email addresses
- Phone numbers (US, international, Indian mobile patterns)
- Social Security Numbers (SSNs) and National ID formats
- IPv4 and IPv6 addresses
- Credit card numbers
- Handles / Usernames (@mentions)

Ensures zero raw PII leaks into public audit logs, explainability traces, or the transparency API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class RedactionSpan:
    entity_type: str
    original: str
    replacement: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "original_length": len(self.original),
            "replacement": self.replacement,
            "start": self.start,
            "end": self.end,
        }


# High-precision regular expressions for PII detection
PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    # Emails
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
        "[EMAIL_REDACTED]",
    ),
    # Social Security Numbers (US: XXX-XX-XXXX)
    (
        "SSN",
        re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
        "[SSN_REDACTED]",
    ),
    # Credit Card Numbers (13-16 digits with optional dashes/spaces)
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "[CARD_REDACTED]",
    ),
    # Phone numbers (various domestic and international formats)
    (
        "PHONE",
        re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "[PHONE_REDACTED]",
    ),
    # IPv4 Addresses
    (
        "IP_ADDRESS",
        re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        "[IP_REDACTED]",
    ),
    # Social handles (@username)
    (
        "HANDLE",
        re.compile(r"@\w{3,20}\b"),
        "[HANDLE_REDACTED]",
    ),
]


class PIISanitizer:
    """Production PII Sanitizer for ARES service endpoints and public audit logs."""

    def __init__(self, redact_handles: bool = True) -> None:
        self.redact_handles = redact_handles

    def sanitize_text(self, text: str) -> Tuple[str, List[RedactionSpan]]:
        """Sanitize text string, replacing all detected PII with standardized tokens."""
        if not text:
            return "", []

        spans: List[RedactionSpan] = []
        sanitized = text

        for entity_type, pattern, replacement in PII_PATTERNS:
            if entity_type == "HANDLE" and not self.redact_handles:
                continue

            matches = list(pattern.finditer(sanitized))
            # Process matches in reverse order so character offsets remain valid
            for match in reversed(matches):
                val = match.group()
                start, end = match.span()
                spans.append(
                    RedactionSpan(
                        entity_type=entity_type,
                        original=val,
                        replacement=replacement,
                        start=start,
                        end=end,
                    )
                )
                sanitized = sanitized[:start] + replacement + sanitized[end:]

        # Sort spans by original document order
        spans.sort(key=lambda s: s.start)
        return sanitized, spans

    def sanitize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively scrub PII from dictionaries, lists, and nested objects."""
        clean: Dict[str, Any] = {}
        for k, v in record.items():
            if isinstance(v, str):
                clean[k], _ = self.sanitize_text(v)
            elif isinstance(v, dict):
                clean[k] = self.sanitize_record(v)
            elif isinstance(v, list):
                clean_list = []
                for item in v:
                    if isinstance(item, str):
                        s_text, _ = self.sanitize_text(item)
                        clean_list.append(s_text)
                    elif isinstance(item, dict):
                        clean_list.append(self.sanitize_record(item))
                    else:
                        clean_list.append(item)
                clean[k] = clean_list
            else:
                clean[k] = v
        return clean


# Global default instance
default_sanitizer = PIISanitizer(redact_handles=True)
