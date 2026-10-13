# M14 — Operational Layer: API, Dashboard, Alerts, Audit Log, PII, Transparency

**Objective:** Build the deployable service and analyst dashboard of Sections XI and XIV **for real**, demonstrate them on replayed public data (never on mocked numbers), and implement the ethics safeguards that can honestly be implemented (tamper-evident log, PII sanitisation, transparency API). Everything else gets reworded as a proposal.
**Ledger IDs:** C031, C040 (UI side), C082, C083 (alert weights), C084, C093.
**Depends on:** M11 (frozen models); M13 uses this service for load tests.
**Compute:** RTX 4060 Laptop (development and demo deployment, in WSL2 + Docker Engine or Docker Desktop with the WSL2 backend).
**Output dir:** `milestones/M14_service_ops/`

---

## Steps

### M14.1 — Service API
- **Action:** `src/ares/service/api.py` (FastAPI): `POST /verify` (text | URL | image | audio | video + optional metadata), `GET /trace/{id}`, `GET /health`, `GET /models`. Response JSON: verdict (True / False / Uncertain), **calibrated confidence** (M05), threat index ρ (M10), escalation flag + reason, evidence trace (each passage with S, N, W, contribution, source, date), module outputs, model versions, snapshot hashes. OpenAPI schema committed. Model registry pins exact checkpoints by SHA256.
- **Tests:** contract tests for every endpoint; a golden-trace test for the J&K showcase in replay mode.
- **VRAM residency plan (8 GB):** `configs/service/residency.yaml` declares which models stay resident on the GPU and which are loaded on demand or run on the CPU. Typical setup:
  - DeBERTa-v3-base, the NLI model and the bi-encoder in fp16 stay resident.
  - The 4-bit LLM member is either a separate llama.cpp server with partial offload, or disabled in the low-latency scope.
  - The vision and audio models load per request type.
  - The `/models` endpoint and every trace report the residency configuration, and M13 latency is reported per configuration.

### M14.2 — Dashboard
- **Action:** Rebuild `index.html` into a proper front-end (Vite + vanilla JS/TS or a lightweight framework; modern dark UI). Panels: live feed (flagged items with verdict, confidence, severity), categorical filters (domain), evidence-trace table (S, N, W, contribution, the Truth' equation), entity links to Wikidata, propagation network view (M09 graphs), regional heatmap T(r) (replaces Fig. 19), and time slider replay.
- **Rule:** every screenshot used in the paper comes from the dashboard running on **replayed public data** (M14.7), with the dataset named in the caption.

### M14.3 — Alerts (Tab. XII; Sec. XI-C)
- **Action:** `service/alerts.py`: A = w1·R_m + w2·I + w3·E with levels Low / Medium / Critical (τ_low, τ_crit). Weights and thresholds are either heuristic (documented) or proxy-fitted (M10). The paper text must say which. Queue: Redis streams (Kafka optional; documented). Detection-to-alert latency instrumented (feeds M13.3, C028).

### M14.4 — Tamper-evident audit log (honest replacement for "blockchain")
- **Action:** `service/auditlog.py`: append-only JSONL where each record contains `prev_hash` and `hash = SHA256(prev_hash ‖ canonical_json(record))`; periodic anchoring of the head hash to a file (optionally to a public timestamping service such as OpenTimestamps; document). `scripts/verify_auditlog.py` detects any modification. Tests: tamper one byte → verification fails.
- **Ledger:** C084 "blockchain logging" → REVISED to "hash-chained tamper-evident log".

### M14.5 — PII sanitisation
- **Action:** Microsoft Presidio analyzer + anonymizer in preprocessing; configurable entity types; evaluate precision/recall on a small public PII benchmark or a synthetic test set (labelled SIMULATED). Verify that logs and traces don't contain raw PII (automated test).

### M14.6 — Transparency API
- **Action:** `GET /transparency/decisions?from=&to=` returns decision traces (PII-scrubbed) with API-key access control and rate limits. Documented in the OpenAPI schema.

### M14.7 — Replay demo on public data
- **Action:** `scripts/m14/replay.py` streams a public dataset in timestamp order (PHEME events, or a dated AVeriTeC/X-FACT subset) through the live service to populate the dashboard. Capture screenshots (replacing Figs. 19, 38, 39, 40) and a short screen recording. Captions name the dataset and date range.

### M14.8 — Analyst console for HITL (enables a future real study)
- **Action:** An analyst review queue UI with per-item timing instrumentation and decision capture. **Any real-user pilot is human-gated:** consent, ethics approval if required by the institution, and a protocol. Write `BLOCKED.md` for the human to decide. Without a real study, C031 stays REMOVE and T_human stays unmeasured.

### M14.9 — Packaging
- **Action:** Dockerfile(s) + docker-compose (api, worker, redis, dashboard); `.env.example`; security basics (no secrets in repo, CORS config, input size limits); `docs/deploy.md`.

### M14.10 — Ethics section reconciliation (C084)
- **Action:** `README` table: each Section XIV safeguard → Implemented (with file) / Proposed (no implementation). Federated learning, ethical licensing, kill switch, and oversight board → "Proposed".

---

## Audit gate — milestone-specific red flags
- RF1: No screenshot or figure built from mocked/hard-coded data (auditor checks the replay logs that produced each screenshot).
- RF2: PII absent from logs/traces (automated test + manual sample of 50 records).
- RF3: Audit-log verification detects tampering (test evidence).
- RF4: Alert weights described truthfully in code comments and README.
- RF5: The confidence shown in the UI is the calibrated probability, not Conf_POC.
- RF6: No secrets committed (`git secrets` / `trufflehog` scan).

## Exit criteria
- [ ] API, dashboard, alerts, audit log, PII, and transparency endpoints implemented and tested.
- [ ] Replay demo screenshots + recording produced from public data.
- [ ] Ethics implemented-vs-proposed table complete; ledger C031, C082, C083, C084 updated.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m14-complete`.
