# Milestone M14 Pre-Registration — Operational Layer: API, Dashboard, Alerts, Audit Log, PII, Transparency

**Milestone:** M14 — Operational Layer: API, Dashboard, Alerts, Audit Log, PII, Transparency  
**Date:** 2026-10-14  
**Status:** PRE-REGISTERED  
**Target Ledger Claims:** `C031`, `C040`, `C082`, `C083`, `C084`, `C093`  
**Execution Environment:** HP OMEN 16 (RTX 4060 Laptop GPU 8 GB, Intel i7-13620H, 16 GB RAM, Windows 11 Home 24H2)  
**Primary Artifacts:**
- Service API (`src/ares/service/api.py`)
- Cryptographic Audit Log & Verifier (`src/ares/service/auditlog.py`, `scripts/verify_auditlog.py`)
- PII Sanitization Engine (`src/ares/service/pii.py`)
- Alert System (`src/ares/service/alerts.py`)
- VRAM Residency Profile (`configs/service/residency.yaml`)
- Analyst Dashboard (`dashboard/index.html`, `dashboard/styles.css`, `dashboard/app.js`)
- Dataset Replay Engine (`scripts/m14/replay.py`)
- Human Study Block Declaration (`milestones/M14_service_ops/BLOCKED.md`)

---

## 1. Experimental Plan & Scope

Milestone M14 transitions the verified offline machine learning models (M01–M13) into an operational, deployable cognitive verification service and analyst dashboard, implementing all feasible ethical safeguards while honestly rewording aspirational concepts as proposals:

1. **Deployable REST API (M14.1, Claim C093):**
   - Endpoints:
     - `POST /verify`: Ingests claim text, URL, media attachments, and contextual metadata; executes calibrated Bayesian fusion; returns verdict, calibrated confidence, threat index $\rho$, escalation flags, and evidence trace.
     - `GET /trace/{claim_id}`: Returns complete explainable evidence graph (similarities $S_i$, stances $N_i$, credibility $W_i$, net contribution to Truth' equation, source URLs, and snapshot hashes; Claim C040).
     - `GET /health`: Reports service status, uptime, GPU VRAM allocation, and thermal telemetry.
     - `GET /models`: Enumerates loaded model architectures, checkpoint SHA256 hashes, and device residency.
     - `GET /transparency/decisions`: Public transparency endpoint returning scrubbed decision traces with rate-limiting.
     - `GET /alerts`: Returns real-time priority alerts with severity classification.
2. **Cryptographic Tamper-Evident Audit Log (M14.4, Claim C084):**
   - Replaces the ungrounded and technically impractical "blockchain" claim with a cryptographically sound append-only Merkle-style hash chain:
     $$\text{hash}_t = \text{SHA256}(\text{hash}_{t-1} \mathbin{\Vert} \text{CanonicalJSON}(\text{record}_t))$$
   - Verification utility `verify_auditlog.py` recalculates hashes across the entire ledger. A single-byte or single-bit modification must cause an immediate verification failure with pinpoint error localisation (Hypothesis H14.1).
3. **PII Sanitization Engine (M14.5, Claim C084):**
   - Preprocessing module sanitizing personally identifiable information (emails, phone numbers, SSNs, IP addresses, full legal names) using regex and named-entity patterns before writing to public audit logs or transparency feeds (Hypothesis H14.2).
4. **Operational Alert Prioritization (M14.3, Claims C082, C083):**
   - Alert severity calculation:
     $$A = w_1 \cdot R_m + w_2 \cdot I + w_3 \cdot E$$
     where $R_m$ is misinformation risk / severity, $I$ is impact potential (reach / virality), and $E$ is escalation likelihood.
   - Classification into Low, Medium, and Critical priority tiers based on thresholds $\tau_{\text{low}}, \tau_{\text{crit}}$.
   - Transparently disclaims "learned model weights" in code and documentation, specifying heuristic proxy-fitted parameters.
5. **Real Replay Demo on Public Data (M14.2, M14.7, Claims C040, C082):**
   - Replay dated claims from public benchmarks (PHEME rumor dataset and X-FACT) through the live API to populate the analyst dashboard.
   - Capture genuine dashboard screenshots and interface demonstrations showing real data traces (replacing manuscript Figures 19, 38, 39, 40).
6. **Analyst Human-in-the-Loop Study Gate (M14.8, Claim C031):**
   - Refuse to fabricate synthetic human analyst response times for Claim C031. Create `BLOCKED.md` stating that measuring human analyst response times requires human participants and institutional ethics / IRB approval.

---

## 2. Red-Flag Anti-Cheat Gates (RF1–RF6)

| Gate ID | Requirement | Verification Protocol | Pass Criteria |
| :--- | :--- | :--- | :--- |
| **RF1** | **No Mocked Screenshot Data** | Dashboard must consume genuine live API responses generated from replayed public benchmark data (PHEME / X-FACT). | Zero hardcoded claim strings or mock verdicts in dashboard templates. |
| **RF2** | **Zero Raw PII in Public Logs** | Automated test scanning audit logs and transparency API outputs for raw PII. | Zero un-anonymized phone numbers, emails, or SSNs in outputs. |
| **RF3** | **Audit Log Tamper-Evidence** | Test tampering 1 byte in audit log JSONL file; verify detection. | `verify_auditlog` must return non-zero exit code and flag corrupted line. |
| **RF4** | **Honest Alert Documentation** | Alert weights $w_1, w_2, w_3$ documented as heuristic / proxy-fitted. | Monograph claim of "learned model weights" explicitly refuted. |
| **RF5** | **Calibrated Posterior Confidence** | Displayed confidence must be Platt/temperature-calibrated $P(Y=y\|x)$. | Softmax heuristics or raw logits prohibited. |
| **RF6** | **Zero Committed Secrets** | Automated scan for API keys, tokens, or credentials. | Clean git repo; `.env.example` template provided. |

---

## 3. Pre-Registered Hypotheses

- **H14.1 (Tamper-Evident Hash Chain Integrity):** Any single-bit or single-byte alteration to any past audit log record causes `verify_auditlog` to fail with $100\%$ detection rate ($p < 0.001$).
- **H14.2 (PII Sanitization Recall):** The PII scrubbing pipeline achieves $\ge 95\%$ recall on standard PII entity types (emails, phone numbers, SSNs, IP addresses), ensuring privacy preservation in transparency feeds.
- **H14.3 (Alert Escalation Monotonicity):** Alert score $A = w_1 R_m + w_2 I + w_3 E$ correlates positively and monotonically with simulated/observed threat levels (Spearman rank correlation $\rho > 0.80, p < 0.001$).
- **H14.4 (8 GB VRAM Operational Stability):** The residency profile maintains GPU memory consumption below $7.5\text{ GB}$ during concurrent multi-modal claim evaluation, avoiding CUDA out-of-memory errors on the RTX 4060 laptop.

---

## 4. Ethics Safeguards Implementation Matrix (Section XIV Audit)

| Section XIV Safeguard | Status in ARES 2.0 | Implementation File / Evidence | Action in Revised Monograph |
| :--- | :---: | :--- | :--- |
| **Tamper-Evident Audit Logging** | **IMPLEMENTED** | `src/ares/service/auditlog.py` | Replace "blockchain" claim with SHA-256 hash-chained log. |
| **PII Sanitisation** | **IMPLEMENTED** | `src/ares/service/pii.py` | Document automated regex/NER scrubbing pipeline. |
| **Public Transparency API** | **IMPLEMENTED** | `GET /transparency/decisions` | Document OpenAPI schema and rate-limited public access. |
| **Analyst Kill Switch** | **IMPLEMENTED** | `POST /admin/killswitch` | Document emergency pipeline pause and routing fallback. |
| **Federated Learning** | **PROPOSED** | N/A (Theoretical Architecture) | Label as "Future Proposal: Federated Edge Aggregation". |
| **Independent Oversight Board** | **PROPOSED** | N/A (Governance Proposal) | Label as "Governance Recommendation". |
| **Ethical Data Licensing** | **PROPOSED** | N/A (Policy Framework) | Label as "Policy Proposal". |

---

## 5. Decision Rules & Success Criteria

1. All FastAPI endpoints return HTTP 200 with valid schema contracts.
2. `verify_auditlog.py` passes on clean log and fails on tampered log.
3. PII tests verify zero leaks.
4. Dashboard runs cleanly in browser and renders live replayed public benchmark data.
5. All 104+ pytest unit tests remain passing.
6. Target ledger claims `C031`, `C040`, `C082`, `C083`, `C084`, `C093` updated in `claims_ledger.csv`.
