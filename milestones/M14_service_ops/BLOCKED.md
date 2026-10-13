# Blocked Decision Gate: Human Analyst User Study (Claim C031)

**Milestone:** M14 — Operational Layer: API, Dashboard, Alerts, Audit Log, PII, Transparency  
**Claim ID:** `C031` ("Average analyst response time by alert severity", Section XIII, Figure 22)  
**Date:** 2026-10-14  
**Status:** **BLOCKED / ACTION: REMOVE**  

---

## 1. Description of Blocked Research Activity

Section XIII (Figure 22) of the monograph reported an alleged empirical human study measuring:
> "Average analyst response time across alert severity levels (~5–30 minutes) under operational deployment."

Under strict scientific integrity protocols:
1. **Institutional Ethics Approval (IRB):** Conducting a genuine study on human subjects (measuring response time, fatigue, cognitive load, or operational decision accuracy) requires formal Institutional Review Board (IRB) / Human Subjects Research approval, informed participant consent, and data protection protocols.
2. **Zero Fabrication Policy:** Antigravity and the ARES research protocol strictly prohibit fabricating synthetic human response latencies, generating mock participant questionnaires, or plotting simulated response distributions as if they were collected from human operators.
3. **No Authorized Study Protocol:** The repository does not possess an active IRB protocol number, participant recruitment pool, or signed consent manifests.

---

## 2. Scientific Decision & Manuscript Action

- **Ledger Status:** `C031` is formally updated to **REMOVE** in `plan/claims_ledger.csv`.
- **Manuscript Correction:**
  - Figure 22 and associated text claiming an empirical user study must be removed from the manuscript.
  - Alternatively, the paper may present a parameterized theoretical queueing sensitivity analysis (e.g., $M/M/k$ queue model simulating analyst queue backlog across hypothetical review rates $\mu$), explicitly labeled as a **mathematical sensitivity simulation**, NOT as an empirical human study.
- **Analyst Queue Implementation:** The operational analyst queue data structures, priority sorting, and review dispatch hooks are fully implemented in the ARES 2.0 software (`src/ares/service/alerts.py` and `dashboard/`), ready to log genuine review times if and when an approved institutional user study is conducted.

---

## 3. Auditor Sign-Off

The decision to block Claim `C031` and mandate the removal of fabricated human response times upholds the highest standards of scientific ethics and institutional compliance.
