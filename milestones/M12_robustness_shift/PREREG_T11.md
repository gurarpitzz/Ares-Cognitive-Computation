# Pre-Registration: Track T11 — Post-Freeze Temporal External Validation

**Track:** T11 — Post-Freeze Temporal External Validation  
**Milestone:** M12 / M15  
**Freeze Reference Milestone:** `ares-v2-freeze-1` (`5652ae8`, October 13, 2026)  
**Target Ledger Claim:** `C060` (Replaces ungrounded manuscript claim of "Live OSINT agreement > 88% on Telegram and regional X")  
**Status:** PRE-REGISTERED  
**Author:** ARES Engineering Team  

---

## 1. Scientific Rationale

Section XXVI-E of the ARES monograph asserted:
> *"Live OSINT deployment across regional Telegram channels and X feeds achieved > 88 % agreement with subsequent IFCN fact-checks."*

As documented in `01_MANUSCRIPT_AUDIT.md` §4, this statement described unverified real-time scraping and unrecorded manual evaluations that cannot be reproduced from code. To preserve absolute scientific integrity, Track T11 replaces this statement with a **genuine, prospective, post-freeze temporal external validation** conducted on fact-checked claims published strictly after the model freeze date.

---

## 2. Prospective Data Collection Protocol

### 2.1 Collection Source & Query Window
- **Primary Source:** Google Fact Check Tools API querying ClaimReview structured data.
- **Participating IFCN Signatories:**
  - United States: *PolitiFact*, *FactCheck.org*, *The Washington Post Fact Checker*, *Lead Stories*
  - India: *BOOM Live*, *Alt News*, *Factly*, *Vishvas News*, *Quint Webqoof*
  - International / Multilingual: *Full Fact (UK)*, *AFP Fact Check (Multilingual)*, *Deutsche Welle (DW)*
- **Temporal Eligibility Criterion:**
  $$\text{reviewDate} > \text{2026-10-13T00:00:00Z} \quad (\text{tag: } \texttt{ares-v2-freeze-1})$$
  Claims published or reviewed on or before October 13, 2026 are strictly excluded.
- **Collection Window:** Minimum prospective window of 4 weeks post-freeze (scheduled evaluation execution in M15).
- **Target Sample Size:** $n \ge 300$ verified unique claims.

---

## 3. Pre-Registered Rating $\to$ Veracity Label Mapping

Different fact-checking organizations employ proprietary rating vernacular. To eliminate subjective post-hoc labeling, the mapping from publisher textual ratings to binary veracity labels is fixed unconditionally:

| Publisher Rating String | Pre-Registered Mapping | Action / Ingestion |
|---|---|---|
| `False`, `Pants on Fire`, `Incorrect`, `Fake`, `Distorted`, `Misleading`, `Fabricated` | **FALSE (0)** | Ingested |
| `True`, `Correct`, `Accurate`, `Verified`, `Supported` | **TRUE (1)** | Ingested |
| `Mostly False`, `Barely True` | **FALSE (0)** | Ingested (Sensitivity: Binary Strict) |
| `Mostly True` | **TRUE (1)** | Ingested (Sensitivity: Binary Strict) |
| `Half True`, `Mixture`, `Unproven`, `Unverified`, `Needs Context`, `No Evidence` | **EXCLUDED** | Dropped from T/F evaluation; logged to `excluded_ratings.csv` |

---

## 4. Strict Contamination & Temporal Leakage Controls (Audit Red Flags RF2, RF4)

1. **Deduplication against Training Memory:**
   Every collected prospective claim is compared against all training claims in ARES-Bench (T1–T9) using MinHash Jaccard similarity and normalized string matching. Any prospective item with Jaccard similarity $\ge 0.80$ to any historical training item is permanently discarded.
2. **Strict Evidence Timestamp Constraint:**
   When running the evidence retrieval engine, an explicit temporal filter is applied:
   $$t_{\text{evidence}} \le t_{\text{claim}}$$
   Pages or articles published after the claim date are excluded from retrieval to prevent hindsight data leakage.
3. **Fact-Check Domain Blacklist:**
   Search queries explicitly exclude fact-checking domains (e.g., `-site:politifact.com`, `-site:factcheck.org`, `-site:boomlive.in`, `-site:altnews.in`) to ensure ARES discovers primary corroborating or refuting evidence rather than merely retrieving the fact-checker's verdict.

---

## 5. Evaluation Metrics & Decision Rules

1. **Primary Metric:** Macro-F1 across binary classes with 95% percentile bootstrap confidence intervals ($B = 1000$).
2. **Agreement Metric:** Empirical agreement rate with IFCN consensus verdicts:
   $$\text{Agreement} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i^{\text{IFCN}})$$
3. **Chow Abstention Evaluation:** Agreement evaluated under cost-asymmetric abstention ($\tau \in [0.50, 0.90]$), reporting coverage vs agreement curves.
4. **Reconciliation of Claim `C060`:**
   The manuscript claim `C060` is marked **REVISED** in the ledger, citing this protocol, the measured prospective agreement rate, and the exact collection window.
