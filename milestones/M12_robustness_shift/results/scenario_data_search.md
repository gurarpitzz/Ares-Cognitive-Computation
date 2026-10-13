# Search Log: Public Labelled Scenario Datasets (M12.6)

**Milestone:** M12 — Robustness, Distribution Shift, Stress Scenarios & Temporal External Validation  
**Date:** October 13, 2026  
**Auditor / Investigator:** ARES Rebuild Engineering Team  
**Scope:** Search for publicly available, gold-labelled misinformation/fact-checking datasets for:
1. India–China border conflict propaganda (Galwan 2020, Pangong Tso disengagement, Yangtse/Tawang 2022) — Claim `C021`
2. Assam 2022 flood misinformation — Claim `C022`
3. Indian general election misinformation (2024 / historical) — Claim `C020` (India part)

---

## 1. Search Methodology & Repositories Queried

The following primary repositories and academic indexes were queried systematically using structured keyword combinations:
- **HuggingFace Hub:** `datasets` search API
- **Zenodo & Figshare:** Open-access research data repositories
- **Google Fact Check Tools API / ClaimReview:** IFCN-certified Indian fact-checkers (BOOM Live, Alt News, Factly, Vishvas News, Quint Webqoof)
- **Kaggle Datasets & Harvard Dataverse**
- **Google Scholar / Semantic Scholar / arXiv:** Literature review of published misinformation benchmark papers (2020–2026)

---

## 2. Findings by Target Scenario

### 2.1 India–China Border Conflict (Claim `C021`)
- **Query Strings:** `"India China border" AND (misinformation OR fake news OR fact check OR propaganda)`, `"Galwan" AND (fact check OR dataset OR tweets)`, `"Tawang" AND (border conflict OR disengagement)`.
- **Search Results:**
  - *Academic Literature:* Academic papers analyze propaganda narratives (e.g., social media bot activity and sentiment analysis during the June 2020 Galwan clash), but **no public repository hosts a structured, curated, gold-labelled claim-veracity benchmark** with ground-truth evidence tuples.
  - *Fact-Check Portals:* Individual debunking articles exist on BOOM Live and Alt News regarding viral videos falsely attributed to Galwan (e.g., old military training clips or unrelated martial arts drills), but these comprise fewer than 25 isolated articles, far below the required $n \ge 100$ threshold.
- **Finding:** **NEGATIVE**. No public labelled dataset exists meeting the sample size and provenance criteria.
- **Protocol Action:** In accordance with `01_MANUSCRIPT_AUDIT.md` §4 and `M12_robustness_shift.md` §M12.6, the manuscript's real-world claim of "91.1% recall on India-China border conflict propaganda" is **REMOVED** as ungrounded. Performance is evaluated instead on **ARES-Synth Scenario S1** (a controlled grounded-synthetic benchmark built from frozen Wikipedia and MEA/MFA official releases), marked **REVISED** with the qualifier *"on a controlled grounded-synthetic benchmark"*, and captioned **SIMULATED**.

---

### 2.2 Assam 2022 Floods (Claim `C022`)
- **Query Strings:** `"Assam floods" AND 2022 AND (misinformation OR "fake news" OR rumors)`, `"Assam disaster" AND "fact check"`.
- **Search Results:**
  - *Disaster Data:* Official daily disaster situation reports are openly published by the Assam State Disaster Management Authority (ASDMA) and ReliefWeb, providing comprehensive ground-truth statistics on flood levels, affected districts, and casualties.
  - *Labelled Misinformation Datasets:* No curated misinformation benchmark specifically annotates or categorizes false claims regarding the 2022 Assam floods. Fact-checking organizations debunked sporadic false claims (e.g., misrepresented dam release rumors or recycled video footage from 2017), yielding only ~15 discrete claims.
- **Finding:** **NEGATIVE**. No public gold-labelled dataset exists with $n \ge 100$ verified items.
- **Protocol Action:** The manuscript's claim of "94.8% accuracy on Assam 2022 flood misinformation" is **REMOVED** as ungrounded. Performance is evaluated on **ARES-Synth Scenario S2** (controlled grounded-synthetic benchmark derived from ASDMA official bulletins), marked **REVISED** with the qualifier *"on a controlled grounded-synthetic benchmark"*, and captioned **SIMULATED**.

---

### 2.3 Indian Election Misinformation (Claim `C020` — India Component)
- **Query Strings:** `"Indian election" AND (misinformation OR fake news OR fact check)`, `"IFND"`, `"FactDrill"`, `"Indian news"`.
- **Search Results:**
  - *IFND (Indian Fake News Dataset; Verma et al., 2021):* Contains thousands of labelled news articles from Indian media, with a substantial political subset.
  - *FactDrill (Gupta et al., 2022):* Curated fact-checked claims from Indian fact-checkers covering political topics.
  - *ClaimReview via Fact Check Tools API:* Substantial historical coverage of election-related claims debunked by BOOM, Alt News, and Factly ($n > 500$).
- **Finding:** **POSITIVE**. Real labelled data exists for Indian political and election misinformation in IFND and FactDrill.
- **Protocol Action:** Real Indian political and election claims are incorporated into the real-world scenario evaluation (Scenario S_INDIA) and cross-domain generalization matrix. ARES-Synth Scenario S3 (2024 Indian general election) provides a complementary controlled grounded-synthetic benchmark.

---

## 3. Summary of Ledger Actions

| Scenario ID | Topic | Real Data Status | Primary Evaluation Vehicle | Ledger Action & Qualifier |
|---|---|---|---|---|
| `C019` | COVID-19 Infodemic | Found ($n > 10,000$) | Constraint + CoAID real data | **REVISED** with measured empirical F1 |
| `C020` | US Election Manipulation | Found ($n > 1,000$) | LIAR Election subset | **REVISED** with measured empirical F1 |
| `C020`-India | Indian Election Misinformation | Found ($n > 500$) | IFND/FactDrill + ARES-Synth S3 | **REVISED** with real data + synthetic complement |
| `C021` | India–China Border Conflict | **Not Found** | ARES-Synth S1 (SIMULATED) | **REVISED**: *"on a controlled grounded-synthetic benchmark"* |
| `C022` | Assam 2022 Floods | **Not Found** | ARES-Synth S2 (SIMULATED) | **REVISED**: *"on a controlled grounded-synthetic benchmark"* |
