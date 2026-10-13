# Pre-Registration: Milestone M10 — Geopolitical & Contextual Intelligence

**Milestone:** M10 — Geopolitical & Contextual Intelligence  
**Repository:** `gurarpitzz/Ares-Cognitive-Computation`  
**Date:** October 13, 2026  
**Status:** PRE-REGISTERED  
**Commit Baseline:** `98fa016` (`m09-complete`)  
**Hardware Target:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Intel Core i7-13620H, 16 GB RAM  
**Execution Environment:** Windows 11 24H2, Python 3.11.9, PyTorch 2.6.0+cu124, spaCy 3.8.16, transformers 4.49.0  

---

## 1. Scientific Objective and Context

Sections V and IX of the ARES monograph introduce geopolitical and contextual intelligence modules designed to ground multimodal claims in world knowledge, geographic coordinates, external event databases (GDELT), structured ontologies (Wikidata), national demographic indicators (CIA World Factbook), and composite geopolitical risk scoring.

In the original manuscript, multiple tables (Table IV: NER examples, Table V: Topic classification examples, Table VI: Geotagging coordinates) were presented as illustrative examples without published code or systematic benchmark validation. Furthermore, Section IX-B claimed that risk alert weights $w_1, w_2, w_3$ were "learned model weights" without specifying training labels, an optimization loss, or an empirical ground truth dataset.

Milestone M10 resolves these deficiencies through rigorous empirical implementation, reproducible benchmark evaluations on public datasets, transparent provenance documentation, and strict temporal integrity guarantees.

---

## 2. Pre-Registered Hypotheses

### Hypothesis H10.1 — Named Entity Recognition (NER) Benchmark Integrity
* **Hypothesis:** spaCy English NER (`en_core_web_sm` / transformer pipeline) achieves competitive extraction fidelity on the standard CoNLL-2003 test set (3,453 sentences; Sang & De Meulder, 2003) across standard entity types (PER, ORG, LOC/GPE, MISC).
* **Metric:** Macro and entity-level Precision, Recall, and F1 evaluated using the standard `seqeval` span-level evaluation framework.
* **Criterion:** Overall entity F1 $\ge 0.80$ on the CoNLL-2003 test split.
* **Monograph Replacement:** Regenerate Table IV from genuine model inference on the exact monograph input sentences, replacing static placeholders with verifiable model outputs (Claim `C007`).

### Hypothesis H10.2 — Toponym Resolution Accuracy on GeoWebNews
* **Hypothesis:** The ARES toponym resolution pipeline (spaCy entity extractor + GeoNames gazetteer matching + population prior and country-context spatial disambiguation) accurately resolves geographical references on the public GeoWebNews benchmark (Gritta et al., 2018; 200 international news articles, 2,601 ground-truth geocoded coordinates).
* **Metric:** Accuracy@161km ($\text{Acc@161km}$), defined as the percentage of resolved toponyms whose predicted coordinates fall within 161 km (100 miles) of the ground-truth latitude/longitude via the great-circle Haversine formula:
  $$d = 2 R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
* **Criterion:** $\text{Acc@161km} \ge 0.70$ on the GeoWebNews benchmark.
* **Monograph Replacement:** Regenerate Table VI from genuine pipeline execution on the exact monograph input sentences (Claim `C009`).

### Hypothesis H10.3 — GDELT Event Corroboration Discriminability
* **Hypothesis:** External event corroboration from GDELT 2.0 (mapping extracted event types to Conflict and Mediation Event Observations [CAMEO] root codes 14: Protest, 18: Assault, 19: Fight, 20: Unconventional Mass Violence, or disaster themes within a temporal window $[t_{\text{claim}} - k, t_{\text{claim}}]$) reliably discriminates true event claims from fabricated event claims.
* **Metric:** Receiver Operating Characteristic Area Under the Curve (ROC-AUC) with 95% bootstrap confidence interval (1,000 resamples).
* **Criterion:** $\text{ROC-AUC} > 0.50$, with the 95% bootstrap confidence interval strictly excluding 0.50 ($\text{CI}_{95\%} \cap (-\infty, 0.50] = \emptyset$).
* **Temporal Integrity (Audit Red Flag RF1):** Strict adherence to temporal non-leakage. No event records with ingestion timestamp after claim publication ($DATEADDED \le t_{\text{claim}}$) are permitted in corroboration scoring.
* **Showcase:** Concrete trace demonstration on the Jammu & Kashmir blast claim ($t = 2026\text{-}02\text{-}01$, location: J&K, event: kinetic explosion/attack).

### Hypothesis H10.4 — Risk Score Formulation and Proxy Validation
* **Hypothesis:** The composite geopolitical threat index $\rho = \alpha_1 I_e + \alpha_2 E_f + \alpha_3 P_g$ (formulated in Section IX-B) positively correlates with empirical misinformation risk proxies (e.g. social virality, retweet velocity, and fact-checker prioritization urgency).
* **Provenance Correction (Claim `C083`):** We formally refute the manuscript's claim that alert weights are "learned model weights". Because no ground-truth "geopolitical risk" labels exist in public corpora, weights $\alpha_1, \alpha_2, \alpha_3$ are explicitly pre-registered as **heuristic / proxy-fitted parameters**.
* **Metric:** Spearman rank correlation $\rho_s$ and Pearson correlation $r$ between computed $\rho$ and empirical virality / severity proxy.
* **Criterion:** Statistically significant positive correlation ($\rho_s > 0, p < 0.05$).

---

## 3. Domain Mapping Specification

Downstream multimodal fusion (M11) and human-in-the-loop decision routing (M05) require domain labeling. We pre-register the following hierarchical mapping:

1. **Taxonomy:** $\mathcal{D} = \{\text{Politics}, \text{Health}, \text{Conflict}, \text{Disaster}, \text{Economy}, \text{Technology}, \text{Science}, \text{Entertainment}, \text{Other}\}$.
2. **Priority 1 (Source Metadata Rules):**
   - COVID-19 / biomedical datasets (CoAID, Constraint-COVID) $\to$ `Health`.
   - Legislative / election datasets (LIAR, PolitiFact election claims) $\to$ `Politics`.
   - Armed conflict monitoring $\to$ `Conflict`.
   - Natural hazard warnings $\to$ `Disaster`.
3. **Priority 2 (Zero-Shot NLI Classification):**
   - Model: Large NLI transformer (`facebook/bart-large-mnli`), running in FP16 on GPU.
   - Hypothesis template: `"This text is about {domain}."`
   - Confidence threshold: $\tau = 0.40$; if $\max_{d \in \mathcal{D}} P(d \mid x) < \tau$, assign `Other`.
* **Monograph Replacement:** Regenerate Table V from genuine model inference on the monograph sentences, reporting calibrated softmax probabilities (Claim `C008`).

---

## 4. Audit Gates & Red Flags

* **RF1 (GDELT Temporal Horizon):** Every GDELT query must enforce $DATEADDED \le t_{\text{claim}}$. Future event records are strictly quarantined.
* **RF2 (Wikidata Temporal Grounding):** Entity properties and relationships must reflect knowledge valid at claim time.
* **RF3 (Toponym Independence):** Coordinates must be extracted solely from claim text / media EXIF, never from fact-checking debunking articles.
* **RF4 (Deterministic Snapshot Replay):** All external API queries (GeoNames, Factbook, GDELT, Wikidata) must support 100% offline replay from cached JSON/CSV snapshots with zero cache misses.
* **RF5 (Honest Weight Provenance):** Refute the "learned weights" claim in `C083`; document heuristic/proxy provenance in `claims_ledger.csv`.
* **RF6 (Reproducible Table Regeneration):** Tables IV, V, VI must be executable via an automated script with zero manual editing.
