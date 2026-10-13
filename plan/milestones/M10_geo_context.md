# M10 — Geopolitical & Contextual Intelligence

**Objective:** Implement Sections V and IX concretely: NER, topic classification, geotagging, Wikidata entity linking, **GDELT event corroboration**, Factbook context, and the risk scores (I_e, E_f, P_g, ρ). Evaluate each component on a public benchmark where one exists. Give the geo module a measurable route into veracity, and be explicit about what can't be validated (no ground-truth "geopolitical risk" labels exist).
**Ledger IDs:** C007, C008, C009, C083 (+ geo inputs for C043, C047, C053, C054 in M11).
**Depends on:** M02.
**Compute:** RTX 4060 Laptop, mostly CPU/network. API snapshotting is the bottleneck, so run it alongside GPU jobs. About 3 GPU-h for NER/geotag inference. GDELT: query only the needed date and keyword windows; never mirror the full GDELT (storage).
**Output dir:** `milestones/M10_geo_context/`

---

## Pre-registration
- **H10.1:** NER (spaCy transformer) F1 on CoNLL-2003 test (descriptive; sanity check vs the published model card).
- **H10.2:** Toponym resolution accuracy@161 km on a public benchmark (e.g. LGL, GeoWebNews, or WikToR; pick one that's accessible, cite it).
- **H10.3:** The GDELT event-corroboration score separates true from false **event claims** (claims with an event type + location + date) with AUC > 0.5 on AVeriTeC/X-FACT event subsets (CI excludes 0.5 to claim it).
- **H10.4:** Risk scores ρ have no ground truth. Pre-register a **proxy** validation (e.g. correlation of ρ with engagement/virality in PHEME/FakeNewsNet, or with fact-checker selection) and state that it's only a proxy.
- **Domain mapping (needed by M05/M11):** pre-register how each item gets a domain label (election / health / conflict / disaster / other): dataset-source rules first (e.g. COVID datasets → health), else the zero-shot topic classifier with a validated threshold.

## Steps

### M10.1 — NER (Tab. IV; Eqs. 5–6)
- **Action:** `src/ares/geo/ner.py`: spaCy `en_core_web_trf` + a multilingual model for non-English. Evaluate on CoNLL-2003 test (or WikiNER if licensing blocks CoNLL). Regenerate Table IV from real model outputs on real sentences (C007).

### M10.2 — Topic classification (Tab. V; Eq. 7)
- **Action:** `geo/topic.py`: (a) fine-tuned DeBERTa-v3-base on a public news-topic dataset (e.g. AG News) and/or (b) zero-shot NLI over the ARES domain set. Evaluate (a) on its test set; validate the domain mapping for (b) against items with known source-domain labels. Regenerate Table V from real outputs (C008).

### M10.3 — Geotagging (Tab. VI; Eq. 5 in §V-C)
- **Action:** `geo/geotag.py`: location entities → GeoNames candidates → disambiguation (population prior + country context + co-occurring toponyms); EXIF GPS parsing for images (Pillow/exifread). Evaluate H10.2. Regenerate Table VI from real outputs (C009).

### M10.4 — Wikidata entity linking
- **Action:** `geo/wikidata_link.py`: candidate generation via the Wikidata search API (snapshotted) + context-embedding re-ranking. Evaluate on a public EL benchmark subset if accessible (e.g. AIDA-CoNLL, licence permitting), else report coverage + precision on a sample annotated by the human (BLOCKED for human annotation, n = 200, two passes if possible). Outputs feed the logic layer (M04) and E_f.

### M10.5 — GDELT event corroboration (the measurable geo contribution)
- **Action:** `geo/gdelt.py`: for a claim with extracted (event type, location, date): map event type to CAMEO root codes (14 protest, 18 assault, 19 fight, 20 unconventional mass violence) or GKG themes for disasters; query GDELT 2.0 (raw 15-min event files for the date window, or BigQuery within free-tier limits) for window [date − k, date + k] (k pre-registered, e.g. 2 days) **but never using data published after the claim's publication/collection time**. Features: event count in region, unique sources, mean Goldstein, mean tone, normalised against the regional baseline rate. All responses snapshotted.
- **Evaluation:** H10.3 on event-claim subsets; cache features out-of-fold for M11.
- **Showcase:** run on the J&K claim (date 2026-02-01, Jammu & Kashmir, explosion/attack event types); include in the trace.

### M10.6 — World Factbook context
- **Action:** `geo/factbook.py`: archived Factbook JSON snapshot (record source and date); country-level features (population, government type, conflict/refugee indicators as available). Test whether they add anything to fusion (likely weak; report honestly).

### M10.7 — Risk scores (Sec. IX-B; Eqs. for I_e, E_f, P_g; ρ)
- **Action:** `geo/risk.py`: implement the formulas as written. Weights: either set heuristically and documented as such, or fitted against the pre-registered proxy (H10.4) and called "fitted to proxy X". Never call them "learned" without naming the label. ρ feeds the escalation rule (M05/M11).
- **Output:** `tables/risk_proxy_validation.csv`; ledger C083 → REVISED with an honest description.

### M10.8 — Feature cache
- **Action:** Per-item context features (entities, geotags, GDELT corroboration, Factbook, ρ, domain) for all fusion datasets → out-of-fold cache with manifest.

---

## Audit gate — milestone-specific red flags
- RF1: **Temporal leakage in GDELT:** no event records with ingestion time after the claim's timestamp (check DATEADDED vs claim date).
- RF2: Wikidata facts evaluated at the claim time (qualifiers), not today's state.
- RF3: Geotags never taken from the fact-check article's metadata.
- RF4: Snapshot completeness: experiments run in replay mode with zero misses.
- RF5: Risk-score weights described truthfully (heuristic vs proxy-fitted).
- RF6: Regenerated Tables IV–VI come from real model outputs (auditor reruns 3 examples).

## Exit criteria
- [ ] Components implemented and evaluated on public benchmarks where available.
- [ ] GDELT corroboration evaluated on event claims; J&K showcase geo trace.
- [ ] Risk scores implemented with honest weight provenance and proxy validation.
- [ ] Ledger C007, C008, C009, C083 updated.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m10-complete`.
