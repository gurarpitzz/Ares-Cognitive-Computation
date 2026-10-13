# M04 — Evidence Engine R·S·N·W + Logic Layer (ARES_POC, done properly)

**Objective:** Rebuild the manuscript's central artefact, the evidence functional Truth'(C) = Σ S·N·W / Σ S, on **frozen, leakage-controlled corpora**, evaluate every operator separately and the pipeline end-to-end (Track T2), add the symbolic logic layer, and operationalise the four "pipeline layers" of Table XV.
**Ledger IDs:** C018, C040, C067, C078 (+ prep for C010).
**Depends on:** M02, M03.
**Compute:** RTX 4060 Laptop, ≈ 15–20 GPU-h (dense encoding of the passages, cross-encoder reranking, NLI inference). BM25 indexing and the logic layer run on the CPU. Index storage is ≈ 15–30 GB on D:.
**Output dir:** `milestones/M04_evidence_engine/`

---

## Pre-registration
- **H4.1:** Replacing BART-large-MNLI (POC) with a stronger NLI model **not trained on FEVER** improves FEVER label accuracy (oracle-evidence setting).
- **H4.2:** Hybrid retrieval (BM25 + dense) has higher evidence recall@5 than either alone on FEVER.
- **H4.3:** The soft-N functional (N = P(ent) − P(contr)) beats hard-N (POC) on FEVER/AVeriTeC macro-F1.
- **H4.4:** Credibility weighting W (Lin et al. ratings) improves AVeriTeC macro-F1 over constant W (exploratory).
- Metrics: FEVER label accuracy, FEVER score, evidence P/R/F1; AVeriTeC official scorer + macro-F1. Thresholds for verdicts are chosen on **dev** only.

## Steps

### M04.1 — Operator interfaces
- **Action:** `src/ares/evidence/{retriever,ranker,nli,credibility,functional}.py` with abstract base classes; each operator returns per-evidence records (`evidence_id, text, source_url, date, score`) so the full trace is preserved. Port the POC formula exactly as `functional.POCFunctional` (hard N ∈ {−1, 0, +1}) for reference.
- **Done when:** unit tests reproduce the POC formula on hand-built examples.

### M04.2 — Frozen corpora & indexes
- **Action:**
  - **FEVER wiki (2017):**
    - Document retrieval with **Pyserini BM25** (a Lucene index on disk, Java 21) over page titles + text. `bm25s` is an alternative only if its memory use stays under ~8 GB.
    - Sentence-level candidate extraction from the top-k docs.
    - Dense retrieval over 100-word passages with FAISS **IVF-PQ** (e.g. `nlist=65536, m=64`, CPU, mmap). Train it on a 1–2 M passage sample, then encode the passages in fp16 batches on the GPU, sharded and resumable. A flat sentence-level dense index does not fit in 16 GB RAM, so don't build one.
    - Hybrid via reciprocal rank fusion.
    - Measure and report the IVF-PQ recall loss against exact search on a 10k-passage sample.
  - **AVeriTeC:** use the official per-claim knowledge store for dev/test only (no open web), streamed per claim.
  - Hash all indexes; record build time, hardware and peak RAM.
- **Done when:** `results/index_build.json` with sizes and hashes; retrieval smoke test returns gold evidence for 10 known dev claims.

### M04.3 — Web retrieval for open claims (showcase + T11 only)
- **Action:** `retriever.WebRetriever` using a search API available at execution time (e.g. Brave Search API or SerpAPI; record which) through `SnapshotClient`. **Evaluation mode enforces:** (1) exclusion of fact-checking domains (list in `configs/evidence/factcheck_domains.yaml`: PolitiFact, Snopes, FactCheck.org, Full Fact, AFP Fact Check, Reuters Fact Check, BOOM, Alt News, Factly, Vishvas News, plus IFCN signatories); (2) evidence published **before** the claim date when known. Log every exclusion.
- **Done when:** replay mode works offline; exclusion filters unit-tested.

### M04.4 — S operator evaluation
- **Action:** Compare `all-MiniLM-L6-v2` (POC), `all-mpnet-base-v2`, `e5-base-v2`, and a cross-encoder re-ranker (e.g. `ms-marco-MiniLM-L-6-v2`) on FEVER dev: evidence recall@{1, 5, 10}, MRR. Choose by dev only.
- **Output:** `tables/ranker_comparison.csv`.

### M04.5 — N operator evaluation
- **Action:** Oracle-evidence NLI on FEVER (gold evidence → label). Compare:
  - BART-large-MNLI (POC);
  - a **public** DeBERTa-v3-large NLI checkpoint whose model card documents training data **without FEVER-NLI**, for example SNLI+MNLI-only cross-encoders (verify the card and record it);
  - fine-tuning DeBERTa-v3-large on MNLI ourselves (bf16 + gradient checkpointing; ≈ 15–25 GPU-h), **only if no qualifying public checkpoint exists**;
  - a FEVER-trained NLI model, reported **only** as a contaminated upper bound and clearly labelled.
  - Report hard vs soft outputs for each.
- **Note for M16:** the NLI model used by ARES here must **not** be the one M16 uses to validate synthetic claims. Record both model IDs.
- **Output:** `tables/nli_comparison.csv`.

### M04.6 — W operator
- **Action:** `credibility.LinRatingsCredibility` using the Lin et al. (2023) aggregated domain-quality ratings (continuous 0–1). Fallback for unrated domains is a single documented constant, chosen on dev. Keep `credibility.POCKeywordCredibility` for comparison. Note: on FEVER (Wikipedia only) W is constant, so W's effect is measurable only on AVeriTeC/web settings.

### M04.7 — Functional variants & verdict thresholds
- **Action:** Implement (a) POC hard-N, (b) soft-N, (c) soft-N with evidence-count features passed to M05's learned aggregator. Verdict thresholds (T/F/Uncertain) are fitted on dev by maximising macro-F1 (or minimising Chow risk with costs from M01, as a variant). The hard-coded −0.4 isn't used.
- **Done when:** dev-fitted thresholds saved in `configs_used/`.

### M04.8 — Logic layer (the "symbolic logic / contradiction" component)
- **Action:** `src/ares/logic/`:
  - `quantities.py`: extract numbers, units, dates from claim and evidence (quantulum3 / dateparser / spaCy); align on shared entities; flag numeric/date mismatches beyond tolerance.
  - `wikidata_checks.py`: relation templates with time qualifiers (head of state P35, head of government P6, capital P36, population P1082 with relative tolerance, date of birth P569, inception P571, located in P131); SPARQL via `SnapshotClient`; return consistent / contradicted / unknown + trace.
  - `logic_score.py`: aggregate into features (n_contradictions, n_confirmations, coverage).
- **Evaluation:** QuanTemp (numerical claims) and the numeric subset of FEVER: precision/recall of contradiction detection; contribution to verdict accuracy (feeds M11 "−Logic" ablation, where logic is replaced by a neural softmax as described in §XXVII-B2).

### M04.9 — Table XV layer operationalisation (C018)
- **Action:** Pre-register this mapping (it's a manuscript change, and it must be stated as one):
  1. **Text filtering** → check-worthiness classification on the ClaimBuster dataset / CheckThat! 2021 Task 1 (DeBERTa-v3-base fine-tune; ClaimBuster's published model as comparison).
  2. **Claim/evidence extraction** → FEVER evidence-sentence P/R/F1.
  3. **Contextual reasoning** → NLI label accuracy given gold evidence (FEVER).
  4. **Cross-source verification** → full-pipeline label macro-F1 (FEVER, AVeriTeC).
- **Output:** `tables/table15_replacement.csv` with P/R/F1 + CIs.

### M04.10 — End-to-end evaluation (Track T2)
- **Action:** Run the full R→S→N→W→Truth' pipeline on FEVER (official dev as test if the test labels are hidden; say so) and AVeriTeC (dev/test per official availability). Report the official metrics with CIs; compare with published baselines from `sota_reference.csv` (cited) and our own reruns where code exists.
- **Cache:** per-claim evidence features for fusion (out-of-fold on train).
- **Tier 2 (descriptive, frozen pipeline, no re-tuning):**
  - CLIMATE-FEVER and SciFact, each with its own corpus indexed the same way.
  - The Snopes corpus (UKP), with its provided evidence.
  - Report macro-F1 with CIs as a domain-shift result.

### M04.11 — Showcase re-run (C067)
- **Action:** Run the J&K claim ("8 terrorist bombs dropped in Jammu and Kashmir on 1st February 2026") through the web pipeline in `record` mode once, then `replay`. Publish the trace JSON, snapshot hash, and a UI screenshot (M14). Report the new numbers. They will differ from −0.4231 / 0.8799, because live web results change. Note the retrieval date.

### M04.12 — Confidence fix (C078)
- **Action:** Keep Conf_POC = |Truth'|·log(1+M) only as a legacy field. The reported confidence becomes the calibrated probability from M05. Show on dev that Conf_POC can exceed 1 and is poorly calibrated (reliability diagram).

---

## Audit gate — milestone-specific red flags
- RF1: **NLI contamination:** any model trained on FEVER-NLI must not be used for headline FEVER results.
- RF2: **Fact-check leakage:** in web mode, zero evidence from fact-check domains in evaluation runs (auditor greps traces).
- RF3: **Temporal leakage:** no evidence dated after the claim date when a date exists.
- RF4: Open-retrieval results never use gold evidence (the oracle setting is separate and labelled).
- RF5: Thresholds/hyperparameters chosen on dev only; test access logged.
- RF6: AVeriTeC knowledge-store usage follows the official rules.
- RF7: Logic-layer Wikidata checks use time-qualified statements valid at the claim date, not today's value.
- RF8: Plausibility vs published FEVER/AVeriTeC numbers.
- RF9: The training data of the NLI checkpoint is verified from its model card or paper. "No FEVER" is evidenced, not assumed.
- RF10: The IVF-PQ approximation loss is measured, and the retrieval recall reported is that of the index actually used.

## Exit criteria
- [ ] Operators implemented, individually evaluated, and selected on dev.
- [ ] T2 end-to-end results with CIs; Table XV replacement produced.
- [ ] Logic layer evaluated on QuanTemp/FEVER-numeric.
- [ ] Showcase trace regenerated from a snapshot.
- [ ] Evidence feature cache (out-of-fold) complete.
- [ ] Ledger: C018, C040, C067, C078 updated.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m04-complete`.
