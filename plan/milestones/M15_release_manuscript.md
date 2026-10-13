# M15 — Release, Reproducibility Package, References & Manuscript Reconciliation

**Objective:** Freeze and publish everything (code, models, manifests, results), verify every reference, regenerate every figure/table from scripts, and produce a section-by-section reconciliation that turns the ledger into exact manuscript changes. Then run a final meta-audit tracing a random sample of manuscript numbers back to raw results.
**Ledger IDs:** C024, C025, C026, C027, C030, C076, C079, C090, C092, C099 + close-out of all rows.
**Depends on:** M00–M14 and M16 (T11 results from M12 included when available).
**Compute:** RTX 4060 Laptop, in a **fresh WSL distro** (`wsl --import Ubuntu-CLEAN …`) or a fresh Docker container, with an empty cache directory. The clean-room must not reuse the builder's venv, caches or data.
**Output dir:** `milestones/M15_release/`

---

## Steps

### M15.1 — Clean-room reproduction
- **Action:** In the fresh WSL distro or container:
  1. Clone the tagged repo and build the env from the lockfile.
  2. Fetch the open datasets with the M02 scripts; restricted datasets are copied from the hashed local store and hash-verified.
  3. Run `make reproduce-all-smoke` (each milestone's ≤ 15-min smoke).
  4. Run **three full experiments** chosen at random by the auditor (`random.seed(20261013)` over the list of headline experiments, restricted to those ≤ 10 GPU-h; the restriction is declared).
  5. Rebuild ARES-Synth from its frozen sources and generator outputs, and confirm that the manifest hash matches `ares-synth-v1`.
  6. Compare with the stored results within tolerance.
- **Output:** `results/cleanroom_report.json`.

### M15.2 — Release artefacts
- **Action:**
  - Public GitHub repo `ARES` (new) with a tagged release `v2.0.0`. Archive `ARES_POC` with a README pointer to the new repo.
  - Zenodo DOI for the release.
  - Hugging Face model repos, only where dataset licences permit redistributing derived weights; otherwise provide training scripts + checksums.
  - Split manifests and snapshot hashes (not restricted raw data).
  - **ARES-Synth v1** (M16): the Wikipedia/ReliefWeb-derived claims under CC BY-SA, plus the datasheet, generator prompts, perturbation code and validation reports. News-derived items are released as URL + hash only.
  - `docs/reproduce.md` with exact commands, hardware (RTX 4060 Laptop) and runtimes per milestone.
  - Docker image.
- **Done when:** DOI minted; links resolve; licences listed per artefact.

### M15.3 — Ledger close-out
- **Action:** Every ledger row has a final status ≠ UNVERIFIED and an existing evidence path (or REMOVE with reason). `python scripts/ledger.py summary` → `ledger_final_summary.md` (counts by status, list of REVISED with old → new values, list of REMOVE).

### M15.4 — Figure regeneration
- **Action:** `scripts/m15/make_figures.py` regenerates every paper figure from results JSON with one consistent style (matplotlib, vector PDF, colour-blind-safe palette, readable font sizes, no overlapping elements, labelled axes with units, descriptive captions). Write a mapping old Fig. N → new figure file (or REMOVED). Directly addresses Reviewers 1–3's formatting complaints.

### M15.5 — Table regeneration
- **Action:** `scripts/m15/make_tables.py` → LaTeX (booktabs) + CSV for every table, values pulled from results JSON with CIs. Mapping old Table N → new (or REMOVED).

### M15.6 — Reference verification (C092)
- **Action:** `scripts/m15/verify_refs.py`: for each reference, query Crossref and Semantic Scholar by title + first author + year; record match score, DOI, venue, authors. Flag mismatches (the audit lists suspected ones: [1], [4], [6], [9], [13], [14], [22], [26], [27], [35], [40], [41], [47], [49]). Produce a verified `references.bib`. Remove references that don't exist or don't support their sentence; add citations for every dataset, model, toolkit, and baseline actually used. Resolve every "[?]" (Reviewers 2 and 3).
- **Output:** `tables/reference_verification.csv`, `references.bib`.

### M15.7 — Related-work & novelty check (C079)
- **Action:** Systematic search (Semantic Scholar/Google Scholar queries recorded) for multimodal misinformation systems, evidence-based fact-checking pipelines, claim-matching systems, and integrated platforms (2018–present). Rebuild the capability comparison tables (I, III, XXV, XXXI) with citations per cell. Replace "first system to unify" with a defensible statement, or remove it.

### M15.8 — Manuscript reconciliation document
- **Action:** `manuscript_reconciliation.md`, organised by manuscript section. For each: KEEP / REVISE (with exact new text fragments and numbers pulled from the ledger) / REMOVE. Required items:
  - **Title** (Reviewer 2): 3 options reflecting the actual contribution, e.g. evidence-grounded, calibrated multimodal verification with mutation-aware claim memory.
  - **Abstract & conclusion:** only measured, ledger-backed numbers; per-track results; no promotional language (Reviewer 3).
  - **Use cases (§XIII):** C024–C027, C030 (real-world outcomes) are removed. The India–China border and Assam flood scenarios are rewritten as **controlled grounded-synthetic evaluations** (ARES-Synth, M12.6b), with measured numbers, the SIMULATED label, and the qualifier "on a controlled grounded-synthetic benchmark". No claim of real-world deployment or impact. Optionally add **retrospective case studies** on real historical claims, showing actual system traces.
  - **Narrative "Story behind ARES":** removed or moved to a short motivation paragraph (Reviewer 3).
  - **Labelling pipeline (C090):** describe only what was done.
  - **Theory sections:** replaced by M01 results, including what the empirical tests in M06 showed about the assumptions.
  - **Statistics:** protocol as executed (K = 10, paired bootstrap, McNemar, Holm); exact p-values.
  - **Limitations:** include negative results and failed assumptions.
  - **Data availability (C076) & code availability:** accurate statements with the DOI, dataset access terms, and what is and isn't redistributable.
  - **Reproducibility appendix:** updated file → equation map, commands, hardware, runtimes, seeds (C099).
- **Rule:** no number may appear in the reconciliation unless it's in the ledger with status REPRODUCED or REVISED and an evidence path.

### M15.9 — Accurate response-to-reviewers v2
- **Action:** Draft `response_to_reviewers_v2.md`. Every claimed change points to a section and an evidence artefact. No claim is made that isn't true. Explicitly note corrected errors from the previous version where relevant.

### M15.10 — Final meta-audit
- **Action:** Independent auditor session: (1) randomly sample 20 numbers from `manuscript_reconciliation.md` and trace each to results JSON → predictions file → recompute the metric independently; (2) re-run the integrity test suite; (3) check that no restricted data is in any public artefact; (4) check every figure/table file has a generating script. Write `META_AUDIT.md`.

---

## Audit gate — milestone-specific red flags
- RF1: Any reconciliation number not traceable to the ledger is a BLOCKER.
- RF2: Any unverified reference left in `references.bib` is a MAJOR.
- RF3: Restricted data found in public artefacts is a BLOCKER.
- RF4: A clean-room mismatch beyond tolerance on a headline result is a BLOCKER.
- RF5: Promotional/unsupported language remaining in abstract/conclusion is a MAJOR.
- RF6: Any synthetic (ARES-Synth or simulated-analyst) number in the reconciliation without the SIMULATED label and qualifier is a BLOCKER.

## Exit criteria
- [ ] Clean-room reproduction passes; release published with DOI.
- [ ] Ledger fully closed; figures/tables regenerated; references verified.
- [ ] Reconciliation + response v2 drafted; META_AUDIT passes.
- [ ] Tag `v2.0.0`. Hand to the human for manuscript rewriting.
