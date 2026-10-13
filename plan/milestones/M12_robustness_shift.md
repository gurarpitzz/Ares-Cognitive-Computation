# M12 — Robustness, Distribution Shift, Stress Scenarios & Temporal External Validation

**Objective:** Ground (or correct) the robustness, cross-domain, stress-test, and external-validation claims of Sections XII-D, XV-D, XXII-E/F, XXVI, and XXVII-D, using real attacks, real domain shifts, and a **genuine post-freeze external evaluation**.
**Ledger IDs:** C003, C019, C020, C021, C022, C023, C037, C044, C045, C056, C059, C060, C066, C073, C091.
**Depends on:** M11 (frozen models, tag `ares-v2-freeze-1`), M16 (frozen ARES-Synth, tag `ares-synth-v1`).
**Compute:** RTX 4060 Laptop, ≈ 50–60 GPU-h. TextAttack on a fixed n = 1000 sample per target dominates, followed by LOSO retraining.
**Output dir:** `milestones/M12_robustness_shift/`

---

## Pre-registration
- **Attacks:** TextFooler (synonym), DeepWordBug (character), homoglyph substitution (custom, e.g. o→0, Latin→Cyrillic confusables), with budgets stated (max % words/characters perturbed; query budget). They are evaluated on a fixed random sample of **n = 1000** correctly-classified test items per target (the same item IDs for every model; seeds fixed). n = 1000 is set by the laptop's budget, and the CI width it implies is reported.
- **Robustness metrics:** attack success rate, accuracy under attack, Δ_adv = F1_clean − F1_attack (C045), and a perturbation-strength curve (C066).
- **Cross-domain matrix (C044):** domains = Politics (LIAR), Health (Constraint-COVID + CoAID), Social rumours (PHEME), Multilingual (X-FACT, as a separate block). Tier-2 domains are added as extra test columns: Science/Climate (SciFact, CLIMATE-FEVER), Health (PUBHEALTH), and Indian news (IFND, FactDrill).
- **Stress scenarios / LOSO (C019–C023, C059):**
  - **Real scenarios:** the domains above **plus the PHEME events as crisis scenarios** (Charlie Hebdo, Sydney siege, Ottawa shooting, Germanwings crash, Ferguson, …), named honestly as "breaking-news crisis events (PHEME)".
  - **K for LOSO:** K = 5 per held-out scenario. These are descriptive per-scenario estimates, not pairwise model comparisons; the justification is written in PREREG.
  - **Grounded-synthetic scenarios (T12, from M16):** S1 India–China border, S2 Assam floods 2022, S3 Indian election 2024 (optional). They are evaluated **test-only** on the frozen models and reported in a separate block labelled SIMULATED. Claims from these scenarios carry the qualifier "on a controlled grounded-synthetic benchmark".
- **Temporal external validation (C060):** claims fact-checked **after** `ares-v2-freeze-1`, collected via the Google Fact Check Tools API (ClaimReview), with a pre-registered rating → label mapping and exclusion rules, collection window (≥ 4 weeks after freeze), languages, and target n (≥ 300 if available).

## Steps

### M12.1 — Text adversarial attacks
- **Action:** `src/ares/robustness/text_attacks.py` via TextAttack recipes + a homoglyph attacker. Targets: DeBERTa-v3-base (T1), the full late-fusion ARES (attack the text input; downstream modules recompute), and the TF-IDF baseline. Report all metrics with bootstrap CIs; produce `figures/adv_strength_curve.pdf` (replaces Fig. 42).

### M12.2 — Adversarial training (Eq. 16–17; C056)
- **Action:** Compare fusion/text models trained with λ_adv = 0.3 (declared) vs 0 on clean F1 and under-attack F1 (K = 10 seeds). Report Δ with CIs.

### M12.3 — LLM-generated misinformation (C037, C056 "GPT-generated", C091)
- **Action:** Evaluate frozen models on LLMFake (Chen & Shu, ICLR 2024) or an equivalent public LLM-generated misinformation set. Report degradation vs human-written counterparts. If Grover discriminator weights were obtained (M03), evaluate Grover **only** on its native task (machine-text detection) here. The "Grover 22.4 %" comparison is then REVISED or REMOVE. **Injection-rate experiment (C091):** train with synthetic false claims injected at rates {0, 5, 10, 20 %} (synthetic = LLMFake train portion), measure the effect on real-test F1.

### M12.4 — Cross-domain matrix + discrepancy (C044, C073)
- **Action:** Train on domain i, test on domain j (all ordered pairs), K = 10 seeds, plus in-domain diagonal. For each pair, estimate the proxy A-distance (domain classifier on frozen DeBERTa embeddings: d_A = 2(1 − 2·err)). Correlate d_A with the performance drop (Spearman, with bootstrap CI). Deduplicate cross-dataset PolitiFact overlaps (from M02) before testing.
- **Output:** `tables/cross_domain.csv`, `figures/cross_domain_heatmap.pdf`, `figures/adist_vs_drop.pdf`.

### M12.5 — Leave-one-scenario-out (C019, C020, C023, C059)
- **Action:** For each real scenario s: train on all the others and test on s; compare with in-scenario performance (scenario-internal split). Report per-scenario F1 with CI and the mean degradation.
  - COVID → C019 measured value.
  - US elections → C020 (the LIAR election-related subset, defined by a pre-registered keyword/topic rule).
  - Indian election → C020-India uses real data if M12.6 finds a labelled subset (e.g. election-tagged IFND/FactDrill items); otherwise it uses the S3 synthetic block from M12.6b.

### M12.6 — Real-data search for the missing scenarios (C020-India, C021, C022)
- **Action:** Search for public labelled datasets on Indian election misinformation, India–China border propaganda and the Assam 2022 floods. Start with IFND, FactDrill, Indian fact-checker ClaimReview items (BOOM, Alt News, Factly, Vishvas News via the Fact Check Tools API), and any dataset papers. Verify existence, licence and labels, with citations, and write the search log to `results/scenario_data_search.md`.
  - **If a real labelled subset with n ≥ 100 per scenario exists:** use it as a real LOSO scenario (in M12.5). It **takes precedence** over the synthetic result for the corresponding claim.
  - **In all cases**, also run the grounded-synthetic evaluation (M12.6b). It is a controlled complement, not a substitute that gets reported as real.

### M12.6b — ARES-Synth evaluation (T12; SIMULATED; frozen models only)
- **Action:** Run the frozen systems on the frozen M16 test sets (`ares-synth-v1`), with the M16 snapshot evidence corpus as the retrieval corpus (replay mode; no web).
  - **Systems:**
    1. The claim-only TF-IDF baseline;
    2. DeBERTa-v3-base text-only;
    3. The Qwen 4-bit LLM;
    4. The evidence engine (M04) alone;
    5. Full ARES (M11 late fusion with the T2 module set);
    6. ARES − Logic, ARES − Geo, ARES − Drift-memory.
  - **Metrics, with bootstrap CIs over claims and clustered by source fact:**
    - macro-F1 (T/F and T/F/NEI);
    - **accuracy per perturbation type** (entity swap, location swap, number inflation, date shift, fabricated event, misattribution, out-of-context);
    - ECE and the coverage–risk curve under Chow abstention;
    - M10 geotag accuracy against the gold locations recorded by M16;
    - logic-layer detection rate on the number/date perturbations.
  - **Also report** the result restricted to the **human-audited subset** (M16.9), and the accuracy on claims the human auditor disagreed with.
  - **Contamination check:** the S1/S2 events (2020, 2022) may be inside the LLM's pre-training data. Report LLM accuracy on TRUE claims vs on FALSE perturbations, and compare with an "evidence-withheld" run, so the effect of memorisation is visible.
- **Rules:**
  - Zero training, tuning or threshold selection on ARES-Synth.
  - Every table and figure is captioned "SIMULATED — grounded synthetic (ARES-Synth v1)".
  - Results JSONs use `data_kind="synthetic_grounded"`.
- **Ledger:** C021 (India–China border) and C022 (Assam floods), and C020-India if no real data was found → **REVISED**, with the measured value and the qualifier "on a controlled grounded-synthetic benchmark". Real-world deployment wording stays REMOVE.
- **Output:** `tables/ares_synth_main.csv/.tex`, `tables/ares_synth_by_perturbation.csv`, `figures/ares_synth_coverage_risk.pdf`.

### M12.7 — Temporal external validation (C060)
- **Action:**
  1. On the freeze date, write `PREREG_T11.md` (mapping, filters, window, n target) and commit.
  2. After the window: collect ClaimReview items with `reviewDate > freeze date` via the API (snapshotted). Map ratings → T/F per the pre-registered table (drop ambiguous ratings, logging counts).
  3. Deduplicate against **all** training data (exact + near-dup) and remove overlaps.
  4. Run the frozen T2 web pipeline with fact-check-domain exclusion and evidence-date ≤ claim-date.
  5. Report macro-F1 and agreement with CIs, per language/publisher.
- **This replaces** the "live Telegram/X OSINT > 88 % agreement" claim with a reproducible one.

### M12.8 — Media robustness summary
- **Action:** Collate compression/noise/cross-dataset results from M07/M08 into one robustness table for the paper.

### M12.9 — "High resilience" statement (C003)
- **Action:** Write the evidence-based replacement sentence from M12.1–M12.8 numbers only.

---

## Audit gate — milestone-specific red flags
- RF1: Attacks run on test items, not training items; budgets reported; same sample across models.
- RF2: The ClaimReview set contains no item reviewed before the freeze; no overlap with training data; evidence excludes fact-check domains and post-claim-date pages.
- RF3: Scenario definitions fixed in PREREG; no scenario selected post hoc.
- RF4: Cross-domain tests deduplicated for cross-dataset PolitiFact overlap.
- RF5: Proxy A-distance classifier trained/evaluated on held-out splits.
- RF6: Removed scenarios really removed from all figures (Figs. 21/23 replacements).
- RF7: The ARES-Synth manifest hash in the M12 results equals the frozen M16 hash. No ARES-Synth item appears in any training, validation or threshold-selection log (auditor greps the configs and the test-set access log).
- RF8: Synthetic results are never pooled with real-data results in one number, table row or figure panel. Every synthetic table carries the SIMULATED caption.
- RF9: TextAttack samples are identical across models (same item IDs, hash-checked).

## Exit criteria
- [ ] Attacks, adversarial training, LLM-generated eval, cross-domain + A-distance, LOSO done with CIs.
- [ ] Real-data scenario search documented. ARES-Synth evaluation complete. Ledger decisions made for C020-India, C021 and C022.
- [ ] Temporal external validation pre-registered and executed (or scheduled with a date, if the window hasn't elapsed; the milestone can close with T11 pending, and T11 is then reported in M15).
- [ ] Ledger IDs in scope updated.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m12-complete`.
