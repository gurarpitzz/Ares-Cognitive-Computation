# M06 — Claim Drift & Mutation: Test the Theory on Real Embeddings + Claim Memory

**Objective:** (1) Test the assumptions of M01's mutation model on real sentence embeddings and compare theoretical with empirical ROC curves. (2) Build the **claim memory** (time-ordered retrieval of previously fact-checked claims with mutation-aware matching), so the temporal module has a concrete, measurable route into veracity performance. (3) Replace the illustrative Figures 13–15, 25–27, 32, 36 and Table VIII with real data.
**Ledger IDs:** C011, C012, C070 (empirical side), C074, C085, C086.
**Depends on:** M01, M02, M16 (frozen synthetic mutation chains, for M06.6b only).
**Compute:** RTX 4060 Laptop, ≈ 3 GPU-h (embeddings, fp16, batched). The statistics run on the CPU.
**Output dir:** `milestones/M06_drift_mutation/`

---

## Pre-registration
- **Operational definitions (state them as definitions; the manuscript's "mutation" is ambiguous):**
  - **H0 pairs (benign evolution):** PAWS label = 1; STS-B score ≥ 4.5; MultiClaim / CheckThat! post–fact-check pairs judged as the same claim.
  - **H1 pairs (mutation), three pre-registered families:**
    - (a) PAWS label = 0 (high lexical overlap, different meaning);
    - (b) claims linked to *different* fact-checks within the same topic cluster;
    - (c) controlled synthetic mutations (entity swap, negation, number change, date change) **taken from the frozen M16 ARES-Synth chains**, labelled SIMULATED and reported separately from (a) and (b).
- **H6.1:** The Gaussian mutation model (isotropic) fits H0 differences ε. *Expected to fail on isotropy, since transformer embeddings are known to be anisotropic. Report it either way.*
- **H6.2:** The theoretical ROC (Theorem A/B with fitted parameters) matches the empirical ROC within a pre-declared tolerance (e.g. |ΔAUC| ≤ 0.02).
- **H6.3:** Under the paper's own model the ‖ε‖² statistic has at least the power of cosine (Theorem B(c)). Test empirically.
- **H6.4:** Claim memory improves veracity macro-F1 on T1/T2 datasets with dated claims (tested in M11's ablation; measured standalone here).
- **H6.5 (C086):** Verification log-odds drift negatively along mutation chains (κ > 0).

## Steps

### M06.1 — Embeddings
- **Action:** Compute embeddings for all pair sets and chain claims with: `bert-base-uncased` CLS and mean-pool (the manuscript says BERT), `all-MiniLM-L6-v2` (POC), `all-mpnet-base-v2`, `intfloat/e5-base-v2`. Store raw **and** L2-normalised. Cache with manifest.

### M06.2 — Build pair sets
- **Action:** `src/ares/drift/pairs.py` → H0/H1 families with train/val/test splits that are **disjoint by claim and by fact-check article**.
- **Done when:** `tables/pairs_summary.csv` (counts per family/split).

### M06.3 — Test the model assumptions (H6.1)
- **Action:** On H0 train pairs, ε = x' − x (raw and normalised):
  - Gaussianity: Anderson–Darling per top-50 PCA component; Henze–Zirkler on top-k PCs (k ∈ {5, 10, 20}); Q-Q plots.
  - Isotropy: eigen-spectrum of Σ̂, effective rank, condition number, ratio Tr(Σ̂)/(d·λ_max).
  - Independence of ε from x: correlation of ‖ε‖ with ‖x‖ and with cosine to the principal direction.
- **Output:** `figures/eps_spectrum.pdf`, `figures/eps_qq.pdf`, `results/assumption_tests.json`.

### M06.4 — Theory vs data (H6.2, H6.3)
- **Action:** Fit s0 (isotropic) and Σ0 (full/low-rank + diagonal) on H0 train; s1/Σ1 on H1 train. Predict the Δ distributions and ROC (Theorem A for isotropic; numeric/MC for general Σ). Compare with empirical Δ and ROC on test pairs: KS distance, ΔAUC. Also compute ROC for (i) cosine threshold, (ii) ‖ε‖² threshold (raw embeddings), (iii) a learned cross-encoder paraphrase model (e.g. `cross-encoder/stsb-roberta-base` or a PAWS-tuned model) as the practical baseline.
- **Output:** `figures/roc_theory_vs_empirical.pdf` (replaces Figs. 25/26/32), `tables/roc_comparison.csv`.

### M06.5 — Lipschitz estimation (Theorem B(e))
- **Action:** On STS-B: d_sem = (5 − score)/5; ratio ρ = ‖E(c) − E(c')‖ / d_sem for d_sem > 0; report the distribution and quantile-Lipschitz constants L_q (q ∈ {0.5, 0.9, 0.95, 0.99}). Evaluate the deterministic H0 bound: empirical FPR when θ is set from L_{0.95}.
- **Output:** `figures/lipschitz_hist.pdf`, `results/lipschitz.json`.

### M06.6 — Real claim chains & drift timelines (C011, C012)
- **Action:** Build chains of claims linked to the same fact-check or narrative, ordered by date (MultiClaim posts with timestamps; AVeriTeC/X-FACT claims grouped by fact-check article; CheckThat! tweets). Compute Δ_t, cumulative drift D_{t,k}, and mutation events at a val-selected θ.
- **Output:** a real replacement for Table VIII (`tables/mutation_examples.csv`, with actual claim texts and cosine values), real Figs. 14/15/36 (`figures/drift_*.pdf`).

### M06.6b — Synthetic chains with known change points (SIMULATED; from M16)
- **Action:** Load the frozen M16 mutation chains, which are sequences of paraphrases with **known mutation steps** and a recorded perturbation type. On these chains, measure:
  1. **Detection delay** (steps between the true mutation and the detection);
  2. **False-alarm rate per benign step**;
  3. **Power by perturbation type** (entity, number, date, negation, misattribution);
  4. **Theory vs empirical:** the ROC predicted by Theorem A/B, using σ fitted on the *real* H0 pairs (M06.4), against the empirical ROC on the synthetic chains.
- **Rules:**
  - θ comes from M06.6 (real val). No tuning on synthetic chains.
  - Results go in a separate table captioned "SIMULATED (ARES-Synth)".
- **Output:** `tables/synth_chain_detection.csv`, `figures/synth_detection_delay.pdf`.
- **Done when:** results JSONs have `data_kind="synthetic_grounded"`, and their M16 manifest hash matches the frozen one.

### M06.7 — Claim memory module (veracity route for the temporal module)
- **Action:** `src/ares/drift/memory.py`: an index of fact-checked claims with (verdict, date). For a new claim with date t: retrieve top-k neighbours with date < t (strict); features = max similarity, similarity-weighted verdict score, count above θ_match; θ_match chosen on val. Evaluate on (1) CheckThat! 2021/22 Task 2 (MAP@5, the official metric), (2) standalone veracity signal on dated T1/T2 datasets (AUC), and (3) cache out-of-fold features for M11.
- **Done when:** results JSON + cache; temporal ordering unit-tested.

### M06.8 — Temporal drift of L(c_t) (H6.5, C086)
- **Action:** Along labelled chains, regress L(c_{t+1}) − L(c_t) on the mutation indicator with claim-chain random effects (or cluster-robust SEs). Report κ with CI. A null result is reported as a null result.

### M06.9 — Robustness to paraphrase attacks (C074)
- **Action:** Paraphrase test claims with:
  - (a) back-translation (MarianMT en→de→en);
  - (b) an open LLM paraphraser (Mistral-7B-Instruct-v0.3 in 4-bit, the same generator family as M16; fixed prompt; temperature 0).
  Measure the claim-memory recall@k drop and the change in mutation-detector false alarms. This number replaces "immunity to paraphrasing".

---

## Audit gate — milestone-specific red flags
- RF1: **Temporal leakage in memory:** no retrieved neighbour has date ≥ the query date (auditor checks 100 random queries).
- RF2: The same fact-check article never appears in both train and test pairs/chains.
- RF3: θ and θ_match chosen on val only.
- RF4: H1 definitions aren't circular (e.g. H1 defined by the same cosine being tested).
- RF5: Raw vs normalised embeddings used consistently with the theory being tested.
- RF6: Synthetic mutations labelled SIMULATED and never mixed into "real" results.
- RF7: If assumptions fail, the README says so in plain language, and the theory's applicability section in `ares_theory.tex` is updated.

## Exit criteria
- [ ] Assumption tests, theory-vs-empirical ROC, and Lipschitz estimates complete.
- [ ] Real chains, real figures/tables replacing the illustrative ones.
- [ ] Claim memory evaluated (CheckThat MAP@5) and cached for fusion.
- [ ] Ledger: C011, C012, C074, C085, C086 updated; C070 empirical note added.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m06-complete`.
