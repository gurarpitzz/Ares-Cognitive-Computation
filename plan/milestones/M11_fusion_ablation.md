# M11 — Track-Level Fusion, Full System, Ablations, Shapley Attribution, Significance

**Objective:** Assemble ARES v2 per ARES-Bench track, train the fusion heads (including the Appendix-A architecture and the Eq. 14 composite loss), and produce the honest replacement for every headline, ablation, interaction, attribution, and significance claim (Sections XXII–XXV, XXVII, XXX).
**Ledger IDs:** C001, C043, C046, C047, C048, C049, C051, C052, C053, C054, C055, C057, C058, C065, C068, C069, C097.
**Depends on:** M03–M10 (caches), M05 (decision layer), M16 (must be **frozen** before the first M11 run; ARES-Synth is never used for fusion training, selection or thresholds).
**Compute:** RTX 4060 Laptop, ≈ 45 GPU-h. End-to-end fusion on T3/T6 dominates (bf16, gradient accumulation to the declared batch, resumable). Late-fusion sweeps and exact Shapley over cached outputs run on the CPU.
**Output dir:** `milestones/M11_fusion_ablation/`

---

## Pre-registration (most important PREREG of the project; commit before any fusion run)
- **Tracks and datasets in scope:** list exactly which of T1–T9 enter the headline table (all with complete caches). No track may be added or dropped after seeing results; any later change gets logged as a deviation.
- **Headline definition (replaces C001's single F1 = 0.92):** per-track primary metric (from `03_IMPLEMENTATION_PLAN.md §3`) for full ARES, best single module, and best baseline, with 95 % CIs. If an aggregate is reported, it's the unweighted mean of per-track macro-F1 over the **pre-listed classification tracks** only.
- **Ablations per track:** remove module X **only where X's input exists** in that track:
  - T1/T2: −Evidence, −Logic (replaced by a neural softmax head, per §XXVII-B2), −Temporal (claim memory), −Geo (GDELT/context), −LLM, Text-only.
  - T3: −Vision (text-only JBC), −Text (image-only), −Metadata.
  - T6: −Audio, −Vision, −AV-sync.
  - T7: −Behavior, −Temporal, −Geo, Text-only.
  - All tracks: −HITL (simulated) for the escalation-enabled system.
- **Ablation method:** retrain the fusion head without module X (not zeroing inputs at test time). Keep everything else fixed.
- **Interaction tests (C055):** 2×2 factorials for (Multimodal × Temporal) on tracks where both exist, and (Logic × Geo) on T1/T2.
- **Attribution (replaces C058):** exact Shapley values over active modules per track.
- **Statistics:** Protocol §8 (K = 10 seeds; paired bootstrap; McNemar; seed-level t + Wilcoxon; d_z; Holm within each track family).

## Steps

### M11.1 — Feature assembly
- **Action:** `src/ares/fusion/assemble.py`: per track, join cached module outputs by item_id (out-of-fold for train, standard for val/test) into `data/cache/fusion/<track>/<dataset>.parquet`; record which modules are present per item; manifest hash.
- **Done when:** a coverage table shows per-module availability for each track; no train row uses in-fold predictions (auditor-checkable flag column).

### M11.2 — Late-fusion heads (primary, cheap, exact ablations)
- **Action:** `fusion/late.py`: (a) calibrated LLR logistic (from M05); (b) gradient-boosted trees; (c) MLP head following App. A where possible (input = concatenated module scores/embeddings → 1024 → 2 hidden layers, GELU, dropout 0.1). Output space: dataset-native labels; "Uncertain" via Chow abstention (M05) **resolves audit I-11**. Hyperparameters chosen on val.

### M11.3 — End-to-end fusion (App. A + Eq. 14) where modalities co-occur
- **Action:** `fusion/e2e.py`:
  - **T3 (Fakeddit):** DeBERTa-v3-base + ResNet-50 → concat → 1024-d fusion MLP; loss L = λ_cls·CE_weighted + λ_adv·KL(f(x) ‖ f(x + δ)) + λ_reg‖θ‖² with λ = (1.0, 0.3, 1e-4), class weights (0.7, 1.3), ε = 0.02 (C069). δ from a single-step virtual-adversarial perturbation in **embedding space** (state the exact procedure; Eq. 16 doesn't specify it). AdamW per C050. Note the double regularisation (AdamW wd + λ_reg) and keep it as declared, flagged (audit I-16).
  - **T6 (FakeAVCeleb, if available):** ResNet-50 video + declared audio CNN → fusion.
  - K = 10 seeds (K = 5 if budget-limited; stated).
- **Ablations of the training mechanism:** unweighted loss (C057), λ_adv = 0 (C056 partial; robustness side in M12).

### M11.4 — Full-system evaluation per track
- **Action:** `scripts/m11/eval_full.py` → per track: full ARES (late and, where available, end-to-end), best single module, best baselines (M03/M04/M07/M08/M09), with metrics, CIs, and the significance family. Also P/R/F1 tables in the format of Tabs. XXVIII–XXX for the relevant ablations (C047–C049).

### M11.5 — Leave-one-module-out ablations (C043, C046, C053, C054)
- **Action:** For each track and each removable module: ΔMetric = full − ablated, with paired bootstrap CI and seed-level stats. Conflict-domain recall drop for −Geo (C054) computed only if the pre-registered domain mapping yields enough conflict items (state n).

### M11.6 — Second-order interactions (C055)
- **Action:** Interaction = Δ_{A∧B} − Δ_A − Δ_B with bootstrap CI. Call it "super-additive" only if the CI is entirely > 0, "sub-additive" only if it's entirely < 0, else "no detectable interaction".

### M11.7 — Shapley attribution (C058)
- **Action:** `fusion/shapley.py`: train the late-fusion head on every subset of active modules (2^m; m ≤ 8 per track), compute exact Shapley values of the primary metric; bootstrap over test items (B = 2000) for CIs; normalised shares. Figure per track.

### M11.8 — Significance & effect sizes (C051, C052)
- **Action:** For each comparison family: instance-level paired bootstrap Δ + McNemar (primary), seed-level paired t + exact Wilcoxon over K = 10 (secondary), Cohen's d_z with CI, Holm-adjusted p. Report exact p-values, never just "p < 0.01". Explicitly note in the README that the manuscript's K = 5 Wilcoxon claim was impossible.

### M11.9 — Baseline comparison tables (replace Tabs. XXI / XXXVII, Fig. 41)
- **Action:** Include only baselines evaluated on the **same task and split** (M03/M04). Latency columns are filled from M13 later. Fig. 41 is regenerated from results (SVM = TF-IDF-SVM, BERT, RoBERTa, best multimodal baseline, ARES) per track.

### M11.10 — Ledger reconciliation
- **Action:** For every ledger ID in scope: measured value + CI + evidence path; status REPRODUCED / REVISED / REMOVE. For C001: status REVISED with a pointer to the per-track headline table.

---

## Audit gate — milestone-specific red flags
- RF1: **Stacking leakage:** the auditor recomputes one track's fusion using only val-trained components and checks for consistency; verifies the out-of-fold flags.
- RF2: **Post-hoc track/metric selection:** headline table matches PREREG exactly (diff check).
- RF3: Ablations retrain the head; no test-time input zeroing passed off as ablation.
- RF4: Ablations only reported where the module's input exists in that track.
- RF5: Shapley computed over the same seeds/splits as the ablations; shares sum to the full-minus-empty difference (efficiency axiom check).
- RF6: Interaction language matches CIs.
- RF7: Exact p-values with the correction method named.
- RF8: Plausibility of each track vs M02's SOTA references; results above SOTA get investigated.

## Exit criteria
- [ ] Per-track full-system results, ablations, interactions, Shapley, and significance complete with CIs.
- [ ] E2E fusion (T3, and T6 if available) trained with the declared loss and config.
- [ ] Ledger IDs in scope all updated with evidence.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m11-complete`. **Model freeze tag `ares-v2-freeze-1`** (starts the clock for M12's temporal external validation).
