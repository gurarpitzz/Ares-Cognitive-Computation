# M03 — Text Baselines & Layer-1 Text Verifiers

**Objective:** Train and evaluate honest text baselines and the manuscript's declared Layer-1 text models on Tracks T1 (text veracity) and T3 (image–text, via JBC). Cache instance-level outputs for fusion. Establish what text alone achieves.
**Ledger IDs:** C034, C038, C039, C050, C057 (prep), C064, C065.
**Depends on:** M02.
**Compute:** RTX 4060 Laptop, ≈ 80–110 GPU-h. Base-model grids, OOF cross-fitting, DeBERTa-v3-large reference, 4-bit LLM inference, JBC. Re-estimate with the M00 `vram_fit.csv` before starting, and queue the jobs overnight.
**Output dir:** `milestones/M03_text_verifiers/`

---

## Pre-registration (write into PREREG.md before training)
- **H3.1:** DeBERTa-v3-base (the declared encoder) beats TF-IDF+LR on macro-F1 on each T1 dataset (paired bootstrap, Holm-corrected).
- **H3.2:** Ranking among {BERT-base, RoBERTa-base, DeBERTa-v3-base} (exploratory, no directional claim).
- **H3.3:** mDeBERTa-v3-base on X-FACT: report in-domain / out-of-domain / zero-shot macro-F1 (descriptive).
- **Primary metric:** macro-F1 on the official test split. Secondary: accuracy, per-class P/R/F1, AUC, ECE.
- **Seeds:** K = 10 for the declared and compared transformer models on the Tier-1 T1 datasets. K = 5 for the DeBERTa-v3-large reference and the Tier-2 datasets; these are labelled descriptive, not used for significance claims. **Model selection:** best validation macro-F1 checkpoint within the declared 15 epochs.
- **Fields allowed:** claim/statement text only (no LIAR speaker/party/context metadata) for the primary results. A metadata variant is reported separately and labelled.

## Steps

### M03.1 — Trivial & classical baselines
- **Action:** Majority class; stratified random; TF-IDF (word 1–2 + char 3–5) + logistic regression; TF-IDF + linear SVM (the "SVM" bar in Fig. 41). Tune C on validation only.
- **Files:** `src/ares/text/classical.py`, `configs/m03/classical_*.yaml`.
- **Output:** results JSON per dataset; `tables/classical.csv`.
- **Done when:** all T1 datasets have classical results with bootstrap CIs.

### M03.2 — Transformer fine-tuning harness
- **Action:** `src/ares/text/finetune.py`, a custom loop on `ares.common.train_loop` (resumable) driven by YAML.
  - **Defaults** = the declared config (C050): AdamW lr 2e‑5, wd 0.01, β (0.9, 0.999), ε 1e‑8, **effective** batch 32, 15 epochs, 10 % linear warm-up then cosine decay, max length cap 512 (C069).
  - **Loss:** weighted CE with class weights (0.7, 1.3) as declared (C069) for binary tasks, plus an **unweighted variant** (needed for C057 in M11).
  - **8 GB VRAM tactics:**
    - bf16 autocast.
    - micro-batch × `grad_accum` = 32. Choose the micro-batch from `vram_fit.csv`. Note that the optimizer step count per epoch is identical to batch 32.
    - Dynamic padding with length-grouped sampling.
    - `max_len` = the train-split 99.5th-percentile token length, capped at 512. Record it, and record the % truncated.
    - Gradient checkpointing only if needed.
  - Deterministic seeding. Save the best-val checkpoint, val/test logits, and per-epoch logs. Apply the checkpoint retention policy (plan §5.4).
- **Done when:**
  - A 1-epoch smoke run on LIAR-binary gives identical val metrics across two runs with the same seed (deterministic GPU mode).
  - A micro-batch 8 × accum 4 run matches a micro-batch 16 × accum 2 run within seed noise (recorded).

### M03.3 — Models × datasets grid
- **Models:**
  - `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base` (declared).
  - `microsoft/deberta-v3-large` as a strong reference, K = 5: bf16 + gradient checkpointing + 8-bit AdamW, micro-batch 2–4 × accumulation to 32. If it doesn't fit, use LoRA r=16 and **name it "DeBERTa-v3-large (LoRA)"**.
  - `microsoft/mdeberta-v3-base` and `xlm-roberta-base` (X-FACT).
- **Datasets (Tier 1):** LIAR (binary + 6-way), Constraint-COVID, CoAID, X-FACT, PHEME-text (LOEO folds), FakeNewsNet/UPFD news content (PolitiFact, GossipCop).
- **Datasets (Tier 2, if acquired in M02):** MultiFC, PUBHEALTH, COVID-Fact (claim-only), WELFake, ISOT (original + shortcut-stripped), IFND, FactDrill, Constraint Hindi (mDeBERTa/XLM-R).
  - Run TF-IDF + DeBERTa-v3-base (mDeBERTa for non-English) with K = 5.
  - Always report the claim-only TF-IDF shortcut baseline next to them.
- **Action:** Run the grid via `scripts/m03/run_grid.py --config configs/m03/grid.yaml`, submitted through the job queue. Order it short datasets first (LIAR, COVID, PHEME); the long-document runs (GossipCop, FakeNewsNet) go last and overnight. Log GPU-h in `milestones/compute_ledger.csv`.
- **Done when:** every (model, dataset, seed) cell has a results JSON + saved predictions, or is listed in the README as skipped with its reason.

### M03.4 — Open LLM zero-shot baseline (the "LLM" member of the Layer-1 ensemble, Sec. VI)
- **Action:** `src/ares/text/llm_zeroshot.py` with **Qwen2.5-7B-Instruct** (Apache-2.0; record the exact revision hash).
  - Run it in **4-bit NF4** (bitsandbytes), or as GGUF Q4_K_M via llama.cpp, so it fits in 8 GB. Name the quantisation in every table.
  - Fixed prompt template in `configs/m03/llm_prompt.txt`; greedy decoding.
  - Compute P(TRUE) from label-token log-probabilities, not by parsing free text. Cache all outputs via the snapshot/cache layer.
  - Evaluate on all T1 test sets.
  - **Family separation:** the M16 synthetic-data generator must come from a *different* model family (Mistral). Don't switch this member to a Mistral model.
  - **Optional quantisation check:** on 500 LIAR validation items, compare 4-bit vs 8-bit P(TRUE) and agreement, and report the difference.
- **Caveat to document:** LLMs may have seen PolitiFact/LIAR during pre-training (contamination). Report it as a limitation; optionally test with post-cutoff claims in M12/T11.
- **Done when:** results + cached outputs exist; prompt committed before evaluation.

### M03.5 — FakeFlow re-run; ClaimBuster & Grover scoping
- **Action:**
  - **FakeFlow** (Ghanem et al., EACL 2021): clone the official code, pin its commit, run on the datasets it supports, with our splits where possible. Document deviations.
  - **ClaimBuster:** it's a *check-worthiness* model. Don't compare it on veracity. Its comparison happens in M04 on the check-worthiness task (Tab. XV "text filtering").
  - **Grover:** its discriminator detects *machine-generated* news. If weights are obtainable, evaluate it only on machine-generated-text detection (LLMFake, M12). Otherwise mark it unavailable.
- **Output:** `README` section "Baseline scoping" explaining why Tabs. XXI/XXXVII can't stay as they are (C034, C038, C064).

### M03.6 — JBC (Joint BERT–CNN) for image–text posts (Track T3)
- **Action:** `src/ares/text/jbc.py`: BERT-base (or DeBERTa-v3-base) text encoder + ResNet-50 image encoder, concatenation → MLP → classifier. Train on the Fakeddit fixed subset (60k/6k/6k; 2-way and 6-way) with K = 10 seeds. Train text-only and image-only variants with the same recipe (needed for the M11 ablation).
  - **8 GB tactic:** bf16; micro-batch 16 with accumulation to 32. Optionally precompute and freeze the ResNet image features for the first epoch; if you do, declare it.
  - **Tier 2:** evaluate the frozen Fakeddit JBC zero-shot on the MOCHEG/Factify/VERITE test splits (descriptive cross-dataset results; VERITE is the hard test).
- **Done when:** JBC + unimodal variants have results and cached predictions.

### M03.7 — Evaluation & statistics
- **Action:** `scripts/m03/aggregate.py` → for each dataset: table of mean ± std (all seeds listed in an appendix CSV), 95 % bootstrap CI for the seed-median model, paired bootstrap Δ vs DeBERTa-v3-base, McNemar, seed-level paired t + Wilcoxon (K = 10), Cohen's d_z, Holm across the family. LaTeX + CSV.
- **Plausibility check:** compare with `M02_data/sota_reference.csv`; anything above published SOTA triggers a leakage investigation **before** reporting.

### M03.8 — Calibration snapshot
- **Action:** ECE, Brier, and reliability diagrams on validation for DeBERTa-v3-base, the LLM, and JBC. These feed M05.

### M03.9 — Multilingual capability (C039)
- **Action:** per-language macro-F1 on X-FACT for mDeBERTa/XLM-R with test support ≥ 50. Report the number of languages meeting a pre-registered minimum F1 (e.g. > majority baseline with CI excluding it). This number replaces "14 languages".

### M03.10 — Cache Layer-1 outputs for fusion
- **Action:** For every L1 model that enters fusion (DeBERTa-v3-base, LLM, JBC; plus mDeBERTa for X-FACT):
  - Save **out-of-fold** predictions on train (5-fold cross-fitting with fixed folds) and standard predictions on val/test → `data/cache/l1/<model>/<dataset>/<split>.parquet` (item_id, logits, probs, seed, fold). This prevents stacking leakage in M05/M11.
  - To fit the budget, OOF cross-fitting runs with **one fixed seed per (model, dataset)**. Fusion uncertainty across L1 seeds is then estimated in M11 from the K test-set prediction sets. Declare this.
- **Done when:** cache complete with a manifest hash.

### M03.11 — Model cards
- **Action:** `models/MODEL_CARD_<model>_<dataset>.md` + SHA256SUMS; upload weights privately to HF Hub (public at release, licence permitting).

---

## Audit gate — milestone-specific red flags
- RF1: LIAR results near or above published SOTA (6-way ≈ 0.27–0.30 text-only, binary ≈ 0.60–0.70) → check for leakage (metadata columns, duplicates, test-set early stopping).
- RF2: Truncation: % of examples truncated at 512 reported; tokenizer matches model.
- RF3: Label mapping in code matches PREREG exactly (auditor diffs `label_maps.yaml` against the loader).
- RF4: Early stopping/checkpoint selection used validation only (inspect trainer config and logs).
- RF5: Different seeds give different results (catch seed bugs where all 10 runs are identical).
- RF6: PHEME LOEO folds really hold out entire events.
- RF7: LLM label-token probability extraction is correct for the tokenizer (multi-token labels handled).
- RF8: Out-of-fold cache really is out-of-fold (auditor re-derives one fold).
- RF9: Weighted vs unweighted loss variants both exist and are labelled.
- RF10: The effective batch really is 32 (micro-batch × accum logged). LoRA and quantised models are labelled as such in every table.
- RF11: Resumed runs (`resumed_from_checkpoint=true`) are flagged and listed; a resume never changed the seed.

## Exit criteria
- [ ] Classical, transformer, LLM, FakeFlow (or documented unavailability), and JBC results complete with CIs.
- [ ] Statistical comparison tables generated; plausibility check done.
- [ ] Out-of-fold L1 cache complete.
- [ ] Ledger: C039 (languages), C050 (config used), C065 (Fig. 41 replacement values), C034/C038/C064 (scoping notes, status REVISED or REMOVE for incomparable entries).
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m03-complete`.
