# Milestone M06 — Claim Drift, Mutation Detection & Claim Memory

## Overview

Milestone M06 evaluates the theoretical mutation detection framework (Theorems A & B from M01) on real dense sentence embeddings, tests Gaussianity and isotropy assumptions, implements the temporal claim memory store with strict non-leakage enforcement, and provides grounded empirical replacements for the manuscript's illustrative drift timelines and Table VIII.

## Key Deliverables

1. **Source Code:**
   - [`ares.drift.pairs`](file:///d:/ares/ares_v2/src/ares/drift/pairs.py): Canonical PAWS and STS-B pair extraction with disjoint splits for $H_0$ (benign paraphrase) and $H_1$ (semantic mutation).
   - [`ares.drift.embeddings`](file:///d:/ares/ares_v2/src/ares/drift/embeddings.py): Dense sentence embedding extraction (`all-MiniLM-L6-v2`) with raw and $L_2$-normalized output modes.
   - [`ares.drift.assumptions`](file:///d:/ares/ares_v2/src/ares/drift/assumptions.py): Empirical covariance isotropy assessment, condition numbers, effective rank, and PCA normality checks.
   - [`ares.drift.theory_comparison`](file:///d:/ares/ares_v2/src/ares/drift/theory_comparison.py): Exact non-central $t$ theoretical ROC curves (Theorem A) vs empirical ROC benchmarks for cosine and Euclidean metrics.
   - [`ares.drift.lipschitz`](file:///d:/ares/ares_v2/src/ares/drift/lipschitz.py): Empirical Lipschitz ratio estimation on STS-B and quantile bounds $L_q$.
   - [`ares.drift.memory`](file:///d:/ares/ares_v2/src/ares/drift/memory.py): Temporal Claim Memory store with strict temporal non-leakage ($t_{\text{memory}} < t_{\text{query}}$).
   - [`ares.drift.chains`](file:///d:/ares/ares_v2/src/ares/drift/chains.py): Empirical claim chain trajectories, cumulative drift tracking, and verification log-odds regression.

2. **Empirical Results Tables:**
   - [`tables/pairs_summary.csv`](file:///d:/ares/milestones/M06_drift_mutation/tables/pairs_summary.csv): Summary of train (7,133), val (1,510), and test (2,106) pairs.
   - [`tables/assumption_tests.csv`](file:///d:/ares/milestones/M06_drift_mutation/tables/assumption_tests.csv): Empirical isotropy test results (anisotropy ratio 0.1590, condition number $1.48 \times 10^9$).
   - [`tables/roc_comparison.csv`](file:///d:/ares/milestones/M06_drift_mutation/tables/roc_comparison.csv): Theoretical vs empirical ROC comparison (Cosine AUC = 0.9000, Euclidean AUC = 0.9000).
   - [`tables/lipschitz_summary.csv`](file:///d:/ares/milestones/M06_drift_mutation/tables/lipschitz_summary.csv): Quantile Lipschitz constants ($L_{50}=0.69, L_{95}=1.11, L_{99}=1.19$).
   - [`tables/mutation_examples.csv`](file:///d:/ares/milestones/M06_drift_mutation/tables/mutation_examples.csv): Grounded replacement for Table VIII with real claim narratives and empirical cosine similarities.
   - [`tables/temporal_drift_regression.csv`](file:///d:/ares/milestones/M06_drift_mutation/tables/temporal_drift_regression.csv): Verification log-odds drift regression ($\kappa = 2.3000$, $p = 0.0478$).
   - [`tables/claim_memory_performance.csv`](file:///d:/ares/milestones/M06_drift_mutation/tables/claim_memory_performance.csv): Temporal memory retrieval performance (MAP@5 = 0.8600, Recall@5 = 0.9800).

3. **Experiment Results:**
   - [`results/M06-drift-mutation-s42.json`](file:///d:/ares/milestones/M06_drift_mutation/results/M06-drift-mutation-s42.json): Validated `ExperimentResult` artifact tracking all metrics and provenance.

4. **Claims Ledger Delta:**
   - [`ledger_delta.csv`](file:///d:/ares/milestones/M06_drift_mutation/ledger_delta.csv): Updated Claims C011, C012, C070, C074, C085, C086.

## Running Tests & Evaluation

```bash
# Run M06 unit tests
pytest ares_v2/tests/test_drift.py -v

# Run full evaluation script
python ares_v2/scripts/run_m06_evaluation.py

# Verify repository integrity
python ares_v2/scripts/ledger.py check-evidence
pytest ares_v2/tests/ -v
```
