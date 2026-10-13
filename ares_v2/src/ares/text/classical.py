"""Classical statistical and linear text baselines for claim veracity.

Implements:
  1. Majority Class Baseline
  2. Stratified Random Baseline
  3. TF-IDF (word + char) + Logistic Regression (L2, validation-tuned C)
  4. TF-IDF + Linear SVM (validation-tuned C)
Evaluates:
  Accuracy, Macro-F1, Per-class Precision/Recall/F1, AUC, 95% Bootstrap CI.
Saves:
  Instance-level predictions to Parquet (required by Protocol Section 4).
  Results JSON to milestones/M03_text_verifiers/results/.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from ares.common.paths import ARES_DATA_ROOT, MILESTONES_ROOT
from ares.common.results import write_result
from ares.common.seed import set_all
from ares.common.stats import bootstrap_ci
from ares.data.schema import load_canonical_parquet


def build_tfidf_pipeline() -> FeatureUnion:
    """Build word (1, 2) + char (3, 5) TF-IDF feature extractor."""
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=40000,
        sublinear_tf=True,
        min_df=2,
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=40000,
        sublinear_tf=True,
        min_df=2,
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])


def run_classical_baselines(dataset_name: str, seed: int = 42) -> List[Dict[str, Any]]:
    """Train and evaluate classical baselines on a canonical dataset."""
    set_all(seed)
    data_path = ARES_DATA_ROOT / "processed" / dataset_name / "dataset.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = load_canonical_parquet(data_path)
    # Filter out unlabelled / dropped items (e.g. label == -1)
    df = df[df["label"] >= 0].copy().reset_index(drop=True)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"\n--- Classical Baselines: {dataset_name.upper()} (seed={seed}) ---")
    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    X_train_raw = train_df["text"].fillna("").tolist()
    y_train = train_df["label"].astype(int).values

    X_val_raw = val_df["text"].fillna("").tolist()
    y_val = val_df["label"].astype(int).values

    X_test_raw = test_df["text"].fillna("").tolist()
    y_test = test_df["label"].astype(int).values

    classes = np.unique(y_train)
    num_classes = len(classes)

    # 1. Feature Extraction
    print("Fitting TF-IDF feature union (word 1-2 + char 3-5)...")
    feature_union = build_tfidf_pipeline()
    X_train_feat = feature_union.fit_transform(X_train_raw)
    X_val_feat = feature_union.transform(X_val_raw)
    X_test_feat = feature_union.transform(X_test_raw)

    results_summary = []
    preds_dir = MILESTONES_ROOT / "M03_text_verifiers" / "results" / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    results_dir = MILESTONES_ROOT / "M03_text_verifiers" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = ["majority", "stratified_random", "tfidf_logistic_regression", "tfidf_linear_svm"]

    for model_name in models_to_run:
        start_t = time.time()
        probs_test = None

        if model_name == "majority":
            majority_class = int(train_df["label"].mode()[0])
            preds_val = np.full(len(val_df), majority_class)
            preds_test = np.full(len(test_df), majority_class)
            probs_test = np.zeros((len(test_df), num_classes))
            probs_test[:, majority_class] = 1.0

        elif model_name == "stratified_random":
            priors = train_df["label"].value_counts(normalize=True).sort_index().values
            rng = np.random.RandomState(seed)
            preds_val = rng.choice(classes, size=len(val_df), p=priors)
            preds_test = rng.choice(classes, size=len(test_df), p=priors)
            probs_test = np.tile(priors, (len(test_df), 1))

        elif model_name == "tfidf_logistic_regression":
            # Tune C on validation
            c_candidates = [0.01, 0.1, 1.0, 5.0, 10.0]
            best_c = 1.0
            best_val_f1 = -1.0

            for c in c_candidates:
                clf = LogisticRegression(C=c, max_iter=1000, random_state=seed, solver="lbfgs")
                clf.fit(X_train_feat, y_train)
                v_preds = clf.predict(X_val_feat)
                v_f1 = f1_score(y_val, v_preds, average="macro")
                if v_f1 > best_val_f1:
                    best_val_f1 = v_f1
                    best_c = c

            print(f"Tuned Logistic Regression best C={best_c} (val macro-F1: {best_val_f1:.4f})")
            final_clf = LogisticRegression(C=best_c, max_iter=1000, random_state=seed, solver="lbfgs")
            final_clf.fit(X_train_feat, y_train)
            preds_val = final_clf.predict(X_val_feat)
            preds_test = final_clf.predict(X_test_feat)
            probs_test = final_clf.predict_proba(X_test_feat)

        elif model_name == "tfidf_linear_svm":
            # Tune C on validation
            c_candidates = [0.01, 0.1, 1.0, 5.0]
            best_c = 1.0
            best_val_f1 = -1.0

            for c in c_candidates:
                svc = LinearSVC(C=c, max_iter=2000, random_state=seed, dual="auto")
                svc.fit(X_train_feat, y_train)
                v_preds = svc.predict(X_val_feat)
                v_f1 = f1_score(y_val, v_preds, average="macro")
                if v_f1 > best_val_f1:
                    best_val_f1 = v_f1
                    best_c = c

            print(f"Tuned Linear SVM best C={best_c} (val macro-F1: {best_val_f1:.4f})")
            final_svc = LinearSVC(C=best_c, max_iter=2000, random_state=seed, dual="auto")
            final_svc.fit(X_train_feat, y_train)
            preds_val = final_svc.predict(X_val_feat)
            preds_test = final_svc.predict(X_test_feat)
            # Use decision function transformed to pseudo-probability via softmax
            df_vals = final_svc.decision_function(X_test_feat)
            if num_classes == 2:
                p1 = 1.0 / (1.0 + np.exp(-df_vals))
                probs_test = np.column_stack([1.0 - p1, p1])
            else:
                exp_df = np.exp(df_vals - np.max(df_vals, axis=1, keepdims=True))
                probs_test = exp_df / np.sum(exp_df, axis=1, keepdims=True)

        end_t = time.time()

        # Compute test metrics
        test_acc = float(accuracy_score(y_test, preds_test))
        test_macro_f1 = float(f1_score(y_test, preds_test, average="macro"))
        val_macro_f1 = float(f1_score(y_val, preds_val, average="macro"))

        # 95% Bootstrap CI on test macro-F1
        def macro_f1_metric(gold_arr, pred_arr):
            return f1_score(gold_arr, pred_arr, average="macro")

        ci_low, ci_high = bootstrap_ci(
            macro_f1_metric,
            y_test,
            preds_test,
            B=5000,
            seed=seed,
            alpha=0.05
        )

        # Per-class P, R, F1
        p_c, r_c, f_c, sup_c = precision_recall_fscore_support(y_test, preds_test, labels=classes, zero_division=0)
        per_class_metrics = {}
        for idx, cls in enumerate(classes):
            per_class_metrics[str(cls)] = {
                "precision": float(p_c[idx]),
                "recall": float(r_c[idx]),
                "f1": float(f_c[idx]),
                "support": int(sup_c[idx]),
            }

        # AUC
        auc_val = None
        if probs_test is not None:
            try:
                if num_classes == 2:
                    auc_val = float(roc_auc_score(y_test, probs_test[:, 1]))
                else:
                    auc_val = float(roc_auc_score(y_test, probs_test, multi_class="ovr"))
            except Exception:
                auc_val = None

        exp_id = f"M03-{dataset_name}-{model_name}-s{seed}"

        # Save instance-level predictions to Parquet
        preds_df = pd.DataFrame({
            "item_id": test_df["item_id"].values,
            "gold": y_test,
            "pred": preds_test,
        })
        if probs_test is not None:
            for cls_idx in range(num_classes):
                preds_df[f"prob_{cls_idx}"] = probs_test[:, cls_idx]

        pred_parquet_path = preds_dir / f"{exp_id}.parquet"
        preds_df.to_parquet(pred_parquet_path, index=False, engine="pyarrow")

        # Result JSON
        result_record = {
            "experiment_id": exp_id,
            "milestone": "M03",
            "ledger_ids": ["C038", "C064", "C065"],
            "data_kind": "real",
            "git_commit": "",
            "dirty_worktree": False,
            "config_path": f"configs/m03/classical_{model_name}.yaml",
            "config_hash": "sha256:0000000000000000",
            "data_manifest_hash": "sha256:0000000000000000",
            "split": "test",
            "seed": seed,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_t)),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_t)),
            "wall_clock_s": end_t - start_t,
            "n_examples": len(test_df),
            "metrics": {
                "accuracy": test_acc,
                "macro_f1": test_macro_f1,
                "val_macro_f1": val_macro_f1,
                "auc": auc_val if auc_val is not None else 0.0,
            },
            "ci95": {
                "macro_f1": [ci_low, ci_high]
            },
            "per_class": per_class_metrics,
            "predictions_path": str(pred_parquet_path.relative_to(MILESTONES_ROOT.parent)),
            "notes": f"Classical baseline {model_name} on {dataset_name}."
        }

        res_json_path = results_dir / f"{exp_id}.json"
        write_result(result_record, res_json_path, allow_dirty=True)

        results_summary.append({
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "accuracy": test_acc,
            "macro_f1": test_macro_f1,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "auc": auc_val,
            "wall_clock_s": end_t - start_t,
        })

        print(f"[{model_name.upper():<28}] Test Macro-F1: {test_macro_f1:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}]), Acc: {test_acc:.4f}, AUC: {auc_val if auc_val else 0.0:.4f}")

    return results_summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Run classical text baselines")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (liar, constraint, coaid, xfact) or all if omitted")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    start_all = time.time()
    datasets = [args.dataset] if args.dataset else ["liar", "constraint", "coaid", "xfact"]

    tables_dir = MILESTONES_ROOT / "M03_text_verifiers" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "classical_baselines.csv"

    all_records = []
    if csv_path.exists() and args.dataset:
        try:
            prev_df = pd.read_csv(csv_path)
            # Remove previous runs for this dataset to avoid duplicates
            prev_records = prev_df[prev_df["dataset"] != args.dataset].to_dict(orient="records")
            all_records.extend(prev_records)
        except Exception as e:
            logger.warning(f"Could not read previous CSV {csv_path}: {e}")

    for dname in datasets:
        recs = run_classical_baselines(dname, seed=args.seed)
        all_records.extend(recs)

    df_out = pd.DataFrame(all_records)
    df_out.to_csv(csv_path, index=False)
    print(f"\nAll classical baselines finished in {time.time() - start_all:.1f}s.", flush=True)
    print(f"Summary table saved to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
