"""Ablation and interaction harness for ARES-Bench.

Enforces:
1. Strict leave-one-out retraining (Audit Red Flag RF3, RF4; no test-time zeroing).
2. Paired bootstrap confidence intervals on metric deltas.
3. 2x2 factorial interaction testing (Claim C055, Audit Red Flag RF6).
4. Conflict-domain recall drop evaluation (Claim C054).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

from ares.common.stats import bootstrap_ci, paired_bootstrap_delta
from ares.fusion.late import LateFusionPipeline

logger = logging.getLogger(__name__)


class TrackAblationHarness:
    """Executes retrained leave-one-out ablations and second-order interaction tests."""

    def __init__(
        self,
        method: str = "calibrated_llr",
        seed: int = 42,
        n_bootstraps: int = 2000,
    ):
        self.method = method
        self.seed = seed
        self.n_bootstraps = n_bootstraps

    def run_leave_one_out_ablations(
        self,
        df: pd.DataFrame,
        active_modules: List[str],
        primary_metric_fn: Callable[[np.ndarray, np.ndarray], float] = None,
    ) -> Dict[str, Any]:
        """Runs retrained leave-one-out ablations across all active modules."""
        if primary_metric_fn is None:
            def primary_metric_fn(y_true, y_pred):
                return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]

        y_train = train_df["label"].values
        y_test = test_df["label"].values

        # 1. Train Full Model
        full_feature_cols = [f"llr_{m}" for m in active_modules]
        full_pipeline = LateFusionPipeline(method=self.method, seed=self.seed)
        full_pipeline.fit(train_df[full_feature_cols].values, y_train, feature_names=full_feature_cols)
        full_preds = full_pipeline.predict(test_df[full_feature_cols].values)

        full_metric = primary_metric_fn(y_test, full_preds)
        full_prec = float(precision_score(y_test, full_preds, average="macro", zero_division=0))
        full_rec = float(recall_score(y_test, full_preds, average="macro", zero_division=0))

        results = {
            "full": {
                "metric": full_metric,
                "precision": full_prec,
                "recall": full_rec,
                "f1": full_metric,
                "active_modules": active_modules,
            },
            "ablations": {},
        }

        # 2. Leave-One-Module-Out Retraining
        for mod in active_modules:
            reduced_modules = [m for m in active_modules if m != mod]
            if not reduced_modules:
                continue

            reduced_cols = [f"llr_{m}" for m in reduced_modules]
            ablated_pipeline = LateFusionPipeline(method=self.method, seed=self.seed)
            ablated_pipeline.fit(train_df[reduced_cols].values, y_train, feature_names=reduced_cols)
            ablated_preds = ablated_pipeline.predict(test_df[reduced_cols].values)

            abl_metric = primary_metric_fn(y_test, ablated_preds)
            abl_prec = float(precision_score(y_test, ablated_preds, average="macro", zero_division=0))
            abl_rec = float(recall_score(y_test, ablated_preds, average="macro", zero_division=0))

            # Delta = Full - Ablated (positive means removing the module drops performance)
            delta, ci_low, ci_high, p_val = paired_bootstrap_delta(
                primary_metric_fn,
                y_test,
                ablated_preds,
                full_preds,
                B=self.n_bootstraps,
                seed=self.seed,
            )

            results["ablations"][f"-{mod}"] = {
                "excluded_module": mod,
                "metric": abl_metric,
                "precision": abl_prec,
                "recall": abl_rec,
                "f1": abl_metric,
                "delta": delta,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "p_value": p_val,
            }

        # 3. Text-Only Baseline (if text is among active modules and m > 1)
        if "text" in active_modules and len(active_modules) > 1:
            text_cols = ["llr_text"]
            text_pipeline = LateFusionPipeline(method=self.method, seed=self.seed)
            text_pipeline.fit(train_df[text_cols].values, y_train, feature_names=text_cols)
            text_preds = text_pipeline.predict(test_df[text_cols].values)

            text_metric = primary_metric_fn(y_test, text_preds)
            text_prec = float(precision_score(y_test, text_preds, average="macro", zero_division=0))
            text_rec = float(recall_score(y_test, text_preds, average="macro", zero_division=0))

            delta_text_only, ci_l, ci_u, p_t = paired_bootstrap_delta(
                primary_metric_fn,
                y_test,
                text_preds,
                full_preds,
                B=self.n_bootstraps,
                seed=self.seed,
            )

            results["ablations"]["text_only"] = {
                "metric": text_metric,
                "precision": text_prec,
                "recall": text_rec,
                "f1": text_metric,
                "delta": delta_text_only,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "p_value": p_t,
            }

        # 4. Conflict Domain Recall Drop (Claim C054)
        if "geo" in active_modules and "domain" in test_df.columns:
            conflict_mask = (test_df["domain"] == "Conflict").values
            if np.sum(conflict_mask) >= 10:
                y_test_conflict = y_test[conflict_mask]
                full_preds_conflict = full_preds[conflict_mask]
                geo_abl_preds = results["ablations"]["-geo"]
                # Get preds of -geo
                reduced_cols = [f"llr_{m}" for m in active_modules if m != "geo"]
                p_geo_abl = LateFusionPipeline(method=self.method, seed=self.seed).fit(
                    train_df[reduced_cols].values, y_train, feature_names=reduced_cols
                )
                geo_preds_conflict = p_geo_abl.predict(test_df[reduced_cols].values)[conflict_mask]

                if len(np.unique(y_test)) > 2:
                    rec_full_conflict = float(recall_score(y_test_conflict, full_preds_conflict, labels=[1], average="macro", zero_division=0))
                    rec_nogeo_conflict = float(recall_score(y_test_conflict, geo_preds_conflict, labels=[1], average="macro", zero_division=0))
                else:
                    rec_full_conflict = float(recall_score(y_test_conflict, full_preds_conflict, pos_label=1, zero_division=0))
                    rec_nogeo_conflict = float(recall_score(y_test_conflict, geo_preds_conflict, pos_label=1, zero_division=0))
                rec_drop_pct = (rec_full_conflict - rec_nogeo_conflict) * 100.0

                results["conflict_geo_ablation"] = {
                    "n_conflict_items": int(np.sum(conflict_mask)),
                    "recall_full": rec_full_conflict,
                    "recall_without_geo": rec_nogeo_conflict,
                    "recall_drop_pct": rec_drop_pct,
                }

        return results

    def run_second_order_interaction(
        self,
        df: pd.DataFrame,
        active_modules: List[str],
        mod_a: str,
        mod_b: str,
        primary_metric_fn: Callable[[np.ndarray, np.ndarray], float] = None,
    ) -> Dict[str, Any]:
        """Runs 2x2 factorial interaction test for modules A and B (Claim C055).
        
        Computes:
        Interaction = Delta_{A and B} - (Delta_A + Delta_B)
        with bootstrap CI and classification (super-additive vs sub-additive).
        """
        if primary_metric_fn is None:
            def primary_metric_fn(y_true, y_pred):
                return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        if mod_a not in active_modules or mod_b not in active_modules:
            raise ValueError(f"Both {mod_a} and {mod_b} must be in active_modules ({active_modules}).")

        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]

        y_train = train_df["label"].values
        y_test = test_df["label"].values

        # 1. Full model
        cols_full = [f"llr_{m}" for m in active_modules]
        p_full = LateFusionPipeline(method=self.method, seed=self.seed).fit(train_df[cols_full].values, y_train)
        pred_full = p_full.predict(test_df[cols_full].values)
        m_full = primary_metric_fn(y_test, pred_full)

        # 2. No A
        cols_no_a = [f"llr_{m}" for m in active_modules if m != mod_a]
        p_no_a = LateFusionPipeline(method=self.method, seed=self.seed).fit(train_df[cols_no_a].values, y_train)
        pred_no_a = p_no_a.predict(test_df[cols_no_a].values)
        m_no_a = primary_metric_fn(y_test, pred_no_a)
        delta_a = m_full - m_no_a

        # 3. No B
        cols_no_b = [f"llr_{m}" for m in active_modules if m != mod_b]
        p_no_b = LateFusionPipeline(method=self.method, seed=self.seed).fit(train_df[cols_no_b].values, y_train)
        pred_no_b = p_no_b.predict(test_df[cols_no_b].values)
        m_no_b = primary_metric_fn(y_test, pred_no_b)
        delta_b = m_full - m_no_b

        # 4. No A and B
        cols_no_ab = [f"llr_{m}" for m in active_modules if m not in (mod_a, mod_b)]
        if not cols_no_ab:
            cols_no_ab = cols_full[:1]  # edge case fallback
        p_no_ab = LateFusionPipeline(method=self.method, seed=self.seed).fit(train_df[cols_no_ab].values, y_train)
        pred_no_ab = p_no_ab.predict(test_df[cols_no_ab].values)
        m_no_ab = primary_metric_fn(y_test, pred_no_ab)
        delta_ab = m_full - m_no_ab

        # Interaction effect: Delta_{AB} - (Delta_A + Delta_B)
        obs_interaction = delta_ab - (delta_a + delta_b)

        # Bootstrap over test items
        rng = np.random.default_rng(self.seed)
        n = len(y_test)
        indices = np.arange(n)
        boot_interactions = []

        for _ in range(self.n_bootstraps):
            b_idx = rng.choice(indices, size=n, replace=True)
            y_b = y_test[b_idx]
            mf_b = primary_metric_fn(y_b, pred_full[b_idx])
            ma_b = primary_metric_fn(y_b, pred_no_a[b_idx])
            mb_b = primary_metric_fn(y_b, pred_no_b[b_idx])
            mab_b = primary_metric_fn(y_b, pred_no_ab[b_idx])

            d_a_b = mf_b - ma_b
            d_b_b = mf_b - mb_b
            d_ab_b = mf_b - mab_b
            boot_interactions.append(d_ab_b - (d_a_b + d_b_b))

        ci_lower = float(np.percentile(boot_interactions, 2.5))
        ci_upper = float(np.percentile(boot_interactions, 97.5))

        if ci_lower > 0:
            classification = "Super-additive"
        elif ci_upper < 0:
            classification = "Sub-additive"
        else:
            classification = "No detectable interaction"

        return {
            "module_a": mod_a,
            "module_b": mod_b,
            "delta_a": float(delta_a),
            "delta_b": float(delta_b),
            "sum_individual_deltas": float(delta_a + delta_b),
            "delta_joint_ab": float(delta_ab),
            "interaction_effect": float(obs_interaction),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "classification": classification,
        }
