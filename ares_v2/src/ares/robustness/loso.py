"""
Milestone M12: Leave-One-Scenario-Out (LOSO) Stress Scenario Engine.

Evaluates system degradation under real-world crisis shifts:
1. Scenario S_COVID: COVID-19 Infodemic (Claim C019).
2. Scenario S_ELECTION: Election Manipulation (Claim C020).
3. Scenario S_CRISIS: Breaking-News Crisis Events (Claim C023).
4. Scenario S_DISASTER: Natural Disasters (Claim C023).

Measures in-scenario vs held-out scenario performance across K = 5 seeds,
computing per-scenario Delta F1 and mean degradation (Claim C059).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class ScenarioDataset:
    name: str
    scenario_id: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class LOSOResult:
    scenario_name: str
    scenario_id: str
    in_scenario_f1: float
    in_scenario_f1_ci: Tuple[float, float]
    held_out_f1: float
    held_out_f1_ci: Tuple[float, float]
    delta_f1: float
    pct_degradation: float


class LOSOEvaluator:
    """Evaluates Leave-One-Scenario-Out (LOSO) performance degradation."""

    def __init__(self, seeds: Optional[List[int]] = None) -> None:
        self.seeds = seeds or [42, 43, 44, 45, 46]

    def evaluate_scenario(
        self,
        target_scenario: ScenarioDataset,
        all_scenarios: List[ScenarioDataset],
    ) -> LOSOResult:
        """
        Evaluate single scenario in-scenario vs trained on all other scenarios held out.
        Averages across K = 5 seeds.
        """
        in_f1s: List[float] = []
        out_f1s: List[float] = []

        # Construct pooled held-out training data (excluding target scenario)
        other_scenarios = [s for s in all_scenarios if s.scenario_id != target_scenario.scenario_id]
        X_heldout_train = np.vstack([s.X_train for s in other_scenarios])
        y_heldout_train = np.concatenate([s.y_train for s in other_scenarios])

        for seed in self.seeds:
            # 1. In-scenario model
            clf_in = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            clf_in.fit(target_scenario.X_train, target_scenario.y_train)
            preds_in = clf_in.predict(target_scenario.X_test)
            in_f1s.append(float(f1_score(target_scenario.y_test, preds_in, average="macro")))

            # 2. Held-out scenario model
            clf_out = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            clf_out.fit(X_heldout_train, y_heldout_train)
            preds_out = clf_out.predict(target_scenario.X_test)
            out_f1s.append(float(f1_score(target_scenario.y_test, preds_out, average="macro")))

        mean_in = float(np.mean(in_f1s))
        mean_out = float(np.mean(out_f1s))
        delta = mean_in - mean_out
        pct_deg = float((delta / max(1e-6, mean_in)) * 100.0)

        # Bootstrap CIs on test predictions of seed 42
        clf_in_base = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf_in_base.fit(target_scenario.X_train, target_scenario.y_train)
        p_in = clf_in_base.predict(target_scenario.X_test)

        clf_out_base = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf_out_base.fit(X_heldout_train, y_heldout_train)
        p_out = clf_out_base.predict(target_scenario.X_test)

        b_in: List[float] = []
        b_out: List[float] = []
        n_te = len(target_scenario.y_test)
        rng = np.random.RandomState(42)
        for _ in range(1000):
            idx = rng.randint(0, n_te, size=n_te)
            b_in.append(float(f1_score(target_scenario.y_test[idx], p_in[idx], average="macro", zero_division=0)))
            b_out.append(float(f1_score(target_scenario.y_test[idx], p_out[idx], average="macro", zero_division=0)))

        ci_in = (float(np.percentile(b_in, 2.5)), float(np.percentile(b_in, 97.5)))
        ci_out = (float(np.percentile(b_out, 2.5)), float(np.percentile(b_out, 97.5)))

        return LOSOResult(
            scenario_name=target_scenario.name,
            scenario_id=target_scenario.scenario_id,
            in_scenario_f1=mean_in,
            in_scenario_f1_ci=ci_in,
            held_out_f1=mean_out,
            held_out_f1_ci=ci_out,
            delta_f1=delta,
            pct_degradation=pct_deg,
        )

    def run_loso(self, scenarios: List[ScenarioDataset]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run full LOSO evaluation across all pre-registered scenarios."""
        results: List[LOSOResult] = []
        records: List[Dict[str, Any]] = []

        for sc in scenarios:
            res = self.evaluate_scenario(sc, scenarios)
            results.append(res)
            records.append({
                "scenario_id": res.scenario_id,
                "scenario_name": res.scenario_name,
                "in_scenario_f1": res.in_scenario_f1,
                "in_scenario_ci_lower": res.in_scenario_f1_ci[0],
                "in_scenario_ci_upper": res.in_scenario_f1_ci[1],
                "held_out_f1": res.held_out_f1,
                "held_out_ci_lower": res.held_out_f1_ci[0],
                "held_out_ci_upper": res.held_out_f1_ci[1],
                "delta_f1": res.delta_f1,
                "pct_degradation": res.pct_degradation,
            })

        df = pd.DataFrame(records)
        deltas = np.array([r.delta_f1 for r in results])
        mean_delta = float(np.mean(deltas))

        # Bootstrap CI on mean degradation
        b_means = []
        rng = np.random.RandomState(42)
        n = len(deltas)
        for _ in range(1000):
            idx = rng.randint(0, n, size=n)
            b_means.append(float(np.mean(deltas[idx])))
        ci_mean = (float(np.percentile(b_means, 2.5)), float(np.percentile(b_means, 97.5)))

        summary = {
            "n_scenarios": len(scenarios),
            "mean_delta_f1": mean_delta,
            "mean_delta_ci95": ci_mean,
            "mean_pct_degradation": float(np.mean([r.pct_degradation for r in results])),
            "max_delta_f1": float(np.max(deltas)),
            "claim_c059_satisfied": bool(mean_delta < 0.048),
        }

        return df, summary
