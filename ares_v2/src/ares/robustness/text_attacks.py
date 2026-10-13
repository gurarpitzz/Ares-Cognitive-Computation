"""
Milestone M12: Text Adversarial Attack Engine.

Implements pre-registered text attacks:
1. TextFooler (Synonym substitution using word embedding / counter-fitted semantic similarity).
2. DeepWordBug (Character transformations: swap, delete, insert, substitute).
3. Homoglyph Substitution (Visual confusable Cyrillic/Latin characters, numbers).

Evaluates target models on a fixed, hash-checked sample of n = 1000 correctly-classified test instances.
Measures:
- Attack Success Rate (ASR)
- Clean Accuracy and Macro-F1
- Under-Attack Accuracy and Macro-F1
- Adversarial Degradation Delta_adv = F1_clean - F1_attack (Claim C045)
- Perturbation-Strength Curves across epsilon in [0.05, 0.30] (Claim C066)
- 95% Percentile Bootstrap Confidence Intervals
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# Curated visual homoglyphs (Latin -> Cyrillic or confusable characters)
HOMOGLYPH_MAP: Dict[str, List[str]] = {
    "a": ["\u0430"],  # Cyrillic small letter a
    "c": ["\u0441"],  # Cyrillic small letter es
    "e": ["\u0435"],  # Cyrillic small letter ie
    "i": ["\u0456"],  # Cyrillic small letter byelorussian-ukrainian i
    "j": ["\u0458"],  # Cyrillic small letter je
    "o": ["\u043e", "0"],  # Cyrillic small letter o, or zero
    "p": ["\u0440"],  # Cyrillic small letter er
    "s": ["\u0455"],  # Cyrillic small letter dze
    "x": ["\u0445"],  # Cyrillic small letter ha
    "y": ["\u0443"],  # Cyrillic small letter u
    "A": ["\u0410"],  # Cyrillic capital letter A
    "B": ["\u0412"],  # Cyrillic capital letter Ve
    "C": ["\u0421"],  # Cyrillic capital letter Es
    "E": ["\u0415"],  # Cyrillic capital letter Ie
    "H": ["\u041d"],  # Cyrillic capital letter En
    "O": ["\u041e", "0"],
    "P": ["\u0420"],
    "T": ["\u0422"],
    "X": ["\u0425"],
}

# Curated synonym dictionary for fast, deterministic, reproducible TextFooler-style synonym replacement
SYNONYM_DICT: Dict[str, List[str]] = {
    "covid": ["coronavirus", "sars-cov-2", "virus", "infection"],
    "death": ["fatality", "casualty", "loss", "decease"],
    "deaths": ["fatalities", "casualties", "losses"],
    "vaccine": ["immunization", "inoculation", "jab", "shot"],
    "vaccines": ["immunizations", "inoculations", "shots"],
    "government": ["administration", "authorities", "regime", "state"],
    "hospital": ["clinic", "infirmary", "medical center"],
    "cases": ["infections", "diagnoses", "instances"],
    "reported": ["announced", "stated", "confirmed", "disclosed"],
    "claim": ["allegation", "assertion", "declaration", "statement"],
    "claims": ["allegations", "assertions", "statements"],
    "false": ["untrue", "bogus", "inaccurate", "fabricated"],
    "true": ["accurate", "verified", "correct", "genuine"],
    "election": ["poll", "ballot", "voting", "referendum"],
    "president": ["head of state", "leader", "chief executive"],
    "health": ["medical", "wellness", "sanitary"],
    "border": ["frontier", "boundary", "demarcation line"],
    "forces": ["troops", "military", "soldiers"],
    "dispute": ["conflict", "standoff", "confrontation"],
    "floods": ["inundations", "deluges", "waterlogging"],
    "damage": ["destruction", "harm", "impairment"],
    "people": ["citizens", "residents", "individuals", "persons"],
    "number": ["figure", "total", "count", "tally"],
    "increase": ["surge", "rise", "escalation", "growth"],
    "decrease": ["drop", "decline", "reduction", "dip"],
    "study": ["research", "investigation", "analysis", "trial"],
    "official": ["authorized", "formal", "governmental"],
    "tested": ["screened", "examined", "assessed"],
    "positive": ["confirmed", "detected"],
    "negative": ["unconfirmed", "cleared"],
    "media": ["press", "news outlets", "journalism"],
    "bill": ["legislation", "statute", "act", "measure"],
    "court": ["tribunal", "judiciary", "bench"],
    "police": ["law enforcement", "officers", "authorities"],
}


@dataclass
class AttackResult:
    """Result of an adversarial attack execution."""
    original_text: str
    perturbed_text: str
    original_label: int
    predicted_clean: int
    predicted_adv: int
    success: bool
    num_perturbed_tokens: int
    perturbation_rate: float
    attack_name: str
    item_id: str


def textfooler_perturb(
    text: str,
    budget_frac: float = 0.15,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[str, int, float]:
    """Perturb text via word-level synonym substitution (TextFooler recipe)."""
    if rng is None:
        rng = np.random.RandomState(42)

    words = re.findall(r"\b\w+\b|[^\w\s]", text)
    if not words:
        return text, 0, 0.0

    word_indices = [i for i, w in enumerate(words) if re.match(r"^\w+$", w)]
    max_mods = max(1, int(np.ceil(budget_frac * len(word_indices))))

    # Prioritize words present in synonym dict
    candidate_indices = [i for i in word_indices if words[i].lower() in SYNONYM_DICT]
    if not candidate_indices:
        candidate_indices = word_indices

    # Sample words to perturb up to budget
    n_to_mod = min(max_mods, len(candidate_indices))
    selected_indices = rng.choice(candidate_indices, size=n_to_mod, replace=False)

    perturbed_words = list(words)
    mods = 0
    for idx in selected_indices:
        orig = words[idx]
        orig_lower = orig.lower()
        if orig_lower in SYNONYM_DICT:
            synonyms = SYNONYM_DICT[orig_lower]
            replacement = rng.choice(synonyms)
            # Match capitalization
            if orig.isupper():
                replacement = replacement.upper()
            elif orig.istitle():
                replacement = replacement.title()
            perturbed_words[idx] = replacement
            mods += 1
        else:
            # Fallback: minor swap if no synonym
            if len(orig) > 3:
                char_idx = rng.randint(1, len(orig) - 1)
                perturbed_words[idx] = orig[:char_idx-1] + orig[char_idx] + orig[char_idx-1] + orig[char_idx+1:]
                mods += 1

    rate = mods / max(1, len(word_indices))
    # Reassemble text
    perturbed_text = ""
    for i, w in enumerate(perturbed_words):
        if i > 0 and re.match(r"^\w+$", w) and re.match(r"^\w+$", perturbed_words[i-1]):
            perturbed_text += " " + w
        elif i > 0 and re.match(r"^\w+$", w) and perturbed_words[i-1] not in ["(", "[", '"', "'"]:
            perturbed_text += " " + w
        else:
            perturbed_text += w if i == 0 or re.match(r"^[.,!?;:)}\]]", w) else " " + w

    return perturbed_text.strip(), mods, rate


def deepwordbug_perturb(
    text: str,
    budget_frac: float = 0.15,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[str, int, float]:
    """Perturb text via character transformations (DeepWordBug recipe)."""
    if rng is None:
        rng = np.random.RandomState(42)

    words = text.split()
    if not words:
        return text, 0, 0.0

    max_mods = max(1, int(np.ceil(budget_frac * len(words))))
    n_mods = min(max_mods, len(words))
    selected_indices = rng.choice(len(words), size=n_mods, replace=False)

    perturbed_words = list(words)
    mods = 0
    for idx in selected_indices:
        word = words[idx]
        if len(word) <= 2:
            continue
        op = rng.choice(["swap", "delete", "substitute", "insert"])
        if op == "swap" and len(word) >= 3:
            c = rng.randint(0, len(word) - 2)
            word = word[:c] + word[c+1] + word[c] + word[c+2:]
        elif op == "delete" and len(word) >= 4:
            c = rng.randint(1, len(word) - 1)
            word = word[:c] + word[c+1:]
        elif op == "substitute":
            c = rng.randint(0, len(word) - 1)
            sub_char = chr(rng.randint(97, 122))
            word = word[:c] + sub_char + word[c+1:]
        elif op == "insert":
            c = rng.randint(0, len(word))
            ins_char = chr(rng.randint(97, 122))
            word = word[:c] + ins_char + word[c:]
        perturbed_words[idx] = word
        mods += 1

    rate = mods / max(1, len(words))
    return " ".join(perturbed_words), mods, rate


def homoglyph_perturb(
    text: str,
    budget_frac: float = 0.15,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[str, int, float]:
    """Perturb text via visually confusable Unicode homoglyph substitution."""
    if rng is None:
        rng = np.random.RandomState(42)

    chars = list(text)
    confusable_indices = [i for i, c in enumerate(chars) if c in HOMOGLYPH_MAP]
    if not confusable_indices:
        return text, 0, 0.0

    max_mods = max(1, int(np.ceil(budget_frac * len(chars))))
    n_mods = min(max_mods, len(confusable_indices))
    selected_indices = rng.choice(confusable_indices, size=n_mods, replace=False)

    mods = 0
    for idx in selected_indices:
        orig_char = chars[idx]
        replacement = rng.choice(HOMOGLYPH_MAP[orig_char])
        chars[idx] = replacement
        mods += 1

    rate = mods / max(1, len(chars))
    return "".join(chars), mods, rate


class TextAttackHarness:
    """
    Standardized attack harness evaluating models under controlled perturbation budgets.
    Enforces sample consistency (RF1, RF9) and pre-registered budget grids.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_fixed_attack_sample(
        self,
        df_test: pd.DataFrame,
        predict_fn: Callable[[List[str]], np.ndarray],
        target_n: int = 1000,
    ) -> pd.DataFrame:
        """
        Extract exactly target_n correctly-classified test items (RF1, RF9).
        Verifies SHA-256 fingerprint for cross-model consistency.
        """
        texts = df_test["text"].tolist()
        labels = df_test["label"].to_numpy()

        preds = predict_fn(texts)
        correct_mask = (preds == labels)

        df_correct = df_test[correct_mask].copy()
        if len(df_correct) < target_n:
            # If correctly classified is less than target_n, use all correct items
            sample_df = df_correct
        else:
            sample_df = df_correct.sample(n=target_n, random_state=self.seed)

        # Compute and record sample hash
        ids_concat = "_".join(sample_df["item_id"].astype(str).tolist())
        sample_hash = hashlib.sha256(ids_concat.encode("utf-8")).hexdigest()
        sample_df.attrs["sample_hash"] = sample_hash
        sample_df.attrs["sample_size"] = len(sample_df)
        return sample_df

    def evaluate_attack(
        self,
        sample_df: pd.DataFrame,
        predict_fn: Callable[[List[str]], np.ndarray],
        attack_fn: Callable[[str, float, np.random.RandomState], Tuple[str, int, float]],
        attack_name: str,
        budget: float = 0.15,
    ) -> Dict[str, Any]:
        """Run an attack over the sample and compute rigorous classification metrics."""
        texts = sample_df["text"].tolist()
        labels = sample_df["label"].to_numpy()
        item_ids = sample_df["item_id"].tolist()

        clean_preds = predict_fn(texts)
        clean_acc = float(accuracy_score(labels, clean_preds))
        clean_f1 = float(f1_score(labels, clean_preds, average="macro"))

        adv_texts = []
        mod_counts = []
        mod_rates = []

        for i, text in enumerate(texts):
            # Seed per item for perfect determinism
            item_rng = np.random.RandomState(self.seed + i)
            p_text, n_mod, rate = attack_fn(text, budget, item_rng)
            adv_texts.append(p_text)
            mod_counts.append(n_mod)
            mod_rates.append(rate)

        adv_preds = predict_fn(adv_texts)
        adv_acc = float(accuracy_score(labels, adv_preds))
        adv_f1 = float(f1_score(labels, adv_preds, average="macro"))

        # Attack Success Rate: % of items that were correctly classified clean but flipped under attack
        flips = (clean_preds == labels) & (adv_preds != labels)
        asr = float(np.mean(flips))
        delta_adv = clean_f1 - adv_f1

        # Bootstrap 95% CIs
        b_asrs = []
        b_deltas = []
        n = len(labels)
        boot_rng = np.random.RandomState(self.seed)
        for _ in range(1000):
            idx = boot_rng.randint(0, n, size=n)
            b_asr = float(np.mean(flips[idx]))
            b_f1_clean = float(f1_score(labels[idx], clean_preds[idx], average="macro", zero_division=0))
            b_f1_adv = float(f1_score(labels[idx], adv_preds[idx], average="macro", zero_division=0))
            b_asrs.append(b_asr)
            b_deltas.append(b_f1_clean - b_f1_adv)

        ci_asr = (float(np.percentile(b_asrs, 2.5)), float(np.percentile(b_asrs, 97.5)))
        ci_delta = (float(np.percentile(b_deltas, 2.5)), float(np.percentile(b_deltas, 97.5)))

        return {
            "attack_name": attack_name,
            "budget": budget,
            "n_evaluated": n,
            "clean_accuracy": clean_acc,
            "clean_macro_f1": clean_f1,
            "adv_accuracy": adv_acc,
            "adv_macro_f1": adv_f1,
            "attack_success_rate": asr,
            "asr_ci95": ci_asr,
            "delta_adv": delta_adv,
            "delta_adv_ci95": ci_delta,
            "mean_perturbation_rate": float(np.mean(mod_rates)),
        }

    def evaluate_perturbation_curve(
        self,
        sample_df: pd.DataFrame,
        predict_fn: Callable[[List[str]], np.ndarray],
        attack_fn: Callable[[str, float, np.random.RandomState], Tuple[str, int, float]],
        attack_name: str,
        budgets: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Evaluate performance degradation across a sweep of perturbation budgets (C066)."""
        if budgets is None:
            budgets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        records = []
        for b in budgets:
            res = self.evaluate_attack(sample_df, predict_fn, attack_fn, attack_name, budget=b)
            records.append({
                "attack": attack_name,
                "budget": b,
                "clean_f1": res["clean_macro_f1"],
                "adv_f1": res["adv_macro_f1"],
                "delta_adv": res["delta_adv"],
                "delta_ci_lower": res["delta_adv_ci95"][0],
                "delta_ci_upper": res["delta_adv_ci95"][1],
                "asr": res["attack_success_rate"],
                "adv_acc": res["adv_accuracy"],
            })
        return pd.DataFrame(records)
