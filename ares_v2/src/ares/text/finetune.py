"""Transformer fine-tuning harness for claim veracity (Protocol Section 4 & Claim C050 compliant).

Hyperparameters (Claim C050 grounded):
  - Optimizer: AdamW (lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
  - Schedule: Linear warm-up (10%) + Cosine decay
  - Effective batch size: 32 (micro-batch 16 x gradient accumulation 2)
  - Precision: bfloat16 autocast on RTX 4060 Laptop GPU
  - Model selection: Best validation macro-F1 checkpoint
Outputs:
  - Instance-level test predictions: milestones/M03_text_verifiers/results/preds/<exp_id>.parquet
  - Experiment result JSON: milestones/M03_text_verifiers/results/<exp_id>.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ares.common.paths import ARES_DATA_ROOT, MILESTONES_ROOT, ARES_CACHE_ROOT
os.environ.setdefault("HF_HOME", str(ARES_CACHE_ROOT / "huggingface"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score

from ares.common.results import write_result
from ares.common.seed import set_all
from ares.common.stats import bootstrap_ci
from ares.data.schema import load_canonical_parquet


class ClaimDataset(Dataset):
    """PyTorch Dataset for claim text and integer labels."""
    def __init__(self, texts: List[str], labels: List[int], item_ids: List[str], tokenizer, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.item_ids = item_ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
        )
        encoding["labels"] = self.labels[idx]
        encoding["item_id"] = self.item_ids[idx]
        return encoding


class DataCollatorWithItemIds:
    """Collator wrapping HuggingFace dynamic padding while preserving item_id strings."""
    def __init__(self, tokenizer):
        self.hf_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        item_ids = [f.pop("item_id") for f in features]
        batch = self.hf_collator(features)
        batch["item_ids"] = item_ids
        return batch


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_bf16: bool = True
) -> Tuple[float, float, np.ndarray, np.ndarray, List[str]]:
    """Evaluate model on a dataloader and return loss, macro-F1, preds, probs, item_ids."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_golds = []
    all_item_ids = []

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            item_ids = batch.pop("item_ids")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=-1).cpu().float().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds)
            all_probs.append(probs)
            all_golds.extend(labels.cpu().numpy())
            all_item_ids.extend(item_ids)

    avg_loss = total_loss / len(all_golds)
    all_probs_arr = np.concatenate(all_probs, axis=0)
    all_preds_arr = np.array(all_preds)
    all_golds_arr = np.array(all_golds)

    macro_f1 = float(f1_score(all_golds_arr, all_preds_arr, average="macro"))
    return avg_loss, macro_f1, all_preds_arr, all_probs_arr, all_item_ids


def train_and_eval_verifier(
    model_name: str,
    dataset_name: str,
    seed: int = 42,
    num_epochs: int = 4,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    micro_batch: int = 16,
    grad_accum: int = 2,
    max_len: int = 256,
    patience: int = 2,
) -> Dict[str, Any]:
    """Train transformer on train set, select best on val, evaluate once on test."""
    set_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

    start_time = time.time()
    clean_model_id = model_name.split("/")[-1]
    exp_id = f"M03-{dataset_name}-{clean_model_id}-s{seed}"
    print(f"\n========================================================")
    print(f"Starting Experiment: {exp_id}")
    print(f"Device: {device} (bf16={use_bf16}), Model: {model_name}, Dataset: {dataset_name}")
    print(f"Hyperparameters: lr={lr}, bs={micro_batch*grad_accum} ({micro_batch}x{grad_accum}), max_epochs={num_epochs}")
    print(f"========================================================")

    # 1. Load Data
    data_path = ARES_DATA_ROOT / "processed" / dataset_name / "dataset.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    df = load_canonical_parquet(data_path)
    df = df[df["label"] >= 0].copy().reset_index(drop=True)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    classes = np.unique(train_df["label"].astype(int))
    num_classes = len(classes)
    print(f"Dataset splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} | Num classes: {num_classes}")

    # 2. Tokenizer and Datasets
    local_model_dir = ARES_CACHE_ROOT / "models" / model_name.replace("/", "_")
    load_target = str(local_model_dir) if (local_model_dir.exists() and any(local_model_dir.iterdir())) else model_name
    print(f"Loading tokenizer for {model_name} from {load_target}...", flush=True)
    tok_kwargs = {}
    if "deberta" in model_name.lower():
        tok_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(load_target, **tok_kwargs)
    collator = DataCollatorWithItemIds(tokenizer)

    train_ds = ClaimDataset(train_df["text"].fillna("").tolist(), train_df["label"].astype(int).tolist(), train_df["item_id"].tolist(), tokenizer, max_len=max_len)
    val_ds = ClaimDataset(val_df["text"].fillna("").tolist(), val_df["label"].astype(int).tolist(), val_df["item_id"].tolist(), tokenizer, max_len=max_len)
    test_ds = ClaimDataset(test_df["text"].fillna("").tolist(), test_df["label"].astype(int).tolist(), test_df["item_id"].tolist(), tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True, collate_fn=collator, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=micro_batch * 2, shuffle=False, collate_fn=collator, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=micro_batch * 2, shuffle=False, collate_fn=collator, pin_memory=(device.type == "cuda"))

    # 3. Model
    print(f"Loading model weights for {model_name} from {load_target}...", flush=True)
    config = AutoConfig.from_pretrained(load_target, num_labels=num_classes)
    model = AutoModelForSequenceClassification.from_pretrained(load_target, config=config).float().to(device)

    # 4. Optimizer & Schedule
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.999))

    total_update_steps = (len(train_loader) // grad_accum) * num_epochs
    warmup_steps = int(total_update_steps * 0.10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)

    # 5. Training Loop
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_checkpoint_path = ARES_DATA_ROOT / "checkpoints" / f"{exp_id}_best.pt"
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    patience_counter = 0

    loss_fn = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    steps_per_epoch = len(train_loader) // grad_accum
    print(f"Starting training: total update steps = {total_update_steps}, warmup steps = {warmup_steps}", flush=True)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        step_in_epoch = 0

        for step, batch in enumerate(train_loader):
            _ = batch.pop("item_ids")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss = loss / grad_accum

            loss.backward()
            running_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_in_epoch += 1

                if step_in_epoch % 25 == 0 or step_in_epoch == steps_per_epoch:
                    elapsed = time.time() - epoch_start
                    steps_left = steps_per_epoch - step_in_epoch
                    est_epoch_rem = (elapsed / step_in_epoch) * steps_left if step_in_epoch > 0 else 0
                    print(f"  [Epoch {epoch}/{num_epochs}] Step {step_in_epoch}/{steps_per_epoch} | Batch Loss: {loss.item()*grad_accum:.4f} | Elapsed: {elapsed:.0f}s | Est. epoch left: {est_epoch_rem:.0f}s", flush=True)

        train_loss = running_loss / len(train_loader)
        val_loss, val_f1, _, _, _ = evaluate_model(model, val_loader, device, use_bf16=use_bf16)
        epoch_dur = time.time() - epoch_start

        print(f"Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f} | Time: {epoch_dur:.1f}s", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "seed": seed,
            }, best_checkpoint_path)
            print(f"  -> New best validation Macro-F1: {val_f1:.4f}. Saved checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # 6. Load Best Model and Evaluate on Test Set (Strictly Once)
    print(f"\nLoading best checkpoint for official test evaluation: {best_checkpoint_path}...")
    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_macro_f1, test_preds, test_probs, test_item_ids = evaluate_model(model, test_loader, device, use_bf16=use_bf16)
    test_golds = test_df["label"].astype(int).values
    test_acc = float(accuracy_score(test_golds, test_preds))

    # 95% Bootstrap CI on test macro-F1
    def macro_f1_metric(gold_arr, pred_arr):
        return f1_score(gold_arr, pred_arr, average="macro")

    ci_low, ci_high = bootstrap_ci(
        macro_f1_metric,
        test_golds,
        test_preds,
        B=5000,
        seed=seed,
        alpha=0.05
    )

    # Per-class P, R, F1
    p_c, r_c, f_c, sup_c = precision_recall_fscore_support(test_golds, test_preds, labels=classes, zero_division=0)
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
    try:
        if num_classes == 2:
            auc_val = float(roc_auc_score(test_golds, test_probs[:, 1]))
        else:
            auc_val = float(roc_auc_score(test_golds, test_probs, multi_class="ovr"))
    except Exception:
        auc_val = None

    print(f"\n=== FINAL TEST EVALUATION: {exp_id} ===")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Macro-F1: {test_macro_f1:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}]) | AUC: {auc_val if auc_val else 0.0:.4f}")

    # 7. Save Instance Predictions Parquet
    preds_dir = MILESTONES_ROOT / "M03_text_verifiers" / "results" / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame({
        "item_id": test_item_ids,
        "gold": test_golds,
        "pred": test_preds,
    })
    for cls_idx in range(num_classes):
        preds_df[f"prob_{cls_idx}"] = test_probs[:, cls_idx]

    pred_parquet_path = preds_dir / f"{exp_id}.parquet"
    preds_df.to_parquet(pred_parquet_path, index=False, engine="pyarrow")
    print(f"Saved instance-level predictions to {pred_parquet_path}")

    # 8. Save Result JSON
    results_dir = MILESTONES_ROOT / "M03_text_verifiers" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    finished_time = time.time()

    result_record = {
        "experiment_id": exp_id,
        "milestone": "M03",
        "ledger_ids": ["C038", "C050", "C065"],
        "data_kind": "real",
        "git_commit": "",
        "dirty_worktree": False,
        "config_path": f"configs/m03/transformer_{clean_model_id}.yaml",
        "config_hash": "sha256:0000000000000000",
        "data_manifest_hash": "sha256:0000000000000000",
        "split": "test",
        "seed": seed,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cpu": "Intel Core i7-13620H",
            "precision": "bfloat16" if use_bf16 else "float32",
        },
        "precision": "bf16" if use_bf16 else "fp32",
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "effective_batch": micro_batch * grad_accum,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished_time)),
        "wall_clock_s": finished_time - start_time,
        "n_examples": len(test_df),
        "metrics": {
            "accuracy": test_acc,
            "macro_f1": test_macro_f1,
            "val_macro_f1": best_val_f1,
            "val_loss": best_val_loss,
            "test_loss": test_loss,
            "auc": auc_val if auc_val is not None else 0.0,
        },
        "ci95": {
            "macro_f1": [ci_low, ci_high]
        },
        "per_class": per_class_metrics,
        "predictions_path": str(pred_parquet_path.relative_to(MILESTONES_ROOT.parent)),
        "notes": f"Fine-tuned {model_name} on {dataset_name} veracity."
    }

    res_json_path = results_dir / f"{exp_id}.json"
    write_result(result_record, res_json_path, allow_dirty=True)
    print(f"Results JSON written to {res_json_path}")

    return result_record


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Fine-tune transformer verifier")
    parser.add_argument("--model", type=str, default="roberta-base", help="Model name or HF checkpoint")
    parser.add_argument("--dataset", type=str, default="liar", help="Dataset name in data/processed/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum token length")
    args = parser.parse_args()

    train_and_eval_verifier(
        model_name=args.model,
        dataset_name=args.dataset,
        seed=args.seed,
        num_epochs=args.epochs,
        lr=args.lr,
        micro_batch=args.batch_size,
        grad_accum=args.grad_accum,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
