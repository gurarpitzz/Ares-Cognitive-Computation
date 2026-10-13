"""Script to aggregate all M03 evaluation results into a master summary table."""

import json
from pathlib import Path
import pandas as pd

def main():
    results_dir = Path("milestones/M03_text_verifiers/results")
    json_files = sorted(results_dir.glob("M03-*.json"))
    
    rows = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        exp_id = data.get("experiment_id", jf.stem)
        parts = exp_id.split("-")
        dataset = parts[1] if len(parts) > 1 else ""
        model = "-".join(parts[2:-1]) if len(parts) > 3 else parts[2]
        seed = parts[-1].replace("s", "") if len(parts) > 1 else ""
        
        metrics = data.get("metrics", {})
        ci95 = data.get("ci95", {}).get("macro_f1", [None, None])
        
        row = {
            "experiment_id": exp_id,
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "accuracy": round(metrics.get("accuracy", 0.0), 4),
            "macro_f1": round(metrics.get("macro_f1", 0.0), 4),
            "f1_ci95_low": round(ci95[0], 4) if ci95[0] is not None else None,
            "f1_ci95_high": round(ci95[1], 4) if ci95[1] is not None else None,
            "auc": round(metrics.get("auc", 0.0), 4) if metrics.get("auc") is not None else None,
            "wall_clock_s": round(data.get("wall_clock_s", 0.0), 1),
            "n_examples": data.get("n_examples", 0)
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.sort_values(by=["dataset", "macro_f1"], ascending=[True, False], inplace=True)
    
    out_path = Path("milestones/M03_text_verifiers/tables/text_verifiers_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Aggregated {len(df)} runs into {out_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
