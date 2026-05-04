import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def parse_seed(name: str):
    m = re.search(r"_seed(\d+)$", name)
    return int(m.group(1)) if m else None

def load_seed_metrics(seed_dir: Path) -> dict | None:
    two_stage_path = seed_dir / "test_metrics_two_stage_final_12class.csv"
    if two_stage_path.exists():
        df = pd.read_csv(two_stage_path)
        row = df.iloc[0].to_dict()

        remapped = {}
        for k, v in row.items():
            if k.startswith("final_12class."):
                flat_key = k[len("final_12class."):]  
                remapped[flat_key] = v
            else:
                remapped[k] = v

        remapped["_source"] = "two_stage_final_12class"
        return remapped

    model_space_path = seed_dir / "test_metrics_model_space.csv"
    if model_space_path.exists():
        df = pd.read_csv(model_space_path)
        row = df.iloc[0].to_dict()
        row["_source"] = "model_space_11class"
        return row

    return None

def aggregate(root: Path) -> pd.DataFrame:
    rows = []

    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue

        seed_metrics = []
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir() or parse_seed(seed_dir.name) is None:
                continue

            m = load_seed_metrics(seed_dir)
            if m is None:
                print(f"  WARN: no test_metrics_* in {seed_dir} - seed not used")
                continue

            m["seed"] = parse_seed(seed_dir.name)
            seed_metrics.append(m)

        if not seed_metrics:
            continue

        sources = {m["_source"] for m in seed_metrics}
        if len(sources) > 1:
            print(f"  WARN: {exp_dir.name} has mixed sources of metrics: {sources}")

        df = pd.DataFrame(seed_metrics)

        numeric = df.select_dtypes(include=[np.number]).drop(columns=["seed"], errors="ignore")

        summary = {
            "experiment": exp_dir.name,
            "n_seeds": len(df),
            "metric_source": ", ".join(sorted(sources)),
        }
        for col in numeric.columns:
            summary[f"{col}_mean"] = numeric[col].mean()
            summary[f"{col}_std"]  = numeric[col].std(ddof=0)

        rows.append(summary)

    return pd.DataFrame(rows)

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    result = aggregate(root)

    if result.empty:
        print("No results - check the directory structure")
        return

    key_cols = ["experiment", "n_seeds", "metric_source"]
    for prefix in ["macro_f1", "balanced_acc", "overall_acc"]:
        for suffix in ["_mean", "_std"]:
            col = prefix + suffix
            if col in result.columns:
                key_cols.append(col)

    sort_col = next((c for c in key_cols if c == "macro_f1_mean"), None)
    result_sorted = result.sort_values(by=sort_col, ascending=False) if sort_col else result

    print("\n=== Experiments results (test metrics) ===")
    print(result_sorted[key_cols].to_string(index=False))

    out_path = root / "phase1_summary.csv"
    result.to_csv(out_path, index=False)
    print(f"\n-> Zapisano: {out_path}")


if __name__ == "__main__":
    main()