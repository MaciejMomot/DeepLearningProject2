import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

LABELS_12 = ["background\nnoise"] + TARGET_WORDS + ["unknown"]
LABELS_11 = TARGET_WORDS + ["unknown"]

SHORT_12 = ["bgnd"] + TARGET_WORDS + ["unk"]
SHORT_11 = TARGET_WORDS + ["unk"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
MEAN_COLOR  = "#d62728"

def parse_seed(name: str):
    m = re.search(r"_seed(\d+)$", name)
    return int(m.group(1)) if m else None


def load_confusion_matrices(exp_dir: Path, filename: str) -> list[np.ndarray]:
    matrices = []
    for seed_dir in sorted(exp_dir.iterdir()):
        if not seed_dir.is_dir() or parse_seed(seed_dir.name) is None:
            continue
        p = seed_dir / filename
        if not p.exists():
            print(f"  WARN: no {filename} in {seed_dir.name}")
            continue
        df = pd.read_csv(p, index_col=0)
        matrices.append(df.values.astype(float))
    return matrices


def load_histories(exp_dir: Path) -> list[pd.DataFrame]:
    histories = []
    for seed_dir in sorted(exp_dir.iterdir()):
        if not seed_dir.is_dir() or parse_seed(seed_dir.name) is None:
            continue
        p = seed_dir / "training_history.csv"
        if not p.exists():
            print(f"  WARN: no training_history.csv in {seed_dir.name}")
            continue
        histories.append(pd.read_csv(p))
    return histories

def print_cm_stats(mean_cm: np.ndarray, labels: list[str], title: str):
    n = len(labels)
    row_sums = mean_cm.sum(axis=1)
    recall = np.where(row_sums > 0, mean_cm.diagonal() / row_sums, 0.0)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    print(f"\n  Recall per class (diagonal):")
    for lbl, rec, total in zip(labels, recall, row_sums):
        bar = "█" * int(rec * 20)
        print(f"    {lbl:>14s}  {rec:.3f}  ({int(total):4d} samples)  {bar}")

    print(f"\n  Macro recall: {recall.mean():.4f}")

    errors = []
    for i in range(n):
        for j in range(n):
            if i != j and mean_cm[i, j] > 0:
                errors.append((mean_cm[i, j], labels[i], labels[j]))
    errors.sort(reverse=True)

    print(f"\n  Top 10 errors (true -> predicted, average over seeds):")
    for count, true_lbl, pred_lbl in errors[:10]:
        print(f"    {true_lbl:>14s} → {pred_lbl:<14s}  {count:.1f}")
    print()


def plot_confusion_matrix(matrices: list[np.ndarray], labels: list[str],
                          title: str, outpath: Path, normalize: bool = True):
    if not matrices:
        print(f"  SKIP: no matrices for {title}")
        return

    mean_cm = np.mean(np.stack(matrices, axis=0), axis=0)
    print_cm_stats(mean_cm, labels, title)

    if normalize:
        row_sums = mean_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 
        cm_plot = mean_cm / row_sums
        fmt_str = "{:.2f}"
        cbar_label = "Recall per true class (row-normalized)"
    else:
        cm_plot = mean_cm
        fmt_str = "{:.0f}"
        cbar_label = "Mean count across seeds"

    n = len(labels)
    fig_size = max(7, n * 0.65)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=1 if normalize else None)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_label, size=8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            if np.isnan(val):
                continue
            text = fmt_str.format(val)
            color = "white" if val > thresh else "black"
            fontsize = 7 if n > 11 else 8
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=fontsize,
                    fontweight="bold" if i == j else "normal")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {outpath}")


def plot_learning_curves(histories: list[pd.DataFrame], title: str, outpath: Path,
                         metric_col: str = "val_macro_f1",
                         train_col: str = "train_macro_f1"):
    if not histories:
        print(f"  SKIP: no history for {title}")
        return

    available = histories[0].columns.tolist()
    print(f"Available columns in training_history: {available}")

    if metric_col not in available:
        candidates = [c for c in available if "val" in c and "f1" in c.lower()]
        metric_col = candidates[0] if candidates else available[-1]
        print(f"I use column val: '{metric_col}'")

    train_available = train_col in available
    if not train_available:
        candidates = [c for c in available if "train" in c and "f1" in c.lower()]
        train_col = candidates[0] if candidates else None
        train_available = train_col is not None
        if train_available:
            print(f"i use column train: '{train_col}'")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    all_val = []
    for idx, hist in enumerate(histories):
        epochs = range(1, len(hist) + 1)
        color = SEED_COLORS[idx % len(SEED_COLORS)]
        seed_label = f"Seed {idx + 1}"

        ax.plot(epochs, hist[metric_col], color=color, alpha=0.5,
                linewidth=1.2, linestyle="-", label=f"Val {seed_label}")

        if train_available and train_col in hist.columns:
            ax.plot(epochs, hist[train_col], color=color, alpha=0.25,
                    linewidth=0.8, linestyle="--")

        all_val.append(hist[metric_col].values)

    min_len = min(len(v) for v in all_val)
    val_array = np.stack([v[:min_len] for v in all_val], axis=0)
    mean_val = val_array.mean(axis=0)
    epochs_mean = range(1, min_len + 1)

    ax.plot(epochs_mean, mean_val, color=MEAN_COLOR, linewidth=2.5,
            linestyle="-", label="Mean val (all seeds)", zorder=5)

    best_epoch = int(np.argmax(mean_val)) + 1
    best_val   = mean_val[best_epoch - 1]
    ax.axhline(best_val, color=MEAN_COLOR, linewidth=0.8, linestyle=":",
               alpha=0.7)
    ax.axvline(best_epoch, color=MEAN_COLOR, linewidth=0.8, linestyle=":",
               alpha=0.7)
    ax.annotate(f"Best: {best_val:.4f}\n(ep. {best_epoch})",
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 0.5, best_val - 0.04),
                fontsize=7.5, color=MEAN_COLOR,
                arrowprops=dict(arrowstyle="->", color=MEAN_COLOR, lw=0.8))

    if train_available:
        ax.plot([], [], color="gray", linewidth=1.2, linestyle="--",
                label="Train (individual seeds)", alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1-score")
    ax.set_title(title)
    ax.set_xlim(1, min_len)
    ax.set_ylim(bottom=max(0, mean_val.min() - 0.05))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {outpath}")

    if all_val:
        min_len = min(len(v) for v in all_val)
        val_array = np.stack([v[:min_len] for v in all_val], axis=0)
        mean_val  = val_array.mean(axis=0)
        std_val   = val_array.std(axis=0)
        best_ep   = int(np.argmax(mean_val)) + 1
        print(f"  Best val epoch:  {best_ep}")
        print(f"  Best val mean:   {mean_val[best_ep-1]:.4f} ± {std_val[best_ep-1]:.4f}")
        print(f"  Final val mean:  {mean_val[-1]:.4f} ± {std_val[-1]:.4f}")
        for idx, h in enumerate(histories):
            if metric_col in h.columns:
                peak = h[metric_col].max()
                print(f"  Seed {idx+1} peak val: {peak:.4f}  (ep {h[metric_col].idxmax()+1})")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Checkpoints directory (contains experiments subdirectories)")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--ext", default="pdf", choices=["pdf", "png"])
    parser.add_argument(
        "--two_stage_exp",
        default="two_stage_cnngru_raw_speed_conv64_lr1e3_drop01_phase3_ep30",
        help="Name of experiment directory Two-Stage Phase 3",
    )
    parser.add_argument(
        "--single_stage_exp",
        default="single_stage_cnngru_raw_speed_conv64_lr1e3_drop01_phase3_ep30",
        help="Name of experiment directory Single-Stage Phase 3",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out  = Path(args.outdir)
    out.mkdir(exist_ok=True, parents=True)
    ext = args.ext

    two_dir  = root / args.two_stage_exp
    sing_dir = root / args.single_stage_exp

    print(f"\nTwo-Stage dir:    {two_dir}  (exists: {two_dir.exists()})")
    print(f"Single-Stage dir: {sing_dir}  (exists: {sing_dir.exists()})")
    print(f"Output dir:       {out}\n")

    print("=== Two-Stage: confusion matrix (12-class final) ===")
    cms_two_12 = load_confusion_matrices(two_dir, "test_confusion_matrix_final_12class.csv")
    plot_confusion_matrix(
        cms_two_12, SHORT_12,
        title="CNN+GRU Two-Stage: 12-class confusion matrix (mean over seeds)",
        outpath=out / f"fig_cm_twostage_12class.{ext}",
    )

    print("\n=== Two-Stage: confusion matrix (11-class model space) ===")
    cms_two_11 = load_confusion_matrices(two_dir, "test_confusion_matrix_model_space.csv")
    plot_confusion_matrix(
        cms_two_11, SHORT_11,
        title="CNN+GRU Two-Stage: 11-class model space confusion matrix (mean over seeds)",
        outpath=out / f"fig_cm_twostage_11class.{ext}",
    )

    print("\n=== Single-Stage: confusion matrix (12-class) ===")
    cms_sing = load_confusion_matrices(sing_dir, "test_confusion_matrix_model_space.csv")
    plot_confusion_matrix(
        cms_sing, SHORT_12,
        title="CNN+GRU Single-Stage: 12-class confusion matrix (mean over seeds)",
        outpath=out / f"fig_cm_singlestage_12class.{ext}",
    )

    print("\n=== Two-Stage: learning curves ===")
    hist_two = load_histories(two_dir)
    plot_learning_curves(
        hist_two,
        title="CNN+GRU Two-Stage: validation Macro F1 per epoch (30 epochs)",
        outpath=out / f"fig_lc_twostage.{ext}",
    )

    print("\n=== Single-Stage: learning curves ===")
    hist_sing = load_histories(sing_dir)
    plot_learning_curves(
        hist_sing,
        title="CNN+GRU Single-Stage: validation Macro F1 per epoch (30 epochs)",
        outpath=out / f"fig_lc_singlestage.{ext}",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()