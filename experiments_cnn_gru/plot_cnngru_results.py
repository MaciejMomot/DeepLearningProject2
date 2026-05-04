import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLOR_MEL  = "#1f77b4"   
COLOR_MFCC = "#ff7f0e"   
COLOR_RAW  = "#2ca02c"   
COLOR_HIGHLIGHT = "#d62728"  
COLOR_NEUTRAL   = "#7f7f7f"


def get(df, name):
    rows = df[df["experiment"] == name]
    if rows.empty:
        raise KeyError(f"No experiment '{name}' in CSV")
    return rows.iloc[0]


def f1(df, name):
    r = get(df, name)
    return r["macro_f1_mean"], r["macro_f1_std"]


def annotate_bars(ax, bars, values, stds, fmt="{:.3f}", offset=0.005):
    for bar, val, std in zip(bars, values, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + std + offset,
            fmt.format(val),
            ha="center", va="bottom", fontsize=8,
        )


def plot_phase1(df, outpath):
    """Grouped bar chart: preprocessing x augmentation."""
    augs    = ["none", "speed", "bgnoise", "specaug"]
    aug_lbl = ["None", "Speed perturb.", "Background noise", "SpecAugment"]

    def lookup(prep, aug):
        candidates = [
            f"two_stage_cnngru_{prep}_{aug}",
            f"two_stage_cnngru_{prep}_{aug}_conv64_lr1e3_drop01", 
        ]
        for c in candidates:
            if (df["experiment"] == c).any():
                return f1(df, c)
        return None 

    preps     = ["mel", "mfcc", "raw"]
    prep_lbl  = ["Mel-spectrogram", "MFCC", "Raw waveform"]
    prep_col  = [COLOR_MEL, COLOR_MFCC, COLOR_RAW]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))

    n_aug = len(augs)
    n_prep = len(preps)
    width = 0.25
    x = np.arange(n_aug)

    for i, (prep, lbl, col) in enumerate(zip(preps, prep_lbl, prep_col)):
        means, stds = [], []
        for aug in augs:
            res = lookup(prep, aug)
            if res is None:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(res[0])
                stds.append(res[1])
        offset = (i - (n_prep - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            label=lbl, color=col, edgecolor="black", linewidth=0.5,
            error_kw=dict(elinewidth=0.8),
        )
        for bar, m, s in zip(bars, means, stds):
            if not np.isnan(m):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    m + s + 0.01,
                    f"{m:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(aug_lbl)
    ax.set_xlabel("Augmentation technique")
    ax.set_ylabel("Test Macro F1-score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Phase 1: Preprocessing $\\times$ Augmentation (CNN+GRU)")
    ax.legend(loc="lower left", framealpha=0.95, ncol=3)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


def plot_phase2a(df, outpath):
    confs = [
        (32,  "two_stage_cnngru_raw_speed_conv32_lr1e3_drop01"),
        (64,  "two_stage_cnngru_raw_speed_conv64_lr1e3_drop01"),
        (128, "two_stage_cnngru_raw_speed_conv128_lr1e3_drop01"),
    ]

    means = [f1(df, n)[0] for _, n in confs]
    stds  = [f1(df, n)[1] for _, n in confs]
    labels = [str(c) for c, _ in confs]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    best = int(np.argmax(means))
    colors = [COLOR_HIGHLIGHT if i == best else COLOR_RAW for i in range(len(means))]

    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6,
                  error_kw=dict(elinewidth=0.8))
    annotate_bars(ax, bars, means, stds, offset=0.015)

    ax.set_xlabel("CNN filters")
    ax.set_ylabel("Test Macro F1-score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Phase 2a: Number of CNN filters")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


def plot_phase2b(df, outpath):
    confs = [
        ("$10^{-3}$",      "two_stage_cnngru_raw_speed_conv64_lr1e3_drop01"),
        ("$10^{-4}$",      "two_stage_cnngru_raw_speed_conv64_lr1e4_drop01"),
        ("Cosine ann.\n($10^{-3}$ init)", "two_stage_cnngru_raw_speed_conv64_cosine_drop01"),
    ]

    means = [f1(df, n)[0] for _, n in confs]
    stds  = [f1(df, n)[1] for _, n in confs]
    labels = [l for l, _ in confs]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))

    best = int(np.argmax(means))
    colors = [COLOR_HIGHLIGHT if i == best else COLOR_RAW for i in range(len(means))]

    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6,
                  error_kw=dict(elinewidth=0.8))
    annotate_bars(ax, bars, means, stds, offset=0.015)

    ax.set_xlabel("Learning rate strategy")
    ax.set_ylabel("Test Macro F1-score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Phase 2b: Learning rate / scheduler")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


def plot_phase2c(df, outpath):
    confs = [
        (0.1, "two_stage_cnngru_raw_speed_conv64_lr1e3_drop01"),
        (0.3, "two_stage_cnngru_raw_speed_conv64_lr1e3_drop03"),
        (0.5, "two_stage_cnngru_raw_speed_conv64_lr1e3_drop05"),
    ]

    means = [f1(df, n)[0] for _, n in confs]
    stds  = [f1(df, n)[1] for _, n in confs]
    labels = [str(d) for d, _ in confs]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    best = int(np.argmax(means))
    colors = [COLOR_HIGHLIGHT if i == best else COLOR_RAW for i in range(len(means))]

    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6,
                  error_kw=dict(elinewidth=0.8))
    annotate_bars(ax, bars, means, stds, offset=0.005)

    ax.set_xlabel("Dropout rate")
    ax.set_ylabel("Test Macro F1-score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Phase 2c: Dropout regularization")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")

def plot_phase3(df, outpath):
    metrics = [
        ("Macro F1",        "macro_f1_mean",     "macro_f1_std"),
        ("Balanced acc.",   "balanced_acc_mean", "balanced_acc_std"),
        ("Overall acc.",    "overall_acc_mean",  "overall_acc_std"),
    ]

    two   = get(df, "two_stage_cnngru_raw_speed_conv64_lr1e3_drop01_phase3_ep30")
    sing  = get(df, "single_stage_cnngru_raw_speed_conv64_lr1e3_drop01_phase3_ep30")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    n = len(metrics)
    width = 0.35
    x = np.arange(n)

    two_means  = [two[m]  for _, m, _ in metrics]
    two_stds   = [two[s]  for _, _, s in metrics]
    sing_means = [sing[m] for _, m, _ in metrics]
    sing_stds  = [sing[s] for _, _, s in metrics]

    bars1 = ax.bar(x - width/2, two_means, width, yerr=two_stds, capsize=3,
                   color=COLOR_RAW, edgecolor="black", linewidth=0.6,
                   label="Approach 1 (Two-Stage, 11 cls. + VAD)",
                   error_kw=dict(elinewidth=0.8))
    bars2 = ax.bar(x + width/2, sing_means, width, yerr=sing_stds, capsize=3,
                   color=COLOR_NEUTRAL, edgecolor="black", linewidth=0.6,
                   label="Approach 2 (Single-Stage, 12 cls.)",
                   error_kw=dict(elinewidth=0.8))

    for bars, means, stds in [(bars1, two_means, two_stds), (bars2, sing_means, sing_stds)]:
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _, _ in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Phase 3: Silence handling approaches comparison (CNN+GRU, 30 epochs)")
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="phase1_summary.csv")
    parser.add_argument("--outdir", default="figures",
                        help="output dir (default: ./figures)")
    parser.add_argument("--ext", default="pdf", choices=["pdf", "png"],
                        help="output format (default: pdf)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = Path(args.outdir)
    out.mkdir(exist_ok=True, parents=True)

    print(f"Loaded {len(df)} experiments from {args.csv}")
    print(f"Saving plots to {out}/ ...")

    plot_phase1(df,  out / f"fig1_phase1.{args.ext}")
    plot_phase2a(df, out / f"fig2_phase2a.{args.ext}")
    plot_phase2b(df, out / f"fig3_phase2b.{args.ext}")
    plot_phase2c(df, out / f"fig4_phase2c.{args.ext}")
    plot_phase3(df,  out / f"fig5_phase3.{args.ext}")

    print("Done.")

if __name__ == "__main__":
    main()