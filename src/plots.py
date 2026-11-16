"""Report figures and the results table, from experiments/results/<run>/.

Figures (experiments/results/figures/):
  training_curves.png     validation loss vs. tokens trained, every run
  canary_exposure.png     mean canary exposure vs. number of copies inserted
  canary_extraction.png   fraction of canaries extracted by greedy decoding
  exposure_vs_seen.png    exposure vs. times the model actually saw each canary
  mia_roc.png             membership-inference ROC curves (loss attack)
Table: experiments/results/results_table.md (privacy vs. utility)

Skips any run that doesn't have results yet.  Usage:  python src/plots.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import length_matched  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = RES / "figures"

# Fixed identity per run (validated categorical slots 1-3; baseline is a neutral reference).
RUNS = {
    "baseline_M": {"label": "Baseline (no canaries)", "color": "#8a8984", "ls": ":"},
    "canary_M": {"label": "No defense", "color": "#2a78d6", "ls": "-"},
    "canary_dedup_M": {"label": "Dedup", "color": "#eb6834", "ls": "-"},
    "dp_eps8_M": {"label": "DP-SGD (ε=8)", "color": "#1baf7a", "ls": "-"},
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "figure.facecolor": "white",
})


def load(run):
    d = RES / run
    out = {}
    if (d / "log.jsonl").exists():
        out["log"] = [json.loads(l) for l in open(d / "log.jsonl")]
    if (d / "config.json").exists():
        out["config"] = json.loads((d / "config.json").read_text())
    a = d / "attacks"
    if (a / "summary.json").exists():
        out["summary"] = json.loads((a / "summary.json").read_text())
        with open(a / "canaries.csv") as f:
            out["canaries"] = list(csv.DictReader(f))
        if (a / "mia_scores.csv").exists():
            with open(a / "mia_scores.csv") as f:
                out["mia"] = list(csv.DictReader(f))
    return out


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def training_curves(data):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for run, d in data.items():
        if "log" not in d:
            continue
        x = [r["tokens"] / 1e6 for r in d["log"]]
        y = [r["val_loss"] for r in d["log"]]
        ax.plot(x, y, color=RUNS[run]["color"], ls=RUNS[run]["ls"], label=RUNS[run]["label"])
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("Tokens trained (millions)")
    ax.set_ylabel("Validation loss (lower is better)")
    ax.set_ylim(top=min(4.0, ax.get_ylim()[1]))
    ax.legend()
    save(fig, "training_curves.png")


def by_reps(rows, key):
    reps = sorted({int(r["reps"]) for r in rows})
    vals = [[float(r[key]) for r in rows if int(r["reps"]) == k] for k in reps]
    return reps, vals


def canary_plots(data):
    have = {r: d for r, d in data.items() if "canaries" in d}
    if not have:
        return
    # Exposure vs repetitions
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for run, d in have.items():
        reps, vals = by_reps(d["canaries"], "exposure")
        xs = np.arange(len(reps))
        means = [np.mean(v) for v in vals]
        ax.plot(xs, means, color=RUNS[run]["color"], ls=RUNS[run]["ls"], marker="o", ms=5,
                label=RUNS[run]["label"])
    ax.axhline(np.log2(1e6), color=MUTED, lw=0.8, ls="--")
    ax.text(0, np.log2(1e6) - 0.3, "max (secret is the model's top guess)", fontsize=8, color=MUTED, va="top")
    ax.axhline(1 / np.log(2), color=MUTED, lw=0.8, ls="--")
    ax.text(0, 1 / np.log(2) + 0.3, "random guessing", fontsize=8, color=MUTED, va="bottom")
    ax.set_xticks(xs, [str(r) if r else "0\n(held out)" for r in reps])
    ax.set_xlabel("Copies of the canary in the training data")
    ax.set_ylabel("Mean exposure (bits)")
    ax.set_ylim(0, 21)
    ax.legend(loc="center left")
    save(fig, "canary_exposure.png")

    # Extraction rate
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    runs = [r for r in have if r != "baseline_M"] or list(have)
    w = 0.8 / len(runs)
    for i, run in enumerate(runs):
        reps = sorted({int(r["reps"]) for r in have[run]["canaries"]})
        rates = [np.mean([str(r["extracted"]) == "True" for r in have[run]["canaries"] if int(r["reps"]) == k]) * 100
                 for k in reps]
        xs = np.arange(len(reps)) + (i - (len(runs) - 1) / 2) * w
        ax.bar(xs, rates, width=w * 0.92, color=RUNS[run]["color"], label=RUNS[run]["label"])
    ax.set_xticks(np.arange(len(reps)), [str(r) if r else "0\n(held out)" for r in reps])
    ax.set_xlabel("Copies of the canary in the training data")
    ax.set_ylabel("Secrets extracted (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left")
    save(fig, "canary_extraction.png")

    # Exposure vs times actually seen
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for run, d in have.items():
        if run == "baseline_M":
            continue
        s = np.array([int(r["times_seen"]) for r in d["canaries"] if int(r["reps"]) > 0])
        e = np.array([float(r["exposure"]) for r in d["canaries"] if int(r["reps"]) > 0])
        ax.scatter(s + 1, e, s=18, color=RUNS[run]["color"], alpha=0.7, edgecolor="white", lw=0.5,
                   label=RUNS[run]["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Times the model saw the canary + 1 (log scale)")
    ax.set_ylabel("Exposure (bits)")
    ax.set_ylim(0, 21)
    ax.legend(loc="upper left")
    save(fig, "exposure_vs_seen.png")


def mia_roc(data):
    have = {r: d for r, d in data.items() if "mia" in d}
    if not have:
        return
    groups = [("seen0", "never fully seen"), ("seen1", "seen once"), ("seen2", "seen twice"),
              ("seen3+", "seen 3+ times")]
    fig, axes = plt.subplots(1, len(have), figsize=(3.2 * len(have), 3.4), squeeze=False)
    shades = ["#b7d3f2", "#79ade8", "#2a78d6", "#0d3f7a"]   # one hue, light -> dark = more exposure
    for ax, (run, d) in zip(axes[0], have.items()):
        non_rows = [r for r in d["mia"] if r["group"] == "nonmember"]
        for (g, label), c in zip(groups, shades):
            rows = [r for r in d["mia"] if r["group"] == g]
            if len(rows) < 10:
                continue
            match = length_matched([int(r["n_tokens"]) for r in rows], [int(r["n_tokens"]) for r in non_rows])
            mem = [-float(r["loss"]) for r in rows]
            non = [-float(non_rows[i]["loss"]) for i in match]
            y = np.r_[np.ones(len(mem)), np.zeros(len(non))]
            fpr, tpr, _ = roc_curve(y, np.r_[mem, non])
            auc = d["summary"]["mia"]["loss"][g]["auc"]
            ax.plot(fpr, tpr, color=c, lw=1.6, label=f"{label} (AUC {auc:.2f})")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
        ax.set_title(RUNS[run]["label"], fontsize=10, color=TEXT)
        ax.text(0.02, 0.98, "members vs. length-matched\nvalidation stories", transform=ax.transAxes,
                fontsize=7, color=MUTED, va="top")
        ax.set_xlabel("False positive rate")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="lower right")
    axes[0][0].set_ylabel("True positive rate")
    save(fig, "mia_roc.png")


def table(data):
    lines = ["| Run | ε | Val loss | Val ppl | Extracted (all inserted) | Extracted (64 copies) | "
             "Mean exposure, 1 copy | Mean exposure, 64 copies | MIA AUC (train vs. val) | MIA AUC (seen 3+) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for run, d in data.items():
        if "summary" not in d:
            continue
        s = d["summary"]
        c = s["canaries"]
        ins = [r for r in d["canaries"] if int(r["reps"]) > 0]
        ext_all = np.mean([r["extracted"] == "True" for r in ins]) * 100
        eps = d.get("config", {}).get("dp", {}).get("epsilon")
        mia = s.get("mia", {}).get("loss", {})
        f = lambda g: f"{mia[g]['auc']:.3f}" if g in mia else "n/a"
        lines.append(f"| {RUNS[run]['label']} | {eps:.1f} | " if eps else f"| {RUNS[run]['label']} | ∞ | ")
        lines[-1] += (f"{s['val_loss']:.3f} | {s['val_perplexity']:.2f} | {ext_all:.0f}% | "
                      f"{c['64']['extracted_frac'] * 100:.0f}% | {c['1']['exposure_mean']:.1f} | "
                      f"{c['64']['exposure_mean']:.1f} | {f('train_any')} | {f('seen3+')} |")
    out = RES / "results_table.md"
    out.write_text("\n".join(lines) + "\n\n(Baseline has no canaries in its data: its canary numbers are "
                   "the 'never trained on it' reference. Exposure: 0 to 19.9 bits; random guessing averages 1.44.)\n",
                   encoding="utf-8")
    print("wrote", out)


def main():
    data = {run: load(run) for run in RUNS}
    data = {r: d for r, d in data.items() if d}
    training_curves(data)
    canary_plots(data)
    mia_roc(data)
    table(data)


if __name__ == "__main__":
    main()
