"""
Generate paper-ready PNGs for the WHCA retrieval evaluation.
Output: results/figures/fig_{1..4}.png

Usage:
    python src/make_figures.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e0deda",
    "grid.linewidth": 0.6,
    "xtick.bottom": False,
    "ytick.left": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f8f5",
})

C_BM25 = "#2a78d6"
C_BI   = "#1baf7a"
C_RNK  = "#c98500"
C_CAP  = "#2a78d6"
C_DOC  = "#c98500"

C_CE = "#9b3fcc"

SYSTEMS = ["BM25", "Bi-encoder", "Dense + Rerank", "Cross-encoder"]
SYS_COLORS = [C_BM25, C_BI, C_RNK, C_CE]

# ── Data (Docling chunks, 4 systems) ───────────────────────────────────────
docling_results = {
    "BM25":           dict(mrr=0.318, h1=0.208, h3=0.333, map=0.130),
    "Bi-encoder":     dict(mrr=0.230, h1=0.167, h3=0.208, map=0.130),
    "Dense + Rerank": dict(mrr=0.360, h1=0.208, h3=0.500, map=0.117),
    "Cross-encoder":  dict(mrr=0.452, h1=0.292, h3=0.542, map=0.136),
}

ablation = [
    dict(size=200,  BM25=0.234, Bi=0.217, Rnk=0.273, CE=0.257),
    dict(size=434,  BM25=0.348, Bi=0.195, Rnk=0.278, CE=0.310),
    dict(size=638,  BM25=0.342, Bi=0.293, Rnk=0.316, CE=0.275),
    dict(size=940,  BM25=0.367, Bi=0.201, Rnk=0.327, CE=0.373),
    dict(size=1384, BM25=0.366, Bi=0.234, Rnk=0.354, CE=0.537),
    dict(size=3000, BM25=0.455, Bi=0.269, Rnk=0.420, CE=0.422),
]

judge_buckets  = ["Bucket A\n(GT positive)", "Bucket B\n(top unlabeled)", "Bucket C\n(low-ranked)"]
judge_capstone = [3.91, 3.21, 1.21]
judge_docling  = [3.81, 3.47, 1.21]
recall_cap, recall_doc = 53.4, 60.3


# ── Fig 1: System comparison (3 metrics, single bars, 4 systems) ──────────
def fig1_system_comparison():
    metrics = [
        ("MRR",   "mrr",  0.58),
        ("Hit@3", "h3",   0.70),
        ("MAP",   "map",  0.20),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)

    x = np.arange(len(SYSTEMS))
    w = 0.55

    for ax, (metric_name, key, ylim) in zip(axes, metrics):
        vals = [docling_results[s][key] for s in SYSTEMS]
        best = max(vals)

        bars = ax.bar(x, vals, width=w, color=SYS_COLORS, linewidth=0, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="#333")

        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(SYSTEMS, fontsize=7.5)
        ax.set_ylim(0, ylim)
        ax.set_ylabel(metric_name if ax == axes[0] else "")

    fig.tight_layout()
    path = OUT / "fig1_system_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 2: Chunk size ablation (line chart, MRR) ──────────────────────────
def fig2_ablation():
    sizes = [d["size"] for d in ablation]
    bm25  = [d["BM25"] for d in ablation]
    bi    = [d["Bi"]   for d in ablation]
    rnk   = [d["Rnk"]  for d in ablation]
    ce    = [d["CE"]   for d in ablation]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.set_title("MRR vs chunk size — chunk size ablation (Docling extraction)",
                 fontweight="bold")

    x = np.arange(len(sizes))

    ax.plot(x, bm25, "o-", color=C_BM25, lw=2, ms=6, label="BM25",          zorder=3)
    ax.plot(x, bi,   "o-", color=C_BI,   lw=2, ms=6, label="Bi-encoder",    zorder=3)
    ax.plot(x, rnk,  "o-", color=C_RNK,  lw=2, ms=6, label="Dense+Rerank",  zorder=3)
    ax.plot(x, ce,   "o-", color=C_CE,   lw=2, ms=6, label="Cross-encoder", zorder=3)

    # Fill under lines
    ax.fill_between(x, bm25, alpha=0.08, color=C_BM25)
    ax.fill_between(x, bi,   alpha=0.08, color=C_BI)
    ax.fill_between(x, rnk,  alpha=0.08, color=C_RNK)
    ax.fill_between(x, ce,   alpha=0.08, color=C_CE)

    # Current default marker
    default_x = sizes.index(940)
    ax.axvline(default_x, color="#aaa", lw=1.2, ls="--", zorder=1)
    ax.text(default_x + 0.08, 0.52, "primary\nconfig", fontsize=7,
            color="#888", va="top")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Chunk size (characters)")
    ax.set_ylabel("MRR")
    ax.set_ylim(0.15, 0.62)
    ax.legend(loc="upper left")

    fig.tight_layout()
    path = OUT / "fig2_ablation_mrr.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 3: LLM judge bucket scores — violin plot ──────────────────────────
def fig3_judge_buckets():
    ljp_path = ROOT / "results/final/data/llm_judge_pairs_docling.csv"
    ljp = pd.read_csv(ljp_path)

    bucket_order  = ["A", "B", "C"]
    bucket_labels = [
        "Bucket A\n(GT positive)",
        "Bucket B\n(top unlabeled)",
        "Bucket C\n(low-ranked)",
    ]
    colors = [C_RNK, C_BM25, "#999999"]

    data_by_bucket = [ljp[ljp["bucket"] == b]["llm_score"].tolist() for b in bucket_order]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.set_title("LLM-as-judge: relevance score distribution by bucket",
                 fontweight="bold")

    parts = ax.violinplot(data_by_bucket, positions=[1, 2, 3],
                          showmedians=True, showextrema=False, widths=0.6)

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.45)
        pc.set_edgecolor(color)

    parts["cmedians"].set_color("#333")
    parts["cmedians"].set_linewidth(1.8)

    # Jittered data points
    rng = np.random.default_rng(42)
    for xi, (data, color) in enumerate(zip(data_by_bucket, colors), start=1):
        jitter = rng.uniform(-0.12, 0.12, len(data))
        ax.scatter(xi + jitter, data, s=14, color=color, alpha=0.6, zorder=3, linewidths=0)

    # Mean annotation
    for xi, data in enumerate(data_by_bucket, start=1):
        mean = np.mean(data)
        std  = np.std(data)
        ax.text(xi, 5.25, f"μ={mean:.2f}\nσ={std:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="#444")

    ax.axhline(4.0, color="#888", lw=0.8, ls="--", zorder=1)
    ax.text(3.35, 4.06, "score = 4\n(relevant)", fontsize=7, color="#888", ha="left", va="bottom")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(bucket_labels, fontsize=8.5)
    ax.set_ylabel("LLM relevance score (1–5)")
    ax.set_ylim(0.5, 6.0)
    ax.set_yticks([1, 2, 3, 4, 5])

    fig.tight_layout()
    path = OUT / "fig3_llm_judge_buckets.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 4: Recall gap (% Bucket B ≥ 4) ───────────────────────────────────
def fig4_recall_gap():
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.set_title("Recall gap: % of top-ranked unlabeled chunks\nscoring ≥ 4/5 relevance (Bucket B)",
                 fontweight="bold", fontsize=9.5)

    labels = ["Capstone", "Docling"]
    vals   = [recall_cap, recall_doc]
    colors = [C_CAP, C_DOC]

    bars = ax.barh(labels, vals, color=colors, linewidth=0, height=0.45, zorder=3)

    for bar, val in zip(bars, vals):
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

    delta = recall_doc - recall_cap
    ax.text(vals[1] / 2, 1,
            f"+{delta:.1f}pp", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

    ax.set_xlim(0, 80)
    ax.set_xlabel("% Bucket B chunks scoring ≥ 4")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color="#e0deda", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = OUT / "fig4_recall_gap.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 5: Per-MP MAP heatmap / grouped bar ───────────────────────────────
def fig5_per_mp_map():
    rpd_path = ROOT / "results/docling_4sys/retrieval_per_doc.csv"
    rpd = pd.read_csv(rpd_path)

    mp_names = {
        0: "Colonies", 1: "LGL", 2: "Eisinga", 3: "Defence Lines",
        4: "Wadden Sea", 5: "Amsterdam", 6: "Van Nelle", 7: "Schokland",
        8: "Kinderdijk", 10: "Rietveld",
    }
    rpd["mp_name"] = rpd["mp_index"].map(mp_names)
    # Sort MPs by Cross-encoder MAP descending
    ce_order = (rpd[rpd["system"] == "Cross-encoder"]
                .sort_values("MAP", ascending=True)["mp_name"].tolist())

    sys_order  = ["BM25", "Bi-encoder", "Reranker", "Cross-encoder"]
    sys_labels = ["BM25", "Bi-encoder", "Dense+Rerank", "Cross-encoder"]
    sys_colors = [C_BM25, C_BI, C_RNK, C_CE]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_title("Per-MP MAP by retrieval system", fontweight="bold")

    n_mp  = len(ce_order)
    n_sys = len(sys_order)
    total_w = 0.72
    w = total_w / n_sys
    offsets = np.linspace(-(total_w - w) / 2, (total_w - w) / 2, n_sys)

    y = np.arange(n_mp)
    for sys, label, color, offset in zip(sys_order, sys_labels, sys_colors, offsets):
        vals = [rpd[(rpd["system"] == sys) & (rpd["mp_name"] == mp)]["MAP"].values
                for mp in ce_order]
        vals = [v[0] if len(v) else 0 for v in vals]
        bars = ax.barh(y + offset, vals, height=w, color=color,
                       linewidth=0, label=label, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(ce_order, fontsize=8.5)
    ax.set_xlabel("MAP")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 0.48)

    fig.tight_layout()
    path = OUT / "fig5_per_mp_map.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 6: CE score histogram (requires ce_scores.csv from retrieval rerun) ─
def fig6_ce_score_histogram(ce_scores_path=None):
    if ce_scores_path is None:
        ce_scores_path = ROOT / "results/docling_4sys_new/ce_scores.csv"
    if not Path(ce_scores_path).exists():
        print(f"[skip fig6] {ce_scores_path} not found — run retrieval.py first")
        return

    df = pd.read_csv(ce_scores_path)
    positive = df[df["in_gt"] == True]["ce_score"]
    negative = df[df["in_gt"] == False]["ce_score"]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.set_title("Cross-encoder score distribution: positive vs negative chunks",
                 fontweight="bold")

    bins = np.linspace(df["ce_score"].min(), df["ce_score"].max(), 60)
    ax.hist(negative, bins=bins, color=C_BI,  alpha=0.55, label="Negative (not in GT)", density=True, zorder=2)
    ax.hist(positive, bins=bins, color=C_RNK, alpha=0.75, label="Positive (GT)",        density=True, zorder=3)

    ax.set_xlabel("Cross-encoder score (logit scale)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUT / "fig6_ce_score_histogram.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce-scores", default=None, help="Path to ce_scores.csv")
    args = parser.parse_args()

    fig1_system_comparison()
    fig2_ablation()
    fig3_judge_buckets()
    fig4_recall_gap()
    fig5_per_mp_map()
    fig6_ce_score_histogram(args.ce_scores)
    print("\nAll figures saved to results/figures/")
