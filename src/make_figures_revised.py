"""Camera-ready figures: colour-blind safe, greyscale safe, honest encodings.

Reviewer:
  - "Figures 1,2,3 are not black/white nor colour-blind friendly."
  - "Figure 4 should not be a violin plot ... the vertical axis is not continuous
     but only has 5 distinct levels (curvy lines suggest continuous). Same with
     the horizontal axis. There are scattered dots that seem to have a
     distribution, but as far as I can tell the x-axis has no continuous scale."

Compiled figure order in the paper (NOT the filenames):
  Fig 1 = fig5_per_mp_map      Fig 2 = fig2_ablation_mrr
  Fig 3 = fig6_ce_score_histogram   Fig 4 = fig3_llm_judge_buckets  <- the violin

Fixes applied:
  * Okabe-Ito palette (deuter/prot/tritanopia safe).
  * Every series is ALSO distinguished without colour -- hatch on bars, marker
    shape + line style on lines -- so the figures survive greyscale printing.
  * Ordered luminance within each palette so greyscale keeps series separable.
  * The violin is replaced by a stacked proportion bar of the five discrete
    score levels, with counts printed. No implied continuity on either axis.

This is a SEPARATE module from make_figures.py, which is left untouched as the
record of the originally published figures.

Usage: python src/make_figures_revised.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "figures_revised"

JUDGE_PAIRS = ROOT / "results/final/data/llm_judge_pairs_docling.csv"
PERDOC = ROOT / "results/docling_4sys_new/retrieval_per_doc.csv"
ABLATION = ROOT / "results/ablation_4sys/ablation_4sys.csv"
CE_SCORES = ROOT / "results/docling_4sys_new/ce_scores.csv.bak_pre_gospel"
HITK = ROOT / "results/revision/hit_at_k_macro.csv"
CMP = ROOT / "results/docling_4sys_new/retrieval_comparison.csv"

# Okabe-Ito: colour-blind safe. Ordered so luminance also separates in greyscale.
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREEN = "#009E73"
OI_VERM = "#D55E00"
OI_SKY = "#56B4E9"
OI_GREY = "#999999"

SYSTEMS = ["BM25", "Bi-encoder", "Dense + Rerank", "Cross-encoder"]
SYS_COLOR = {
    "BM25": OI_SKY,
    "Bi-encoder": OI_GREEN,
    "Dense + Rerank": OI_ORANGE,
    "Cross-encoder": OI_BLUE,
}
# Redundant encoding: hatch for bars, (marker, linestyle) for lines.
SYS_HATCH = {"BM25": "///", "Bi-encoder": "\\\\\\", "Dense + Rerank": "...", "Cross-encoder": ""}
SYS_MARKER = {"BM25": "s", "Bi-encoder": "^", "Dense + Rerank": "D", "Cross-encoder": "o"}
SYS_LS = {"BM25": (0, (4, 2)), "Bi-encoder": (0, (1, 1.5)), "Dense + Rerank": (0, (6, 2, 1, 2)), "Cross-encoder": "-"}

MP_NAMES = {
    0: "Colonies", 1: "LGL", 2: "Eisinga", 3: "Defence Lines", 4: "Wadden Sea",
    5: "Amsterdam", 6: "Van Nelle", 7: "Schokland", 8: "Kinderdijk", 10: "Rietveld",
}

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
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "hatch.linewidth": 0.6,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def fig_per_mp_map():
    """Paper Fig 1. Grouped horizontal bars + hatching."""
    rpd = pd.read_csv(PERDOC)
    rpd["mp_name"] = rpd.mp_index.map(MP_NAMES)

    order = (
        rpd[rpd.system == "Cross-encoder"].sort_values("MAP").mp_name.tolist()
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.set_title("Per-MP MAP by retrieval system")

    y = np.arange(len(order))
    total_w, n = 0.78, len(SYSTEMS)
    w = total_w / n
    offsets = np.linspace(-(total_w - w) / 2, (total_w - w) / 2, n)

    for sysname, off in zip(SYSTEMS, offsets):
        vals = [
            rpd[(rpd.system == sysname) & (rpd.mp_name == mp)].MAP.squeeze()
            for mp in order
        ]
        ax.barh(
            y + off, vals, height=w, label=sysname,
            color=SYS_COLOR[sysname], hatch=SYS_HATCH[sysname],
            edgecolor="white", linewidth=0.5, zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xlabel("MAP")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig1_per_mp_map.png")


def fig_ablation():
    """Paper Fig 2. Marker + linestyle carry the series without colour."""
    ab = pd.read_csv(ABLATION)
    ab["system"] = ab.system.replace({"Dense+Rerank": "Dense + Rerank"})
    sizes = sorted(ab.chunk_size.unique())

    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    ax.set_title("MRR vs. chunk size")

    x = np.arange(len(sizes))
    for sysname in SYSTEMS:
        sub = ab[ab.system == sysname].set_index("chunk_size").loc[sizes]
        ax.plot(
            x, sub.MRR.values, label=sysname,
            color=SYS_COLOR[sysname], marker=SYS_MARKER[sysname],
            linestyle=SYS_LS[sysname], lw=1.8, ms=6,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3,
        )

    # The ablation grid's own primary point is 940 chars (2,881 chunks) -- a
    # separate re-chunking from the paper's 1,000-char main config (2,926).
    ax.axvline(sizes.index(940), color="#666", lw=1.0, ls=":", zorder=1)
    # Label sits inside the axes just right of the line, below the legend band,
    # so it clears both the title and the plotted series.
    ax.annotate(
        "ablation reference",
        xy=(sizes.index(940), 0.88), xycoords=("data", "axes fraction"),
        xytext=(4, 0), textcoords="offset points",
        fontsize=7, color="#666", ha="left", va="center",
    )

    n_chunks = ab.drop_duplicates("chunk_size").set_index("chunk_size").n_chunks
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{s:,}\n({n_chunks[s]:,} chunks)" for s in sizes], fontsize=7.5
    )
    ax.set_xlabel("chunk size (characters)")
    ax.set_ylabel("MRR")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "fig2_ablation_mrr.png")


def fig_ce_histogram():
    """Paper Fig 3. Two overlaid step histograms; hatch + linestyle separate them."""
    ce = pd.read_csv(CE_SCORES)

    # The score file covers 26 labels (it also scores 'Landscape dynamics' and
    # 'Management'). The paper evaluates 24, so restrict to those before
    # plotting: 24 x 2,926 = 70,224 pairs, not 26 x 2,926 = 76,076.
    evaluable = set(pd.read_csv(CMP).label.unique())
    ce = ce[ce.label.isin(evaluable)]

    ce["in_gt"] = ce["in_gt"].astype(bool)

    pos = ce[ce.in_gt].ce_score
    neg = ce[~ce.in_gt].ce_score

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.set_title("Cross-encoder score distribution")

    bins = np.linspace(ce.ce_score.min(), ce.ce_score.max(), 45)
    ax.hist(
        neg, bins=bins, density=True, color=OI_GREY, alpha=0.55,
        label=f"not in ground truth (n={len(neg):,})", zorder=2,
    )
    ax.hist(
        pos, bins=bins, density=True, histtype="step", color=OI_BLUE,
        lw=2.0, label=f"ground-truth positive (n={len(pos):,})", zorder=3,
    )
    ax.hist(
        pos, bins=bins, density=True, color=OI_BLUE, alpha=0.18,
        hatch="///", edgecolor=OI_BLUE, linewidth=0, zorder=3,
    )

    ax.set_xlabel("cross-encoder score")
    ax.set_ylabel("density")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_ce_score_histogram.png")


def fig_judge_buckets():
    """Paper Fig 4 -- the violin replacement.

    The data is 144 ratings on a 1-5 ordinal scale across 3 named buckets.
    Neither axis is continuous, so a violin (and its jittered x-scatter) implies
    structure that does not exist. A stacked proportion bar over the five levels
    shows the actual distribution, with n printed per cell.
    """
    ljp = pd.read_csv(JUDGE_PAIRS)

    # The judge run covers 25 labels, including 'Landscape dynamics', which is
    # NOT one of the paper's 24 evaluable labels. Restrict to the evaluated set
    # so the figure matches the numbers reported in the text (n=140, not 144).
    evaluable = set(pd.read_csv(CMP).label.unique())
    ljp = ljp[ljp.label.isin(evaluable)]

    buckets = ["A", "B", "C"]
    titles = {
        "A": "A\nGT positive",
        "B": "B\ntop-ranked, unlabelled",
        "C": "C\nlow-ranked, unlabelled",
    }
    # Sequential, ordered by luminance so greyscale preserves the 1->5 ramp.
    level_color = {1: "#f0f0f0", 2: "#cfcfcf", 3: "#9ecae1", 4: "#4292c6", 5: "#08519c"}
    level_hatch = {1: "xxx", 2: "\\\\\\", 3: "", 4: "", 5: ""}

    ct = pd.crosstab(ljp.bucket, ljp.llm_score).reindex(
        index=buckets, columns=[1, 2, 3, 4, 5], fill_value=0
    )
    prop = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.set_title("LLM-as-judge relevance ratings by bucket")

    y = np.arange(len(buckets))
    left = np.zeros(len(buckets))
    for lvl in [1, 2, 3, 4, 5]:
        vals = prop[lvl].values
        ax.barh(
            y, vals, left=left, height=0.55,
            color=level_color[lvl], hatch=level_hatch[lvl],
            edgecolor="white", linewidth=0.8, zorder=3,
            label=f"score {lvl}",
        )
        for yi, (v, l0) in enumerate(zip(vals, left)):
            if v > 0.045:
                ax.text(
                    l0 + v / 2, yi, str(int(ct.iloc[yi][lvl])),
                    ha="center", va="center", fontsize=7.5, zorder=4,
                    color="white" if lvl >= 4 else "#222",
                    fontweight="bold",
                )
        left += vals

    # Share of ratings >= 4, the quantity the paper actually argues about.
    for yi, b in enumerate(buckets):
        rel = prop.loc[b, [4, 5]].sum()
        ax.text(1.015, yi, f"{rel:.0%} rated ≥4", va="center", fontsize=8, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels([titles[b] for b in buckets], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("proportion of ratings")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()

    handles = [
        Patch(facecolor=level_color[l], hatch=level_hatch[l], edgecolor="white", label=f"{l}")
        for l in [1, 2, 3, 4, 5]
    ]
    ax.legend(
        handles=handles, title="LLM relevance score", ncol=5,
        loc="upper center", bbox_to_anchor=(0.5, -0.28), title_fontsize=8,
    )
    fig.tight_layout()
    save(fig, "fig4_llm_judge_buckets.png")


def fig_hit_at_k():
    """New figure for the reviewer's annotation-prioritisation point."""
    if not HITK.exists():
        print("  skipped hit@k figure -- run src/hit_at_k.py first")
        return

    m = pd.read_csv(HITK, index_col=0)
    ks = [1, 3, 5, 10, 20, 50, 100]

    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    ax.set_title("Hit@K: does scanning deeper find a positive?")

    x = np.arange(len(ks))
    for sysname in SYSTEMS:
        vals = [m.loc[sysname, f"Hit@{k}"] for k in ks]
        ax.plot(
            x, vals, label=sysname,
            color=SYS_COLOR[sysname], marker=SYS_MARKER[sysname],
            linestyle=SYS_LS[sysname], lw=1.8, ms=6,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("K (candidates inspected per concept)")
    ax.set_ylabel("Hit@K")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig5_hit_at_k.png")


def main():
    print("regenerating figures (colour-blind + greyscale safe)")
    fig_per_mp_map()
    fig_ablation()
    fig_ce_histogram()
    fig_judge_buckets()
    fig_hit_at_k()
    print(f"\nall figures -> {OUT}")


if __name__ == "__main__":
    main()
