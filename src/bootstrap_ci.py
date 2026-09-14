"""Bootstrap confidence intervals for the system comparison (paper Table 3).

Reviewer T1gn: "Only 24 labels and 10 documents are evaluated, with some concepts
represented by one to four sentences. No confidence intervals, paired tests, or
label/document bootstrap is provided for Table 1. The cross-encoder's MAP
advantage over BM25 and the bi-encoder is only 0.006, which may not be robust."

The reviewer is right. This resamples the 24 labels with replacement and
recomputes each macro-average, giving a 95% CI for the difference between the
Cross-encoder and each other system.

Result: the MRR and Hit@3 advantages over BM25 and the Bi-encoder are real; the
MAP and Hit@1 differences are not distinguishable from zero for any pair, and the
Cross-encoder is not separable from Dense + Rerank on anything.

Reads only the paper's canonical July results file. Seeded, so the numbers in the
paper are reproducible.

Usage: python src/bootstrap_ci.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
CMP = ROOT / "results/docling_4sys_new/retrieval_comparison.csv"
OUT = ROOT / "results" / "revision"

SEED = 20260914
N_BOOT = 10_000
REFERENCE = "Cross-encoder"
OTHERS = ["BM25", "Bi-encoder", "Dense + Rerank"]
# (display name, column in the results file)
METRICS = [("MRR", "MRR"), ("Hit@1", "Hit@1"), ("Hit@3", "Hit@3"), ("MAP", "AP")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(CMP)
    rng = np.random.default_rng(SEED)

    rows = []
    for name, col in METRICS:
        piv = d.pivot(index="label", columns="system", values=col)
        n = len(piv)
        # One resampling draw shared across systems per replicate: the labels are
        # paired, so the same bootstrap sample must be used for both arms.
        idx = rng.integers(0, n, (N_BOOT, n))
        ref = piv[REFERENCE].to_numpy()
        for other in OTHERS:
            oth = piv[other].to_numpy()
            delta = ref.mean() - oth.mean()
            boots = ref[idx].mean(axis=1) - oth[idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            # Strict: a bound sitting exactly on zero does NOT exclude zero.
            # Hit@K is a mean of binary indicators over 24 labels, so its
            # bootstrap distribution is coarse and bounds land on 0 exactly.
            rows.append({
                "metric": name,
                "comparison": f"CE - {other}",
                "delta": delta,
                "ci_lo": lo,
                "ci_hi": hi,
                "excludes_zero": lo > 0 or hi < 0,
            })

    res = pd.DataFrame(rows)

    print(f"bootstrap over {len(d.label.unique())} labels, "
          f"{N_BOOT:,} replicates, seed={SEED}\n")
    for name, _ in METRICS:
        sub = res[res.metric == name]
        print(f"{name}:")
        for _, r in sub.iterrows():
            mark = " *" if r.excludes_zero else "  "
            print(f"  {r.comparison:24s} {r.delta:+.4f}  "
                  f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]{mark}")
        print()

    print("* = 95% CI excludes zero")
    print("\nReading: the Cross-encoder's MRR and Hit@3 margins over BM25 and the")
    print("Bi-encoder are reliable. MAP and Hit@1 separate nothing. CE vs")
    print("Dense + Rerank is not separable on any metric.")

    res.round(4).to_csv(OUT / "bootstrap_ci.csv", index=False)
    print(f"\nwritten -> {OUT / 'bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()
