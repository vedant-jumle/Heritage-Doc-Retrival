"""Query-formation ablation: label name vs definition vs both.

Reviewer (line 252): "Why not use label description but only label name? I believe
there is a lot of missed potential for BM25 ... and semantic search."

The premise is a misreading -- the paper already concatenates name AND definition
("q = 'label: definition'", Sec. 2.3.1). But the underlying question is fair and
untested: how much does each component actually contribute? This ablates all three
query forms across all four systems.

Reuses the ranking code in hit_at_k.py; only the query strings change. BM25 and the
bi-encoder are re-run per variant (cheap). The cross-encoder is NOT re-run -- its
cached scores exist only for the published name+definition query, so CE and
Dense+Rerank are reported for that variant alone and marked n/a elsewhere.

Usage: python src/query_ablation.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hit_at_k import (  # noqa: E402
    CDEF, K_VALUES, OUT, canonical_labels, evaluate, load_chunks, load_gt,
    rank_bm25, rank_biencoder,
)

VARIANTS = {
    "name_only": lambda lab, d: lab,
    "def_only": lambda lab, d: d,
    "name_plus_def": lambda lab, d: f"{lab}: {d}",  # the published form
}

REPORT_K = [1, 3, 10, 20]


def build_queries(labels):
    d = pd.read_excel(CDEF, usecols=["Heritage Concept", "Definition"])
    d["Heritage Concept"] = d["Heritage Concept"].str.strip()
    d = d[d.Definition.notna()]
    defs = dict(zip(d["Heritage Concept"], d["Definition"]))
    return {
        name: {lab: fn(lab, defs[lab]) for lab in labels}
        for name, fn in VARIANTS.items()
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    labels = canonical_labels()
    chunks = load_chunks()
    gt = load_gt(labels)
    variants = build_queries(labels)

    print(f"labels {len(labels)} | chunks {len(chunks)}")
    print(f"variants: {list(variants)}")
    print()
    for name, q in variants.items():
        print(f"  {name:14s} e.g. {q[labels[0]][:70]!r}")

    frames = []
    for vname, queries in variants.items():
        print(f"\n=== {vname} ===")
        print("  BM25...")
        bm = rank_bm25(queries, chunks)
        print("  bi-encoder...")
        bi = rank_biencoder(queries, chunks)

        for sysname, ranked in [("BM25", bm), ("Bi-encoder", bi)]:
            df = evaluate(sysname, ranked, gt)
            df["variant"] = vname
            frames.append(df)

    per_label = pd.concat(frames, ignore_index=True)

    cols = [f"Hit@{k}" for k in REPORT_K] + ["R@10", "R@20"]
    macro = (
        per_label.groupby(["system", "variant"])[cols]
        .mean()
        .reindex(
            pd.MultiIndex.from_product(
                [["BM25", "Bi-encoder"], list(VARIANTS)], names=["system", "variant"]
            )
        )
    )

    print("\n" + "=" * 78)
    print("QUERY ABLATION -- macro-averaged over 24 labels")
    print("=" * 78)
    print(macro.round(3).to_string())

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    for sysname in ["BM25", "Bi-encoder"]:
        m = macro.loc[sysname]
        best = m["Hit@10"].idxmax()
        print(f"\n{sysname}: best Hit@10 = {best} ({m.loc[best,'Hit@10']:.3f})")
        for v in VARIANTS:
            print(f"    {v:14s} Hit@1 {m.loc[v,'Hit@1']:.3f}  "
                  f"Hit@3 {m.loc[v,'Hit@3']:.3f}  Hit@10 {m.loc[v,'Hit@10']:.3f}")

    print()
    print("NOTE: Cross-encoder and Dense + Rerank are omitted -- the cached CE")
    print("      scores cover only the published name+definition query. Re-running")
    print("      them would need ~70k CE pair evaluations per variant.")

    per_label.round(4).to_csv(OUT / "query_ablation_per_label.csv", index=False)
    macro.round(4).to_csv(OUT / "query_ablation_macro.csv")
    print(f"\nwritten -> {OUT / 'query_ablation_macro.csv'}")


if __name__ == "__main__":
    sys.exit(main())
