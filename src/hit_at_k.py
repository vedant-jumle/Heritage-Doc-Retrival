"""Hit@K / Recall@K at larger K (reviewer: "Why only k in {1,3}?").

The reviewer's point: under an annotation-prioritisation framing, an expert
scanning the top 100 of a ranked list is entirely reasonable -- far cheaper than
reading the whole corpus. So the paper's K in {1,3} understates the practical
value of the ranking. This recomputes the depth curve for every system.

No model is re-run. Ranked lists are rebuilt from cached artefacts:
  Cross-encoder  : results/docling_4sys_new/ce_scores.csv   (all 70,224 pairs)
  BM25           : recomputed locally (cheap, no GPU, deterministic)
  Bi-encoder     : cached embeddings in chunks_docling_embedded.parquet
  Dense + Rerank : bi-encoder top-50 order, then CE scores for those 50

The label set is PINNED to the 24 labels behind the published results. The
definitions file on disk has since moved to a 26-label "gospel" taxonomy
(adds 'Management ' and 'Landscape dynamics'), so reading it directly would
silently produce numbers that do not match the paper.

Usage: python src/hit_at_k.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "revision"

# The live ce_scores.csv was overwritten in the Aug-2026 "gospel" taxonomy update,
# which re-scored 'Assessment' against a revised definition. Every other label is
# byte-identical. To reproduce the PUBLISHED results we must read the pre-gospel
# backup; using the live file silently changes Assessment's ranking.
CE_SCORES = ROOT / "results/docling_4sys_new/ce_scores.csv.bak_pre_gospel"
CE_SCORES_LIVE = ROOT / "results/docling_4sys_new/ce_scores.csv"
CMP = ROOT / "results/docling_4sys_new/retrieval_comparison.csv"
PSI = ROOT / "results/new_chunks/psi_docling.csv"
CHUNKS = ROOT / "results/new_chunks/chunks_docling.parquet"
EMBEDDED = ROOT / "results/new_chunks/chunks_docling_embedded.parquet"
CDEF = ROOT / "Datasets/Datasets/Heritage concepts/Definitions/New/Heritage Labels & definition.xlsx"

BEEMSTER_MP = 9
TOP_K_RERANK = 50
K_VALUES = [1, 3, 5, 10, 20, 50, 100]


def canonical_labels():
    """The 24 labels actually evaluated in the paper."""
    return sorted(pd.read_csv(CMP).label.unique())


def load_queries(labels):
    """query = 'label: definition', matching the paper's query formation."""
    d = pd.read_excel(CDEF, usecols=["Heritage Concept", "Definition"])
    d["Heritage Concept"] = d["Heritage Concept"].str.strip()
    d = d[d.Definition.notna()]
    defs = dict(zip(d["Heritage Concept"], d["Definition"]))

    missing = [l for l in labels if l not in defs]
    if missing:
        raise SystemExit(f"definitions missing for: {missing}")
    return {l: f"{l}: {defs[l]}" for l in labels}


def load_chunks():
    c = pd.read_parquet(CHUNKS)
    return c[c.mp_index != BEEMSTER_MP].reset_index(drop=True)


def load_gt(labels, strip_labels=False):
    """Positive chunk sets per label, from the psi alignment.

    One psi row carries 'Legislation ' with a trailing space. The published run
    did NOT strip it, so that row was dropped and Legislation was evaluated with
    34 positives rather than 35. `strip_labels=False` reproduces the published
    behaviour; True is the corrected ground truth (see revision_audit.py).
    """
    d = pd.read_csv(PSI)
    if strip_labels:
        d["Label"] = d["Label"].str.strip()
    d = d.dropna(subset=["Chunk ID"])
    d = d[d.Label.isin(labels)]
    return {lab: set(g["Chunk ID"].astype(int)) for lab, g in d.groupby("Label")}


def hit_at_k(ranked, positives, k):
    return 1 if any(c in positives for c in ranked[:k]) else 0


def recall_at_k(ranked, positives, k):
    return sum(1 for c in ranked[:k] if c in positives) / len(positives)


def rank_crossencoder(labels, chunk_ids):
    """Ranked lists from the cached CE scores."""
    ce = pd.read_csv(CE_SCORES)
    ce["label"] = ce["label"].str.strip()
    valid = set(chunk_ids)
    ce = ce[ce.chunk_id.isin(valid)]

    out = {}
    for lab in labels:
        sub = ce[ce.label == lab]
        if sub.empty:
            continue
        out[lab] = sub.sort_values("ce_score", ascending=False).chunk_id.tolist()
    return out


def rank_bm25(queries, chunks):
    """BM25 using retrieval.py's own tokenizer -- r'\\w+' on lowercased text.
    Naive .split() does not reproduce the published ranking."""
    from rank_bm25 import BM25Okapi
    from retrieval import tokenize

    corpus = [tokenize(t) for t in chunks.text_chunk]
    bm25 = BM25Okapi(corpus)
    ids = chunks.chunk_id.tolist()

    out = {}
    for lab, q in queries.items():
        scores = bm25.get_scores(tokenize(q))
        order = np.argsort(-scores)
        out[lab] = [ids[i] for i in order]
    return out


def rank_biencoder(queries, chunks):
    """Cosine ranking from cached chunk embeddings; only the query is encoded."""
    from sentence_transformers import SentenceTransformer

    emb = pd.read_parquet(EMBEDDED)
    emb = emb[emb.mp_index != BEEMSTER_MP]
    emb = emb.set_index("chunk_id").loc[chunks.chunk_id].reset_index()

    M = np.vstack(emb.embedding.values).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    ids = emb.chunk_id.tolist()

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    labs = list(queries)
    Q = model.encode([queries[l] for l in labs], normalize_embeddings=True)

    sims = Q @ M.T
    return {lab: [ids[i] for i in np.argsort(-row)] for lab, row in zip(labs, sims)}


def rank_dense_rerank(bienc, ce_ranked, labels):
    """Bi-encoder top-50 reordered by CE score, then the bi-encoder tail."""
    ce_pos = {
        lab: {cid: i for i, cid in enumerate(ranked)} for lab, ranked in ce_ranked.items()
    }
    out = {}
    for lab in labels:
        if lab not in bienc or lab not in ce_pos:
            continue
        ranked = bienc[lab]
        head, tail = ranked[:TOP_K_RERANK], ranked[TOP_K_RERANK:]
        pos = ce_pos[lab]
        head = sorted(head, key=lambda c: pos.get(c, 10**9))
        out[lab] = head + tail
    return out


def evaluate(system, ranked, gt):
    rows = []
    for lab, r in ranked.items():
        if lab not in gt or not gt[lab]:
            continue
        row = {"system": system, "label": lab, "n_positives": len(gt[lab])}
        for k in K_VALUES:
            row[f"Hit@{k}"] = hit_at_k(r, gt[lab], k)
            row[f"R@{k}"] = recall_at_k(r, gt[lab], k)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    labels = canonical_labels()
    queries = load_queries(labels)
    chunks = load_chunks()
    gt = load_gt(labels)

    print(f"labels {len(labels)} | chunks {len(chunks)} | labels with GT {len(gt)}")
    print(f"K values: {K_VALUES}")

    print("\n[1/4] cross-encoder (cached scores)")
    ce = rank_crossencoder(labels, set(chunks.chunk_id))
    print("[2/4] BM25")
    bm = rank_bm25(queries, chunks)
    print("[3/4] bi-encoder (cached embeddings)")
    bi = rank_biencoder(queries, chunks)
    print("[4/4] dense + rerank")
    dr = rank_dense_rerank(bi, ce, labels)

    per_label = pd.concat(
        [
            evaluate("BM25", bm, gt),
            evaluate("Bi-encoder", bi, gt),
            evaluate("Dense + Rerank", dr, gt),
            evaluate("Cross-encoder", ce, gt),
        ],
        ignore_index=True,
    )

    order = ["BM25", "Bi-encoder", "Dense + Rerank", "Cross-encoder"]
    cols = [f"Hit@{k}" for k in K_VALUES] + [f"R@{k}" for k in K_VALUES]
    macro = per_label.groupby("system")[cols].mean().loc[order]

    print("\n" + "=" * 78)
    print("HIT@K -- macro-averaged over 24 labels")
    print("=" * 78)
    print(macro[[f"Hit@{k}" for k in K_VALUES]].round(3).to_string())
    print()
    print("RECALL@K")
    print(macro[[f"R@{k}" for k in K_VALUES]].round(3).to_string())

    # Sanity: K in {1,3} must reproduce the published table.
    print("\n" + "=" * 78)
    print("SANITY CHECK vs published table")
    print("=" * 78)
    pub = pd.read_csv(CMP).groupby("system")[["Hit@1", "Hit@3"]].mean()
    ok = True
    for s in order:
        for k in ["Hit@1", "Hit@3"]:
            a, b = macro.loc[s, k], pub.loc[s, k]
            good = abs(a - b) < 1e-9
            ok &= good
            print(f"  {s:16s} {k}: recomputed {a:.3f} vs published {b:.3f}"
                  f"{'' if good else '   <-- MISMATCH'}")
    print(f"\n{'PASS' if ok else 'FAIL'}: ranked lists reproduce the published results.")

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    ce_row = macro.loc["Cross-encoder"]
    print(f"Cross-encoder Hit@3   = {ce_row['Hit@3']:.3f}  (published)")
    print(f"Cross-encoder Hit@10  = {ce_row['Hit@10']:.3f}")
    print(f"Cross-encoder Hit@20  = {ce_row['Hit@20']:.3f}")
    print(f"Cross-encoder Hit@100 = {ce_row['Hit@100']:.3f}")
    print()
    print("An expert willing to scan 20 candidates per concept finds a")
    print(f"true positive for {ce_row['Hit@20']:.0%} of concepts, against "
          f"{ce_row['Hit@3']:.0%} at K=3.")
    print("This is the annotation-prioritisation argument the reviewer asked for,")
    print("and it is far more favourable to the system than the published K in {1,3}.")

    per_label.round(4).to_csv(OUT / "hit_at_k_per_label.csv", index=False)
    macro.round(4).to_csv(OUT / "hit_at_k_macro.csv")
    print(f"\nwritten -> {OUT / 'hit_at_k_macro.csv'}")
    print(f"           {OUT / 'hit_at_k_per_label.csv'}")


if __name__ == "__main__":
    sys.exit(main())
