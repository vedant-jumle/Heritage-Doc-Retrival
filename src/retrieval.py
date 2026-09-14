"""
Retrieval system comparison: BM25 vs Bi-encoder vs Bi-encoder+Reranker vs Cross-encoder (standalone).
Task: given a heritage concept query (label + definition), retrieve relevant MP text chunks.

Usage:
    python src/retrieval.py              # full run (user does this)
    python src/retrieval.py --smoke-test # quick validation on 2 labels + 100 chunks
"""

import sys
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
PARQUET_FILE   = ROOT / "Capstone-Applied-AI-project_12/MP_Embeddings/WG_MPs_mpnet_Embeddings.parquet"
DEFAULT_PARQUET = PARQUET_FILE
CDEF_FILE      = ROOT / "Datasets/Datasets/Heritage concepts/Definitions/New/Heritage Labels & definition.xlsx"
PSI_FILE       = ROOT / "Datasets/Datasets/Heritage concepts/Sentence Matching/New/Reference sentences Matching final+Label.xlsx"
RESULTS_DIR    = ROOT / "results"

BIENCODER_MODEL   = "sentence-transformers/all-mpnet-base-v2"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EXCLUDE_MP        = {9}   # Beemster — unreliable translation
TOP_K_RERANK      = 50


# ── Data loading ───────────────────────────────────────────────────────────────
def load_chunks(smoke_test=False, parquet_path=None):
    path = Path(parquet_path) if parquet_path else DEFAULT_PARQUET
    df = pd.read_parquet(path)
    if "chunk_id" not in df.columns:
        df = df.reset_index(drop=True)
        df.index.name = "chunk_id"
        df = df.reset_index()  # chunk_id as column
    else:
        df = df.reset_index(drop=True)
    df = df[~df["mp_index"].isin(EXCLUDE_MP)].reset_index(drop=True)
    if smoke_test:
        df = df.head(100)
    print(f"[data] {len(df)} chunks loaded (MP{list(EXCLUDE_MP)} excluded)")
    return df


def load_labels(smoke_test=False):
    df = pd.read_excel(CDEF_FILE, usecols=["Heritage Concept", "Definition"])
    df = df[df["Definition"].notna()].reset_index(drop=True)
    df["query"] = df["Heritage Concept"] + ": " + df["Definition"]
    if smoke_test:
        df = df.head(2)
    label_names  = df["Heritage Concept"].tolist()
    label_queries = df["query"].tolist()
    print(f"[data] {len(label_names)} labels loaded")
    return label_names, label_queries


def load_ground_truth(chunk_ids_valid, smoke_test=False, psi_path=None):
    path = Path(psi_path) if psi_path else PSI_FILE
    df = pd.read_excel(path)
    df = df[df["Chunk ID"].notna()]
    df = df[df["Match Type"] != "no match"]
    df["Chunk ID"] = df["Chunk ID"].astype(int)
    df["MP Index"] = df["MP Index"].astype(int)
    df = df[~df["MP Index"].isin(EXCLUDE_MP)]
    df = df[df["Chunk ID"].isin(chunk_ids_valid)]
    # ground truth: label → set of positive chunk_ids
    gt = {}
    for label, grp in df.groupby("Label"):
        gt[label] = set(grp["Chunk ID"].unique())
    print(f"[data] ground truth: {len(gt)} labels, {sum(len(v) for v in gt.values())} total positives")
    return gt, df


# ── BM25 ───────────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"\w+", text.lower())


def build_bm25(chunks_df):
    corpus = [tokenize(t) for t in chunks_df["text_chunk"]]
    return BM25Okapi(corpus)


def retrieve_bm25(bm25, query, chunk_ids):
    scores = bm25.get_scores(tokenize(query))
    ranked_idx = np.argsort(-scores)
    return [chunk_ids[i] for i in ranked_idx]


# ── Bi-encoder ─────────────────────────────────────────────────────────────────
def build_biencoder(chunks_df, label_queries):
    chunk_embs = np.vstack(chunks_df["embedding"].values).astype(np.float32)
    norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    chunk_embs = chunk_embs / np.clip(norms, 1e-10, None)

    print(f"[bienc] embedding {len(label_queries)} labels...")
    model = SentenceTransformer(BIENCODER_MODEL)
    label_embs = model.encode(label_queries, convert_to_numpy=True, normalize_embeddings=True)
    print(f"[bienc] done. chunk_embs={chunk_embs.shape}, label_embs={label_embs.shape}")
    return label_embs, chunk_embs, model


def retrieve_biencoder(label_embs, chunk_embs, chunk_ids):
    scores = label_embs @ chunk_embs.T  # (n_labels, n_chunks)
    ranked_indices = np.argsort(-scores, axis=1)
    return [[chunk_ids[i] for i in row] for row in ranked_indices]


# ── Cross-encoder reranker ─────────────────────────────────────────────────────
def rerank(crossencoder, query, top_chunk_ids, chunks_df):
    chunk_id_to_text = dict(zip(chunks_df["chunk_id"], chunks_df["text_chunk"]))
    pairs = [(query, chunk_id_to_text[cid]) for cid in top_chunk_ids if cid in chunk_id_to_text]
    if not pairs:
        return top_chunk_ids
    ce_scores = crossencoder.predict(pairs)
    reranked_idx = np.argsort(-ce_scores)
    return [top_chunk_ids[i] for i in reranked_idx]


# ── Cross-encoder standalone retriever ────────────────────────────────────────
def retrieve_crossencoder_full(crossencoder, query, chunk_ids, chunks_df):
    """Score ALL chunks with cross-encoder — no bi-encoder pre-filter.
    Returns (ranked_chunk_ids, scores_dict) where scores_dict maps chunk_id -> ce_score."""
    chunk_id_to_text = dict(zip(chunks_df["chunk_id"], chunks_df["text_chunk"]))
    valid_ids = [cid for cid in chunk_ids if cid in chunk_id_to_text]
    pairs = [(query, chunk_id_to_text[cid]) for cid in valid_ids]
    ce_scores = crossencoder.predict(pairs, batch_size=128, show_progress_bar=False)
    ranked_idx = np.argsort(-ce_scores)
    ranked_ids = [valid_ids[i] for i in ranked_idx]
    scores_dict = {valid_ids[i]: float(ce_scores[i]) for i in range(len(valid_ids))}
    return ranked_ids, scores_dict


# ── Eval metrics ───────────────────────────────────────────────────────────────
def compute_metrics(ranked_chunk_ids, positives, k_values=(5, 10, 20)):
    """Given a ranked list and set of positives, compute all metrics."""
    n_pos = len(positives)
    if n_pos == 0:
        return None

    ranked = list(ranked_chunk_ids)
    n = len(ranked)

    metrics = {"n_positives": n_pos}

    for k in k_values:
        top_k = ranked[:k]
        hits = sum(1 for cid in top_k if cid in positives)
        metrics[f"P@{k}"] = hits / k
        metrics[f"R@{k}"] = hits / n_pos

    # AP
    ap, n_hits = 0.0, 0
    for rank, cid in enumerate(ranked, 1):
        if cid in positives:
            n_hits += 1
            ap += n_hits / rank
    metrics["AP"] = ap / n_pos

    # MRR — rank of first positive
    mrr = 0.0
    for rank, cid in enumerate(ranked, 1):
        if cid in positives:
            mrr = 1.0 / rank
            break
    metrics["MRR"] = mrr

    # Hit@1, Hit@3
    metrics["Hit@1"] = 1 if ranked[0] in positives else 0
    metrics["Hit@3"] = 1 if any(cid in positives for cid in ranked[:3]) else 0

    return metrics


def eval_system(system_name, ranked_lists, label_names, gt, chunks_df):
    """ranked_lists: list of ranked chunk_id lists, one per label (same order as label_names)."""
    rows = []
    for label, ranked in zip(label_names, ranked_lists):
        if label not in gt:
            continue
        m = compute_metrics(ranked, gt[label])
        if m is None:
            continue
        m["system"] = system_name
        m["label"]  = label
        rows.append(m)
    return pd.DataFrame(rows)


# ── Per-document breakdown ─────────────────────────────────────────────────────
def per_doc_map(system_name, ranked_lists, label_names, psi_df, chunks_df):
    chunk_to_mp = dict(zip(chunks_df["chunk_id"], chunks_df["mp_index"]))
    rows = []
    for mp_idx in sorted(psi_df["MP Index"].unique()):
        mp_psi = psi_df[psi_df["MP Index"] == mp_idx]
        gt_mp  = {}
        for label, grp in mp_psi.groupby("Label"):
            gt_mp[label] = set(grp["Chunk ID"].unique())
        aps = []
        for label, ranked in zip(label_names, ranked_lists):
            if label not in gt_mp:
                continue
            # filter ranked list to chunks from this MP only
            ranked_mp = [cid for cid in ranked if chunk_to_mp.get(cid) == mp_idx]
            m = compute_metrics(ranked_mp, gt_mp[label])
            if m:
                aps.append(m["AP"])
        if aps:
            rows.append({"system": system_name, "mp_index": mp_idx, "MAP": np.mean(aps), "n_labels": len(aps)})
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────
def main(smoke_test=False, args=None):
    print(f"{'[SMOKE TEST] ' if smoke_test else ''}Starting retrieval pipeline...")

    parquet_path = args.parquet if args and args.parquet else None
    psi_path     = args.psi     if args and args.psi     else None

    chunks_df = load_chunks(smoke_test, parquet_path=parquet_path)
    chunk_ids = chunks_df["chunk_id"].tolist()
    label_names, label_queries = load_labels(smoke_test)
    gt, psi_df = load_ground_truth(set(chunk_ids), smoke_test, psi_path=psi_path)

    # ── BM25 ──
    print("\n[BM25] building index...")
    bm25 = build_bm25(chunks_df)
    bm25_ranked = [retrieve_bm25(bm25, q, chunk_ids) for q in label_queries]
    print("[BM25] done")

    # ── Bi-encoder ──
    print("\n[Bi-encoder] building...")
    label_embs, chunk_embs, bienc_model = build_biencoder(chunks_df, label_queries)
    bienc_ranked = retrieve_biencoder(label_embs, chunk_embs, chunk_ids)
    print("[Bi-encoder] done")

    # ── Cross-encoder reranker (bi-encoder top-50 → cross-encoder) ──
    print(f"\n[Reranker] loading {CROSSENCODER_MODEL}...")
    crossencoder = CrossEncoder(CROSSENCODER_MODEL)
    reranked_lists = []
    for i, (query, bienc_top) in enumerate(zip(label_queries, bienc_ranked)):
        top50 = bienc_top[:TOP_K_RERANK]
        reranked_top = rerank(crossencoder, query, top50, chunks_df)
        rest = bienc_top[TOP_K_RERANK:]
        reranked_lists.append(reranked_top + rest)
        if (i + 1) % 5 == 0:
            print(f"[Reranker] {i+1}/{len(label_queries)} labels reranked")
    print("[Reranker] done")

    # ── Cross-encoder standalone (all chunks) ──
    print(f"\n[Cross-encoder] scoring all {len(chunk_ids)} chunks per query (slow)...")
    ce_full_lists = []
    ce_scores_rows = []  # for histogram: label, chunk_id, score, in_gt
    for i, (query, label) in enumerate(zip(label_queries, label_names)):
        ranked, scores_dict = retrieve_crossencoder_full(crossencoder, query, chunk_ids, chunks_df)
        ce_full_lists.append(ranked)
        gt_set = gt.get(label, set())
        for cid, score in scores_dict.items():
            ce_scores_rows.append({"label": label, "chunk_id": cid, "ce_score": score,
                                   "in_gt": cid in gt_set})
        if (i + 1) % 5 == 0:
            print(f"[Cross-encoder] {i+1}/{len(label_queries)} queries done")
    print("[Cross-encoder] done")
    df_ce_scores = pd.DataFrame(ce_scores_rows)

    # ── Eval ──
    print("\n[Eval] computing metrics...")
    df_bm25   = eval_system("BM25",           bm25_ranked,    label_names, gt, chunks_df)
    df_bienc  = eval_system("Bi-encoder",     bienc_ranked,   label_names, gt, chunks_df)
    df_rerank = eval_system("Reranker",       reranked_lists, label_names, gt, chunks_df)
    df_ce     = eval_system("Cross-encoder",  ce_full_lists,  label_names, gt, chunks_df)

    df_all = pd.concat([df_bm25, df_bienc, df_rerank, df_ce], ignore_index=True)

    # Summary
    summary_rows = []
    for system, grp in df_all.groupby("system"):
        summary_rows.append({
            "system":     system,
            "MRR":        grp["MRR"].mean(),
            "Hit@1":      grp["Hit@1"].mean(),
            "Hit@3":      grp["Hit@3"].mean(),
            "P@5":        grp["P@5"].mean(),
            "P@10":       grp["P@10"].mean(),
            "P@20":       grp["P@20"].mean(),
            "R@5":        grp["R@5"].mean(),
            "R@10":       grp["R@10"].mean(),
            "R@20":       grp["R@20"].mean(),
            "MAP":        grp["AP"].mean(),
        })
    df_summary = pd.DataFrame(summary_rows)

    # Per-doc breakdown
    df_perdoc = pd.concat([
        per_doc_map("BM25",          bm25_ranked,    label_names, psi_df, chunks_df),
        per_doc_map("Bi-encoder",    bienc_ranked,   label_names, psi_df, chunks_df),
        per_doc_map("Reranker",      reranked_lists, label_names, psi_df, chunks_df),
        per_doc_map("Cross-encoder", ce_full_lists,  label_names, psi_df, chunks_df),
    ], ignore_index=True)

    # ── Print summary ──
    print("\n" + "="*60)
    print("SUMMARY (macro averages)")
    print("="*60)
    print(df_summary.to_string(index=False))

    # ── Write results ──
    RESULTS_DIR.mkdir(exist_ok=True)
    out = Path(args.outdir) if args and getattr(args, "outdir", None) else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke_test else ""
    df_all.to_csv(out / f"retrieval_comparison{suffix}.csv", index=False)
    df_summary.to_csv(out / f"retrieval_summary{suffix}.csv", index=False)
    df_perdoc.to_csv(out / f"retrieval_per_doc{suffix}.csv", index=False)
    df_ce_scores.to_csv(out / f"ce_scores{suffix}.csv", index=False)
    print(f"\n[done] written to {out}/ (suffix='{suffix}')")

    if smoke_test:
        assert len(df_summary) == 4, f"Expected 4 systems, got {len(df_summary)}"
        assert "MAP" in df_summary.columns
        assert len(df_all) > 0
        print("[smoke test] PASSED ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--parquet", default=None, help="Override path to chunks parquet file")
    parser.add_argument("--psi",     default=None, help="Override path to PSI ground-truth Excel file")
    parser.add_argument("--outdir",  default=None, help="Output directory for result CSVs (default: results/)")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test, args=args)
