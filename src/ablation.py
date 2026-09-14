"""
ablation.py
-----------
Chunk-size ablation across 6 log-spaced configurations.
Runs all 4 retrieval systems (BM25, Bi-encoder, Dense+Rerank, Cross-encoder)
at each chunk size and saves MRR/Hit@1/Hit@3/MAP per system per size.

Uses already-extracted Docling text (reconstructed from chunks_docling.csv)
to avoid re-running Docling. Re-chunks, remaps GT, runs retrieval.

Usage:
    python src/ablation.py
    python src/ablation.py --smoke-test   # 2 sizes, 2 labels, fast
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunk_pdfs import chunk_text as chunk_text_snap


def chunk_text_hard(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Hard character-level sliding window — no sentence snapping.
    Needed for ablation so sentences always appear fully within a chunk or merged pair."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - chunk_overlap
    return chunks
from remap_gt import load_sentences, match_sentences
from retrieval import (
    build_bm25,
    compute_metrics,
    retrieve_bm25,
    retrieve_biencoder,
    rerank,
    retrieve_crossencoder_full,
)

PDF_DIR      = ROOT / "Datasets" / "MPs" / "NL"
CDEF_FILE    = ROOT / "Datasets" / "Datasets" / "Heritage concepts" / "Definitions" / "New" / "Heritage Labels & definition.xlsx"
OUT_DIR      = ROOT / "results" / "ablation_4sys"
BIENC_MODEL  = "sentence-transformers/all-mpnet-base-v2"
CE_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EXCLUDE_MP   = {9}
TOP_K_RERANK = 50

CHUNK_SIZES = [200, 434, 638, 940, 1384, 3000]
OVERLAP_FRAC = 0.2  # overlap = 20% of chunk_size


def load_labels(smoke_test=False):
    df = pd.read_excel(CDEF_FILE, usecols=["Heritage Concept", "Definition"])
    df = df[df["Definition"].notna()].reset_index(drop=True)
    df["query"] = df["Heritage Concept"] + ": " + df["Definition"]
    if smoke_test:
        df = df.head(2)
    return df["Heritage Concept"].tolist(), df["query"].tolist()


MP_TEXTS_CACHE = OUT_DIR / "mp_texts"


def extract_mp_texts() -> dict[int, str]:
    """Extract per-MP clean text via Docling (cached to disk after first run)."""
    from chunk_pdfs import build_converter, extract_text, clean_text

    MP_TEXTS_CACHE.mkdir(parents=True, exist_ok=True)
    mp_texts = {}
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    needs = [p for p in pdf_paths
             if not (MP_TEXTS_CACHE / f"{int(p.stem.split('_')[0]):02d}.txt").exists()
             and int(p.stem.split('_')[0]) not in EXCLUDE_MP]

    if needs:
        print(f"Extracting {len(needs)} MPs via Docling...")
        converter = build_converter()
        for pdf_path in needs:
            mp_idx = int(pdf_path.stem.split("_")[0])
            text = clean_text(extract_text(pdf_path, converter))
            (MP_TEXTS_CACHE / f"{mp_idx:02d}.txt").write_text(text, encoding="utf-8")
            print(f"  MP{mp_idx:02d}: {len(text):,} chars")

    for txt_path in sorted(MP_TEXTS_CACHE.glob("*.txt")):
        mp_idx = int(txt_path.stem)
        if mp_idx in EXCLUDE_MP:
            continue
        mp_texts[mp_idx] = txt_path.read_text(encoding="utf-8")

    print(f"Loaded {len(mp_texts)} MP texts from cache.")
    return mp_texts


def build_chunks_df(mp_texts: dict, chunk_size: int, overlap: int) -> pd.DataFrame:
    rows = []
    chunk_id = 0
    for mp_idx in sorted(mp_texts.keys()):
        chunks = chunk_text_hard(mp_texts[mp_idx], chunk_size=chunk_size, chunk_overlap=overlap)
        for ch in chunks:
            rows.append({"chunk_id": chunk_id, "mp_index": mp_idx, "text_chunk": ch})
            chunk_id += 1
    return pd.DataFrame(rows)


def build_gt(chunks_df: pd.DataFrame, sentences_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    psi_df = match_sentences(sentences_df, chunks_df)
    psi_df = psi_df[psi_df["Match Type"] != "no match"]
    psi_df = psi_df[psi_df["Chunk ID"].notna()]
    psi_df["Chunk ID"] = psi_df["Chunk ID"].astype(int)
    psi_df = psi_df[~psi_df["MP Index"].isin(EXCLUDE_MP)]
    chunk_ids_valid = set(chunks_df["chunk_id"].tolist())
    psi_df = psi_df[psi_df["Chunk ID"].isin(chunk_ids_valid)]
    gt = {}
    for label, grp in psi_df.groupby("Label"):
        gt[label] = set(grp["Chunk ID"].unique())
    return gt, psi_df


def eval_system(system_name, ranked_lists, label_names, gt):
    rows = []
    for label, ranked in zip(label_names, ranked_lists):
        if label not in gt:
            continue
        m = compute_metrics(ranked, gt[label])
        if m is None:
            continue
        m["system"] = system_name
        m["label"] = label
        rows.append(m)
    return pd.DataFrame(rows)


def run_size(chunk_size, overlap, mp_texts, label_names, label_queries, sentences_df,
             bienc_model, ce_model, smoke_test=False):
    print(f"\n{'='*60}")
    print(f"chunk_size={chunk_size}, overlap={overlap}")

    chunks_df = build_chunks_df(mp_texts, chunk_size, overlap)
    chunk_ids = chunks_df["chunk_id"].tolist()
    print(f"  {len(chunks_df)} chunks across {chunks_df['mp_index'].nunique()} MPs")

    if smoke_test:
        chunks_df = chunks_df.head(200)
        chunk_ids = chunks_df["chunk_id"].tolist()

    print("  Remapping GT...")
    gt, _ = build_gt(chunks_df, sentences_df)
    print(f"  GT: {len(gt)} labels, {sum(len(v) for v in gt.values())} positives")

    # BM25
    bm25 = build_bm25(chunks_df)
    bm25_ranked = [retrieve_bm25(bm25, q, chunk_ids) for q in label_queries]

    # Bi-encoder — encode chunks directly (no embedding column needed)
    print("  [Bi-encoder] encoding chunks...")
    chunk_embs = bienc_model.encode(
        chunks_df["text_chunk"].tolist(),
        convert_to_numpy=True, normalize_embeddings=True,
        batch_size=256, show_progress_bar=False,
    )
    print("  [Bi-encoder] encoding labels...")
    label_embs = bienc_model.encode(
        label_queries, convert_to_numpy=True, normalize_embeddings=True,
    )
    bienc_ranked = retrieve_biencoder(label_embs, chunk_embs, chunk_ids)

    # Dense + Rerank
    reranked_lists = []
    for query, bienc_top in zip(label_queries, bienc_ranked):
        top50 = bienc_top[:TOP_K_RERANK]
        reranked_top = rerank(ce_model, query, top50, chunks_df)
        rest = bienc_top[TOP_K_RERANK:]
        reranked_lists.append(reranked_top + rest)

    # Cross-encoder full scan
    ce_full_lists = []
    for i, query in enumerate(label_queries):
        ranked = retrieve_crossencoder_full(ce_model, query, chunk_ids, chunks_df)
        ce_full_lists.append(ranked)
        if (i + 1) % 5 == 0:
            print(f"  [CE] {i+1}/{len(label_queries)} done")

    df_bm25  = eval_system("BM25",           bm25_ranked,    label_names, gt)
    df_bi    = eval_system("Bi-encoder",      bienc_ranked,   label_names, gt)
    df_rnk   = eval_system("Dense+Rerank",    reranked_lists, label_names, gt)
    df_ce    = eval_system("Cross-encoder",   ce_full_lists,  label_names, gt)

    df_all = pd.concat([df_bm25, df_bi, df_rnk, df_ce], ignore_index=True)

    if df_all.empty or "system" not in df_all.columns:
        print("  WARNING: no GT matches at this chunk size, skipping.")
        return pd.DataFrame()

    summary_rows = []
    for system, grp in df_all.groupby("system"):
        summary_rows.append({
            "chunk_size": chunk_size,
            "overlap":    overlap,
            "n_chunks":   len(chunks_df),
            "system":     system,
            "MRR":        grp["MRR"].mean(),
            "Hit@1":      grp["Hit@1"].mean(),
            "Hit@3":      grp["Hit@3"].mean(),
            "MAP":        grp["AP"].mean(),
        })

    return pd.DataFrame(summary_rows)


def main(smoke_test=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    label_names, label_queries = load_labels(smoke_test)
    sentences_df = load_sentences()

    print("Extracting/loading MP texts...")
    mp_texts = extract_mp_texts()

    print(f"Loading models...")
    bienc_model = SentenceTransformer(BIENC_MODEL)
    ce_model    = CrossEncoder(CE_MODEL)
    print("Models loaded.")

    sizes = CHUNK_SIZES[:2] if smoke_test else CHUNK_SIZES
    all_rows = []

    for size in sizes:
        overlap = int(size * OVERLAP_FRAC)
        df = run_size(size, overlap, mp_texts, label_names, label_queries, sentences_df,
                      bienc_model, ce_model, smoke_test=smoke_test)
        all_rows.append(df)
        # save incrementally
        pd.concat(all_rows).to_csv(OUT_DIR / "ablation_4sys.csv", index=False)
        print(f"  Saved intermediate results.")

    final = pd.concat(all_rows, ignore_index=True)
    final.to_csv(OUT_DIR / "ablation_4sys.csv", index=False)

    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    pivot = final.pivot_table(index=["chunk_size", "system"], values="MRR")
    print(pivot.to_string())
    print(f"\nSaved → {OUT_DIR / 'ablation_4sys.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
