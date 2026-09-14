"""
Evaluation pipeline for climate-adaptive heritage retrieval system.
Computes MRR, Hit@K (Part 1) and NDCG@K (Part 2) metrics.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
PSI_FILE       = ROOT / "Datasets/Datasets/Heritage concepts/Sentence Matching/New/Reference sentences Matching final+Label.xlsx"
CDEF_FILE      = ROOT / "Datasets/Datasets/Heritage concepts/Definitions/New/Heritage Labels & definition.xlsx"
PHI_FILE       = ROOT / "Datasets/Datasets/Heritage concepts/Definitions/New/Heritage Labels  combined - Fuzzy + Cosine.xlsx"
EMBEDDINGS_FILE= ROOT / "Capstone-Applied-AI-project_12/MP_Embeddings/WG_MPs_mpnet_Embeddings.parquet"
SIM_FILE       = ROOT / "Capstone-Applied-AI-project_12/Similarity_Results/normalized_similarity_results.csv"
LABEL_EMB_CACHE= ROOT / "results/label_embeddings.npy"
LABEL_NAMES_CACHE = ROOT / "results/label_names.npy"
OUTPUT_FILE    = ROOT / "results"  # output directory for CSVs

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ── Step 1: Load & clean alignment (psi) ──────────────────────────────────────
def load_psi():
    df = pd.read_excel(PSI_FILE)
    df = df[df["Chunk ID"].notna()]
    df = df[df["Match Type"] != "no match"]
    df["Chunk ID"] = df["Chunk ID"].astype(int)
    df["MP Index"] = df["MP Index"].astype(int)
    df = df[["Label", "Sentence", "Chunk ID", "MP Index"]].reset_index(drop=True)
    print(f"[psi] {len(df)} valid sentence-chunk pairs, {df['Sentence'].nunique()} unique sentences")
    return df


# ── Step 2: Load heritage label definitions (C_def) ───────────────────────────
def load_cdef():
    df = pd.read_excel(CDEF_FILE, usecols=["Heritage Concept", "Definition"])
    df = df[df["Definition"].notna()].reset_index(drop=True)
    df["query"] = df["Heritage Concept"] + ": " + df["Definition"]
    label_to_query = dict(zip(df["Heritage Concept"], df["query"]))
    print(f"[C_def] {len(label_to_query)} labels with definitions")
    return label_to_query


# ── Step 3: Embed heritage labels ─────────────────────────────────────────────
def embed_labels(label_to_query):
    if LABEL_EMB_CACHE.exists() and LABEL_NAMES_CACHE.exists():
        label_names = list(np.load(LABEL_NAMES_CACHE, allow_pickle=True))
        label_embs  = np.load(LABEL_EMB_CACHE)
        print(f"[embed] loaded cached label embeddings ({len(label_names)} labels)")
        return label_names, label_embs

    print("[embed] encoding label definitions with MPNet...")
    model = SentenceTransformer(MODEL_NAME)
    label_names = list(label_to_query.keys())
    queries     = [label_to_query[l] for l in label_names]
    label_embs  = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    np.save(LABEL_EMB_CACHE, label_embs)
    np.save(LABEL_NAMES_CACHE, np.array(label_names, dtype=object))
    print(f"[embed] cached {len(label_names)} label embeddings")
    return label_names, label_embs


# ── Step 4: Load chunk embeddings ─────────────────────────────────────────────
def load_chunk_embeddings():
    df = pd.read_parquet(EMBEDDINGS_FILE)
    embs = np.vstack(df["embedding"].values).astype(np.float32)
    # L2-normalize for cosine via dot product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-10, None)
    print(f"[chunks] {embs.shape[0]} chunks, dim={embs.shape[1]}")
    return embs  # chunk_id == row index


# ── Step 5: Eval Part 1 — MRR and Hit@K ───────────────────────────────────────
def eval_mrr_hitk(psi, label_names, label_embs, chunk_embs):
    label_idx = {l: i for i, l in enumerate(label_names)}
    n_labels  = len(label_names)

    # Group by unique sentence
    grouped = psi.groupby("Sentence")

    per_sentence = []  # (label, rank)

    for sentence, grp in grouped:
        gt_label = grp["Label"].iloc[0]
        if gt_label not in label_idx:
            continue  # label has no definition, skip

        chunk_ids = grp["Chunk ID"].tolist()

        # Get embeddings for all chunks of this sentence
        chunk_vecs = chunk_embs[chunk_ids]  # (n_chunks, 768)

        # Cosine sim: dot product (already normalized)
        scores = chunk_vecs @ label_embs.T  # (n_chunks, n_labels)

        # Best-chunk: max score per label across chunks
        best_scores = scores.max(axis=0)  # (n_labels,)

        # Rank labels descending
        ranked = np.argsort(-best_scores)
        rank = int(np.where(ranked == label_idx[gt_label])[0][0]) + 1  # 1-indexed

        per_sentence.append({"sentence": sentence, "label": gt_label, "rank": rank})

    df_ps = pd.DataFrame(per_sentence)
    df_ps["reciprocal_rank"] = 1.0 / df_ps["rank"]
    df_ps["hit1"] = (df_ps["rank"] == 1).astype(int)
    df_ps["hit3"] = (df_ps["rank"] <= 3).astype(int)

    # Micro averages
    micro = {
        "MRR":   df_ps["reciprocal_rank"].mean(),
        "Hit@1": df_ps["hit1"].mean(),
        "Hit@3": df_ps["hit3"].mean(),
        "n_sentences": len(df_ps),
    }

    # Macro averages (per label)
    macro_rows = []
    for label, grp in df_ps.groupby("label"):
        macro_rows.append({
            "label": label,
            "MRR":   grp["reciprocal_rank"].mean(),
            "Hit@1": grp["hit1"].mean(),
            "Hit@3": grp["hit3"].mean(),
            "n_sentences": len(grp),
        })
    df_macro = pd.DataFrame(macro_rows).sort_values("label")
    macro_avg = {
        "MRR":   df_macro["MRR"].mean(),
        "Hit@1": df_macro["Hit@1"].mean(),
        "Hit@3": df_macro["Hit@3"].mean(),
    }

    print(f"\n[Part 1] Micro  — MRR={micro['MRR']:.4f}  Hit@1={micro['Hit@1']:.4f}  Hit@3={micro['Hit@3']:.4f}  (n={micro['n_sentences']})")
    print(f"[Part 1] Macro  — MRR={macro_avg['MRR']:.4f}  Hit@1={macro_avg['Hit@1']:.4f}  Hit@3={macro_avg['Hit@3']:.4f}")

    return df_ps, df_macro, micro, macro_avg


# ── Step 6: Load IPCC mapping (phi) ───────────────────────────────────────────
def load_phi():
    df = pd.read_excel(PHI_FILE)

    def parse_concepts(val):
        if pd.isna(val) or str(val).strip() == "":
            return set()
        return {c.strip().lower() for c in str(val).split(",")}

    phi = {}
    for _, row in df.iterrows():
        label = row["heritage_concept"]
        strict = (
            parse_concepts(row.get("Fuzzy exact_match (100)")) |
            parse_concepts(row.get("Fuzzy strong_match (90+)")) |
            parse_concepts(row.get("Cosine exact_match (0.99+)")) |
            parse_concepts(row.get("Cosine strong_match (0.90+)"))
        )
        standard = strict | (
            parse_concepts(row.get("Fuzzy moderate_match (70+)")) |
            parse_concepts(row.get("Cosine moderate_match (0.70+)"))
        )
        phi[label] = {"strict": strict, "standard": standard}

    print(f"[phi] loaded IPCC mappings for {len(phi)} labels")
    return phi


# ── Step 7: Eval Part 2 — NDCG@K ─────────────────────────────────────────────
def eval_ndcg(psi, phi, k_values=(5, 10)):
    print("[Part 2] loading normalized similarity results (this may take a moment)...")
    sim_df = pd.read_csv(SIM_FILE)
    sim_df = sim_df[sim_df["Model"] == "MPNet"].copy()
    sim_df["Concept_lower"] = sim_df["Concept"].str.strip().str.lower()

    def dcg(relevances, k):
        relevances = np.array(relevances[:k], dtype=float)
        if len(relevances) == 0:
            return 0.0
        positions = np.arange(1, len(relevances) + 1)
        return np.sum(relevances / np.log2(positions + 1))

    def ndcg(relevances, k):
        actual  = dcg(sorted(relevances, reverse=True), k)  # ideal
        if actual == 0:
            return 0.0
        achieved = dcg(relevances, k)
        return achieved / actual

    ndcg_rows = []

    for label, grp in psi.groupby("Label"):
        if label not in phi:
            continue

        concept_rel = {}
        for c in phi[label]["strict"]:
            concept_rel[c] = 2
        for c in phi[label]["standard"] - phi[label]["strict"]:
            concept_rel[c] = 1

        if not concept_rel:
            continue

        chunk_ids = grp["Chunk ID"].unique().tolist()
        chunk_sim = sim_df[sim_df["Chunk ID"].isin(chunk_ids)]

        row = {"label": label, "n_chunks": len(chunk_ids), "n_ipcc_mapped": len(concept_rel)}

        for k in k_values:
            ndcg_scores = []
            for chunk_id, csim in chunk_sim.groupby("Chunk ID"):
                csim_sorted = csim.sort_values("Z_Score", ascending=False).head(k)
                rels = [concept_rel.get(c, 0) for c in csim_sorted["Concept_lower"]]
                # pad to k
                rels += [0] * (k - len(rels))
                # ideal = top-k relevances sorted
                ideal_rels = sorted(concept_rel.values(), reverse=True)
                ndcg_scores.append(ndcg(rels, k) if dcg(sorted(ideal_rels, reverse=True), k) > 0 else 0.0)
            row[f"NDCG@{k}"] = np.mean(ndcg_scores) if ndcg_scores else 0.0

        ndcg_rows.append(row)

    df_ndcg = pd.DataFrame(ndcg_rows).sort_values("label")
    for k in k_values:
        print(f"[Part 2] Macro NDCG@{k} = {df_ndcg[f'NDCG@{k}'].mean():.4f}")

    return df_ndcg


# ── Part 3: Full-corpus IR eval ───────────────────────────────────────────────
def eval_full_corpus(psi, label_names, label_embs, chunk_embs, k_values=(5, 10, 20)):
    # Build label → positive chunk_id set
    label_positives = {}
    for label, grp in psi.groupby("Label"):
        label_positives[label] = set(grp["Chunk ID"].unique())

    # Score all labels vs all chunks in one matrix multiply: (26, 2636)
    scores_matrix = label_embs @ chunk_embs.T  # already L2-normalized

    n_chunks = chunk_embs.shape[0]
    rows = []

    for i, label in enumerate(label_names):
        if label not in label_positives:
            continue

        positives = label_positives[label]
        n_pos = len(positives)
        if n_pos == 0:
            continue

        scores = scores_matrix[i]  # (2636,)
        ranked_ids = np.argsort(-scores)  # descending

        row = {"label": label, "n_positives": n_pos}

        for k in k_values:
            top_k = ranked_ids[:k]
            hits = sum(1 for cid in top_k if cid in positives)
            row[f"P@{k}"] = hits / k
            row[f"R@{k}"] = hits / n_pos

        # Average Precision
        ap, n_hits = 0.0, 0
        for rank, cid in enumerate(ranked_ids, 1):
            if cid in positives:
                n_hits += 1
                ap += n_hits / rank
        row["AP"] = ap / n_pos if n_pos > 0 else 0.0

        rows.append(row)

    df_ir = pd.DataFrame(rows).sort_values("label")
    map_macro = df_ir["AP"].mean()
    # micro MAP: weight by n_positives
    map_micro = (df_ir["AP"] * df_ir["n_positives"]).sum() / df_ir["n_positives"].sum()

    print(f"\n[Part 3] MAP (macro)={map_macro:.4f}  MAP (micro)={map_micro:.4f}")
    for k in k_values:
        print(f"[Part 3] P@{k} (macro)={df_ir[f'P@{k}'].mean():.4f}  R@{k} (macro)={df_ir[f'R@{k}'].mean():.4f}")

    return df_ir, map_macro, map_micro


# ── Step 8: Write results ──────────────────────────────────────────────────────
def write_results(df_per_sentence, df_macro, micro, macro_avg, df_ndcg, df_ir, map_macro, map_micro):
    summary = pd.DataFrame([
        {"Metric": "MRR (micro)",    "Value": micro["MRR"]},
        {"Metric": "Hit@1 (micro)",  "Value": micro["Hit@1"]},
        {"Metric": "Hit@3 (micro)",  "Value": micro["Hit@3"]},
        {"Metric": "MRR (macro)",    "Value": macro_avg["MRR"]},
        {"Metric": "Hit@1 (macro)",  "Value": macro_avg["Hit@1"]},
        {"Metric": "Hit@3 (macro)",  "Value": macro_avg["Hit@3"]},
        {"Metric": "NDCG@5 (macro)", "Value": df_ndcg["NDCG@5"].mean() if "NDCG@5" in df_ndcg else None},
        {"Metric": "NDCG@10 (macro)","Value": df_ndcg["NDCG@10"].mean() if "NDCG@10" in df_ndcg else None},
        {"Metric": "MAP (macro)",    "Value": map_macro},
        {"Metric": "MAP (micro)",    "Value": map_micro},
    ])

    results_dir = ROOT / "results"
    summary.to_csv(results_dir / "summary.csv", index=False)
    df_per_sentence.to_csv(results_dir / "per_sentence.csv", index=False)
    df_macro.to_csv(results_dir / "per_label_macro.csv", index=False)
    df_ndcg.to_csv(results_dir / "per_label_ndcg.csv", index=False)
    df_ir.to_csv(results_dir / "full_corpus_ir.csv", index=False)

    print(f"\n[done] results written to {results_dir}/ (5 CSV files)")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    psi            = load_psi()
    label_to_query = load_cdef()
    label_names, label_embs = embed_labels(label_to_query)
    chunk_embs     = load_chunk_embeddings()

    df_ps, df_macro, micro, macro_avg = eval_mrr_hitk(psi, label_names, label_embs, chunk_embs)

    phi     = load_phi()
    df_ndcg = eval_ndcg(psi, phi)

    df_ir, map_macro, map_micro = eval_full_corpus(psi, label_names, label_embs, chunk_embs)

    write_results(df_ps, df_macro, micro, macro_avg, df_ndcg, df_ir, map_macro, map_micro)
