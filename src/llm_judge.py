"""
LLM-as-judge: test whether the ground truth (high precision, low recall) is
underestimating retrieval system performance.

Samples chunks across 3 buckets per label using the cached reranked lists
from analysis_top_docs.py:
  - Bucket A: ground-truth positive chunks
  - Bucket B: top-ranked chunks NOT in ground truth (candidate "missed positives")
  - Bucket C: low-ranked chunks NOT in ground truth (expected negatives)

Asks an LLM (via TU Delft TULIP API) to rate 1-5 relevance of each sampled
chunk to its label, then checks whether Bucket B scores look more like
Bucket A (evidence of recall gap) or more like Bucket C (evidence ground
truth is fine).

Usage:
    python src/llm_judge.py                # full run (144 pairs)
    python src/llm_judge.py --smoke-test   # quick validation, ~12 pairs
"""

import os
import re
import json
import random
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
import retrieval
from analysis_top_docs import build_and_cache_reranked
from meeting_common import RESULTS_DIR

CE_CACHE_PATH = RESULTS_DIR / "new_chunks" / "ce_ranked_lists_cache.pkl"


def build_ce_ranked(chunks_df, label_names, label_queries, force=False) -> dict:
    """Build (or load) Cross-encoder full-scan ranked lists per label."""
    import pickle
    from sentence_transformers import CrossEncoder

    CE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CE_CACHE_PATH.exists() and not force:
        with open(CE_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"[CE cache] loaded {len(cache)} labels from {CE_CACHE_PATH}")
        return cache

    print(f"[CE] building full-scan ranked lists for {len(label_names)} labels...")
    ce_model = CrossEncoder(retrieval.CROSSENCODER_MODEL)
    chunk_ids = chunks_df["chunk_id"].tolist()
    cache = {}
    for i, (label, query) in enumerate(zip(label_names, label_queries)):
        ranked = retrieval.retrieve_crossencoder_full(ce_model, query, chunk_ids, chunks_df)
        cache[label] = ranked
        if (i + 1) % 5 == 0:
            print(f"  [CE] {i+1}/{len(label_names)} done")
    with open(CE_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"[CE cache] wrote {CE_CACHE_PATH}")
    return cache

TULIP_BASE_URL = "https://api.tulip.tudelft.nl/chat/v1/"
TOP_N_FOR_BUCKET_B = 20  # "top of ranked list" window to draw Bucket B from
MAX_WORKERS = 50  # concurrent TULIP calls


# ── Sampling ──────────────────────────────────────────────────────────────────
def sample_pairs(cache, gt, chunks_df, label_names, seed=42, n_a=2, n_b=3, n_c=1) -> pd.DataFrame:
    """
    For each label, sample from 3 buckets using the label's ranked chunk list
    (cache[label]):
      - Bucket A: ground-truth positive chunks (gt[label]), sample up to n_a.
      - Bucket B: top-ranked chunks (top TOP_N_FOR_BUCKET_B of cache[label])
        NOT in gt[label], sample up to n_b.
      - Bucket C: low-ranked chunks (bottom half of cache[label]) NOT in
        gt[label], sample up to n_c.
    If a label has fewer than n_a ground-truth positives, take what's available.
    """
    rng = random.Random(seed)
    chunk_id_to_text = dict(zip(chunks_df["chunk_id"], chunks_df["text_chunk"]))
    chunk_id_to_mp = dict(zip(chunks_df["chunk_id"], chunks_df["mp_index"]))

    rows = []
    for label in label_names:
        if label not in cache:
            continue
        ranked = cache[label]
        rank_of = {cid: i + 1 for i, cid in enumerate(ranked)}
        positives = gt.get(label, set())

        # Bucket A: ground-truth positives
        a_candidates = [cid for cid in ranked if cid in positives]
        # include any positives not present in ranked list (shouldn't normally happen)
        a_candidates += [cid for cid in positives if cid not in rank_of]
        a_candidates = list(dict.fromkeys(a_candidates))  # dedupe, preserve order
        a_sample = rng.sample(a_candidates, min(n_a, len(a_candidates)))

        # Bucket B: top-ranked, not in ground truth
        top_window = ranked[:TOP_N_FOR_BUCKET_B]
        b_candidates = [cid for cid in top_window if cid not in positives]
        b_sample = rng.sample(b_candidates, min(n_b, len(b_candidates)))

        # Bucket C: bottom half, not in ground truth
        half = len(ranked) // 2
        bottom_window = ranked[half:]
        c_candidates = [cid for cid in bottom_window if cid not in positives]
        c_sample = rng.sample(c_candidates, min(n_c, len(c_candidates)))

        for bucket, sample in (("A", a_sample), ("B", b_sample), ("C", c_sample)):
            for cid in sample:
                rows.append({
                    "label": label,
                    "chunk_id": cid,
                    "mp_index": chunk_id_to_mp.get(cid),
                    "text_chunk": chunk_id_to_text.get(cid, ""),
                    "rank": rank_of.get(cid),
                    "bucket": bucket,
                    "in_ground_truth": cid in positives,
                })

    return pd.DataFrame(rows)


# ── Prompt ────────────────────────────────────────────────────────────────────
def build_judge_prompt(label: str, definition: str, chunk_text: str) -> str:
    truncated = chunk_text[:1500]
    return f"""You are an expert in heritage management and climate adaptation.

Heritage concept label: {label}
Definition: {definition}

Text chunk:
\"\"\"{truncated}\"\"\"

Task: Rate how relevant the text chunk is to the heritage concept label above,
on a scale of 1-5:
1 = not relevant
2 = slightly relevant
3 = moderately relevant
4 = relevant
5 = highly relevant

Respond with ONLY JSON, no other text: {{"score": <1-5>, "reason": "<one short sentence>"}}"""


# ── LLM call ──────────────────────────────────────────────────────────────────
def call_tulip(client, prompt: str, model="chat") -> str:
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == 1:
                print(f"  [error] TULIP call failed twice: {e}")
                return None
    return None


def parse_judge_response(response: str) -> dict:
    if not response:
        return {"score": None, "reason": "<parse failed>"}
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            score = obj.get("score")
            reason = obj.get("reason", "")
            return {"score": int(score) if score is not None else None, "reason": str(reason)}
        except Exception:
            pass
    # Fallback: bare digit 1-5
    digit_match = re.search(r"\b([1-5])\b", response)
    if digit_match:
        return {"score": int(digit_match.group(1)), "reason": "<parse failed>"}
    return {"score": None, "reason": "<parse failed>"}


# ── Agreement / summary ─────────────────────────────────────────────────────
def compute_judge_agreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for bucket, grp in df.groupby("bucket"):
        rows.append({
            "stat": f"mean_llm_score_bucket_{bucket}",
            "value": grp["llm_score"].mean(),
            "n": len(grp),
        })

    bucket_b = df[df["bucket"] == "B"]
    if len(bucket_b) >= 2 and bucket_b["llm_score"].nunique() > 1 and bucket_b["rank"].nunique() > 1:
        corr, pval = spearmanr(bucket_b["llm_score"], bucket_b["rank"])
    else:
        corr, pval = np.nan, np.nan
    rows.append({"stat": "spearman_corr_score_vs_rank_bucketB", "value": corr, "n": len(bucket_b)})
    rows.append({"stat": "spearman_pvalue_bucketB", "value": pval, "n": len(bucket_b)})

    if len(bucket_b) > 0:
        frac_high = (bucket_b["llm_score"] >= 4).mean()
    else:
        frac_high = np.nan
    rows.append({"stat": "frac_bucketB_score_ge_4_recall_gap_estimate", "value": frac_high, "n": len(bucket_b)})

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(model="chat", n_total=144, seed=42, smoke_test=False, system="reranker"):
    load_dotenv(Path(__file__).parent.parent / ".env")
    client = OpenAI(api_key=os.environ["TUDELFT_TULIP_API_KEY"], base_url=TULIP_BASE_URL)

    chunks_df = retrieval.load_chunks()
    label_names, label_queries = retrieval.load_labels()

    # label_queries are "Label: Definition" strings; split off the definition
    label_defs = {}
    for name, query in zip(label_names, label_queries):
        prefix = f"{name}: "
        label_defs[name] = query[len(prefix):] if query.startswith(prefix) else query

    gt, _ = retrieval.load_ground_truth(set(chunks_df["chunk_id"]))

    if system == "crossencoder":
        cache = build_ce_ranked(chunks_df, label_names, label_queries)
    else:
        cache = build_and_cache_reranked()  # loads existing pkl instantly

    df = sample_pairs(cache, gt, chunks_df, label_names, seed=seed, n_a=2, n_b=3, n_c=1)
    if smoke_test:
        df = df.head(12)
    elif n_total and len(df) > n_total:
        df = df.head(n_total)

    print(f"{'[SMOKE TEST] ' if smoke_test else ''}Judging {len(df)} (label, chunk) pairs "
          f"with model='{model}' ({MAX_WORKERS} concurrent calls)...")

    def judge_row(idx_row):
        idx, row = idx_row
        prompt = build_judge_prompt(row["label"], label_defs.get(row["label"], ""), row["text_chunk"])
        resp = call_tulip(client, prompt, model=model)
        parsed = parse_judge_response(resp) if resp else {"score": None, "reason": "<api call failed>"}
        return idx, parsed["score"], parsed["reason"]

    results = {}
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(judge_row, item) for item in df.iterrows()]
        for future in as_completed(futures):
            idx, score, reason = future.result()
            results[idx] = (score, reason)
            done += 1
            if done % 10 == 0:
                print(f"  [{done}/{len(df)}] done")

    df = df.copy()
    df["llm_score"] = [results[i][0] for i in df.index]
    df["llm_reason"] = [results[i][1] for i in df.index]

    RESULTS_DIR.mkdir(exist_ok=True)
    sys_suffix = f"_{system}" if system != "reranker" else ""
    suffix = ("_smoke" if smoke_test else "") + sys_suffix
    df.to_csv(RESULTS_DIR / f"llm_judge_pairs{suffix}.csv", index=False)

    n_ok = df["llm_score"].notna().sum()
    print(f"\n[done] {n_ok}/{len(df)} pairs scored successfully")

    summary = compute_judge_agreement(df.dropna(subset=["llm_score"]))
    summary.to_csv(RESULTS_DIR / f"llm_judge_summary{suffix}.csv", index=False)
    print(f"\nPairs  → {RESULTS_DIR}/llm_judge_pairs{suffix}.csv")
    print(f"Summary → {RESULTS_DIR}/llm_judge_summary{suffix}.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chat")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--system", default="reranker", choices=["reranker", "crossencoder"],
                        help="Which ranked lists to use for Bucket B/C sampling")
    args = parser.parse_args()
    main(model=args.model, smoke_test=args.smoke_test, system=args.system)
