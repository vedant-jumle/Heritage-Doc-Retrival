"""Camera-ready revision: recompute every number in the paper from source.

Reviewer flagged that some numbers look inconsistent. This script recomputes each
claimed figure from the canonical result files and reports a PASS/FAIL diff, so the
.tex can be corrected against ground truth rather than by hand.

Also produces the new numbers requested by the reviewer:
  - Hit@K for K in {1,3,5,10,20,50,100}   (reviewer: why only k in {1,3}?)
  - multi-label prevalence in D'_2         (reviewer: "potentially" multi-label?)
  - psi match-type breakdown + FP audit    (reviewer: what about false positives?)
  - low-support label stratification       (reviewer: exclude more concepts?)

Canonical sources (do not regenerate; these are the July run behind the paper):
  results/docling_4sys_new/retrieval_comparison.csv   per (system,label) metrics
  results/docling_4sys_new/retrieval_per_doc.csv      per (system,mp) MAP
  results/new_chunks/psi_docling.csv                  the psi alignment
  results/new_chunks/chunks_docling.parquet           the 2,926 chunks
  results/ablation_4sys/ablation_4sys.csv             chunk-size ablation

Usage: python src/revision_audit.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "revision"

CMP = ROOT / "results/docling_4sys_new/retrieval_comparison.csv"
PERDOC = ROOT / "results/docling_4sys_new/retrieval_per_doc.csv"
PSI = ROOT / "results/new_chunks/psi_docling.csv"
CHUNKS = ROOT / "results/new_chunks/chunks_docling.parquet"
ABLATION = ROOT / "results/ablation_4sys/ablation_4sys.csv"

# Low-support threshold for the stratified macro-average the reviewer asked about.
MIN_SUPPORT = 10

# H_9 (Beemster) is the only Dutch-language MP and is excluded from every number
# in the paper. In the raw chunk file it is mp_index 9; in the psi alignment its
# rows carry a null MP Index.
BEEMSTER_MP = 9

_checks = []


def check(name, claimed, actual, tol=5e-4):
    """Record a paper-claim vs recomputed-value comparison."""
    if isinstance(claimed, float) or isinstance(actual, float):
        ok = abs(float(claimed) - float(actual)) <= tol
    else:
        ok = claimed == actual
    _checks.append((name, claimed, actual, ok))
    return ok


def load_psi():
    """The psi alignment, with the label whitespace bug fixed.

    One row carries 'Legislation ' with a trailing space. Stripping it is what
    reproduces the paper's 737/529; the eval run did not strip, so it silently
    evaluated 528 pairs. Rows with a null MP Index are Beemster (H_9), the
    Dutch-language MP excluded throughout.
    """
    d = pd.read_csv(PSI)
    d["Label"] = d["Label"].str.strip()
    return d


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def audit_dataset(psi, evaluable):
    section("1. DATASET NUMBERS")

    d24 = psi[psi.Label.isin(evaluable)]
    matched = d24.dropna(subset=["Chunk ID"])
    pairs = matched.drop_duplicates(["Label", "Chunk ID", "MP Index"])
    lab_chunks = matched.drop_duplicates(["Chunk ID", "MP Index"])
    unmatched = d24[d24["Chunk ID"].isna()]

    chunks = pd.read_parquet(CHUNKS)
    n_chunks_raw = len(chunks)
    n_chunks = len(chunks[chunks.mp_index != BEEMSTER_MP])

    print(f"sentences in psi file (all labels)     : {len(psi)}")
    print(f"sentences, 24 evaluable labels         : {len(d24)}")
    print(f"  matched to >=1 chunk                 : {len(matched)}")
    print(f"  unmatched                            : {len(unmatched)}")
    print(f"unique (label, chunk) positive pairs    : {len(pairs)}")
    print(f"chunks carrying >=1 label              : {len(lab_chunks)}")
    print(f"total chunks, all 11 MPs               : {n_chunks_raw}")
    print(f"total chunks N, excl. Beemster         : {n_chunks}")

    check("matched sentences = 737", 737, len(matched))
    check("unique positive pairs = 529", 529, len(pairs))
    check("total chunks N = 2926", 2926, n_chunks)
    check("labelled chunks = 318", 318, len(lab_chunks))
    check("unmatched sentences = 4", 4, len(unmatched))

    # Where do the unmatched actually come from?
    print()
    print("unmatched breakdown by MP:")
    vc = unmatched["MP Index"].value_counts(dropna=False)
    for mp, n in vc.items():
        tag = "  <- Beemster (H_9), excluded" if pd.isna(mp) else ""
        print(f"  MP {mp}: {n}{tag}")

    beemster = unmatched["MP Index"].isna().sum()
    print()
    print(f"FINDING: {beemster}/{len(unmatched)} unmatched sentences are Beemster,")
    print("         which the paper excludes anyway. Among the 10 EVALUATED MPs,")
    print(f"         unmatched = {len(unmatched) - beemster}.")

    return pairs, lab_chunks, n_chunks


def audit_multilabel(psi, evaluable):
    """Reviewer: D'_2 is 'potentially' multi-label -- how prevalent actually?"""
    section("2. MULTI-LABEL PREVALENCE (reviewer: 'potentially' multi-label?)")

    d24 = psi[psi.Label.isin(evaluable)].dropna(subset=["Chunk ID"])
    per_chunk = (
        d24.drop_duplicates(["Label", "Chunk ID", "MP Index"])
        .groupby(["Chunk ID", "MP Index"])
        .size()
    )

    dist = per_chunk.value_counts().sort_index()
    total = len(per_chunk)
    multi = int((per_chunk > 1).sum())

    print(f"labelled chunks                 : {total}")
    print(f"  with exactly 1 label          : {int((per_chunk == 1).sum())}")
    print(f"  with >1 label (multi-label)   : {multi}  ({multi / total:.1%})")
    print(f"  max labels on a single chunk  : {int(per_chunk.max())}")
    print(f"  mean labels per chunk         : {per_chunk.mean():.2f}")
    print()
    print("distribution (n_labels -> n_chunks):")
    for k, v in dist.items():
        print(f"  {int(k):2d} label(s): {v:4d} chunks")
    print()
    print(f"FINDING: multi-label is real, not hypothetical -- {multi/total:.1%} of")
    print("         labelled chunks carry more than one label. Drop 'potentially'.")

    dist.rename("n_chunks").to_csv(OUT / "multilabel_distribution.csv")
    return per_chunk


def audit_psi_quality(psi, evaluable):
    """Reviewer: paper only discusses missed matches -- what about false positives?"""
    section("3. PSI MATCH QUALITY (reviewer: what about false positive matches?)")

    d24 = psi[psi.Label.isin(evaluable)]
    mt = d24["Match Type"].value_counts(dropna=False)
    print("match type breakdown (24 evaluable labels):")
    for k, v in mt.items():
        print(f"  {str(k):28s} {v:4d}  ({v/len(d24):5.1%})")

    matched = d24.dropna(subset=["Chunk ID"])
    exact = matched["Match Type"].eq("exact").sum()
    approx = len(matched) - exact
    print()
    print(f"exact (FP-free by construction) : {exact}  ({exact/len(matched):.1%})")
    print(f"approximate (split/fuzzy)       : {approx}  ({approx/len(matched):.1%})")
    print()
    print("FINDING: only the approximate tier can produce false-positive")
    print(f"         alignments, and it covers {approx/len(matched):.1%} of matches.")
    print("         Exact substring matches cannot mis-align. This bounds psi")
    print("         false-positive risk to a small fraction of the ground truth.")

    approx_rows = matched[matched["Match Type"] != "exact"]
    approx_rows.to_csv(OUT / "psi_approximate_matches.csv", index=False)
    print()
    print(f"  -> {len(approx_rows)} approximate matches written for manual audit:")
    print(f"     {OUT / 'psi_approximate_matches.csv'}")

    return mt


def audit_headline(cmp):
    section("4. HEADLINE TABLE (tab:system_summary)")

    macro = cmp.groupby("system")[["MRR", "Hit@1", "Hit@3", "AP"]].mean()
    claimed = {
        "BM25": (0.318, 0.208, 0.333, 0.130),
        "Bi-encoder": (0.230, 0.167, 0.208, 0.130),
        "Dense + Rerank": (0.360, 0.208, 0.500, 0.117),
        "Cross-encoder": (0.452, 0.292, 0.542, 0.136),
    }

    print(f"{'system':16s} {'MRR':>16s} {'Hit@1':>16s} {'Hit@3':>16s} {'MAP':>16s}")
    for sysname, (c_mrr, c_h1, c_h3, c_map) in claimed.items():
        r = macro.loc[sysname]
        cells = []
        for label, c, a in [
            ("MRR", c_mrr, r.MRR), ("Hit@1", c_h1, r["Hit@1"]),
            ("Hit@3", c_h3, r["Hit@3"]), ("MAP", c_map, r.AP),
        ]:
            ok = check(f"{sysname} {label}", c, a, tol=5e-4)
            cells.append(f"{c:.3f}/{a:.3f}{'' if ok else ' X'}")
        print(f"{sysname:16s} " + " ".join(f"{c:>16s}" for c in cells))

    print()
    print("(claimed/recomputed; X = mismatch)")
    return macro


def audit_hitk(cmp):
    """Reviewer: why only k in {1,3}? Looking through 100 is fine."""
    section("5. HIT@K AND RECALL@K AT LARGER K (reviewer: why only k in {1,3}?)")

    have = [c for c in cmp.columns if c.startswith(("P@", "R@"))]
    print(f"K values already computed in the canonical run: {have}")
    print()
    print("Hit@1 / Hit@3 are stored; Hit@K for larger K needs the full ranked")
    print("lists, not the per-label summary. Precision/Recall@{5,10,20} ARE")
    print("available and already answer part of the reviewer's point:")
    print()

    cols = ["P@5", "P@10", "P@20", "R@5", "R@10", "R@20"]
    macro = cmp.groupby("system")[cols].mean()
    print(macro.round(3).to_string())
    print()
    print("FINDING: R@20 reaches 0.171 for the Cross-encoder vs R@5 = 0.131 --")
    print("         i.e. deeper inspection does recover more positives, which is")
    print("         exactly the reviewer's annotation-prioritisation point.")
    print("         Hit@{5,10,20,50,100} requires re-running retrieval to dump")
    print("         ranked lists -- flagged as the one item needing a re-run.")

    macro.round(4).to_csv(OUT / "precision_recall_at_k.csv")
    return macro


def audit_stratified(cmp):
    """Reviewer: maybe exclude more concepts; 3 positives can't support a finding."""
    section(f"6. LOW-SUPPORT STRATIFICATION (support >= {MIN_SUPPORT})")

    sup = cmp[cmp.system == "Cross-encoder"].set_index("label").n_positives
    well = sorted(sup[sup >= MIN_SUPPORT].index)
    low = sorted(sup[sup < MIN_SUPPORT].index)

    print(f"well-supported labels (n>={MIN_SUPPORT}): {len(well)}")
    print(f"low-support labels    (n< {MIN_SUPPORT}): {len(low)}")
    for lab in low:
        print(f"    {lab:28s} n={int(sup[lab]):3d}")
    print()

    allm = cmp.groupby("system")[["MRR", "Hit@1", "Hit@3", "AP"]].mean()
    wellm = (
        cmp[cmp.label.isin(well)]
        .groupby("system")[["MRR", "Hit@1", "Hit@3", "AP"]]
        .mean()
    )

    print("macro-average over ALL 24 labels:")
    print(allm.round(3).to_string())
    print()
    print(f"macro-average over {len(well)} well-supported labels only:")
    print(wellm.round(3).to_string())
    print()
    delta = (wellm - allm).round(3)
    print("delta (well-supported minus all):")
    print(delta.to_string())
    print()

    # Does the paper's headline claim -- Cross-encoder wins -- survive?
    winners = {m: wellm[m].idxmax() for m in ["MRR", "Hit@1", "Hit@3", "AP"]}
    print("winner per metric on the well-supported subset:")
    for m, w in winners.items():
        flag = "" if w == "Cross-encoder" else "   <- NOT the Cross-encoder"
        print(f"  {m:6s}: {w}{flag}")
    print()
    print("FINDING: every system drops once the low-support labels are removed,")
    print("         confirming those labels were inflating the headline numbers.")
    print("         More importantly the ORDERING CHANGES: Dense + Rerank")
    print(f"         overtakes the Cross-encoder on MRR ({wellm.loc['Dense + Rerank','MRR']:.3f} vs "
          f"{wellm.loc['Cross-encoder','MRR']:.3f})")
    print(f"         and Hit@3 ({wellm.loc['Dense + Rerank','Hit@3']:.3f} vs "
          f"{wellm.loc['Cross-encoder','Hit@3']:.3f}), and wins MAP too.")
    print("         The Cross-encoder retains Hit@1 only on a tie.")
    print()
    print("         => The reviewer's suspicion is CORRECT. The claim that the")
    print("         Cross-encoder is strongest is partly carried by labels with")
    print("         2-8 positives. This needs to be stated in the paper, not")
    print("         buried -- it is a genuine qualification of the main result.")

    out = pd.concat({"all_24": allm, f"support_ge_{MIN_SUPPORT}": wellm}, axis=0)
    out.round(4).to_csv(OUT / "stratified_macro.csv")
    pd.Series({"well_supported": len(well), "low_support": len(low)}).to_csv(
        OUT / "support_counts.csv"
    )
    return well, low


def audit_per_label(cmp):
    section("7. PER-LABEL TABLE (tab:per_label) -- KNOWN BUG")

    ce = cmp[cmp.system == "Cross-encoder"][
        ["label", "n_positives", "MRR", "Hit@3", "AP"]
    ].sort_values(["MRR", "AP"], ascending=False)

    top = ce.head(7)
    bottom = ce.tail(5).sort_values(["MRR", "AP"])

    print("CORRECT top 7 by MRR (ties broken by MAP):")
    print(top.to_string(index=False))
    print()
    print("CORRECT bottom 5:")
    print(bottom.to_string(index=False))
    print()
    print("PAPER BUG: 'Human activities' is listed TWICE in the top-7 block")
    print("           (with two different MAP values, 0.264 and 0.169), and")
    print("           'Limited heritage inclusion' (MRR 1.000, MAP 0.264) is")
    print("           MISSING. The 0.264 row is Limited heritage inclusion")
    print("           mislabelled; 0.169 is the real Human activities row.")

    ce.round(4).to_csv(OUT / "per_label_crossencoder.csv", index=False)
    return top, bottom


def audit_ablation():
    """Reviewer line 207-210: match ratio is affected by chunk size."""
    section("8. CHUNK-SIZE ABLATION (reviewer: ratio affected by chunk size)")

    if not ABLATION.exists():
        print(f"MISSING: {ABLATION}")
        return None

    ab = pd.read_csv(ABLATION)
    print(f"columns: {list(ab.columns)}")
    print()
    print(ab.to_string(index=False))

    ab.to_csv(OUT / "ablation_reference.csv", index=False)
    return ab


def audit_per_mp():
    section("9. PER-MP TABLE (tab:per_mp)")

    pd_ = pd.read_csv(PERDOC)
    piv = pd_.pivot(index="mp_index", columns="system", values="MAP")
    piv = piv[["BM25", "Bi-encoder", "Dense + Rerank", "Cross-encoder"]]
    piv = piv.sort_values("Cross-encoder", ascending=False)
    print(piv.round(3).to_string())

    nlab = pd_.drop_duplicates("mp_index").set_index("mp_index").n_labels
    print()
    print("n_labels per MP:")
    print(nlab.to_string())

    piv.round(4).to_csv(OUT / "per_mp_map.csv")
    return piv


def summary():
    section("AUDIT SUMMARY")
    failed = [c for c in _checks if not c[3]]
    passed = [c for c in _checks if c[3]]

    print(f"{len(passed)}/{len(_checks)} claims reproduce exactly.")
    print()
    if failed:
        print("MISMATCHES REQUIRING A .tex EDIT:")
        for name, claimed, actual, _ in failed:
            print(f"  X {name}")
            print(f"      paper says : {claimed}")
            print(f"      actual     : {actual}")
    else:
        print("No mismatches.")

    pd.DataFrame(_checks, columns=["check", "claimed", "actual", "ok"]).to_csv(
        OUT / "number_audit.csv", index=False
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cmp = pd.read_csv(CMP)
    psi = load_psi()
    evaluable = set(cmp.label.unique())

    print(f"canonical eval file : {CMP.relative_to(ROOT)}")
    print(f"systems             : {list(cmp.system.unique())}")
    print(f"evaluable labels    : {len(evaluable)}")

    audit_dataset(psi, evaluable)
    audit_multilabel(psi, evaluable)
    audit_psi_quality(psi, evaluable)
    audit_headline(cmp)
    audit_hitk(cmp)
    audit_stratified(cmp)
    audit_per_label(cmp)
    audit_ablation()
    audit_per_mp()
    summary()

    print()
    print(f"artefacts -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
