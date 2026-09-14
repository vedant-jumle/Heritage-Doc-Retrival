"""
Generate per-MP dataset statistics and save to results/mp_dataset_stats.csv.

Columns:
  mp_index            : MP index (0-10)
  full_name           : Full UNESCO name
  short_name          : Short name used in paper
  excluded            : True for Beemster (MP9, Dutch-language)
  n_chunks            : Docling chunks after extraction
  n_labels_psi        : Distinct labels annotated in that MP (from PSI)
  n_labels_evaluable  : Subset of PSI labels with an available definition
  n_labelled_chunks   : Unique chunks linked to >=1 labelled sentence via psi
  n_positive_pairs    : Unique (evaluable label, chunk) positive pairs

Usage:
    python src/mp_stats.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent

CHUNKS_PATH = ROOT / "results/new_chunks/chunks_docling_embedded.parquet"
PSI_PATH    = ROOT / "results/new_chunks/psi_docling.xlsx"
CDEF_PATH   = ROOT / "Datasets/Datasets/Heritage concepts/Definitions/New/Heritage Labels & definition.xlsx"
OUT_PATH    = ROOT / "results/mp_dataset_stats.csv"

EXCLUDE_MP = {9}

MP_NAMES = {
    0:  ("Colonies of Benevolence", "Colonies"),
    1:  ("Frontiers of the Roman Empire -- The Lower German Limes", "LGL"),
    2:  ("Eise Eisinga Planetarium in Franeker", "Eisinga"),
    3:  ("Dutch Water Defence Lines", "Defence Lines"),
    4:  ("The Wadden Sea", "Wadden Sea"),
    5:  ("The seventeenth-century canal ring area of Amsterdam inside the Singelgracht", "Amsterdam"),
    6:  ("Van Nellefabriek", "Van Nelle"),
    7:  ("Schokland and surroundings", "Schokland"),
    8:  ("Mill Network at Kinderdijk-Elshout", "Kinderdijk"),
    9:  ("Droogmakerij de Beemster", "Beemster"),
    10: ("Rietveld Schröder House", "Rietveld"),
}


def main():
    chunks = pd.read_parquet(CHUNKS_PATH)
    chunks = chunks[~chunks["mp_index"].isin(EXCLUDE_MP)]

    psi = pd.read_excel(PSI_PATH)
    psi = psi[psi["Chunk ID"].notna()]
    psi = psi[psi["Match Type"] != "no match"]
    psi["Chunk ID"] = psi["Chunk ID"].astype(int)
    psi["MP Index"] = psi["MP Index"].astype(int)
    psi = psi[~psi["MP Index"].isin(EXCLUDE_MP)]
    psi["Label"] = psi["Label"].str.strip()

    cdef = pd.read_excel(CDEF_PATH, usecols=["Heritage Concept", "Definition"])
    cdef = cdef[cdef["Definition"].notna()]
    cdef["Heritage Concept"] = cdef["Heritage Concept"].str.strip()
    evaluable = set(cdef["Heritage Concept"].tolist())

    rows = []
    for mp_idx, (full, short) in sorted(MP_NAMES.items()):
        excluded = mp_idx in EXCLUDE_MP
        if excluded:
            rows.append({"mp_index": mp_idx, "full_name": full, "short_name": short,
                         "excluded": True, "n_chunks": None, "n_labels_psi": None,
                         "n_labels_evaluable": None, "n_labelled_chunks": None,
                         "n_positive_pairs": None})
            continue

        mp_psi = psi[psi["MP Index"] == mp_idx]
        mp_psi_eval = mp_psi[mp_psi["Label"].isin(evaluable)]

        rows.append({
            "mp_index":           mp_idx,
            "full_name":          full,
            "short_name":         short,
            "excluded":           False,
            "n_chunks":           int(len(chunks[chunks["mp_index"] == mp_idx])),
            "n_labels_psi":       int(mp_psi["Label"].nunique()),
            "n_labels_evaluable": int(mp_psi_eval["Label"].nunique()),
            "n_labelled_chunks":  int(mp_psi["Chunk ID"].nunique()),
            "n_positive_pairs":   int(mp_psi_eval.groupby(["Label", "Chunk ID"]).ngroups),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
