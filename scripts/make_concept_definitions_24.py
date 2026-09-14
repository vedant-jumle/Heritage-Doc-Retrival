"""Derive `results/revision/concept_definitions_24.csv`: the 24 evaluated labels + definitions.

The source spreadsheet (`Datasets/.../Heritage Labels & definition.xlsx`) currently holds
27 concept rows -- it has accumulated entries (`Management `, `Landscape dynamics`,
`Restricted greenery`) that the published evaluation does not cover, and one row
has an empty definition. The paper evaluates exactly the 24 labels present in
`results/docling_4sys_new/retrieval_comparison.csv`, so that file -- not the spreadsheet -- is the
authority on the label set. This script pins to those 24.

Label names are stripped of surrounding whitespace HERE ONLY, for the definition
lookup. The evaluation data itself is left untouched: `psi_docling.csv` contains
one row whose label is 'Legislation ' (trailing space) and the published run did
not strip it. See README, "Correctness constraints".

Usage: python scripts/make_concept_definitions_24.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "Datasets/Datasets/Heritage concepts/Definitions/New/Heritage Labels & definition.xlsx"
CMP = ROOT / "results/docling_4sys_new/retrieval_comparison.csv"
OUT = ROOT / "results/revision/concept_definitions_24.csv"


def main() -> None:
    labels_24 = sorted(pd.read_csv(CMP)["label"].unique())
    if len(labels_24) != 24:
        raise SystemExit(f"expected 24 evaluated labels, found {len(labels_24)}")

    defs = pd.read_excel(XLSX)
    defs["label"] = defs["Heritage Concept"].astype(str).str.strip()
    defs = defs.drop_duplicates("label").set_index("label")

    rows = []
    for label in labels_24:
        if label not in defs.index:
            raise SystemExit(f"label {label!r} has no row in {XLSX.name}")
        definition = defs.loc[label, "Definition"]
        rows.append({
            "label": label,
            "definition": "" if pd.isna(definition) else str(definition).strip(),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    missing = out[out["definition"] == ""]["label"].tolist()
    print(f"wrote {len(out)} rows -> {OUT.relative_to(ROOT)}")
    if missing:
        print(f"NOTE: empty definition for: {missing}")


if __name__ == "__main__":
    main()
