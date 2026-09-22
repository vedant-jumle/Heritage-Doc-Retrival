"""
remap_gt.py
-----------
Re-maps labeled sentences from the Capstone-era ground truth to new Docling
chunk IDs produced by chunk_pdfs.py.

Requires: results/new_chunks/chunks_docling.parquet to exist first.

Output schema: Label, Sentence, Chunk ID, MP Index, Match Type
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
PSI_FILE = (
    ROOT
    / "Datasets"
    / "Datasets"
    / "Heritage concepts"
    / "Sentence Matching"
    / "New"
    / "Reference sentences Matching final+Label.xlsx"
)
CHUNKS_FILE = ROOT / "results" / "new_chunks" / "chunks_docling.parquet"
OUT_DIR = ROOT / "results" / "new_chunks"


# ── Normalization (copied verbatim from notebook cell-4) ──────────────────────

def remove_duplicate_fragment(text: str) -> str:
    min_fragment = 40
    for start in range(20, len(text) - min_fragment):
        fragment = text[start:start + min_fragment]
        later_pos = text.find(fragment, start + 1)
        if later_pos != -1:
            before = text[:later_pos].strip()
            end_of_repeat = later_pos + len(fragment)
            remainder = text[end_of_repeat:].strip()
            return (before + " " + remainder).strip()
    return text


def normalize(text: str) -> str:
    text = remove_duplicate_fragment(text)
    text = re.sub(r'\s*-\s+', '', text)
    text = re.sub(r'\s+-\s*', '', text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\s\xa0​­  ]+', ' ', text)
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('­', '')
    text = re.sub(r"[^\w\s.,;:!?'\"()\-]", '', text)
    text = re.sub(r'[\d.,;:!?()\"\'\-]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', '', text)
    return text


def fuzzy_match(
    norm_sentence: str,
    norm_chunk: str,
    min_coverage: float = 0.85,
    max_gap: int = 60,
) -> bool:
    piece_size = 8
    pieces = [
        norm_sentence[i: i + piece_size]
        for i in range(0, len(norm_sentence) - piece_size, piece_size)
    ]
    if not pieces:
        return False
    found = 0
    pos = 0
    for piece in pieces:
        idx = norm_chunk.find(piece, pos)
        if idx == -1:
            continue
        if idx - pos > max_gap and pos > 0:
            pos = idx
        found += 1
        pos = idx + piece_size
    return (found / len(pieces)) >= min_coverage


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_sentences() -> pd.DataFrame:
    """
    Load PSI_FILE, filter out rows whose original Match Type was 'no match',
    keep Label + Sentence, deduplicate on Sentence.
    """
    df = pd.read_excel(PSI_FILE)
    # Filter out "no match" rows (any capitalisation)
    df = df[~df["Match Type"].str.strip().str.lower().eq("no match")]
    # Normalise label whitespace. The source spreadsheet contains one row with
    # 'Legislation ' (trailing space); left unstripped it becomes a 28th label
    # that no downstream step matches, silently dropping a valid positive pair.
    df["Label"] = df["Label"].astype(str).str.strip()
    df = df[["Label", "Sentence"]].drop_duplicates(subset=["Sentence"])
    df = df.reset_index(drop=True)
    return df


# ── Matching ──────────────────────────────────────────────────────────────────

def match_sentences(
    sentences_df: pd.DataFrame,
    new_chunks_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    5-pass matching of each sentence against new Docling chunks.
    Returns DataFrame with columns: Label, Sentence, Chunk ID, MP Index, Match Type
    """
    chunks = new_chunks_df.to_dict("records")  # list of dicts

    # Normalize all chunks once
    normalized_chunks = [(c, normalize(c["text_chunk"])) for c in chunks]

    # Build merged pairs (N + N+1) — only within same MP
    merged_pairs: dict[int, str] = {}
    for i in range(len(chunks) - 1):
        if chunks[i]["mp_index"] == chunks[i + 1]["mp_index"]:
            merged_pairs[i] = normalize(
                chunks[i]["text_chunk"] + " " + chunks[i + 1]["text_chunk"]
            )

    # Build merged triplets (N + N+1 + N+2) — only within same MP
    merged_triplets: dict[int, str] = {}
    for i in range(len(chunks) - 2):
        if (
            chunks[i]["mp_index"] == chunks[i + 1]["mp_index"]
            and chunks[i]["mp_index"] == chunks[i + 2]["mp_index"]
        ):
            merged_triplets[i] = normalize(
                chunks[i]["text_chunk"]
                + " "
                + chunks[i + 1]["text_chunk"]
                + " "
                + chunks[i + 2]["text_chunk"]
            )

    results: list[dict] = []

    for _, row in sentences_df.iterrows():
        label = row["Label"]
        sentence = row["Sentence"]
        if not isinstance(sentence, str) or not sentence.strip():
            continue
        norm_sentence = normalize(sentence)
        found = False

        # Pass 1: exact match in single chunk
        for chunk, norm_chunk in normalized_chunks:
            if norm_sentence in norm_chunk:
                results.append(
                    {
                        "Label": label,
                        "Sentence": sentence,
                        "Chunk ID": chunk["chunk_id"],
                        "MP Index": chunk["mp_index"],
                        "Match Type": "exact",
                    }
                )
                found = True
                break  # take first hit only (avoid duplicates per pass)

        # Pass 2: exact match in merged pair
        if not found:
            for i, norm_merged in merged_pairs.items():
                if norm_sentence in norm_merged:
                    for j, part in enumerate(
                        ["split (first half)", "split (second half)"]
                    ):
                        results.append(
                            {
                                "Label": label,
                                "Sentence": sentence,
                                "Chunk ID": chunks[i + j]["chunk_id"],
                                "MP Index": chunks[i + j]["mp_index"],
                                "Match Type": part,
                            }
                        )
                    found = True
                    break

        # Pass 3: exact match in merged triplet
        if not found:
            for i, norm_triplet in merged_triplets.items():
                if norm_sentence in norm_triplet:
                    for j, part in enumerate(
                        ["split (part 1)", "split (part 2)", "split (part 3)"]
                    ):
                        results.append(
                            {
                                "Label": label,
                                "Sentence": sentence,
                                "Chunk ID": chunks[i + j]["chunk_id"],
                                "MP Index": chunks[i + j]["mp_index"],
                                "Match Type": part,
                            }
                        )
                    found = True
                    break

        # Pass 4: fuzzy match in single chunk
        if not found:
            for chunk, norm_chunk in normalized_chunks:
                if fuzzy_match(norm_sentence, norm_chunk):
                    results.append(
                        {
                            "Label": label,
                            "Sentence": sentence,
                            "Chunk ID": chunk["chunk_id"],
                            "MP Index": chunk["mp_index"],
                            "Match Type": "fuzzy",
                        }
                    )
                    found = True
                    break

        # Pass 5: fuzzy match in merged pair
        if not found:
            for i, norm_merged in merged_pairs.items():
                if fuzzy_match(norm_sentence, norm_merged):
                    for j, part in enumerate(
                        ["fuzzy split (first half)", "fuzzy split (second half)"]
                    ):
                        results.append(
                            {
                                "Label": label,
                                "Sentence": sentence,
                                "Chunk ID": chunks[i + j]["chunk_id"],
                                "MP Index": chunks[i + j]["mp_index"],
                                "Match Type": part,
                            }
                        )
                    found = True
                    break

        if not found:
            results.append(
                {
                    "Label": label,
                    "Sentence": sentence,
                    "Chunk ID": None,
                    "MP Index": None,
                    "Match Type": "no match",
                }
            )

    results_df = pd.DataFrame(
        results, columns=["Label", "Sentence", "Chunk ID", "MP Index", "Match Type"]
    )

    # ── Statistics ─────────────────────────────────────────────────────────────
    total = sentences_df["Sentence"].nunique()
    exact = results_df[results_df["Match Type"] == "exact"]["Sentence"].nunique()
    split = results_df[
        results_df["Match Type"].str.contains("split", na=False)
        & ~results_df["Match Type"].str.startswith("fuzzy", na=False)
    ]["Sentence"].nunique()
    fuzzy = results_df[
        results_df["Match Type"].str.startswith("fuzzy", na=False)
    ]["Sentence"].nunique()
    no_match = results_df[results_df["Match Type"] == "no match"]["Sentence"].nunique()

    print(f"\nMatch statistics ({total} unique sentences searched):")
    print(f"  Exact matches : {exact:4d}  ({exact / total * 100:.1f}%)")
    print(f"  Split matches : {split:4d}  ({split / total * 100:.1f}%)")
    print(f"  Fuzzy matches : {fuzzy:4d}  ({fuzzy / total * 100:.1f}%)")
    print(f"  No match      : {no_match:4d}  ({no_match / total * 100:.1f}%)")

    matched_sentences = total - no_match
    print(f"\nOriginal GT: 890 pairs.  New GT: {len(results_df)} pairs "
          f"({matched_sentences} sentences matched).")

    return results_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"New chunks file not found: {CHUNKS_FILE}\n"
            "Run chunk_pdfs.py first."
        )

    print(f"Loading new chunks from {CHUNKS_FILE} …")
    new_chunks_df = pd.read_parquet(CHUNKS_FILE)
    print(f"  {len(new_chunks_df):,} chunks loaded "
          f"across {new_chunks_df['mp_index'].nunique()} MPs.")

    print(f"Loading labeled sentences from {PSI_FILE} …")
    sentences_df = load_sentences()
    print(f"  {len(sentences_df):,} unique sentences loaded (excluding original 'no match').")

    results_df = match_sentences(sentences_df, new_chunks_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = OUT_DIR / "psi_docling.xlsx"
    csv_path = OUT_DIR / "psi_docling.csv"
    results_df.to_excel(xlsx_path, index=False)
    results_df.to_csv(csv_path, index=False)

    print(f"\nSaved → {xlsx_path}")
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
