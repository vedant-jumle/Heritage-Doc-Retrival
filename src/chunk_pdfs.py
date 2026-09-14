"""
chunk_pdfs.py
-------------
Reads all 11 MP PDFs via Docling, cleans markdown output, slides a window to
produce text chunks, and saves results as parquet + CSV.

Output schema: chunk_id (int), mp_index (int), text_chunk (str)
"""

import re
from pathlib import Path

import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "Datasets" / "MPs" / "NL"
OUT_DIR = ROOT / "results" / "new_chunks"


# ── Docling helpers ───────────────────────────────────────────────────────────

def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.CUDA
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def extract_text(pdf_path: Path, converter: DocumentConverter) -> str:
    """Convert PDF → markdown via Docling, strip image artefacts."""
    result = converter.convert(str(pdf_path))
    md = result.document.export_to_markdown()

    # Strip <!-- image --> comment tags
    md = re.sub(r"<!--\s*image\s*-->", "", md, flags=re.IGNORECASE)
    # Strip markdown image syntax  ![alt](url)
    md = re.sub(r"!\[.*?\]\(.*?\)", "", md)

    return md


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(md_text: str) -> str:
    """
    Clean markdown output into plain text:
      1. Strip markdown table-of-contents tables (pipe-delimited lines near start of doc)
      2. Replace markdown headers with plain text
      3. Normalize whitespace
    """
    lines = md_text.split("\n")

    # Pass 1: remove contiguous blocks of pipe-table lines that appear before the
    # first non-table, non-empty content (table of contents at document start)
    cleaned_lines = []
    in_toc = True  # assume we start in potential TOC region
    for line in lines:
        stripped = line.strip()
        if in_toc:
            # Skip pure table rows (start/end with |) and table separator rows
            if re.match(r"^\|", stripped) or re.match(r"^[-|: ]+$", stripped):
                continue
            elif stripped == "":
                # blank lines during TOC skipping are fine to skip too
                continue
            else:
                # first real content line — stop TOC stripping
                in_toc = False
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Pass 2: strip markdown headers — keep header text only
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Pass 3: collapse multiple blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Sliding-window character-level chunking (same params as Capstone).
    At each chunk boundary, backtrack to the last sentence-ending character
    (.!?) within the final `chunk_overlap` characters to avoid mid-sentence
    splits.  Falls back to hard cut if no sentence boundary is found.
    """
    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = start + chunk_size

        if end >= n:
            # Last chunk — take whatever remains
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to snap to the last sentence boundary in the look-back window
        look_back_start = max(start + chunk_size - chunk_overlap, start)
        window = text[look_back_start:end]
        # Find the last .!? in the window
        match = None
        for m in re.finditer(r"[.!?]", window):
            match = m
        if match is not None:
            snap = look_back_start + match.end()
        else:
            snap = end

        chunk = text[start:snap].strip()
        if chunk:
            chunks.append(chunk)

        # Next window starts with overlap before the snap point
        start = max(snap - chunk_overlap, start + 1)

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")

    print(f"Found {len(pdf_paths)} PDFs in {PDF_DIR}")
    print("Initialising Docling converter (once)…")
    converter = build_converter()

    rows: list[dict] = []
    global_chunk_id = 0

    for pdf_path in pdf_paths:
        # Parse MP index from filename prefix e.g. "06_..." → 6
        prefix = pdf_path.stem.split("_")[0]
        mp_index = int(prefix)

        text = extract_text(pdf_path, converter)
        text = clean_text(text)
        chunks = chunk_text(text)

        print(f"MP{mp_index:02d}: {len(chunks)} chunks, {len(text):,} chars")

        for chunk in chunks:
            rows.append(
                {
                    "chunk_id": global_chunk_id,
                    "mp_index": mp_index,
                    "text_chunk": chunk,
                }
            )
            global_chunk_id += 1

    df = pd.DataFrame(rows, columns=["chunk_id", "mp_index", "text_chunk"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUT_DIR / "chunks_docling.parquet"
    csv_path = OUT_DIR / "chunks_docling.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    total = len(df)
    print(f"\nTotal chunks: {total:,}  (Capstone baseline: 2 636)")
    print(f"Saved → {parquet_path}")
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
