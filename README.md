# Retrieving Climate Change Adaptation Concepts from World Heritage Management Plans in the Netherlands

Code and data for the ClimateNLP workshop paper of the same name.

The paper asks a practical question: can standard information-retrieval systems
surface the passages of a World Heritage Management Plan (MP) that express a
given climate-change-adaptation concept, using only the concept's name and expert
definition as the query? We evaluate four systems — BM25, a bi-encoder, a dense
retriever with a cross-encoder reranker, and a full cross-encoder — against
expert sentence-level annotations from 11 Dutch World Heritage properties, and
complement the sparse ground truth with an LLM-as-judge study.

This repository contains the code, the extracted chunks, the concept definitions,
the alignment output, and the LLM-as-judge prompts and judgements.

## Citation

```bibtex
@inproceedings{jumle2026retrieving,
  title     = {Retrieving Climate Change Adaptation Concepts from World Heritage
               Management Plans in the Netherlands},
  author    = {Jumle, Vedant and
               Agao{\u{g}}lu, Orhan and
               Mol, Laurens and
               {Salas Giron{\'e}s}, Edgar and
               Bai, Nan},
  booktitle = {Proceedings of the Workshop on Natural Language Processing
               for Climate Change (ClimateNLP)},
  year      = {2026},
  note      = {TODO: fill in volume, pages and publisher once proceedings appear}
}
```

The underlying annotation dataset is by Cheang, Bai and Pereira Roders, published
under CC BY 4.0. Please cite it alongside this work — see `CITATION.cff` and
`DATA_PROVENANCE.md`.

## Headline results

Macro-averaged over the 24 evaluated labels, on 2,926 chunks from 10 MPs.
Reproduced from `results/docling_4sys_new/retrieval_summary.csv`; per-label values in
`results/docling_4sys_new/retrieval_comparison.csv`.

| System | MRR | Hit@1 | Hit@3 | MAP |
| --- | --- | --- | --- | --- |
| BM25 | 0.32 | 0.21 | 0.33 | 0.13 |
| Bi-encoder | 0.23 | 0.17 | 0.21 | 0.13 |
| Dense + Rerank | 0.36 | 0.21 | 0.50 | 0.12 |
| **Cross-encoder** | **0.45** | **0.29** | **0.54** | **0.14** |

Read these margins with the bootstrap in mind. With only 24 labels, resampling
(10,000 replicates, seed 20260914 — `src/bootstrap_ci.py`) separates the
cross-encoder from BM25 and the bi-encoder on MRR and from the bi-encoder on
Hit@3, but **Hit@1 and MAP differences are indistinguishable from zero for every
pair of systems**, and the cross-encoder is not separable from Dense + Rerank on
any metric. See `results/revision/bootstrap_ci.csv`.

Models: bi-encoder `sentence-transformers/all-mpnet-base-v2`, cross-encoder
`cross-encoder/ms-marco-MiniLM-L-6-v2`, rerank depth 50.

## Dataset

Cheang et al. annotated sentences in 11 Dutch World Heritage MPs with a 27-code
taxonomy of climate-adaptive management strategies. We extract each MP to text
chunks with Docling (chunk size 1,000, overlap 200) and re-map each annotated
sentence onto the chunk containing it — the alignment we call ψ.

Chunks: **3,102** total, **2,926** after excluding MP 9 (Beemster).

The annotation count chain, reconciled in the paper and verifiable from the
shipped files:

| Stage | Count |
| --- | --- |
| Unique annotated sentences (`psi_docling.csv`) | 763 |
| … matched onto Docling chunks | 731 |
| … excluding the 3 labels without definitions | 724 |
| Sentence-chunk rows on the 24 evaluated labels | 737 |
| **Unique (label, chunk) positive pairs** | **529** |

Of the 763 annotated sentences, 731 are matched onto the Docling chunks and 724
remain after excluding the three labels without expert definitions. Those 724
annotations yield 737 sentence-chunk rows, because 13 sentences each span two
overlapping chunks. Deduplicating (label, chunk) gives the 529 positives used
throughout.

`results/new_chunks/psi_docling.csv` as shipped has 776 rows — 744 matched plus 32 `no match`.
Filtering to matched rows whose label is one of the 24 evaluated labels gives
737; deduplicating on (label, chunk) gives 529. Every figure in the chain above
is reproducible from `psi_docling.csv` in this bundle.

## Correctness constraints

Four non-obvious facts. Ignoring any of them silently produces numbers that
disagree with the published paper.

**1. `ce_scores.csv` is the pre-August-2026 version.** The upstream project's
live `ce_scores.csv` was overwritten in August 2026 by later work that re-scored
the label `Assessment`. Every other label is byte-identical. The file shipped
here as `results/docling_4sys_new/ce_scores.csv.bak_pre_gospel` is a copy of the upstream
`ce_scores.csv.bak_pre_gospel`, which is the version the paper's numbers were
computed from. It is the canonical one. Do not substitute a regenerated file
without re-checking `Assessment`.

**2. The label set is pinned to 24.** The definitions spreadsheet has grown and
now holds 27 concept rows — it has gained `Management ` and `Landscape dynamics`
beyond the set the paper evaluates, and `Restricted greenery` likewise carries a
definition but is never evaluated. The authority on the evaluated label set is
`results/docling_4sys_new/retrieval_comparison.csv`, which contains exactly 24. Any script or
analysis must pin to those 24, not to whatever the spreadsheet currently yields.
`scripts/make_concept_definitions_24.py` does this and emits
`results/revision/concept_definitions_24.csv`.

**3. BM25 tokenisation is `re.findall(r"\w+", text.lower())`**, not
`str.split()` — see `src/retrieval.py:80`. This is not cosmetic: using `.split()`
changes Hit@3 by up to 8 points, because punctuation stays welded to tokens and
query terms stop matching.

**4. MP index 9 (Beemster) is excluded everywhere**, as the only Dutch-language
plan; the rest are English. It accounts for 176 of the 3,102 chunks, leaving
2,926. Note the representation differs by file: in `chunks_docling.parquet`
Beemster appears as `mp_index == 9` and must be filtered out, whereas in
`psi_docling.csv` its rows carry a **NULL** `MP Index` rather than 9 — filtering
on `MP Index != 9` there is a no-op and will not remove them. `retrieval_per_doc.csv`
already covers only the 10 retained MPs.

**5. One label has a trailing space, and it was not stripped.** One row in
the upstream spreadsheet `Reference sentences Matching final+Label.xlsx`
(row 778) carries the label `'Legislation '`. Left unstripped it becomes a 28th
label that no downstream step matches, silently dropping one valid positive pair.

`src/remap_gt.py` now normalises label whitespace, and the shipped
`psi_docling.csv` is regenerated with that fix: 27 labels, and `Legislation` has
**35** positives rather than 34.

`results/docling_4sys_new/retrieval_comparison.csv` and `ce_scores.csv` predate
the fix and still reflect 34. This does **not** affect any reported result:
re-running all four systems against the corrected ground truth reproduces every
published MRR, Hit@1, Hit@3 and MAP to three decimal places, because the affected
chunk ranks 212th for `Legislation` while the first ground-truth hit is already
at rank 18.

## File manifest

Row counts below were read from the shipped files, not copied from the paper.

### `results/` — evaluation data

| File | Rows | Columns |
| --- | --- | --- |
| `chunks_docling.parquet` | 3,102 | `chunk_id`, `mp_index`, `text_chunk` |
| `psi_docling.csv` | 776 | `Label`, `Sentence`, `Chunk ID`, `MP Index`, `Match Type` |
| `ce_scores.csv` | 76,076 | `label`, `chunk_id`, `ce_score`, `in_gt` |
| `retrieval_comparison.csv` | 96 | `n_positives`, `P@5`, `R@5`, `P@10`, `R@10`, `P@20`, `R@20`, `AP`, `MRR`, `Hit@1`, `Hit@3`, `system`, `label` |
| `retrieval_per_doc.csv` | 40 | `system`, `mp_index`, `MAP`, `n_labels` |
| `retrieval_summary.csv` | 4 | `system`, `MRR`, `Hit@1`, `Hit@3`, `P@{5,10,20}`, `R@{5,10,20}`, `MAP` |
| `ablation_4sys.csv` | 24 | `chunk_size`, `overlap`, `n_chunks`, `system`, `MRR`, `Hit@1`, `Hit@3`, `MAP` |
| `llm_judge_pairs_docling.csv` | 144 | `label`, `chunk_id`, `mp_index`, `text_chunk`, `rank`, `bucket`, `in_ground_truth`, `llm_score`, `llm_reason` |
| `concept_definitions_24.csv` | 24 | `label`, `definition` |
| `Heritage Labels & definition.xlsx` | 27 | `Heritage Concept`, `Definition`, `Source`, `Definition2`, `Source3` |

Notes on the larger files:

- **`ce_scores.csv`** is 26 labels × 2,926 chunks = 76,076 rows exactly. It scores
  two labels (`Landscape dynamics`, `Management`) beyond the evaluated 24; filter
  to the 24 before computing anything reported in the paper. The `ce_score`
  column is the original published output; the `in_gt` flag has been recomputed
  from the corrected `psi_docling.csv` and is true on **529** rows.
- **`retrieval_comparison.csv`** is 4 systems × 24 labels. `AP` is the per-label
  average precision that macro-averages to the MAP column of the summary table.
  This file defines the canonical 24-label set.

  **It is deliberately frozen at the published run** and therefore records 34
  positives for `Legislation` rather than the corrected 35. Every other file in
  this bundle uses the corrected ground truth. Re-running all four systems
  against it reproduces every published MRR, Hit@1, Hit@3 and MAP to three
  decimal places, because the affected chunk ranks 212th for `Legislation`
  while the first ground-truth hit is already at rank 18 — but per-label
  recall does move (`Legislation` R@20 goes from 0.088 to 0.114), so derive
  per-label figures from `results/revision/hit_at_k_per_label.csv`, not from
  this file. `results/revision/per_label_crossencoder.csv` is a view of this
  frozen run and likewise shows 34.
- **`llm_judge_pairs_docling.csv`** holds 144 ratings across three buckets:
  A = ground-truth positives (47), B = top-ranked chunks not in ground truth (73),
  C = low-ranked chunks not in ground truth (24). Scores are integers 1–5 with no
  nulls. It covers 25 labels — the evaluated 24 plus `Landscape dynamics`.
- **`psi_docling.csv`** `Match Type` values: `exact` 708, `no match` 32, `fuzzy`
  10, `split (first half)` 9, `split (second half)` 9, `fuzzy split (first half)`
  4, `fuzzy split (second half)` 4.

### `results/revision/` — reviewer-response analyses

| File | Rows | Content |
| --- | --- | --- |
| `bootstrap_ci.csv` | 12 | 95% bootstrap CIs, cross-encoder vs each other system × 4 metrics |
| `hit_at_k_macro.csv` | 4 | Hit@k and R@k for k ∈ {1,3,5,10,20,50,100}, per system |
| `hit_at_k_per_label.csv` | 96 | the same, per (system, label) |
| `query_ablation_macro.csv` | 6 | query-formulation ablation (label only vs label+definition) |
| `query_ablation_per_label.csv` | 144 | the same, per label |
| `psi_approximate_matches.csv` | 35 | the non-exact alignments, offered for inspection |
| `per_label_crossencoder.csv` | 24 | per-label cross-encoder MRR / Hit@3 / AP |
| `per_mp_map.csv` | 10 | per-MP MAP for all four systems |
| `stratified_macro.csv` | 8 | metrics stratified by label support |
| `multilabel_distribution.csv` | 5 | chunks by number of labels assigned |
| `precision_recall_at_k.csv` | 4 | P@k / R@k per system |
| `number_audit.csv` | 21 | automated check of claimed vs actual numbers in the paper |
| `ablation_reference.csv` | 24 | ablation reference copy |
| `support_counts.csv` | 2 | label support summary |

`psi_approximate_matches.csv` holds exactly the 35 fuzzy/split alignments
(10 + 9 + 9 + 4 + 4 from the `Match Type` counts above) that the paper explicitly
offers for reader inspection.

### `figures/`

`fig1_per_mp_map.png`, `fig2_ablation_mrr.png`, `fig3_ce_score_histogram.png`,
`fig4_llm_judge_buckets.png`, `fig5_hit_at_k.png` — produced by
`src/make_figures_revised.py`.

### `prompts/`

`llm_judge_prompt.txt` — the judge prompt template, extracted verbatim from
`build_judge_prompt()` in `src/llm_judge.py`. Placeholders `{label}`,
`{definition}` and `{truncated}` are filled per pair; chunk text is truncated to
1,500 characters.

### `src/` and `scripts/`

`retrieval.py` (the four systems and all metrics), `chunk_pdfs.py` (Docling
extraction), `remap_gt.py` (builds ψ), `llm_judge.py` (LLM-as-judge),
`ablation.py`, `eval_pipeline.py`, `mp_stats.py`, `make_figures.py`,
`make_figures_revised.py`, and the revision scripts `revision_audit.py`,
`hit_at_k.py`, `query_ablation.py`, `bootstrap_ci.py`.
`scripts/make_concept_definitions_24.py` derives the pinned 24-label definitions.

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Versions in `requirements.txt` are pinned to the environment that produced the
published run.

**Repository layout and script paths.** The analysis scripts compute
`ROOT = Path(__file__).parent.parent` and read paths like
`results/docling_4sys_new/...` and
`Datasets/Datasets/Heritage concepts/Definitions/New/...`. Rather than rewrite
them — which would make them diverge from the code that produced the paper —
this repository ships the data under exactly those paths. **No analysis script in
`src/` was edited**; they run unmodified from the repository root.

Data therefore lives under `results/` and `Datasets/`, not under a flattened
`data/` directory. The cross-encoder scores ship only as
`ce_scores.csv.bak_pre_gospel`, which is the canonical pre-August file (see
Correctness constraints); scripts that need it reference that name directly.

These run here and were verified to do so:

```bash
python src/bootstrap_ci.py       # -> results/revision/bootstrap_ci.csv
python src/revision_audit.py     # -> results/revision/*.csv
python src/hit_at_k.py           # -> results/revision/hit_at_k_*.csv
python src/query_ablation.py     # -> results/revision/query_ablation_*.csv
python src/make_figures_revised.py   # -> figures
python scripts/make_concept_definitions_24.py
```

Each of the commands above was executed in this layout and exited successfully.
`src/bootstrap_ci.py` reproduced the paper's confidence intervals exactly
(MRR CE − BM25 = +0.13, 95% CI [+0.03, +0.26]).

`src/make_figures_revised.py` writes to `results/figures_revised/`, not to
`figures/`; the copies in `figures/` are the ones used in the paper. Re-running
it creates that directory rather than overwriting `figures/`.

Note that `src/revision_audit.py` is an audit tool and deliberately reports
mismatches between numbers claimed in the paper draft and numbers recomputed from
the data — a non-empty mismatch list is its normal output, not a failure. Its
findings are captured in `results/revision/number_audit.csv`.

**What does not run here.** `src/retrieval.py`, `src/chunk_pdfs.py`,
`src/remap_gt.py`, `src/ablation.py`, `src/eval_pipeline.py` and `src/mp_stats.py`
need inputs that are not redistributable or not shipped: the 11 MP PDFs, the
precomputed embedding parquet
(`Capstone-Applied-AI-project_12/MP_Embeddings/WG_MPs_mpnet_Embeddings.parquet`),
the raw sentence-matching spreadsheet, and
`results/new_chunks/chunks_docling_embedded.parquet`. They are included because
they are the code that generated the shipped artefacts and are needed to audit
how those artefacts were made — not because they will execute as-is. See
`DATA_PROVENANCE.md`.

## LLM-as-judge

Sparse expert annotation under-counts relevant passages: a chunk can be genuinely
on-topic yet unannotated, so ground-truth metrics understate retrieval quality.
To quantify this we sampled 144 (label, chunk) pairs across three buckets and had
an LLM rate relevance 1–5.

- **Model**: `google/gemma-4-31B-it-qat-w4a16-ct` — a 31B instruction-tuned model
  quantised to 4-bit weights.
- **Serving**: on-premises via TU Delft's **TULIP** API, an OpenAI-compatible
  chat endpoint at `https://api.tulip.tudelft.nl/chat/v1/`. Running the judge
  on-prem avoids sending heritage policy text to third-party providers.
- **Decoding**: temperature 0, `max_tokens` 300, two retries per call.
- **Prompt**: `prompts/llm_judge_prompt.txt`.

**Reproducing the judge requires institutional API access.** TULIP is available
only to TU Delft members and authenticates with a `TUDELFT_TULIP_API_KEY` read
from a local `.env`. There is no public endpoint, and the exact quantised build
is not published, so an external reader **cannot** re-run this step and should not
expect to. That is why the raw judgements are shipped: all 144 ratings, with the
model's one-sentence reason for each, are in
`results/final/data/llm_judge_pairs_docling.csv`, so the analysis is auditable end-to-end even
though the generation step is not repeatable outside TU Delft. No API key is
present in this repository.

## Licence

Code is MIT (© Vedant Jumle, 2026) — see `LICENSE`. The underlying annotation
dataset derives from Cheang et al. and is published under **CC BY 4.0**; its
creators are Kai Cheang, Nan Bai and Ana Pereira Roders. Derived data files
remain subject to CC BY 4.0 and require attribution.

The original Management Plan PDFs are **not** redistributed here — their
redistribution rights are not established. See `DATA_PROVENANCE.md`, which also
flags an open question about the extracted chunk text that should be resolved
before this repository is made public.

Technische Universiteit Delft hereby disclaims all copyright interest in the program “Heritage-Doc-Retrival” written by the Author(s). 

-- Machiel van Dorst, Dean of Faculty of Architecture and the Built Environment
