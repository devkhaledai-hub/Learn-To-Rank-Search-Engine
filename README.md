# Semantrix: Learning-to-Rank Search Engine

Semantrix is a production-style, two-stage search engine built on MS MARCO-style data.  
It combines:

- `Tantivy` for fast first-pass retrieval
- `SentenceTransformers` for semantic similarity
- `LightGBM LambdaRank` for final relevance ranking
- `Modal` for serverless GPU-backed inference/training
- `Streamlit` for the interactive web UI

The result is a search experience that is both fast and semantically aware.

## Homepage (UI)

Your homepage is the neon-themed **Semantrix** interface with:

- Search query input + example query chips
- `Resonate` trigger button
- Sidebar controls:
  - max results (`top_k`)
  - deep ML feature breakdown toggle
- Ranked result cards (`DOC_<id>`)
- Confidence bars + per-result feature explainability
- CSV export (`semantrix_export.csv`)

The main UI implementation is in:
- `notebooks/app.py`

Preview:

![Semantrix Homepage](inspiration_ui.png)

## Project Architecture

Semantrix follows a classic **2-stage retrieval + re-ranking** architecture:

1. Stage 1 (Recall): Tantivy searches the index and returns top candidates (up to 100).
2. Stage 2 (Precision): candidates are featurized and re-ranked with LightGBM.
3. Semantic layer: MiniLM embeddings add neural similarity signals.
4. Optional spelling assist: SymSpell dictionary + custom MS MARCO vocabulary.

## Core Ranking Features

The production ranking pipeline uses:

- `query_len`
- `doc_len`
- `overlap_ratio`
- `bm25_proxy`
- `exact_match_count`
- `tf_idf_proxy`
- `semantic_sim`

These are passed to the trained ranker for final ordering.

## Repository Structure

Key paths:

- `notebooks/app.py`  
  Production Streamlit UI + Modal `ProductionSearchEngine` class
- `notebooks/MS_LTR_code_only.py`  
  End-to-end LTR experimentation/training/inference script
- `build_ltr_notebook.py`  
  Generator that writes the full professional LTR notebook
- `notebooks/upload_dict.py`  
  Uploads local `ms_marco_dictionary.txt` to Modal volume
- `data/checkpoints/ms_marco_v2/`  
  Training pipeline checkpoints
- `data/models/ms_marco_v2/`  
  Trained ranker artifacts
- `data/tantivy_index/`  
  Tantivy search index

## Data + Artifacts

Observed local artifacts include:

- `data/collection.parquet`
- `data/queries.parquet`
- `data/qrels.parquet`
- `data/checkpoints/ms_marco_v2/01_training_pairs.parquet`
- `data/checkpoints/ms_marco_v2/02_features.parquet`
- `data/models/ms_marco_v2/lgbm_ranker.joblib`
- `data/tantivy_index/`

## Setup

### 1. Create environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install streamlit modal pandas numpy pyarrow tantivy lightgbm sentence-transformers torch symspellpy joblib
```

### 3. Configure Modal + Hugging Face access

Set credentials via environment variables or Modal secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `HF_TOKEN`

Also ensure the Modal volume exists (project expects `ltr-ms-data`).

## Running the App

From project root:

```bash
streamlit run notebooks/app.py
```

The UI connects to Modal class:

- App: `ltr-search-engine-ui`
- Class: `ProductionSearchEngine`

## Model/Index Expectations

`notebooks/app.py` currently expects:

- Ranker at `/data/models/ms_marco_v2/lgbm_ranker_jamal.joblib`
- Tantivy index at `/data/tantivy_index`
- Optional custom spell dictionary at `/data/ms_marco_dictionary.txt`

If your trained model is named `lgbm_ranker.joblib`, either:

- rename/copy it to `lgbm_ranker_jamal.joblib`, or
- update the path in `notebooks/app.py`.

## Training Pipeline (High-Level)

Pipeline flow used in notebooks/scripts:

1. Load MS MARCO collection/queries/qrels.
2. Build positive pairs + hard negatives.
3. Engineer lexical + semantic features.
4. Train `LGBMRanker` with query grouping.
5. Save model to `data/models/...`.
6. Build Tantivy index for fast retrieval.
7. Serve production class via Modal + Streamlit frontend.

## Troubleshooting

- `Model not found`: check ranker filename/path mismatch.
- `No results`: verify Tantivy index exists and is populated.
- Slow first request: expected cold-start while Modal container warms.
- SymSpell custom dictionary missing: app still runs, but with default dictionary only.

## Security Note

Do not commit API tokens/secrets in source files.  
Prefer `.env`, Modal Secrets, or secure CI secret stores.

## Version

Current UI footer shows: `Semantrix Search Engine v2.0`.
