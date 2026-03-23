# Adobe Leadership Document Intelligence Agent

This repository is my submission for the Adobe leadership-question assignment. The system ingests long-form company reports, builds persistent search artifacts once, and then answers leadership-style questions with grounded evidence, citations, and fallback behavior that still works without an API key.

## What I Implemented

- Parser chain with fallback: `docling -> marker -> native`
- Structure-preserving ingestion: documents -> sections -> chunks -> tables
- Tables treated as first-class retrieval objects with normalized row/value structure
- Hybrid retrieval:
  - BM25 for lexical match
  - dense embeddings for semantic match
  - DuckDB for metadata filtering
  - score fusion plus task-aware heuristics
- Grounded answer generation with:
  - deterministic offline fallback
  - optional OpenAI classification and synthesis path
- Evaluation utilities and a small regression test suite

## Core Workflow

```text
data/raw/*.pdf
  -> ingest
  -> cleaned markdown + parsed artifacts
  -> sections + chunks + tables
  -> build-index
  -> BM25 indexes + dense indexes + DuckDB metadata DB
  -> ask
  -> question planning + filtered retrieval + evidence pack
  -> computed answer + citations + report JSON
```

The intended operating model is "parse once, answer many times." Ingestion and index building are persistent stages. Question answering then reuses those stored artifacts instead of reparsing documents every time.

## Data and Runtime Artifacts

Source documents are stored directly in `data/raw/`.

Included sample corpus:

- `data/raw/adobe_fy2024_annual_report.pdf`
- `data/raw/adobe_q2_fy2025_earnings_release.pdf`
- `data/raw/adobe_q2_fy2025_investor_datasheet.pdf`

Generated at runtime:

- `data/parsed/`
- `data/indexes/`
- `outputs/sample_answers/`
- `outputs/sample_plots/`

The repo keeps only source inputs under `data/`. Parsed data, indexes, and outputs are regenerated when commands are run.

## Pipeline by Stage

### 1. Ingestion

Implemented in:

- `leadership_agent/ingest.py`

What happens:

- raw files are discovered from `data/raw`
- parser chain tries `docling`, then `marker`, then native parsing
- extracted markdown is cleaned and quality-scored
- document metadata is inferred
- headings are converted into hierarchical sections
- sections are split into token-limited chunks
- numeric and markdown tables are extracted and normalized
- parsed artifacts are saved to JSON/JSONL under `data/parsed`

Key output records:

- `DocumentRecord`
- `SectionRecord`
- `ChunkRecord`
- `TableRecord`

### 2. Indexing

Implemented in:

- `leadership_agent/indexing.py`

What happens:

- BM25 bundles are built for sections, chunks, and tables
- dense embeddings are built for sections, chunks, and tables
- FAISS indexes are created when available
- DuckDB metadata tables are rebuilt for documents, sections, chunks, and tables

This gives the system three retrieval layers:

- lexical search
- semantic search
- metadata filtering

### 3. Retrieval

Implemented in:

- `leadership_agent/retrieval.py`

What happens:

- the question is classified into a `QuestionPlan`
- the system decides whether to prioritize tables, latest reports, or specific report types
- DuckDB narrows the eligible document set
- BM25 and dense search run over sections, chunks, and tables
- scores are fused with retrieval heuristics
- an evidence pack is built for answer generation

### 4. Answering

Implemented in:

- `leadership_agent/answering.py`

What happens:

- retrieved evidence is converted into a compact evidence pack
- table rows are analyzed to compute trends and comparisons
- risk factors and strategy signals are extracted from text
- the system decides whether evidence is sufficient
- if an API key is present, it can synthesize a grounded answer with OpenAI
- otherwise it returns a deterministic fallback answer

The final report contains:

- `direct_answer`
- `key_evidence`
- `sources`
- `caveats`
- `analytics`
- `markdown_report`

### 5. Evaluation

Implemented in:

- `leadership_agent/evaluate.py`

What happens:

- runs a question set end to end through the same `ask` path
- captures retrieval hit, citation coverage, groundedness, latency, and answer length
- optionally writes simple plots

## Main Data Structures

Shared schemas live in `leadership_agent/models.py`.

The important ones are:

- `DocumentRecord`: one parsed source document
- `SectionRecord`: one structured section with heading path and pages
- `ChunkRecord`: one retrieval-sized chunk linked to a section
- `TableRecord`: one extracted table with `normalized_rows`
- `QuestionPlan`: retrieval strategy derived from the question
- `RetrievalCandidate`: one ranked evidence object
- `AnswerReport`: final output contract written to JSON

## Stack Used

Parsing and extraction:

- `docling`
- `marker-pdf`
- `pypdf`
- `python-docx`

Retrieval and indexing:

- `rank-bm25`
- `fastembed`
- `faiss-cpu`
- `duckdb`
- `scikit-learn` for TF-IDF fallback
- `numpy`

Answering and runtime:

- `openai` for optional JSON classification and synthesis
- `pyyaml` for config loading
- `tiktoken` for token-aware chunking
- `matplotlib` for plots

Testing:

- `pytest`

## Assumptions

- The evaluator may or may not provide an OpenAI API key.
- The system must still run without an API key.
- Local embeddings should be free and persistent across runs.
- The highest-value questions are grounded leadership questions, not open-ended chat.
- Parser quality can vary across PDFs, so parser fallback is necessary.
- FAISS is used for dense retrieval acceleration when index artifacts have been built; retrieval still works without FAISS by falling back to NumPy scoring.

## Configuration

Primary runtime settings live in:

- `config.yaml`
- `.env.example`

Important controls:

- parser choice and fallback
- chunk size and overlap
- retrieval weights and top-k values
- answer evidence threshold
- OpenAI model names and timeout
- embedding model name

Config precedence is:

```text
dataclass defaults < config.yaml < environment variables
```

## Repository Layout

```text
.
|- README.md
|- requirements.txt
|- .env.example
|- config.yaml
|- adobe_leadership_demo.ipynb
|- main.py
|- run_demo.sh
|- data/
|  |- raw/
|  `- eval/questions.json
|- leadership_agent/
|  |- config.py
|  |- ingest.py
|  |- indexing.py
|  |- retrieval.py
|  |- answering.py
|  |- evaluate.py
|  |- models.py
|  `- utils.py
`- tests/
```

## Fastest Review Path

If a reviewer only wants to test one or two questions, the easiest entrypoint is:

- `adobe_leadership_demo.ipynb`

The notebook is designed so the reviewer can:

- set `OPENAI_API_KEY`, `OPENAI_NANO_MODEL`, and `OPENAI_MINI_MODEL` in one editable cell
- switch parser preference without editing project files
- optionally force a fresh ingest/index rebuild
- ask a natural-language question directly
- view the markdown answer in the notebook and save the JSON report automatically

Launch it with:

```bash
python -m notebook adobe_leadership_demo.ipynb
```

## How To Run

Set up the environment:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Linux/macOS activation:

```bash
source .venv/bin/activate
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

Optional environment file:

```bash
cp .env.example .env
```

Run the pipeline step by step:

```bash
python main.py ingest --config config.yaml
python main.py build-index --config config.yaml
python main.py ask --config config.yaml --question "What is our current revenue trend?"
```

The `ask` command always writes the JSON answer under:

- `outputs/sample_answers/`

Run evaluation and plots:

```bash
python main.py eval --config config.yaml --results_path data/eval/results.json
python main.py plot --config config.yaml --results_path data/eval/results.json --output_dir outputs/sample_plots
```

There is also a bash helper for Unix-like environments:

```bash
./run_demo.sh
```

## Example Questions Covered

- `What is our current revenue trend?`
- `Which departments are underperforming?`
- `What were the key risks highlighted in the last quarter?`

The pipeline is designed for these question types, but the implementation is not hardcoded to those exact three prompts.

## What Makes This Different From A Toy RAG Script

- it preserves document hierarchy instead of flattening everything into chunks
- it extracts tables separately from prose
- it uses metadata-aware retrieval rather than only vector similarity
- it supports offline deterministic answering
- it can compute trends and comparisons from normalized numeric rows

## Limitations

- Table extraction is robust for the included sample corpus, but it is not a universal finance-table parser for every filing layout.
- Dense retrieval quality depends on the local embedding model.
- The parser chain is resilient, but output quality still depends on the underlying PDF structure.
- The sample corpus is intentionally small so the repo is reviewable quickly.

## Requirement Alignment

The repo does not mirror the original long folder tree exactly, but the requested system behaviors are implemented:

- Docling primary parser
- Marker fallback parser
- native fallback parser
- section-aware chunking
- separate table artifacts
- BM25 plus dense hybrid retrieval
- DuckDB metadata filtering
- FAISS dense search acceleration
- grounded answer format with citations and caveats
- evaluation utilities
- tests for core regression coverage
