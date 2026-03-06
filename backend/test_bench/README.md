# Test-Bench RAG System

Complete test-bench system for comparing three RAG architectures against your new retrieval model.

## Architecture Overview

1. **Hybrid Two-Stage RAG with Cross-Encoder Reranking**
   - Stage 1: Hybrid retrieval (60% vector similarity + 40% BM25)
   - Stage 2: Cross-encoder reranking
   - Returns top 3 documents
   - BM25 backend can be switched between `pyserini` (default) and `rank-bm25`

2. **Fusion-in-Decoder (FiD) RAG Architecture**
   - Hybrid retrieval (70% vector + 30% BM25)
   - Fusion encoding of query-document pairs
   - Mean pooling and similarity-based selection

3. **Agentic RAG (Tool-Orchestrated / Multi-Step RAG)**
   - Multi-step retrieval with tool orchestration
   - Relevance checking
   - Answer synthesis

## Setup

```bash
cd backend/test_bench
pip install -r requirements.txt
```

### BM25 backend selection

```bash
# Default backend
export TEST_BENCH_BM25_BACKEND=pyserini

# Optional fallback backend
# export TEST_BENCH_BM25_BACKEND=rank_bm25

# Optional override for where the Lucene index is created
# export TEST_BENCH_PYSERINI_INDEX_DIR=./data/indices/pyserini
```

By default, each architecture builds a Lucene index from the benchmark corpus and uses Pyserini for BM25 scoring. If Pyserini is unavailable, the benchmark falls back to `rank-bm25` automatically.

> Note: Pyserini depends on Java 21 in addition to Python.

## Usage

```bash
python main.py
```

**Options:**
- `python main.py` - Automatically scrapes if data doesn't exist
- `python main.py --force-scrape` - Force re-scraping
- `python main.py --skip-scrape` - Skip scraping, use existing data

### Pyserini indexing and retrieval example

This is the same indexing/retrieval flow used internally when `TEST_BENCH_BM25_BACKEND=pyserini`:

```bash
# 1) Build a Lucene index from JSONL docs
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /path/to/jsonl-dir \
  --index /path/to/lucene-index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

# 2) Run sparse retrieval
python - <<'PY'
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('/path/to/lucene-index')
for hit in searcher.search('what is retrieval augmented generation?', k=5):
    print(hit.docid, hit.score)
PY
```

### Benchmark / evaluation workflow with Pyserini

```bash
export TEST_BENCH_BM25_BACKEND=pyserini
python main.py --skip-scrape
```

The benchmark will train all three architectures with Pyserini-backed BM25 retrieval and print comparative latency/accuracy metrics.

## Output Format

For each architecture, the system outputs:

```
================================================================================
Architect name: [Architecture Name]
Input query: [Your Query]
Output: [Generated Answer]
Latency to retrieve the data: [seconds]
Accuracy: [score]
Confidence score: [score]
Evidence score: [score]
Average accuracy: [score]
================================================================================
```

## System Flow

1. **Data Scraping**: Automatically scrapes Wikipedia data if needed
2. **Data Processing**: Loads PDFs and converts to document chunks
3. **Training**: All three architectures train on the same scraped data
4. **Query Processing**: Single query → all three architectures → formatted output

## File Structure

```
test-bench/
├── main.py                    # Main entry point
├── orchestrator.py            # Coordinates all architectures
├── scraper_runner.py          # Runs Wikipedia scraper
├── data_processor.py          # Processes scraped PDFs
├── output_formatter.py        # Formats results
├── requirements.txt           # Dependencies
├── core/
│   ├── document.py           # Document data structure
│   └── result.py             # Result data structure
├── architectures/
│   ├── base.py               # Base architecture class
│   ├── architecture1_hybrid_rag.py
│   ├── architecture2_fid_rag.py
│   └── architecture3_agentic_rag.py
└── wikipedia_scraper/        # Wikipedia scraper
```

## Metrics Explained

- **Latency**: Time taken to retrieve and process data (seconds)
- **Accuracy**: Overall accuracy score (0-1)
- **Confidence Score**: Model's confidence in the answer (0-1)
- **Evidence Score**: Quality/quantity of supporting evidence (0-1)
- **Average Accuracy**: Average of accuracy metrics

## Notes

- All architectures train on the same Wikipedia scraped data
- Each query is processed by all three architectures simultaneously
- Results are formatted for easy comparison
- The system handles errors gracefully and reports them in the output
