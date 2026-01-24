# Test-Bench RAG System

Complete test-bench system for comparing three RAG architectures against your new retrieval model.

## Architecture Overview

1. **Hybrid Two-Stage RAG with Cross-Encoder Reranking**
   - Stage 1: Hybrid retrieval (60% vector similarity + 40% BM25)
   - Stage 2: Cross-encoder reranking
   - Returns top 3 documents

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
cd backend/test-bench
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

**Options:**
- `python main.py` - Automatically scrapes if data doesn't exist
- `python main.py --force-scrape` - Force re-scraping
- `python main.py --skip-scrape` - Skip scraping, use existing data

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
