
#!/bin/bash

set -e  # Exit immediately if any command fails

echo "🚀 Installing Python dependencies..."
pip install -r requirements.txt

echo "📚 Running test bench Wikipedia scraper..."
python test_bench/wikipedia_scraper/main.py

echo "📚 Running RAG_TCRL_X Wikipedia scraper..."
python RAG_TCRL_X/wikipedia_scraper/main.py

echo "🌐 Starting FastAPI server with Uvicorn..."
uvicorn main:app --reload
