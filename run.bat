
@echo off
REM Exit immediately if any command fails
SETLOCAL ENABLEEXTENSIONS
SET ERRORLEVEL=0

echo 🚀 Installing Python dependencies...
pip install -r requirements.txt
IF ERRORLEVEL 1 GOTO :error

echo 📚 Running test bench Wikipedia scraper...
python test_bench\wikipedia_scraper\main.py
IF ERRORLEVEL 1 GOTO :error

echo 📚 Running RAG_TCRL_X Wikipedia scraper...
python RAG_TCRL_X\wikipedia_scraper\main.py
IF ERRORLEVEL 1 GOTO :error

echo 🌐 Starting FastAPI server with Uvicorn...
uvicorn main:app --reload
IF ERRORLEVEL 1 GOTO :error

GOTO :eof

:error
echo ❌ An error occurred. Exiting script.
exit /b 1
