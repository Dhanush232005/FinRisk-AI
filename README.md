@'
# FinRisk AI

## Financial Risk Signals & Portfolio Stress Testing

FinRisk AI is a hackathon prototype that analyzes synthetic financial news and social media posts, generates sentiment signals using FinBERT, classifies financial events, and demonstrates potential portfolio impacts through stress testing.

## Features

- Financial news and social media analysis
- FinBERT sentiment analysis
- Rule-based financial event classification
- FastAPI backend with API documentation
- Streamlit dashboard with Plotly charts
- Four portfolio stress scenarios: Baseline, Market Downturn, Credit Crisis, and Regulatory Shock
- CSV and JSON result exports

## Technology Stack

Python, Pandas, FinBERT, Hugging Face Transformers, FastAPI, Streamlit, Plotly.

## Project Structure

- `data/` — synthetic news and social media datasets
- `outputs/` — generated risk signals and stress-test results
- `src/risk_engine.py` — rule-based event classification
- `src/finbert_engine.py` — FinBERT sentiment analysis
- `src/integrate_signals.py` — combines analysis outputs
- `src/stress_test.py` — portfolio stress testing
- `src/api.py` — FastAPI backend
- `src/app.py` — Streamlit dashboard

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt