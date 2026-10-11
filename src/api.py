
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

app = FastAPI(
    title="FinRisk AI API",
    description="Financial risk signals and portfolio stress-test results",
    version="1.0.0",
)


def load_signals():
    path = OUTPUT_DIR / "risk_signals_enriched.csv"

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Risk signals are unavailable. Run the pipeline first.",
        )

    return pd.read_csv(path).fillna("").to_dict(orient="records")


@app.get("/")
def home():
    return {
        "project": "FinRisk AI",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/signals")
def get_signals(
    company: str | None = Query(default=None, description="Filter by company"),
    limit: int = Query(default=100, ge=1, le=1000),
):
    records = load_signals()

    if company:
        records = [
            record for record in records
            if str(record.get("company", "")).lower() == company.lower()
        ]

    return {
        "count": min(len(records), limit),
        "signals": records[:limit],
    }


@app.get("/portfolio")
def get_portfolio():
    path = OUTPUT_DIR / "portfolio_stress_results.csv"

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Portfolio results are unavailable. Run the stress test first.",
        )

    portfolio = pd.read_csv(path)
    return {
        "asset_count": len(portfolio),
        "initial_value": float(portfolio["value"].sum()),
        "stressed_value": float(portfolio["stressed_value"].sum()),
        "estimated_change": float(portfolio["estimated_change"].sum()),
        "assets": portfolio.fillna("").to_dict(orient="records"),
    }

