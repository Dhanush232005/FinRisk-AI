from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

# Illustrative assumptions, not calibrated market forecasts.
EVENT_SHOCKS = {
    "credit_risk": -0.08,
    "regulatory_risk": -0.10,
    "operational_risk": -0.06,
    "market_risk": -0.05,
    "positive_event": 0.04,
    "general": -0.01,
}

NEGATIVE_SENTIMENT_ADJUSTMENT = -0.02

# Additional scenario shocks applied to each asset.
# These are demonstration assumptions only.
SCENARIOS = {
    "baseline": {
        "description": "Current event-based risk signals",
        "sector_shocks": {},
        "government_shock": -0.005,
    },
    "market_downturn": {
        "description": "Broad market downturn",
        "sector_shocks": {
            "Banking": -0.08,
            "Technology": -0.15,
            "Energy": -0.10,
            "Automotive": -0.12,
        },
        "government_shock": 0.02,
    },
    "credit_crisis": {
        "description": "Credit and lending crisis",
        "sector_shocks": {
            "Banking": -0.18,
            "Technology": -0.05,
            "Energy": -0.08,
            "Automotive": -0.10,
        },
        "government_shock": 0.03,
    },
    "regulatory_shock": {
        "description": "Regulatory disruption",
        "sector_shocks": {
            "Banking": -0.12,
            "Technology": -0.08,
            "Energy": -0.15,
            "Automotive": -0.10,
        },
        "government_shock": -0.005,
    },
}


def load_signals():
    input_path = OUTPUT_DIR / "risk_signals_enriched.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            "Run risk_engine.py, finbert_engine.py, and "
            "integrate_signals.py first."
        )

    signals = pd.read_csv(input_path)

    required = {
        "date", "company", "sentiment_label",
        "sentiment_score", "event_type", "impact_score"
    }
    missing = required - set(signals.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    signals["date"] = pd.to_datetime(
        signals["date"], errors="coerce"
    )

    latest = (
        signals.dropna(subset=["date"])
        .sort_values("date")
        .groupby("company")
        .tail(1)
        .set_index("company")
    )

    return latest


def create_portfolio():
    # Synthetic portfolio for the hackathon demonstration.
    return pd.DataFrame([
        {"asset": "ABC Bank Bonds", "company": "ABC Bank",
         "value": 2_000_000, "sector": "Banking"},
        {"asset": "TechNova Equity", "company": "TechNova",
         "value": 1_500_000, "sector": "Technology"},
        {"asset": "GreenEnergy Bonds", "company": "GreenEnergy",
         "value": 1_000_000, "sector": "Energy"},
        {"asset": "AutoWorks Equity", "company": "AutoWorks",
         "value": 1_200_000, "sector": "Automotive"},
        {"asset": "Government Bonds", "company": "Government",
         "value": 2_500_000, "sector": "Government"},
    ])


def run_scenario(portfolio, latest, scenario_name):
    scenario = SCENARIOS[scenario_name]
    rows = []

    for asset in portfolio.to_dict("records"):
        company = asset["company"]
        sector = asset["sector"]

        if company == "Government":
            event_type = "general"
            sentiment_label = "neutral"
            sentiment_score = 0.0
            event_shock = 0.0
        elif company in latest.index:
            signal = latest.loc[company]
            event_type = signal["event_type"]
            sentiment_label = signal["sentiment_label"]
            sentiment_score = float(signal["sentiment_score"])

            event_shock = EVENT_SHOCKS.get(event_type, -0.01)

            if sentiment_label == "negative":
                event_shock += NEGATIVE_SENTIMENT_ADJUSTMENT
        else:
            event_type = "general"
            sentiment_label = "neutral"
            sentiment_score = 0.0
            event_shock = EVENT_SHOCKS["general"]

        # Preserve the existing baseline calculation.
        if scenario_name == "baseline":
            if company == "Government":
                shock = -0.005
            else:
                shock = event_shock
        else:
            # Scenario shocks replace baseline shocks for non-government
            # assets. Government uses its scenario-specific assumption.
            if company == "Government":
                shock = scenario["government_shock"]
            else:
                shock = scenario["sector_shocks"].get(
                    sector, -0.01
                )

        change = asset["value"] * shock

        rows.append({
            **asset,
            "scenario": scenario_name,
            "event_type": event_type,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "shock_pct": shock,
            "estimated_change": change,
            "stressed_value": asset["value"] + change,
        })

    results = pd.DataFrame(rows)

    initial_value = results["value"].sum()
    total_change = results["estimated_change"].sum()
    stressed_value = results["stressed_value"].sum()

    summary = {
        "scenario": scenario_name,
        "description": scenario["description"],
        "initial_value": initial_value,
        "estimated_change": total_change,
        "stressed_value": stressed_value,
        "return_pct": (
            total_change / initial_value * 100
            if initial_value else 0.0
        ),
    }

    return results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Run FinRisk AI portfolio stress scenarios."
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="baseline",
        help="Scenario to run; use 'all' to compare every scenario.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    latest = load_signals()
    portfolio = create_portfolio()

    selected_scenarios = (
        list(SCENARIOS.keys())
        if args.scenario == "all"
        else [args.scenario]
    )

    all_results = []
    summaries = []

    for scenario_name in selected_scenarios:
        results, summary = run_scenario(
            portfolio, latest, scenario_name
        )
        all_results.append(results)
        summaries.append(summary)

        print(f"\n{'=' * 55}")
        print(f"SCENARIO: {scenario_name.upper()}")
        print(SCENARIOS[scenario_name]["description"])
        print(results.to_string(index=False))
        print(f"\nInitial portfolio: ${summary['initial_value']:,.2f}")
        print(f"Estimated change: ${summary['estimated_change']:,.2f}")
        print(f"Stressed portfolio: ${summary['stressed_value']:,.2f}")
        print(f"Scenario return: {summary['return_pct']:.2f}%")

    combined_results = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    # Keep the existing API/dashboard file compatible for baseline runs.
    if args.scenario in ("baseline", "all"):
        baseline_results = combined_results[
            combined_results["scenario"] == "baseline"
        ].drop(columns=["scenario"])
        baseline_results.to_csv(
            OUTPUT_DIR / "portfolio_stress_results.csv",
            index=False,
        )

    combined_results.to_csv(
        OUTPUT_DIR / "portfolio_stress_scenarios.csv",
        index=False,
    )
    summary_df.to_csv(
        OUTPUT_DIR / "portfolio_scenario_summary.csv",
        index=False,
    )

    print("\nSaved scenario results:")
    print("outputs/portfolio_stress_scenarios.csv")
    print("outputs/portfolio_scenario_summary.csv")

    if args.scenario in ("baseline", "all"):
        print("outputs/portfolio_stress_results.csv (baseline/API)")

    print("\nNote: scenario shocks are illustrative assumptions.")


if __name__ == "__main__":
    main()