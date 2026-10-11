
from pathlib import Path

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

st.set_page_config(
    page_title="FinRisk AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 FinRisk AI")
st.subheader("Financial Risk Signals & Portfolio Stress Testing")
st.caption(
    "Hackathon prototype | Synthetic data | Illustrative scenarios"
)


@st.cache_data(ttl=30)
def fetch_signals():
    response = requests.get(f"{API_URL}/signals", timeout=10)
    response.raise_for_status()
    return response.json()["signals"]


@st.cache_data(ttl=30)
def fetch_portfolio():
    response = requests.get(f"{API_URL}/portfolio", timeout=10)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30)
def fetch_scenario_summary():
    path = OUTPUT_DIR / "portfolio_scenario_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=30)
def fetch_scenario_assets():
    path = OUTPUT_DIR / "portfolio_stress_scenarios.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


try:
    signals = pd.DataFrame(fetch_signals())
    portfolio_data = fetch_portfolio()
    baseline_portfolio = pd.DataFrame(portfolio_data["assets"])

except requests.RequestException as exc:
    st.error(f"Could not connect to the FinRisk API: {exc}")
    st.info(
        "Start the API in a separate terminal using: "
        "python -m uvicorn src.api:app --reload"
    )
    st.stop()

if signals.empty:
    st.warning("No risk signals are available.")
    st.stop()

summary = fetch_scenario_summary()
scenario_assets = fetch_scenario_assets()

scenario_names = {
    "baseline": "Baseline",
    "market_downturn": "Market Downturn",
    "credit_crisis": "Credit Crisis",
    "regulatory_shock": "Regulatory Shock",
}

st.header("Portfolio Overview")

if not summary.empty:
    available_scenarios = summary["scenario"].tolist()

    selected_scenario = st.selectbox(
        "Select stress scenario",
        available_scenarios,
        format_func=lambda value: scenario_names.get(
            value, value.replace("_", " ").title()
        ),
    )

    selected_summary = summary[
        summary["scenario"] == selected_scenario
    ].iloc[0]

    initial_value = float(selected_summary["initial_value"])
    stressed_value = float(selected_summary["stressed_value"])
    change = float(selected_summary["estimated_change"])
    return_pct = float(selected_summary["return_pct"])

    if not scenario_assets.empty:
        selected_portfolio = scenario_assets[
            scenario_assets["scenario"] == selected_scenario
        ].copy()
    else:
        selected_portfolio = pd.DataFrame()

    if selected_portfolio.empty and selected_scenario == "baseline":
        selected_portfolio = baseline_portfolio.copy()

    st.caption(
        selected_summary.get("description", "Illustrative scenario")
    )

else:
    st.warning(
        "Scenario comparison files are missing. Run "
        "`python .\\src\\stress_test.py --scenario all` "
        "from your project root."
    )

    selected_scenario = "baseline"
    initial_value = float(portfolio_data["initial_value"])
    stressed_value = float(portfolio_data["stressed_value"])
    change = float(portfolio_data["estimated_change"])
    return_pct = (
        change / initial_value * 100 if initial_value else 0
    )
    selected_portfolio = baseline_portfolio.copy()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Initial Portfolio", f"${initial_value:,.0f}")
c2.metric("Stressed Portfolio", f"${stressed_value:,.0f}")
c3.metric(
    "Estimated Change",
    f"${change:,.0f}",
    delta=f"{return_pct:.2f}%",
    delta_color="normal",
)
c4.metric("Scenario Return", f"{return_pct:.2f}%")

left, right = st.columns(2)

with left:
    st.subheader("Risk Event Distribution")

    event_counts = (
        signals["event_type"]
        .value_counts()
        .rename_axis("Event Type")
        .reset_index(name="Records")
    )

    fig = px.bar(
        event_counts,
        x="Event Type",
        y="Records",
        color="Event Type",
        title="Risk Events by Category",
    )

    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("FinBERT Sentiment Distribution")

    fig = px.histogram(
        signals,
        x="sentiment_score",
        color="sentiment_label",
        nbins=20,
        title="Financial Text Sentiment",
        labels={
            "sentiment_score": "Sentiment Score",
            "sentiment_label": "Sentiment",
        },
    )

    fig.add_vline(x=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

st.header("Portfolio Stress Test")

if not selected_portfolio.empty:
    chart_data = selected_portfolio.melt(
        id_vars=["asset"],
        value_vars=["value", "stressed_value"],
        var_name="Value Type",
        value_name="Portfolio Value",
    )

    chart_data["Value Type"] = chart_data["Value Type"].map({
        "value": "Initial Value",
        "stressed_value": "Stressed Value",
    })

    fig = px.bar(
        chart_data,
        x="asset",
        y="Portfolio Value",
        color="Value Type",
        barmode="group",
        title=(
            "Initial vs Stressed Values — "
            f"{scenario_names.get(selected_scenario, selected_scenario)}"
        ),
        labels={
            "asset": "Asset",
            "Portfolio Value": "Value ($)",
        },
    )

    # Plotly Express uses bargroupgap in update_layout().
    fig.update_layout(
        bargroupgap=0.15,
        xaxis_tickangle=-20,
    )

    st.plotly_chart(fig, use_container_width=True)

    display_columns = [
        "asset",
        "sector",
        "event_type",
        "sentiment_label",
        "value",
        "shock_pct",
        "estimated_change",
        "stressed_value",
    ]

    existing_columns = [
        column for column in display_columns
        if column in selected_portfolio.columns
    ]

    st.dataframe(
        selected_portfolio[existing_columns],
        use_container_width=True,
        hide_index=True,
    )

else:
    st.warning("No asset-level results are available for this scenario.")

if not summary.empty:
    st.header("Scenario Comparison")

    comparison = summary.copy()
    comparison["Scenario Name"] = comparison["scenario"].map(
        lambda value: scenario_names.get(
            value, value.replace("_", " ").title()
        )
    )

    fig = px.bar(
        comparison,
        x="Scenario Name",
        y="stressed_value",
        color="Scenario Name",
        title="Stressed Portfolio Value by Scenario",
        labels={
            "Scenario Name": "Scenario",
            "stressed_value": "Stressed Value ($)",
        },
        text="stressed_value",
    )

    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    comparison_display = comparison[
        [
            "Scenario Name",
            "initial_value",
            "estimated_change",
            "stressed_value",
            "return_pct",
        ]
    ].rename(columns={
        "initial_value": "Initial Value ($)",
        "estimated_change": "Estimated Change ($)",
        "stressed_value": "Stressed Value ($)",
        "return_pct": "Return (%)",
    })

    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Initial Value ($)": st.column_config.NumberColumn(
                format="$%.0f"
            ),
            "Estimated Change ($)": st.column_config.NumberColumn(
                format="$%.0f"
            ),
            "Stressed Value ($)": st.column_config.NumberColumn(
                format="$%.0f"
            ),
            "Return (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
        },
    )

st.header("Risk Signals Explorer")

companies = ["All"] + sorted(
    signals["company"].dropna().unique().tolist()
)

selected_company = st.selectbox(
    "Filter by company",
    companies,
)

filtered = signals.copy()

if selected_company != "All":
    filtered = filtered[
        filtered["company"] == selected_company
    ]

display_columns = [
    "date",
    "company",
    "source",
    "text",
    "sentiment_label",
    "sentiment_score",
    "sentiment_confidence",
    "event_type",
    "impact_score",
]

existing_columns = [
    column for column in display_columns
    if column in filtered.columns
]

column_config = {}

if "sentiment_score" in existing_columns:
    column_config["sentiment_score"] = (
        st.column_config.NumberColumn(
            "Sentiment Score",
            format="%.4f",
        )
    )

if "sentiment_confidence" in existing_columns:
    column_config["sentiment_confidence"] = (
        st.column_config.NumberColumn(
            "Confidence",
            format="%.4f",
        )
    )

if "impact_score" in existing_columns:
    column_config["impact_score"] = (
        st.column_config.NumberColumn(
            "Impact Score",
            format="%.0f",
        )
    )

st.dataframe(
    filtered[existing_columns],
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
)

st.download_button(
    "Download Risk Signals CSV",
    filtered.to_csv(index=False).encode("utf-8"),
    "risk_signals.csv",
    "text/csv",
)

st.info(
    "Sentiment is model-generated by FinBERT. Event classification "
    "uses heuristic rules, and portfolio shocks use illustrative "
    "assumptions. The current records and portfolio are synthetic."
)

