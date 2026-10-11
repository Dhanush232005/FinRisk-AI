
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"


def main():
    risk_path = OUTPUT_DIR / "risk_signals.csv"
    sentiment_path = OUTPUT_DIR / "finbert_sentiment.csv"

    if not risk_path.exists() or not sentiment_path.exists():
        raise FileNotFoundError(
            "Run risk_engine.py and finbert_engine.py first."
        )

    risk = pd.read_csv(risk_path)
    sentiment = pd.read_csv(sentiment_path)

    keys = ["date", "company", "source"]

    if risk.duplicated(keys).any() or sentiment.duplicated(keys).any():
        raise ValueError("Duplicate records found for date, company and source.")

    # Remove the old rule-based sentiment columns before merging.
    risk = risk.drop(
        columns=["sentiment_score"],
        errors="ignore"
    )

    sentiment_columns = keys + [
        "sentiment_label",
        "sentiment_score",
        "sentiment_confidence"
    ]

    merged = risk.merge(
        sentiment[sentiment_columns],
        on=keys,
        how="left",
        validate="one_to_one"
    )

    if merged["sentiment_label"].isna().any():
        raise ValueError("Some records did not match between the files.")

    output_path = OUTPUT_DIR / "risk_signals_enriched.csv"
    merged.to_csv(output_path, index=False)

    print("FinRisk AI integration completed!")
    print("Records merged:", len(merged))
    print("Output:", output_path)

    print("\nSentiment summary:")
    print(merged["sentiment_label"].value_counts().to_string())

    print("\nSample integrated signals:")
    print(
        merged[
            [
                "company",
                "sentiment_label",
                "sentiment_score",
                "sentiment_confidence",
                "event_type",
                "impact_score"
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
