
from pathlib import Path
import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def main():
    news_path = DATA_DIR / "news.csv"
    social_path = DATA_DIR / "social_posts.csv"

    if not news_path.exists() or not social_path.exists():
        raise FileNotFoundError(
            "Both data/news.csv and data/social_posts.csv must exist."
        )

    print("Loading financial sentiment model...")

    model = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None
    )

    news = pd.read_csv(news_path)
    social = pd.read_csv(social_path)
    combined = pd.concat([news, social], ignore_index=True)

    results = []

    for _, row in combined.iterrows():
        text = str(row["text"])
        predictions = model(
            text,
            truncation=True,
            max_length=512
        )[0]

        best = max(predictions, key=lambda item: item["score"])
        scores = {item["label"].lower(): item["score"] for item in predictions}

        # Continuous sentiment score in [-1, 1].
        sentiment_score = scores.get("positive", 0.0) - scores.get("negative", 0.0)

        results.append({
            "date": row["date"],
            "company": row["company"],
            "source": row["source"],
            "text": text,
            "sentiment_label": best["label"].lower(),
            "sentiment_score": round(sentiment_score, 4),
            "sentiment_confidence": round(best["score"], 4)
        })

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "finbert_sentiment.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)

    print("\nFinBERT analysis completed!")
    print("Records analyzed:", len(results))
    print("Output saved to:", output_path)
    print("\nSample predictions:")
    print(pd.DataFrame(results)[
        ["company", "sentiment_label", "sentiment_score", "sentiment_confidence"]
    ].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
