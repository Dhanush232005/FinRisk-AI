
from transformers import pipeline

print("Loading FinBERT model...")

sentiment_model = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    top_k=None
)

samples = [
    "The bank reported rising loan defaults and declining profits.",
    "The company announced record revenue and strong earnings.",
    "The central bank maintained its policy rate as expected."
]

for text in samples:
    predictions = sentiment_model(text, truncation=True, max_length=512)[0]
    best = max(predictions, key=lambda item: item["score"])

    print("\nText:", text)
    print("Sentiment:", best["label"])
    print("Confidence:", round(best["score"], 4))
