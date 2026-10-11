
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

print("Project folder:", ROOT)
print("Data folder:", DATA_DIR)

news_path = DATA_DIR / "news.csv"
social_path = DATA_DIR / "social_posts.csv"

print("News file exists:", news_path.exists())
print("Social file exists:", social_path.exists())

news = pd.read_csv(news_path)
social = pd.read_csv(social_path)

combined = pd.concat([news, social], ignore_index=True)

print("\nNews records:", len(news))
print("Social records:", len(social))
print("Total records:", len(combined))

print("\nData ingestion test successful!")
