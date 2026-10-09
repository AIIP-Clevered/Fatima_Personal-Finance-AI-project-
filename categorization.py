"""
categorization.py
-----------------
Feature 1: auto-categorize a transaction from its description text.

The idea: a bank statement gives you a messy description ("Starbucks",
"SHELL OIL 4457", "Amazon mktp") and you want to tag it with a spending
category (Coffee Shops, Gas & Fuel, Shopping). Doing that by hand for
hundreds of rows is the boring part -- so we train a model to do it.

How it works (this is a text-classification problem):
  1. TF-IDF turns each description into numbers. It weights words by how
     telling they are: "starbucks" is rare and informative, "payment" is
     common and less so. Think of it as scoring which words actually
     identify the merchant.
  2. Logistic Regression learns, from labelled examples, which word-scores
     map to which category. It's the same "weighted sum -> decision" idea
     from your neural-network workshop, just a single linear layer.

We hold out 25% of the data as a test set so the accuracy we report is on
transactions the model has NOT seen -- otherwise it's just memorising.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def train_categorizer(df: pd.DataFrame):
    """Train on Description -> Category. Returns (model, accuracy, report)."""
    data = df[["Description", "Category"]].dropna()

    # Some categories appear only once or twice in the real data (e.g.
    # "Entertainment"). You can't train AND test on a class with one example,
    # so drop categories with fewer than `min_count` rows. In a real product
    # you'd instead collect more data or merge them into a parent category.
    min_count = 3
    counts = data["Category"].value_counts()
    keep = counts[counts >= min_count].index
    dropped = sorted(set(counts.index) - set(keep))
    data = data[data["Category"].isin(keep)]
    if dropped:
        print(f"(Skipping {len(dropped)} rare categories: {', '.join(dropped)})\n")

    X = data["Description"]
    y = data["Category"]

    # stratify keeps the same category mix in train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),   # single words and pairs, e.g. "olive garden"
            min_df=1,
        )),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    return model, acc, report


def predict_category(model, description: str) -> str:
    """Categorize a single new transaction description."""
    return model.predict([description])[0]


if __name__ == "__main__":
    df = pd.read_csv("data/personal_transactions.csv")
    model, acc, report = train_categorizer(df)
    print(f"Test accuracy: {acc:.1%}\n")
    print(report)

    # try it on descriptions the model was never trained on
    for desc in ["Starbucks Coffee", "Shell Gas Station", "Amazon Marketplace"]:
        print(f"{desc:25s} -> {predict_category(model, desc)}")
