from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import build_text_features

EVASIVE_PATTERNS = [
    r"\b(can(?:not|'t)\s+comment)\b",
    r"\b(too\s+early\s+to\s+say)\b",
    r"\b(we\s+do\s+not\s+disclose)\b",
    r"\b(no\s+update)\b",
    r"\b(as\s+we\s+said\s+before)\b",
    r"\b(hard\s+to\s+predict)\b",
    r"\b(macro\s+uncertainty)\b",
]

DIRECT_OPEN_PATTERNS = [
    r"^\s*(yes|no)\b",
    r"^\s*(we\s+expect|guidance\s+is|our\s+outlook)\b",
]

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "your",
    "about",
    "what",
    "when",
    "where",
    "which",
    "into",
    "there",
    "would",
    "could",
    "should",
    "will",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def _keyword_overlap_ratio(question: str, answer: str) -> float:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    a_tokens = _tokenize(answer)
    overlap = q_tokens.intersection(a_tokens)
    return len(overlap) / len(q_tokens)


def predict_heuristic(df: pd.DataFrame) -> list[str]:
    preds: list[str] = []
    for _, row in df.iterrows():
        q = str(row["question"])
        a = str(row["answer"])
        a_lower = a.lower().strip()

        if any(re.search(p, a_lower) for p in EVASIVE_PATTERNS):
            preds.append("evasive")
            continue

        opens_direct = any(re.search(p, a_lower) for p in DIRECT_OPEN_PATTERNS)
        has_numeric = bool(re.search(r"\b\d+(\.\d+)?%?\b", a))
        overlap = _keyword_overlap_ratio(q, a)

        if (opens_direct and has_numeric) or overlap >= 0.35:
            preds.append("direct")
        elif opens_direct or overlap >= 0.20:
            preds.append("partial")
        else:
            preds.append("evasive")

    return preds


def train_tfidf_logreg(df_train: pd.DataFrame) -> Pipeline:
    x_train = build_text_features(df_train)
    y_train = df_train["label"]

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def predict_tfidf_logreg(model: Pipeline, df_test: pd.DataFrame) -> list[str]:
    x_test = build_text_features(df_test)
    return list(model.predict(x_test))

