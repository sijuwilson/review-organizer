"""Evaluation, reporting and plotting helpers shared across steps.

    * classification_metrics  - acc / precision / recall / macro-F1 dict
    * full_report             - + sklearn classification_report & confusion matrix
    * plot_confusion_matrix
    * sentiment_per_category  - VADER (or rating) -> %pos/%neg per category + bar chart
    * wordcloud_images        - overall + per-category word clouds
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
# Agg is file-only: forcing it here would blank every plt.show() in a notebook,
# so pin it only outside a Jupyter kernel (Streamlit / scripts, which may have
# no display at all).
if not any(name.startswith("ipykernel") for name in sys.modules):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

from src import DATA_PROCESSED, SEED, set_seed

FIG_DIR = os.path.join(DATA_PROCESSED, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------
def classification_metrics(y_true, y_pred) -> dict:
    """Scalar summary: accuracy, (macro) precision/recall/F1."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": round(acc, 4), "precision_macro": round(p, 4),
            "recall_macro": round(r, 4), "f1_macro": round(f1, 4)}


def full_report(y_true, y_pred, target_names: list | None = None) -> str:
    """Pretty multi-line report (per-class + macro/weighted)."""
    return classification_report(y_true, y_pred, target_names=target_names,
                                 zero_division=0)


# --------------------------------------------------------------------------
# Confusion matrix
# --------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, labels: list,
                          filename: str = "confusion_matrix.png",
                          normalize: bool = True):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1) \
        if normalize else cm
    plt.figure(figsize=(max(6, len(labels) * 0.7), max(5, len(labels) * 0.6)))
    sns.heatmap(disp, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=120); plt.close()
    return path


# --------------------------------------------------------------------------
# Sentiment per category
# --------------------------------------------------------------------------
def _vader_scores(texts: pd.Series) -> np.ndarray:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] for t in texts])


def sentiment_per_category(df: pd.DataFrame,
                           category_col: str = "category",
                           text_col: str = "raw_text",
                           score_col: str | None = "score",
                           pos_thresh: float = 0.05,
                           neg_thresh: float = -0.05) -> pd.DataFrame:
    """Return % positive / neutral / negative per category.

    Uses VADER on the text. If a star ``score_col`` exists it is used as a
    cross-check anchor but VADER drives the label so it also works on the
    freshly-scraped reviews (which always have a star rating, but text is
    the more robust signal).
    """
    set_seed()
    compound = _vader_scores(df[text_col].astype(str))
    sent = np.where(compound >= pos_thresh, "positive",
                    np.where(compound <= neg_thresh, "negative", "neutral"))
    out = pd.crosstab(df[category_col], sent, normalize="index") * 100
    for col in ("positive", "neutral", "negative"):
        if col not in out.columns:
            out[col] = 0.0
    return out[["positive", "neutral", "negative"]].round(1)


def plot_sentiment_by_category(sent_df: pd.DataFrame,
                               filename: str = "sentiment_by_category.png"):
    ax = sent_df.sort_values("positive", ascending=False).plot(
        kind="bar", stacked=True, figsize=(10, 6),
        color=["#2a9d8f", "#adb5bd", "#e63946"])
    ax.set_ylabel("% of reviews")
    ax.set_xlabel("Category")
    ax.set_title("Sentiment distribution per category")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=120); plt.close()
    return path


# --------------------------------------------------------------------------
# Word clouds (EDA + per-category)
# --------------------------------------------------------------------------
def wordcloud_images(texts: pd.Series, title: str = "wordcloud",
                     save_dir: str = FIG_DIR):
    """Render one word cloud for ``texts`` and return the PNG path."""
    from wordcloud import WordCloud
    blob = " ".join(texts.astype(str))
    if not blob.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color="white",
                   random_state=SEED).generate(blob)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
    plt.title(title)
    safe = "".join(ch if ch.isalnum() else "_" for ch in title)
    path = os.path.join(save_dir, f"wordcloud_{safe}.png")
    plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close()
    return path


def wordcloud_per_category(df: pd.DataFrame, category_col: str,
                           text_col: str = "raw_text",
                           save_dir: str = FIG_DIR) -> list:
    paths = []
    for cat, sub in df.groupby(category_col):
        p = wordcloud_images(sub[text_col], title=f"category_{cat}", save_dir=save_dir)
        if p:
            paths.append(p)
    return paths
