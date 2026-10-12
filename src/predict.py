"""Loading saved artifacts and labelling new reviews (serving path).

A single ``Reviewer`` object wraps: preprocessing -> feature extraction
(TF-IDF or embeddings) -> the best trained pipeline -> cluster-id-to-name
mapping from ``config/cluster_names.json``.

Used by the Streamlit app and the generalization test (step 9).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src import CONFIG_DIR, MODELS_DIR
from src import preprocess as pp
from src import features as ft


# --------------------------------------------------------------------------
# Editable cluster-name config (step 5)
# --------------------------------------------------------------------------
# Named from the KMeans k=8 solution on data/processed/reviews_clustered.csv
# (top tf-idf terms per cluster); edit config/cluster_names.json to re-taxonomise.
DEFAULT_CLUSTER_NAMES = {
    "0": "app crashes & won't open",
    "1": "mixed / general impressions",
    "2": "reminders & notifications broken",
    "3": "pricing & ads complaints",
    "4": "task & scheduling feature requests",
    "5": "calendar & event integration",
    "6": "usability & ease of use",
    "7": "positive feedback / praise",
}


def cluster_names_path() -> str:
    return os.path.join(CONFIG_DIR, "cluster_names.json")


def load_cluster_names(path: str | None = None) -> dict[int, str]:
    """Read cluster id -> human name. Creates the file with defaults if absent."""
    path = path or cluster_names_path()
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(DEFAULT_CLUSTER_NAMES, f, indent=2)
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_cluster_names(names: dict, path: str | None = None) -> str:
    path = path or cluster_names_path()
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in names.items()}, f, indent=2)
    return path


# --------------------------------------------------------------------------
# Reviewer
# --------------------------------------------------------------------------
class Reviewer:
    """End-to-end predictor.

    Parameters
    ----------
    feature_type : 'tfidf' | 'embedding'
    pipeline_path : optional explicit path to a saved best_*_pipeline.joblib
    vectorizer_path : optional explicit path to tfidf_vectorizer.joblib
    """

    def __init__(self, feature_type: str = "tfidf",
                 pipeline_path: str | None = None,
                 vectorizer_path: str | None = None):
        import joblib
        self.feature_type = feature_type
        self.names = load_cluster_names()
        self.name_to_id = {v: int(k) for k, v in self.names.items()}

        if pipeline_path:
            bundle = joblib.load(pipeline_path)
        else:
            exact = os.path.join(MODELS_DIR, f"best_{feature_type}_pipeline.joblib")
            if os.path.exists(exact):
                bundle = joblib.load(exact)
            else:
                # 03 saves only the winning representation, so fall back to
                # whichever best_*_pipeline.joblib actually exists on disk.
                from src.classify import load_best
                bundle = load_best()
        self.pipeline = bundle["pipeline"]
        self.pipeline_feature_type = bundle.get("feature_type", feature_type)

        self.vectorizer = None
        if self.pipeline_feature_type == "tfidf":
            self.vectorizer = joblib.load(
                vectorizer_path or os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

    # ----------------------------------------------------------------------
    def _featurize(self, cleaned: pd.Series, raw: pd.Series):
        if self.pipeline_feature_type == "tfidf":
            return ft.transform_tfidf(cleaned, self.vectorizer)
        return ft.build_embeddings(raw)   # embeddings read the original text

    def predict_dataframe(self, df: pd.DataFrame,
                          review_col: str = "review") -> pd.DataFrame:
        """Add category_id, category_name and predicted features to ``df``.

        ``min_words`` is relaxed to 1 for serving: the embedding pipeline
        classifies raw text, and dropping a user's one-word reviews ("Good
        app") would silently shrink their upload.
        """
        processed = pp.preprocess_dataframe(df, review_col=review_col,
                                            min_words=1)
        if processed.empty:
            raise ValueError(
                "No reviews survived cleaning: rows were blank/duplicate or "
                f"shorter than {pp.MIN_WORDS} words. Check that the selected "
                "column really contains review text.")
        X = self._featurize(processed["text"], processed["raw_text"])
        pred = self.pipeline.predict(X)

        # 03 trains on the readable category names, so the pipeline emits names
        # rather than cluster ids; recover the id through the name config.
        processed["category_name"] = list(pred)
        processed["category_id"] = [self.name_to_id.get(v) for v in pred]
        return processed

    def predict_text(self, text: str) -> str:
        one = pd.DataFrame({"review": [text]})
        return self.predict_dataframe(one)["category_name"].iloc[0]


# --------------------------------------------------------------------------
# Convenience for the generalization test (step 9)
# --------------------------------------------------------------------------
def distribution(df: pd.DataFrame, name_col: str = "category_name") -> pd.Series:
    return df[name_col].value_counts(normalize=True).mul(100).round(1)


def spot_check_accuracy(df: pd.DataFrame, gold_col: str,
                        pred_col: str = "category_name") -> float:
    """Manual spot-check: % agreement on the rows the reviewer filled in.

    NaN when the column is missing or nobody has labeled anything yet: hand
    labeling is a separate human pass, so an unlabeled scrape is the normal
    case rather than a caller error.
    """
    if gold_col not in df.columns:
        return float("nan")
    judged = df[df[gold_col].notna()]
    if judged.empty:
        return float("nan")
    return round((judged[gold_col] == judged[pred_col]).mean() * 100, 1)
