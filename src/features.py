"""Feature extraction: two representations compared side by side.

    a) TF-IDF over the cleaned+lemmatised ``text`` column
       (unigrams + bigrams, ``max_features`` tunable).
    b) Sentence embeddings from ``all-MiniLM-L6-v2`` (sentence-transformers),
       computed on the *raw* text because the model was pretrained on
       natural language and handles emojis/typos itself.

Both return dense/sparse matrices ready for clustering or classification.
The vectorizers and fitted models are saved with joblib in ``models/``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src import MODELS_DIR, SEED

# all-MiniLM-L6-v2 -> 384 dims, ~90MB, fast on CPU, good quality/speed tradeoff
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    max_features=20000,     # tuned in notebook 02; 20k is a sane 8GB-RAM cap
    min_df=2,               # ignore words appearing in <2 reviews
    max_df=0.95,            # ignore words in >95% of reviews (near-stopwords)
    sublinear_tf=True,      # 1 + log(tf) damping, helps long reviews
)


def build_tfidf(texts: pd.Series, params: dict | None = None,
                fit: bool = True) -> tuple[np.ndarray, TfidfVectorizer]:
    """Return (tfidf_matrix, vectorizer).

    fit=True  -> learn vocabulary from ``texts`` (training time).
    fit=False -> not supported here; always call with training data the first
                 time and reuse the saved vectorizer for new data.
    """
    params = {**TFIDF_PARAMS, **(params or {})}
    vec = TfidfVectorizer(**params)
    if fit:
        matrix = vec.fit_transform(texts)
    else:  # transform only (vectorizer must be pre-fitted elsewhere)
        raise ValueError("Use transform_tfidf() to apply a saved vectorizer.")
    return matrix, vec


def transform_tfidf(texts: pd.Series, vectorizer: TfidfVectorizer) -> np.ndarray:
    """Apply an already-fitted vectorizer to new text (no refitting)."""
    return vectorizer.transform(texts)


def build_embeddings(texts: pd.Series,
                     model_name: str = DEFAULT_EMBED_MODEL,
                     batch_size: int = 64) -> np.ndarray:
    """Encode ``texts`` into a (n_samples, dim) float32 array.

    Loads the SentenceTransformer lazily and caches it on the function
    object so repeated calls in one session (notebooks, Streamlit) do not
    reload the weights.
    """
    cache = getattr(build_embeddings, "_cache", {})
    if model_name not in cache:
        from sentence_transformers import SentenceTransformer
        cache[model_name] = SentenceTransformer(model_name)
        build_embeddings._cache = cache
    model = cache[model_name]

    # show_progress_bar keeps long laptop runs from looking frozen.
    vecs = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors -> cosine == euclidean
    )
    return vecs.astype(np.float32)


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------
def save_vectorizer(vectorizer: TfidfVectorizer,
                    path: str | None = None) -> str:
    import joblib
    path = path or os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    return path


def load_vectorizer(path: str | None = None) -> TfidfVectorizer:
    import joblib
    path = path or os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    return joblib.load(path)


def features_summary(tfidf, embeddings: np.ndarray | None = None) -> dict:
    """Small dict of shapes/stats useful for printing in notebooks."""
    info = {
        "tfidf_shape": getattr(tfidf, "shape", None),
        "tfidf_n_features": tfidf.shape[1] if hasattr(tfidf, "shape") else None,
    }
    if embeddings is not None:
        info["embedding_shape"] = embeddings.shape
        info["embedding_dim"] = embeddings.shape[1]
    return info
