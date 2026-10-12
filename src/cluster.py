"""Unsupervised clustering of review features + comparison of methods.

Provides:
    * kmeans_elbow_silhouette  - sweep k=4..12, return scores for both curves
    * fit_kmeans / fit_agglomerative / fit_dbscan
    * fit_lda                  - gensim LDA with topic-coherence scoring
    * compare_clusterings      - silhouette / Davies-Bouldin / coherence table
    * top_terms_per_cluster    - the most tf-idf-weighted words per cluster
    * samples_per_cluster      - N raw example reviews per cluster
    * project_2d               - PCA (always) or UMAP (if installed)

Everything takes a feature matrix (dense embeddings *or* sparse tf-idf).
Silhouette/DB are only computed on dense matrices; for tf-idf we project
to SVD-reduced dense space first to keep them meaningful and fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize

from src import SEED, set_seed


# --------------------------------------------------------------------------
# Feature conditioning
# --------------------------------------------------------------------------
def _to_dense_for_metrics(X: np.ndarray, n_comp: int = 50) -> np.ndarray:
    """Dense, L2-normalised version of X for distance-based metrics.

    Sparse tf-idf -> TruncatedSVD (latent semantics, keeps cosine geometry);
    dense embeddings -> used as-is. Normalising makes euclidean == cosine.
    """
    from scipy import sparse
    if sparse.issparse(X):
        n_comp = min(n_comp, X.shape[1] - 1, X.shape[0] - 1)
        X = TruncatedSVD(n_components=n_comp, random_state=SEED).fit_transform(X)
    return normalize(np.asarray(X, dtype=np.float32))


# --------------------------------------------------------------------------
# K-means model selection
# --------------------------------------------------------------------------
def kmeans_elbow_silhouette(X, k_range=range(4, 13), dense_for_metrics=None):
    """Sweep k, returning Inertia (elbow) + Silhouette + Davies-Bouldin.

    Pass the same X you will cluster on. Metrics are computed on a dense
    projection so the numbers are comparable across tf-idf and embeddings.
    """
    set_seed()
    Xm = dense_for_metrics if dense_for_metrics is not None else _to_dense_for_metrics(X)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        labels = km.labels_
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(Xm, labels),
            "davies_bouldin": davies_bouldin_score(Xm, labels),
        })
    return pd.DataFrame(rows)


def fit_kmeans(X, k: int):
    set_seed()
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    return km.fit(X), km.predict(X)


def fit_agglomerative(X, n_clusters: int = 8, linkage: str = "ward"):
    set_seed()
    Xd = _to_dense_for_metrics(X)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(Xd)
    return model, labels


def fit_dbscan(X, eps: float = 0.75, min_samples: int = 15):
    """DBSCAN on the dense, normalised projection (eps in cosine-distance units).

    eps=0.75 is the knee of the 15-NN k-distance curve for MiniLM vectors on
    short reviews (median 15-NN distance is ~0.90, so anything near 0.35 makes
    every single point noise). Noise points are label -1; silhouette on labels
    that include -1 is still valid but should be reported alongside the noise
    ratio.
    """
    set_seed()
    Xd = _to_dense_for_metrics(X)
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(Xd)
    return model, model.labels_


# --------------------------------------------------------------------------
# LDA topic model (+ coherence)
# --------------------------------------------------------------------------
def _tokens(texts: pd.Series) -> list[list[str]]:
    return [str(t).split() for t in texts]


def fit_lda(texts: pd.Series, n_topics: int = 8, passes: int = 8):
    """Train gensim LdaMulticore and return (model, dictionary, corpus).

    Kept as a gensim LDA (not LDA2Vec) so coherence uses the standard
    ConfirmMeasure / UMass implementations.
    """
    import gensim.corpora as gcorpus
    import gensim.models.ldamulticore as ldamm

    token_texts = _tokens(texts)
    dictionary = gcorpus.Dictionary(token_texts)
    dictionary.filter_extremes(no_below=10, no_above=0.9)
    corpus = [dictionary.doc2bow(t) for t in token_texts]

    model = ldamm.LdaMulticore(
        corpus=corpus, id2word=dictionary,
        num_topics=n_topics, random_state=SEED,
        passes=passes, workers=2,
    )
    return model, dictionary, corpus


def lda_coherence(model, dictionary, texts: pd.Series, coherence: str = "c_v") -> float:
    """Topic coherence averaged over topics, using the model's top terms.

    c_v is the standard human-judgement-correlated measure; UMass is a
    faster PMAI-based alternative. gensim scores both from the tokenised
    documents, so ``texts`` is the same Series handed to ``fit_lda``.
    """
    from gensim.models.coherencemodel import CoherenceModel
    cm = CoherenceModel(model=model,
                        texts=_tokens(texts),
                        dictionary=dictionary,
                        coherence=coherence,
                        topn=10)
    return cm.get_coherence()


def lda_topic_terms(model, dictionary, n_words: int = 10) -> pd.DataFrame:
    rows = []
    for topic_id, terms_str in model.show_topics(num_words=n_words, formatted=False):
        words = [w for w, _ in terms_str]
        rows.append({"cluster": topic_id, "top_terms": " ".join(words)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Comparison table
# --------------------------------------------------------------------------
def compare_clusterings(results: dict[str, tuple]) -> pd.DataFrame:
    """results maps method-name -> (X, labels). Builds the metrics table.

    Silhouette / Davies-Bouldin on the dense projection; coherence is only
    available for LDA (NaN here, filled separately in the notebook).
    """
    from scipy import sparse
    rows = []
    for name, (X, labels) in results.items():
        X = np.asarray(X) if not sparse.issparse(X) else X
        Xm = _to_dense_for_metrics(X)
        valid = labels != -1  # ignore DBSCAN noise in the metrics
        if len(set(labels[valid])) < 2:
            sil = db = float("nan")
        else:
            sil = silhouette_score(Xm[valid], labels[valid])
            db = davies_bouldin_score(Xm[valid], labels[valid])
        rows.append({
            "method": name,
            "n_clusters": int(len(set(labels[valid]))),
            "noise_pct": float((~valid).mean() * 100),
            "silhouette": sil,
            "davies_bouldin": db,
        })
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False)


# --------------------------------------------------------------------------
# Interpretation helpers
# --------------------------------------------------------------------------
def top_terms_per_cluster(tfidf, vectorizer, labels: np.ndarray,
                          n_terms: int = 12) -> pd.DataFrame:
    """Rank words by mean tf-idf weight inside each cluster."""
    import scipy.sparse as sp
    mean = np.asarray(tfidf.mean(axis=0)).ravel() if sp.issparse(tfidf) \
        else tfidf.mean(axis=0)
    vocab = np.array(vectorizer.get_feature_names_out())
    terms_by_cluster = {}
    for c in sorted(set(labels)) if -1 not in set(labels) else sorted(set(labels) - {-1}):
        mask = labels == c
        cluster_mean = np.asarray(tfidf[mask].mean(axis=0)).ravel() if sp.issparse(tfidf) \
            else tfidf[mask].mean(axis=0)
        order = np.argsort(cluster_mean)[::-1][:n_terms]
        terms_by_cluster[c] = " ".join(vocab[order])
    return (pd.DataFrame({"cluster": list(terms_by_cluster),
                          "top_terms": list(terms_by_cluster.values())})
            .sort_values("cluster"))


def samples_per_cluster(raw_texts: pd.Series, labels: np.ndarray,
                        n: int = 5) -> pd.DataFrame:
    """Return up to ``n`` original reviews per cluster for manual naming."""
    rows = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        # Take a spread across the cluster, not just the first n.
        pick = idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
        for i in pick:
            rows.append({"cluster": c, "raw_text": raw_texts.iloc[i]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2D projection for visualisation
# --------------------------------------------------------------------------
def project_2d(X, labels: np.ndarray, method: str = "auto") -> pd.DataFrame:
    """Reduce X to 2 columns. UMAP if importable, else PCA. Sparse-aware."""
    import scipy.sparse as sp
    Xd = np.asarray(X) if not sp.issparse(X) else \
        TruncatedSVD(n_components=min(50, X.shape[1] - 1),
                     random_state=SEED).fit_transform(X)

    use_umap = method in ("auto", "umap")
    if use_umap:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=SEED)
            coords = reducer.fit_transform(Xd)
            which = "umap"
        except ImportError:
            coords = PCA(n_components=2, random_state=SEED).fit_transform(Xd)
            which = "pca"
    else:
        coords = PCA(n_components=2, random_state=SEED).fit_transform(Xd)
        which = "pca"

    df = pd.DataFrame(coords, columns=["x", "y"])
    df["cluster"] = labels
    df.attrs["projection"] = which
    return df
