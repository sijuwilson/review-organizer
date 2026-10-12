"""Supervised classifier training on the cluster-derived labels.

Labels here come from the chosen clustering solution mapped through the
editable ``config/cluster_names.json`` (see predict.load_cluster_names).
In the notebook the 300-review ``labeled_sample.csv`` (manually corrected)
is the ground truth used for the *final* honest evaluation.

Provides:
    * stratified_split          - 70/15/15 train/val/test, seed-fixed
    * model_bank                - LR / LinearSVM / RF / NB with search grids
    * train_and_compare         - RandomizedSearchCV (5-fold) over every
                                  model x feature-representation, returns a
                                  ranked results table + best pipeline
    * evaluate_best             - metrics + report on the held-out test set
    * save_best / load_best     - joblib round-trip of the winning pipeline

Naive Bayes variant is chosen per feature type: ComplementNB for non-negative
TF-IDF, GaussianNB for (signed) dense embeddings.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RandomizedSearchCV,
                                     train_test_split, StratifiedKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.calibration import CalibratedClassifierCV
import scipy.sparse as sp

from src import MODELS_DIR, SEED, set_seed
from src.evaluate import classification_metrics, full_report, plot_confusion_matrix

SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / val / test
CV_FOLDS = 5


def _plain_labels(y) -> np.ndarray:
    """Return ``y`` as a plain NumPy array.

    pandas 3 backs its string column with a pyarrow array, and sklearn's
    internal ``_safe_indexing`` raises
    ``TypeError: only integer scalar arrays can be converted to a scalar
    index`` on it, so every CV split dies before fitting anything.
    """
    if isinstance(y, np.ndarray):
        return y
    to_numpy = getattr(y, "to_numpy", None)
    return np.asarray(y if to_numpy is None else to_numpy())


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
def stratified_split(X, y, ratios=SPLIT_RATIOS, seed: int = SEED):
    """Return (X_tr, X_val, X_te, y_tr, y_val, y_te) stratified.

    First carve out test (15%), then split the remainder into train/val so
    the effective train:val:test is 70:15:15 with every class represented.
    """
    y = _plain_labels(y)
    test_frac = ratios[2]
    val_frac_of_rest = ratios[1] / (ratios[0] + ratios[1])

    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=val_frac_of_rest,
        stratify=y_tr_full, random_state=seed)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# --------------------------------------------------------------------------
# Model bank (grids sized for an 8GB laptop)
# --------------------------------------------------------------------------
def model_bank(feature_type: str = "tfidf") -> dict[str, tuple]:
    """Return {name: (estimator, param_distributions)}.

    Grid keys are pipeline-qualified (``clf__...``) because
    ``train_and_compare`` wraps every estimator in ``Pipeline([("clf", est)])``.
    """
    if feature_type == "tfidf":
        nb = ComplementNB()
        nb_grid = {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}
        svm_base = LinearSVC(class_weight="balanced", max_iter=2000)
    else:  # dense embeddings
        nb = GaussianNB()
        nb_grid = {"clf__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]}
        svm_base = LinearSVC(class_weight="balanced", max_iter=2000)

    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED),
            # saga, not liblinear: liblinear dropped multiclass support in
            # sklearn 1.9 and we have 8 categories. l1_ratio is only read when
            # penalty='elasticnet', so it must be paired with it here -- under
            # the default penalty sklearn ignores it and L1 is never tried.
            {"clf__C": [0.1, 1, 10], "clf__penalty": ["elasticnet"],
             "clf__l1_ratio": [0.0, 1.0], "clf__solver": ["saga"]}),
        "LinearSVM": (
            CalibratedClassifierCV(svm_base),   # gives predict_proba for free
            {"clf__estimator__C": [0.1, 1, 10]}),
        "RandomForest": (
            # n_jobs=1: the outer RandomizedSearchCV is already parallel, and
            # nesting -1 oversubscribes the cores and inflates peak RAM.
            RandomForestClassifier(random_state=SEED, n_jobs=1,
                                   class_weight="balanced"),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 20, 40],
             "clf__min_samples_split": [2, 5]}),
        "NaiveBayes": (nb, nb_grid),
    }
    return models


# --------------------------------------------------------------------------
# Train + compare across all models and both feature sets
# --------------------------------------------------------------------------
def train_and_compare(features: dict[str, object],
                      y: np.ndarray,
                      n_iter: int = 10,
                      random_state: int = SEED) -> tuple[pd.DataFrame, dict]:
    """features maps 'tfidf'/'embedding' -> (X_tr, X_val). y is train labels.

    Runs RandomizedSearchCV (5-fold, fit on train, scored on val) per model.
    Returns (results_table, best_pipelines_by_row) where a row is keyed by
    f"{feature_type}::{model_name}".
    """
    set_seed(random_state)
    y = _plain_labels(y)
    labels = np.unique(y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)

    rows, best = [], {}
    for feat_type, (X_tr, X_val) in features.items():
        for name, (est, grid) in model_bank(feat_type).items():
            pipe = Pipeline([("clf", est)])
            searcher = RandomizedSearchCV(
                pipe, param_distributions=grid, n_iter=n_iter,
                scoring="f1_macro", cv=cv, random_state=random_state,
                n_jobs=-1, refit=True)
            # Need val labels for a clean held-out signal; fall back to CV score.
            searcher.fit(X_tr, y)
            best_pipe = searcher.best_estimator_
            key = f"{feat_type}::{name}"
            best[key] = best_pipe
            rows.append({
                "features": feat_type,
                "model": name,
                "cv_f1_macro": round(searcher.best_score_, 4),
                "best_params": searcher.best_params_,
            })
    table = pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False)
    return table, best


def score_on(table: pd.DataFrame, best: dict,
             features_val: dict, y_val: np.ndarray) -> pd.DataFrame:
    """Add accuracy/precision/recall/F1 columns by predicting on validation."""
    metrics = []
    for _, r in table.iterrows():
        key = f"{r['features']}::{r['model']}"
        X_val = features_val[r["features"]]
        pred = best[key].predict(X_val)
        m = classification_metrics(y_val, pred)
        metrics.append(m)
    return pd.concat([table.reset_index(drop=True),
                      pd.DataFrame(metrics)], axis=1)


# --------------------------------------------------------------------------
# Final evaluation + persistence
# --------------------------------------------------------------------------
def evaluate_best(pipeline, X_test, y_test, labels: list,
                  filename: str = "confusion_best.png") -> dict:
    pred = pipeline.predict(X_test)
    report = full_report(y_test, pred, target_names=[str(l) for l in labels])
    path = plot_confusion_matrix(y_test, pred, labels=labels, filename=filename)
    summary = classification_metrics(y_test, pred)
    summary["confusion_matrix_png"] = path
    summary["report"] = report
    return summary


def save_best(pipeline, feature_type: str,
              path: str | None = None) -> str:
    import joblib
    path = path or os.path.join(MODELS_DIR, f"best_{feature_type}_pipeline.joblib")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_type": feature_type}, path)
    return path


def load_best(path: str | None = None) -> dict:
    import joblib
    if path is None:
        # Prefer the tf-idf pipeline (lighter for CPU serving) if present.
        cand = os.path.join(MODELS_DIR, "best_tfidf_pipeline.joblib")
        if not os.path.exists(cand):
            cand = os.path.join(MODELS_DIR, "best_embedding_pipeline.joblib")
        path = cand
    return joblib.load(path)
