"""Review Organizer - Streamlit app (step 10).

Run:  streamlit run app.py

Two input modes:
  1. Upload a CSV of reviews (needs a text column, e.g. 'review').
  2. Enter one or more Google Play app ids and scrape fresh reviews.

Output: category counts (bar + pie), sentiment per category, top keywords per
category, example reviews, and a downloadable results CSV.

The app loads the best saved pipeline + TF-IDF vectorizer from models/ and the
editable category names from config/cluster_names.json.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import MODELS_DIR  # noqa: E402
from src.predict import Reviewer  # noqa: E402
from src.evaluate import sentiment_per_category  # noqa: E402


st.set_page_config(page_title="Review Organizer", page_icon="🗂️", layout="wide")


@st.cache_resource
def get_reviewer(feature_type: str = "tfidf") -> Reviewer:
    """Build once; the vectorizer/pipeline are read from disk."""
    return Reviewer(feature_type=feature_type)


@st.cache_data(show_spinner="Scraping Play Store...")
def scrape(app_id: str) -> pd.DataFrame:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
    from scrape_reviews import scrape_app  # reuse the page-throttling logic
    return scrape_app(app_id, count=200)


def top_keywords(clean: pd.DataFrame) -> pd.DataFrame:
    """Mean tf-idf terms per category from the fitted vectorizer."""
    from src import features as ft
    vec = get_reviewer().vectorizer
    if vec is None:
        return pd.DataFrame()
    mat = ft.transform_tfidf(clean["text"], vec)
    vocab = vec.get_feature_names_out()
    rows = {}
    for cat, idx in clean.groupby("category_name").groups.items():
        mean = np.asarray(mat[list(idx)].mean(axis=0)).ravel()
        order = mean.argsort()[::-1][:10]
        rows[cat] = ", ".join(vocab[order])
    return pd.DataFrame({"category": list(rows), "keywords": list(rows.values())})


def detect_text_column(df: pd.DataFrame) -> str:
    """Pick the column most likely to hold review prose: text dtype with the
    longest average entry. Numeric/date/short-label columns lose."""
    best, best_score = df.columns[0], -1.0
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if s.empty or pd.api.types.is_numeric_dtype(df[c]):
            continue
        score = s.str.len().mean()
        if score > best_score:
            best, best_score = c, score
    return best


def main():
    st.title("🗂️ Review Organizer")
    st.caption("Google Play reviews auto-grouped into meaningful categories.")

    if not any(os.path.exists(os.path.join(MODELS_DIR, f"best_{ft}_pipeline.joblib"))
               for ft in ("tfidf", "embedding")):
        st.warning("No trained model found in models/. Run notebooks 02 and 03 first, "
                   "or upload reviews to explore clustering-only mode.")

    mode = st.radio("Input mode", ["Upload CSV", "Scrape by app id"], horizontal=True)

    raw = None
    if mode == "Upload CSV":
        up = st.file_uploader("Upload a reviews CSV", type="csv")
        if up:
            raw = pd.read_csv(up)
            detected = detect_text_column(raw)
            col = st.selectbox(
                "Review text column",
                list(raw.columns),
                index=list(raw.columns).index(detected),
                help=f"Auto-detected \"{detected}\" as the text column — "
                     "change it only if wrong.")
            raw = raw.rename(columns={col: "review"})
    else:
        apps = st.text_input("Google Play app ids (comma-separated)",
                             "com.spotify.music")
        if st.button("Scrape"):
            frames = [scrape(a.strip()) for a in apps.split(",") if a.strip()]
            raw = pd.concat(frames, ignore_index=True) if frames else None

    if raw is None or raw.empty:
        st.info("Provide reviews to continue.")
        return

    with st.spinner("Categorising..."):
        reviewer = get_reviewer()
        try:
            clean = reviewer.predict_dataframe(raw, review_col="review")
        except ValueError as exc:
            st.error(str(exc))
            return

    st.subheader("Category counts")
    counts = clean["category_name"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    c1, c2 = st.columns(2)
    c1.bar_chart(counts.set_index("category"))
    c2.plotly_chart(px.pie(counts, names="category", values="count", hole=0.4),
                    use_container_width=True)

    st.subheader("Sentiment per category")
    sent = sentiment_per_category(clean, category_col="category_name",
                                  text_col="raw_text")
    st.bar_chart(sent)

    st.subheader("Top keywords per category")
    st.dataframe(top_keywords(clean), use_container_width=True)

    st.subheader("Example reviews")
    for cat in counts["category"]:
        with st.expander(f"{cat}  ({int(counts.loc[counts.category==cat,'count'].iloc[0])})"):
            for t in clean[clean["category_name"] == cat]["raw_text"].head(5):
                st.write("-", t)

    st.subheader("Download results")
    buf = io.StringIO()
    clean.to_csv(buf, index=False)
    st.download_button("Download results CSV", buf.getvalue().encode(),
                       file_name="review_organizer_results.csv",
                       mime="text/csv")


if __name__ == "__main__":
    main()
