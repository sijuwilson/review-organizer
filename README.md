# Review Organizer

Groups Google Play Store app reviews into meaningful categories using NLP and
clustering, trains a supervised classifier on those categories, and serves
results in a Streamlit app.

---

## 1. Problem statement

App developers get thousands of free-text reviews with a 1–5 star rating but no
structure. Star ratings tell you *that* people are unhappy, not *what about*.
This project turns raw review text into a small set of interpretable categories
(e.g. *app crashes*, *ads*, *payment issues*, *UI/design*, *login problems*,
*positive feedback*) **without hand-labeling at scale**, then trains a fast
classifier so new reviews can be auto-routed to a category.

Two-stage design:
1. **Unsupervised** — embed reviews and cluster them to *discover* categories.
2. **Supervised** — treat cluster assignments as labels, correct a sample by
   hand, and train a lightweight classifier that generalises to new apps.

## 2. Datasets

| Purpose | Source | File (put in `data/raw/`) |
|---|---|---|
| Training | Kaggle *Google Play Store Reviews* (`prakharrathi25`) | `User Reviews.csv` |
| Training (alt) | Kaggle *google-play-store-apps* user reviews (`lava18`) | `user_reviews.csv` |
| Extra test | `scripts/scrape_reviews.py` (google-play-scraper) | `data/raw/scraped_reviews.csv` |

The scrape script pulls 500+ fresh reviews for **3 apps not in the training
corpora** (Spotify, Pokémon GO, Discord by default) to test generalisation.

## 3. Method / pipeline

```
raw CSV ──EDA (01)──▶ cleaned text ──features──▶ TF-IDF / MiniLM embeddings
                                                      │
                                        clustering (02): KMeans vs Agglomerative
                                        vs DBSCAN vs LDA  →  category labels
                                                      │
                            name clusters (config/cluster_names.json, editable)
                                                      │
                    manual fix 300-review sample → labeled_sample.csv
                                                      │
                        supervised (03): LR / LinearSVM / RF / NB × both feature
                        sets, RandomizedSearchCV 5-fold → best pipeline (joblib)
                                                      │
                sentiment per category (VADER) · generalisation test · Streamlit app
                                                      │
                                    optional DistilBERT fine-tune (04, Colab GPU)
```

Reproducibility: single fixed seed (`src.SEED = 42`) applied to Python/NumPy/
Torch via `set_seed()`; splits and clustering are stratified/deterministic.

## 4. Project structure

```
review-organizer/
  data/raw/                training CSVs + scraped_reviews.csv
  data/processed/          reviews_clean.csv, reviews_clustered.csv,
                           labeled_sample.csv, figures/
  notebooks/  01_eda  02_clustering  03_classification  04_bert_colab
  src/        preprocess  features  cluster  classify  evaluate  predict
  models/     tfidf_vectorizer.joblib, best_*_pipeline.joblib
  config/     cluster_names.json   ← edit category names here
  scripts/    scrape_reviews.py
  app.py      Streamlit UI
  requirements.txt
```

## 5. How to run

```bash
# 0. env (Python 3.10+; CPU-only stack)
python -m venv .venv && source .venv/bin/activate
# torch first, from the CPU index — the default PyPI wheel drags in ~5GB of
# nvidia-*-cu12 CUDA runtime libs that this project never uses
pip install --index-url https://download.pytorch.org/whl/cpu torch
# TMPDIR: pip unpacks wheels in /tmp, which on many Linux setups is a small
# quota-limited tmpfs that fails with OSError [Errno 122] Disk quota exceeded
TMPDIR=$HOME/.tmp pip install -r requirements.txt
# first-run NLTK data downloads automatically; or pre-fetch:
python -m nltk.downloader stopwords wordnet punkt punkt_tab vader_lexicon

# 1. drop Kaggle CSV(s) into data/raw/ , then run notebooks in order
jupyter lab notebooks/01_eda.ipynb          # EDA + cleaning
jupyter lab notebooks/02_clustering.ipynb   # cluster + name + export sample
#   -> edit config/cluster_names.json and data/processed/labeled_sample.csv
jupyter lab notebooks/03_classification.ipynb  # train/compare/save best

# optional GPU step
#   open notebooks/04_bert_colab.ipynb in Google Colab (T4 runtime)

# generalisation scrape (used by 03 and the app)
python scripts/scrape_reviews.py

# serve
streamlit run app.py
```

## 6. Results

> Fill these in after running notebooks 02/03/04.

### Clustering comparison (embeddings, chosen k)
| Method | k | Silhouette ↑ | Davies-Bouldin ↓ | Coherence (c_v) |
|---|---|---|---|---|
| KMeans |  |  |  | n/a |
| Agglomerative |  |  |  | n/a |
| DBSCAN |  |  |  | n/a |
| LDA |  | n/a | n/a |  |

### Classifier comparison (test set)
| Features | Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| TF-IDF | Logistic Regression |  |  |  |  |
| TF-IDF | Linear SVM |  |  |  |  |
| TF-IDF | Random Forest |  |  |  |  |
| TF-IDF | Naive Bayes |  |  |  |  |
| Embeddings | Logistic Regression |  |  |  |  |
| Embeddings | Linear SVM |  |  |  |  |
| Embeddings | Random Forest |  |  |  |  |
| Embeddings | Naive Bayes |  |  |  |  |
| DistilBERT (Colab) | fine-tuned |  |  |  |  |

### Generalisation (fresh scraped apps)
| Metric | Value |
|---|---|
| Category distribution (top 3) |  |
| Manual spot-check accuracy (100 reviews) |  |

### Screenshots
_Add to `data/processed/figures/` and reference below._
- `![EDA](data/processed/figures/wordcloud_all_reviews.png)`
- `![Sentiment](data/processed/figures/sentiment_by_category.png)`
- `![Confusion](data/processed/figures/confusion_best.png)`

## 7. Limitations

- Cluster labels are a weak supervision source; the 300-review manual pass only
  partly removes noise, so reported classifier metrics are optimistic on rare
  categories.
- Category taxonomy is fixed to one granularity; a review mentioning both
  "crashes" and "great UI" gets a single label (multi-label is out of scope).
- `google-play-scraper` is an unofficial API — rate-limited, English/US biased,
  and can break when the store changes.
- VADER is lexicon-based and rough on domain sarcasm ("great, another crash 🙃").
- Sentiment uses star-less text where ratings exist; the two can disagree.

## 8. Future work

- Multi-label / hierarchical categories (aspect-based sentiment, ABSA).
- Online/incremental clustering so new apps extend rather than retrain.
- Active learning to prioritise which reviews a human should label.
- Quantise the embedding model (MiniLM → ONNX) for a lighter serving path.
- Evaluate against a held-out, fully human-labelled gold set.
