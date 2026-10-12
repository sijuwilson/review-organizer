"""Text preprocessing for Google Play reviews.

Pipeline (order matters):
    1. keep a copy of the original text in ``raw_text`` (for display)
    2. lowercase
    3. strip URLs, @mentions and #hashtags
    4. expand/normalise emojis -> ``:emoticon:`` style tokens (nltk)
    5. remove standalone numbers and punctuation
    6. tokenise, drop English stopwords
    7. lemmatise (WordNet)

``preprocess_dataframe`` glues all of this together and also does the
EDA cleaning: drop rows with null review text, exact duplicates, and
reviews shorter than MIN_WORDS words.
"""

from __future__ import annotations

import re
import string

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources on first use (cached in ~/nltk_data).
for _res, _pkg in [("stopwords", "stopwords"),
                   ("wordnet", "wordnet"),
                   ("averaged_perceptron_tagger", "averaged_perceptron_tagger"),
                   ("punkt", "punkt"),
                   ("punkt_tab", "punkt_tab"),
                   ("emoji_university", None)]:
    if _pkg is None:
        continue
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:  # network optional; ignore if already cached
        pass

STOP_WORDS = set(stopwords.words("english"))
# Domain-specific words that NLTK would strip but carry signal for reviews.
KEEP_WORDS = {"not", "no", "nor", "cant", "wont", "didnt", "noone", "nothing"}
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"http\S+|www\.\S+|https?://\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NUMBER_RE = re.compile(r"\b\d+\b")
EMOJI_RE = re.compile(  # broad emoji + dingbat ranges
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u2190-\u21FF\u2B00-\u2BFF]"
)

MIN_WORDS = 3  # reviews with fewer tokens than this are dropped as noise


def emoji_to_words(text: str) -> str:
    """Replace emoji characters with readable tokens.

    Uses the ``emoji`` package when available; otherwise just strips them.
    Keeping a textual trace helps clustering ("love it :heart:" groups with
    other praise) without the raw unicode polluting the vocabulary.
    """
    try:
        import emoji
        return emoji.demojize(text)
    except Exception:
        return EMOJI_RE.sub(" ", text)


def clean_text(text: str, keep_raw: bool = False) -> str:
    """Apply the cleaning pipeline above and return a space-joined string."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = URL_RE.sub(" url ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)      # drop '#', keep the tag word
    text = emoji_to_words(text)
    text = NUMBER_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    if keep_raw:
        return text

    tokens = word_tokenize(text)
    tokens = [t for t in tokens
              if t not in STOP_WORDS or t in KEEP_WORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if len(t) > 1]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame,
                         review_col: str = "review",
                         min_words: int = MIN_WORDS) -> pd.DataFrame:
    """Clean + tokenise a reviews DataFrame.

    Parameters
    ----------
    df : raw dataframe with at least ``review_col``.
    review_col : column holding the original review text.
    min_words : drop reviews shorter than this many whitespace tokens.

    Returns a new frame with columns:
        raw_text   original text (never cleaned, for display / word clouds)
        text       cleaned+lemmatised text (for features/clustering)
        + any other original columns that survive (e.g. 'score')
    """
    df = df.copy()
    df[review_col] = df[review_col].astype(str)

    # --- EDA cleaning: nulls, duplicates, very short reviews ------------
    df = df[df[review_col].notna() & (df[review_col].str.strip() != "")
            & (df[review_col].str.strip().str.lower() != "nan")]
    df = df.drop_duplicates(subset=[review_col])
    df = df[df[review_col].str.split().str.len() >= min_words]

    # --- Build clean / raw columns --------------------------------------
    out = pd.DataFrame(index=df.index)
    out["raw_text"] = df[review_col]
    out["text"] = df[review_col].map(clean_text)

    # Carry through useful metadata if present. "gold" is the hand-label column
    # the generalization spot-check compares against (predict.spot_check_accuracy).
    for col in ("score", "installs", "generated_timestamp", "translated", "gold"):
        if col in df.columns:
            out[col] = df[col]

    # Final safety net: preprocessing must not create empty strings.
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out
