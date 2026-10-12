"""Scrape fresh Google Play reviews for generalization testing (step 9).

Pulls 500+ recent reviews for three apps that are NOT in the training set,
writes one combined CSV to data/raw/scraped_reviews.csv.

Usage:
    python scripts/scrape_reviews.py
    python scripts/scrape_reviews.py --apps com.spotify.music com.nianticlabs.pokemongo ...

google-play-scraper is an unofficial API; it is rate-limited and can break
when the store changes. We throttle, retry once per page, and de-duplicate.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import DATA_RAW  # noqa: E402

from google_play_scraper import reviews, Sort  # noqa: E402

# Three apps deliberately outside the Kaggle training corpora (prakharrathi25
# and lava18 both skew toward the top-10k free US apps).
DEFAULT_APPS = [
    "com.spotify.music",          # Spotify
    "com.nianticlabs.pokemongo",  # Pokemon GO
    "com.discord",                # Discord
]

TARGET_PER_APP = 200   # 3 apps x 200 = 600 raw -> de-dupe leaves well over 500
PAGE_SIZE = 100


def scrape_app(app_id: str, count: int = TARGET_PER_APP,
               lang: str = "en", country: str = "us") -> pd.DataFrame:
    """Page through reviews for one app until ``count`` are collected."""
    collected: list[dict] = []
    continuation: str | None = None
    while len(collected) < count:
        for attempt in range(2):  # one retry per page
            try:
                result, continuation = reviews(
                    app_id,
                    lang=lang, country=country,
                    sort=Sort.NEWEST,
                    count=min(PAGE_SIZE, count - len(collected)),
                    continuation_token=continuation,
                )
                break
            except Exception as exc:  # network / schema hiccup
                print(f"  [{app_id}] page error (try {attempt+1}): {exc}")
                time.sleep(3)
                result = []
        if not result:
            break
        collected.extend(result)
        time.sleep(1.5)  # be polite to the unofficial endpoint
        if continuation is None:
            break

    rows = [{
        "app": app_id,
        "review_id": r.get("review_id"),
        "username": r.get("username"),
        "score": r.get("score"),
        "review": r.get("content"),
        "reviews_count": r.get("n_ratings"),
        "at": r.get("at"),
        "helpful_count": r.get("helpful_count"),
    } for r in collected]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", nargs="+", default=DEFAULT_APPS,
                    help="Google Play package ids to scrape.")
    ap.add_argument("--per-app", type=int, default=TARGET_PER_APP)
    ap.add_argument("--out", default=os.path.join(DATA_RAW, "scraped_reviews.csv"))
    args = ap.parse_args()

    os.makedirs(DATA_RAW, exist_ok=True)
    frames = []
    for app_id in args.apps:
        print(f"Scraping {app_id} ...")
        df = scrape_app(app_id, count=args.per_app)
        print(f"  -> {len(df)} reviews")
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["review"]).drop_duplicates(subset=["review"])
    all_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(all_df)} de-duplicated reviews -> {args.out}")
    if len(all_df) < 500:
        print("WARNING: under 500 reviews. Re-run with --per-app 300 "
              "or add more apps.")


if __name__ == "__main__":
    main()
