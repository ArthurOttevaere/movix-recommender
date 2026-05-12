"""Encode (title + tagline + overview) into 384-d sentence embeddings.

Run after fetch_tmdb.py :
    pip install sentence-transformers
    python encode_overviews.py

Writes data/hackathon/content/tmdb_embeddings.npz that is loaded by
ContentBased._load_overview_embeddings().
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_PATH = Path('data/hackathon/content/tmdb_cache.json')
EMB_PATH = Path('data/hackathon/content/tmdb_embeddings.npz')
MOVIES_PATH = Path('data/hackathon/content/movies.csv')
MODEL_NAME = 'all-MiniLM-L6-v2'  # 90 MB, 384-d output


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "sentence-transformers not installed.\n"
            "Run:  pip install sentence-transformers"
        )

    if not CACHE_PATH.exists():
        raise SystemExit(f"{CACHE_PATH} not found. Run python fetch_tmdb.py first.")

    print('Loading TMDB cache and movies catalog...')
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    df_movies = pd.read_csv(MOVIES_PATH).set_index('movieId')

    movie_ids, texts = [], []
    n_skipped = 0
    for k, v in cache.items():
        if v is None:
            n_skipped += 1
            continue
        mid = int(k)
        title = str(df_movies.title.get(mid, '')).strip()
        overview = (v.get('overview') or '').strip()
        tagline = (v.get('tagline') or '').strip()
        if not (title or overview or tagline):
            n_skipped += 1
            continue
        # Compose : title takes always; tagline acts as a short hook; overview is the plot
        parts = [title]
        if tagline:
            parts.append(tagline)
        if overview:
            parts.append(overview)
        text = '. '.join(parts).strip()
        movie_ids.append(mid)
        texts.append(text)
    print(f'Texts to encode: {len(texts)} (skipped {n_skipped} empty/missing)')

    print(f'Loading model {MODEL_NAME} (downloads on first run)...')
    model = SentenceTransformer(MODEL_NAME)

    print('Encoding...')
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    print(f'Saving to {EMB_PATH}')
    np.savez_compressed(
        EMB_PATH,
        movie_ids=np.array(movie_ids, dtype=np.int64),
        embeddings=embeddings,
    )
    print(f'Done. Shape: {embeddings.shape}, file: {EMB_PATH.stat().st_size / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()
