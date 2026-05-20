"""Fetch TMDB metadata for every movie in data/hackathon/content/links.csv.

Run once from the repo root :
    python fetch_tmdb.py

Reads TMDB_API_KEY from .env. Writes to data/hackathon/content/tmdb_cache.json.
Incrementally cached so the script is safe to interrupt / resume.
"""
import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# ── .env loading (dotenv if available, else manual) ───────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = Path('.env')
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

API_KEY = os.environ.get('TMDB_API_KEY')
if not API_KEY:
    raise SystemExit("TMDB_API_KEY missing. Add it to .env at the repo root.")

CACHE_PATH = Path('data/hackathon/content/tmdb_cache.json')
LINKS_PATH = Path('data/hackathon/content/links.csv')
N_WORKERS = 10
SAVE_EVERY = 250


def extract_features(data):
    """Pick the useful subset from the full TMDB response (drops popularity / vote_*)."""
    if data is None or 'id' not in data:
        return None
    cast = (data.get('credits') or {}).get('cast') or []
    crew = (data.get('credits') or {}).get('crew') or []
    directors = [c.get('name') for c in crew if c.get('job') == 'Director' and c.get('name')]

    # Writers : Writing department, dedup preserving order
    seen = set()
    writers = []
    for c in crew:
        if c.get('department') == 'Writing':
            name = c.get('name')
            if name and name not in seen:
                seen.add(name)
                writers.append(name)
        if len(writers) >= 5:
            break

    kw_block = (data.get('keywords') or {})
    keywords = [k.get('name') for k in (kw_block.get('keywords') or []) if k.get('name')]
    collection = (data.get('belongs_to_collection') or {}).get('name')
    return {
        'runtime': data.get('runtime'),
        'original_language': data.get('original_language'),
        'spoken_languages': [l.get('iso_639_1') for l in (data.get('spoken_languages') or []) if l.get('iso_639_1')],
        'production_countries': [c.get('iso_3166_1') for c in (data.get('production_countries') or []) if c.get('iso_3166_1')],
        'production_companies': [c.get('name') for c in (data.get('production_companies') or []) if c.get('name')][:5],
        'directors': directors,
        'cast_top5': [c.get('name') for c in cast[:5] if c.get('name')],
        'keywords': keywords,
        'collection': collection,
        'budget': data.get('budget'),
        'overview': data.get('overview') or '',
        'tagline': data.get('tagline') or '',
        'release_date': data.get('release_date'),
        'writers': writers,
    }


def needs_refetch(entry):
    """True if cache entry is from an older schema (missing newer fields)."""
    if entry is None:
        return False  # 404 stays 404
    return 'overview' not in entry


def fetch_one(tmdb_id, session, retries=4):
    url = f'https://api.themoviedb.org/3/movie/{tmdb_id}'
    params = {'api_key': API_KEY, 'append_to_response': 'credits,keywords'}
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code == 429:
                wait = int(r.headers.get('Retry-After', '1'))
                time.sleep(wait + 0.2)
                continue
            if r.status_code == 404:
                return None
            if r.status_code >= 500:
                time.sleep(0.4 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.4 * (attempt + 1))
        except Exception:
            return None
    return None


def main():
    df_links = pd.read_csv(LINKS_PATH).dropna(subset=['tmdbId'])
    df_links['tmdbId'] = df_links['tmdbId'].astype(int)

    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    print(f'Existing cache: {len(cache)} entries')

    todo = []
    for _, row in df_links.iterrows():
        movie_id_str = str(int(row['movieId']))
        if movie_id_str in cache and not needs_refetch(cache[movie_id_str]):
            continue
        todo.append((movie_id_str, int(row['tmdbId'])))

    print(f'To fetch: {len(todo)} movies')
    if not todo:
        n_with_data = sum(1 for v in cache.values() if v is not None)
        print(f'Done. {n_with_data}/{len(cache)} entries with TMDB data.')
        return

    session = requests.Session()
    start = time.time()

    def worker(item):
        movie_id, tmdb_id = item
        data = fetch_one(tmdb_id, session)
        return movie_id, extract_features(data)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(worker, item): item for item in todo}
        for i, future in enumerate(as_completed(futures), 1):
            movie_id, features = future.result()
            cache[movie_id] = features

            if i % SAVE_EVERY == 0 or i == len(todo):
                with open(CACHE_PATH, 'w') as f:
                    json.dump(cache, f)
                elapsed = time.time() - start
                rate = i / elapsed
                eta_min = (len(todo) - i) / rate / 60 if rate > 0 else 0
                print(f'[{i}/{len(todo)}] rate={rate:.1f}/s, ETA={eta_min:.1f} min')

    n_success = sum(1 for v in cache.values() if v is not None)
    print(f'\nDone. Cache: {len(cache)} entries, {n_success} with TMDB data.')


if __name__ == '__main__':
    main()
