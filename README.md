# Movix — A Movie Recommender System

*Recommender Systems — MLSMM2156, Q2 2026 - Andry Lenny, El Mohcine Moahmed Amine & Ottevaere Arthur*

---

Movix is an academic recommender-systems project built on the
[MovieLens](https://grouplens.org/datasets/movielens/) dataset, enriched with
[TMDB](https://www.themoviedb.org/) metadata. It pairs a **real-time recommendation
web app** with an **offline evaluation pipeline**, so the same models can be both
*served* to users and *measured* against standard RecSys metrics.

A new user rates a handful of films during onboarding; the backend then folds that
user into several pre-trained models and returns personalized, themed carousels —
much like a streaming service home page.

---

## What it does

- **Onboarding → instant recommendations.** A new user rates a few movies; their
  taste vector is estimated on the fly (folding-in) without retraining any model.
- **Five recommendation carousels**, each backed by a different model (see below).
- **Content-aware UI**: posters, metadata, "More Like This", a watchlist, a profile
  page with taste statistics, and an optional LLM-powered search assistant.
- **A reproducible evaluation pipeline** (`evaluator.ipynb`) reporting rating
  accuracy, ranking quality, catalogue coverage, novelty and diversity.

### The models

| Carousel | Model | Algorithm |
| --- | --- | --- |
| *Recommended for You* | **Content-based** | Per-user `RidgeCV` regression on content features (genome tags + TMDB) |
| *Viewers Like You Also Watched* | **User-based kNN** | Surprise `KNNBaseline`, Pearson-baseline similarity, popularity re-ranking |
| *Top Picks For You* | **iALS** | Implicit Alternating Least Squares (weighted matrix factorization) |
| *Discover Something New* | **BPR** | Bayesian Personalized Ranking + novelty re-ranking (β = 0.2) |
| *Trending* | live TMDB feed | (not a learned model) |

The offline pipeline additionally evaluates a **custom Jaccard** user-based variant
implemented from scratch (a similarity measure not provided by Surprise).

> **Note on serving.** A live user is not in the training set, so the backend
> approximates their latent vector via a closed-form *folding-in* step. The serving
> models therefore reproduce the *algorithms* evaluated offline, not their exact
> numeric scores.

---

## Project structure

```
.
├── backend/                  FastAPI app — serves the API *and* the frontend
│   ├── main.py               endpoints, startup, static mount
│   ├── models/               serving models: content, userbased, ials, bpr
│   ├── artifacts/            trained model files (gitignored — see "Data")
│   ├── store.py              in-memory user store + "Lenny" demo profile
│   └── utils.py              shared lookups (movies, links, popularity)
├── frontend/                 static web UI (vanilla HTML / CSS / JS)
│   ├── *.html, js/, css/
│   └── config.js             public config; real keys go in config.local.js
├── models.py                 model classes used for offline training/evaluation
├── configs.py                evaluation configuration (models + metrics)
├── evaluator.ipynb           offline evaluation pipeline
├── analytics.ipynb           dataset exploration & descriptive statistics
├── *_based.ipynb / latent_factor.ipynb   per-model development notebooks
├── recommender_building.py   builds implicit ratings (powers the demo profile)
├── loaders.py / constants.py data loading + dataset configuration
├── library_lenny.csv         demo user library
└── requirements.txt
```

---

## Getting started

### 1. Prerequisites
- Python 3.10+ (developed on 3.12)
- **To run the app:** nothing else — the trained models are fetched in step 3.
- **To retrain or run the evaluation:** the MovieLens/TMDB dataset under `data/`
  (see [Data](#data)).

### 2. Install
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the trained models
The trained models live in `backend/artifacts/`, but they are too large for git, so
they are published as a single archive on the project's
[GitHub Release](https://github.com/ArthurOttevaere/Recommender_System_Assignments/releases/tag/v1.0-artifacts).
**You don't need the dataset for this — just run:**
```bash
python download_artifacts.py
```
That's the whole procedure. The script:
- downloads ~54 MB from the Release (the URL is **already configured** — nothing to set up),
- unpacks the model files into `backend/artifacts/`,
- is safe to re-run (already-present files are skipped; use `--force` to re-download).

The downloaded files are **byte-identical** to the authors' trained models, so model
performance is exactly the same — no training happens on your machine.

### 4. Run the app
The backend serves both the REST API and the web UI from a single process:
```bash
uvicorn backend.main:app --reload
```
Open the **localhost link** given in the terminal and pick the **"Lenny"** demo
profile to see a fully populated home page immediately, or create a profile and go
through onboarding.

> On startup the backend loads trained artifacts from `backend/artifacts/`. If an
> artifact is missing, that model gracefully falls back to a popularity baseline, so
> the app still runs — but for full personalization, make sure step 3 succeeded (or
> regenerate the artifacts from the modeling notebooks).

### 5. (Recommended) external API keys
The app runs without keys, but **for the full experience** — real movie posters
and the LLM-powered search assistant — add your own free API keys:

```bash
cp frontend/config.local.js.example frontend/config.local.js
```
Then open `frontend/config.local.js` and fill in:

| Key | Where to get it (free) | Enables |
| --- | --- | --- |
| `TMDB_API_KEY` | https://www.themoviedb.org/settings/api | Real posters & movie metadata |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | LLM search assistant (optional; falls back to a keyword matcher) |

`config.local.js` is **gitignored** — your keys stay local and are never committed.
Without these keys the app still works, but posters show placeholders and the
search assistant uses a basic keyword fallback.

### 6. Offline evaluation
Open `evaluator.ipynb` and run it. Models and metrics are configured in
`configs.py`; results are reported per model in a single table.

---

## Data

Movix uses the course "hackathon" MovieLens dataset enriched with TMDB content.
Files are expected under the path defined in `constants.py` (`data/hackathon/`):

```
data/hackathon/
├── content/        movies.csv, links.csv, genome-scores.csv, tags.csv,
│                   tmdb_embeddings.npz, tmdb_cache.json, …
└── evidence/       ratings.csv  (userId, movieId, rating, timestamp)
```

A helper, `unzip_data.py`, extracts the provided archive. Ratings use the
**0.5–5.0** scale.

Trained model artifacts (`*.pkl`, `*.npz`) live in `backend/artifacts/` and are
**gitignored** (too large for git). End users simply run `python download_artifacts.py`
(step 3); the files can also be regenerated from the modeling notebooks.

*Maintainers — to publish an updated set:*
1. Zip the eight required files into `artifacts.zip`:
   ```bash
   cd backend/artifacts && zip -1 -j artifacts.zip \
     content_features.pkl ials_model.pkl bpr_model.pkl userbased_model.pkl \
     rating_matrix.npz movies.csv links.csv popularity.csv
   ```
2. Attach `artifacts.zip` to a (public) GitHub Release.
3. Update `ARTIFACTS_URL` at the top of `download_artifacts.py` with the new asset link.

> Important : the data folder can not be found in the repository due to size constraints. Therefore it can be directly downloaded via this [link](https://drive.google.com/drive/folders/1kgFw3-6LNSjYfG2pesnUn81kON82n8rx?usp=share_link). Please make sure to locate it at the root of the project.

---

## Evaluation metrics

The pipeline reports complementary metric families (no single model wins all):

- **Rating accuracy** — RMSE, MAE (meaningful only for rating-prediction models).
- **Ranking** — Hit-Rate@K and NDCG@K, both over the full catalogue and under a
  negative-sampling protocol (1 positive vs. 99 negatives).
- **Beyond-accuracy** — catalogue **coverage**, **MIUF** (novelty) and **ILD**
  (intra-list diversity).

---

## Tech stack

FastAPI · scikit-surprise · implicit · scikit-learn · NumPy / pandas / SciPy ·
Jupyter · vanilla JavaScript frontend.

## Acknowledgments

Built on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset
(GroupLens) with metadata from [The Movie Database (TMDB)](https://www.themoviedb.org/).
Developed as part of the MLSMM2156 Recommender Systems course.
