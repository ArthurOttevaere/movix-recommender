
# MLSMM2156 - Recommender Systems - Group 3

## Analytics Module (`analytics.ipynb`)

This notebook is the **first module** of the recommender system project.
Its purpose is to provide a thorough exploration and descriptive statistics
of the MovieLens dataset before any modelling takes place.

---

### Goals

- Load and explore the data (movies and ratings)
- Compute descriptive statistics on users, items, and ratings
- Visualise the **long-tail property** of the rating distribution
- Assess and visualise the **sparsity** of the user–item matrix

---

### Notebook structure

#### 1. Content (`df_items`)

Loads the movie catalogue via `load_items()`.  
Information extracted:

- Total number of movies
- Release year range (min – max)
- List of available genres

#### 2. Ratings (`df_ratings`)

Loads the ratings matrix via `load_ratings()`.  

Statistics computed:

- Total number of ratings
- Number of unique users
- Number of unique rated movies
- Most/least rated movie (rating count)
- All possible rating values
- Number of movies with no ratings at all

#### 3. Long-tail Property

Visualises the distribution of rating frequencies per movie (sorted by decreasing popularity).  
Highlights that the vast majority of ratings are concentrated on a small number of popular movies.

#### 4. Matrix Sparsity

- Converts `df_ratings` into a sparse matrix (`csr_matrix`)
- Computes the sparsity score: proportion of empty cells in the users × movies matrix
- Visualises non-zero entries via `.spy()` on the first 100 users and 100 movies

---

### Available datasets

| Size    | Description                                                      |
|---------|------------------------------------------------------------------|
| `small` | Full MovieLens data — use this for model evaluation              |
| `tiny`  | Reduced version — useful for faster debugging                    |
| `test`  | Toy example (6 users, 10 items) — matches the lecture slides     |

The active dataset is configured in `constants.py`.

---

### Main dependencies

- `numpy`, `pandas` — data manipulation
- `matplotlib` — visualisations
- `scipy.sparse` — sparse matrix representation
- `constants`, `loaders` — local project modules

---

## Evaluator Module (`evaluator.ipynb`)

This notebook is the **second module** of the recommender system project.
Its purpose is to evaluate a set of recommender system models by producing a report
of relevant metrics across different validation strategies.

---

### Evaluator goals

- Load ratings data in a format compatible with the `surprise` library
- Implement three crossvalidation strategies (split, leave-one-out, full)
- Compute evaluation metrics for each strategy
- Compare baseline models on those metrics
- Export the evaluation report as a versioned CSV file

---

### Evaluator notebook structure

#### 1. Model validation functions

Three functions implement the three crossvalidation strategies:

| Function | Strategy | Output |
| --- | --- | --- |
| `generate_split_predictions` | Random 75/25 train–test split | Raw predictions on the test set |
| `generate_loo_top_n` | Leave-One-Out — one item held out per user | Top-N recommendations + LOO testset |
| `generate_full_top_n` | Full trainset — no held-out ratings | Top-N recommendations from the anti-testset |

#### 2. Evaluation metrics

Three metric categories, each tied to one validation strategy:

| Metric | Category | Description |
| --- | --- | --- |
| `mae` | split | Mean Absolute Error on rating predictions (from `surprise.accuracy`) |
| `rmse` | split | Root Mean Squared Error on rating predictions (from `surprise.accuracy`) |
| `hit_rate` | loo | Share of users for whom the hidden item appeared in the top-N list |
| `novelty` | full | Average (across users) of the sum of popularity ranks of recommended items — higher means more obscure picks |

Metrics are registered in the `AVAILABLE_METRICS` dictionary as
`metric_name : (metric_function, metric_parameters)` and declared in `EvalConfig`.

#### 3. Evaluation workflow

- `precompute_information(df_ratings)` builds `item_to_rank` (popularity rank per movie) before
  the crossvalidation loop to avoid any data leakage.
- `create_evaluation_report(eval_config, sp_ratings, precomputed_dict, available_metrics)`
  iterates over all models in `EvalConfig`, runs the three validation strategies, and returns a
  `DataFrame` where rows are models and columns are metrics.
- `export_evaluation_report(df)` saves the report to `evaluation/<YYYY_MM_DD>.csv`, one file
  per day for versioning purposes.

---

### Baseline models (`models.py`)

| Model | Class | Description |
| --- | --- | --- |
| `baseline_1` | `ModelBaseline1` | Always predicts a constant rating of 2 |
| `baseline_2` | `ModelBaseline2` | Predicts a random rating uniformly drawn from the rating scale |
| `baseline_3` | `ModelBaseline3` | Predicts the global mean rating of the training set |
| `baseline_4` | `ModelBaseline4` | SVD with 100 latent factors (`random_state=1` for reproducibility) |
| `baseline_5` | `ModelBaseline5` | KNNWithMeans using MSD baseline similarity and user-based CF |
| `user_based` | `UserBased` | Custom User-Based CF algorithm with custom similarity metrics |
| `content_based`| `ContentBased` | Learns a specific regression model per user based on item features |

`get_top_n(predictions, n)` converts a list of `Prediction` objects into a per-user
top-N dictionary, using random tie-breaking.

---

### Configuration (`configs.py` / `constants.py`)

`EvalConfig` centralises all evaluation hyper-parameters:

| Parameter | Value | Role |
| --- | --- | --- |
| `test_size` | `0.25` | Fraction of ratings reserved for the split test set |
| `top_n_value` | `40` | Length of the recommendation list for LOO and full evaluations |
| `split_metrics` | `["mae", "rmse"]` | Metrics computed on split predictions |
| `loo_metrics` | `["hit_rate"]` | Metrics computed on LOO top-N recommendations |
| `full_metrics` | `["novelty"]` | Metrics computed on full top-N recommendations |

`constants.py` additions:

| Constant | Value | Role |
| --- | --- | --- |
| `RATINGS_SCALE` | `(0.5, 5.0)` | Rating scale passed to the `surprise` `Reader` |
| `EVALUATION_PATH` | `Path('evaluation')` | Directory where evaluation reports are saved |

---

### New `loaders.py` additions

- `load_ratings(surprise_format=False)` — when `surprise_format=True`, wraps the pandas
  DataFrame in a `surprise.Dataset` using `Dataset.load_from_df()` and `C.RATINGS_SCALE`.
- `export_evaluation_report(df)` — saves the evaluation `DataFrame` to
  `evaluation/<YYYY_MM_DD>.csv`, creating the directory if needed.

---

### Additional notebook sections

The `evaluator.ipynb` notebook also contains two additional discussion sections:

- **Pitfalls of using a sum of ranks as a novelty metric** — discusses the limitations of the `get_novelty` implementation (sensitivity to N, linearity of ranks, quality–novelty trade-off, dependence on catalogue size).
- **Evaluation report observations** — comments on the results produced by the four baseline models (precision metrics, hit rate at 1.0 on the test dataset, and identical novelty scores).

---

### Evaluator dependencies

- `numpy`, `pandas` — data manipulation
- `surprise` — crossvalidation, accuracy metrics, and algorithm base classes
- `constants`, `loaders`, `models`, `configs` — local project modules

---

## User-Based Module (`user_based.ipynb`)

This notebook is the **third module** of the recommender system project.
Its purpose is to implement a customizable user-based collaborative filtering algorithm
with support for custom similarity metrics, starting from Surprise's built-in `KNNWithMeans`
and culminating in a self-made `UserBased` class.

---

### User-Based Goals

- Load ratings in Surprise format and build a full trainset + anti-testset
- Explore Surprise's `KNNWithMeans` and understand the effect of its parameters
- Implement a custom `UserBased` class that replicates `KNNWithMeans` exactly
- Extend the class with a Jaccard similarity metric and compare predictions

---

### User-Based Notebook structure

#### 1. Loading Data

Data is loaded via `load_ratings()`, wrapped in a Surprise `Dataset` using `Reader(rating_scale=C.RATINGS_SCALE)`. The notebook works on the **test dataset** (6 users, 10 items) to facilitate debugging. A full trainset (`build_full_trainset`) and an anti-testset are built.

#### 2. Exploring Surprise's `KNNWithMeans`

Configures and trains `KNNWithMeans` with the following options:

| Parameter | Value | Role |
| --- | --- | --- |
| `k` | `3` | Maximum peer group size |
| `min_k` | `2` | Minimum neighbors required for weighted average (fallback: user mean) |
| `name` | `'msd'` | Similarity metric (Mean Squared Difference) |
| `min_support` | `3` | Minimum co-rated items required between two users |
| `user_based` | `True` | User-based (vs. item-based) mode |

**Reference prediction:** user 11, item 364 → **Est = 2.49**

**Key observations:**

- Increasing `min_k` causes more estimates to fall back to the user's mean rating (fewer pairs meet the minimum neighbor threshold).
- Increasing `min_support` reduces `actual_k` (the number of neighbors effectively used) because fewer user pairs share enough co-rated items to obtain a non-zero similarity.

#### 3. Custom `UserBased` class

Extends `AlgoBase` from Surprise. Four core methods are implemented:

| Method | Description |
| --- | --- |
| `compute_rating_matrix()` | Builds an m×n NumPy array (NaN for missing ratings) from `trainset.ur` |
| `compute_similarity_matrix()` | Builds an m×m similarity matrix (eye initialisation); supports `msd` and `jacard` metrics; enforces `min_support` |
| `fit()` | Calls the two above, then computes per-user mean ratings |
| `estimate(u, i)` | Builds the peer group via `trainset.ir[i]`, selects top-k neighbors with `heapq.nlargest`, and computes a similarity-weighted average deviation from neighbor means |

**MSD similarity** (replicates Surprise): `sim = 1 / (msd + 1)` where `msd` is the mean squared difference over co-rated items.

**Jaccard similarity**: `sim = |intersection| / |union|` over rated item sets, where intersection = items rated by both users and union = items rated by at least one.

#### 4. Validation against `KNNWithMeans`

With identical parameters (`msd`, `min_support=3`, `k=3`, `min_k=2`), `UserBased` produces **bit-for-bit identical predictions** to `KNNWithMeans` across all 30 anti-testset entries.

#### 5. MSD vs. Jaccard comparison

Running `UserBased` with each metric shows that Jaccard can produce different estimates because it measures overlap in the set of rated items rather than rating-value proximity. Example for (user 11, item 364):

| Metric  | Estimate |
| ------- | -------- |
| MSD     | 2.4920   |
| Jaccard | 2.1667   |

---

### User-Based dependencies

- `numpy`, `heapq` — matrix operations and efficient top-k selection
- `surprise` (`AlgoBase`, `KNNWithMeans`, `PredictionImpossible`, `Dataset`, `Reader`) — algorithm base classes and data loading
- `constants`, `loaders` — local project modules

---

## Content-Based Module (`content_based.ipynb`)

This notebook is the **fourth module** of the recommender system project.
Its purpose is to extract item features (from MovieLens and TMDB) and train a personalised regression model (e.g. Ridge Regression, Random Forest) per user to predict ratings based on item features.

---

### Content-Based Goals

- Extract and pre-process content features (genome scores, tags, genres, release year, TMDB metadata, embeddings)
- Build a unified feature matrix representing items dynamically using different strategies
- Implement a `ContentBased` algorithm that learns a specific regressor per user based on their historical ratings and the features of the items they rated
- Compare the performance of various feature sets and regression strategies (e.g., Ridge CV, Stacked Models, ElasticNet)

---

### Content-Based Notebook structure

#### 1. Feature Extraction and Preprocessing

The notebook investigates several approaches to build item representations:
- **MovieLens features**: `genome-scores`, TF-IDF on user `tags`, one-hot/TF-IDF on `genres`, and normalised release year.
- **TMDB metadata**: Additional features extracted via TMDB API (budget, runtime, cast, crew, etc.).
- **Visuals and Embeddings**: Precomputed visual features and text embeddings from `Sentence-Transformer`.
Various `features_method` parameters (e.g., `all_content`, `all_content_tmdb`, `all_content_v2`) combine these blocks into a unified item-feature matrix. 

#### 2. Custom `ContentBased` class

Extends `AlgoBase` from Surprise. Key components include:

| Component | Description |
| --- | --- |
| `create_content_features()` | Assembles the global item-feature matrix based on the selected `features_method` |
| `_build_user_frame(u)` | Extracts the specific training data (X, y) for a given user `u` based on the items they have rated |
| `fit()` | Iterates over all users and trains a personalised regression model (e.g. `RidgeCV`, `RandomForestRegressor`) using their individual `user_frame` |
| `estimate(u, i)` | Predicts a rating by passing the features of item `i` to the pre-trained regressor of user `u` |

The class also supports user-level explainability via `self.user_profile_explain` and the `explain(u)` method. This computes a weighted average of the features a user has rated and returns normalized importance scores for each feature, making the content-based predictions interpretable.

#### 3. Regression Strategies

The algorithm supports different regression methods to model user profiles (`regressor_method`):
- Simple baselines (`linear_regression`, `ridge`, `random_forest`)
- Cross-validated linear models (`ridge_cv`, `elastic_cv`)
- Advanced approaches (`stacking_groups`, `ridge_knn_blend`)

#### 4. Model Evaluation

The notebook explores the impact of different combinations of `features_method` and `regressor_method` on rating prediction accuracy using evaluation strategies previously defined in the `evaluator.ipynb` module.

---

### Content-Based dependencies

- `numpy`, `pandas` — data manipulation
- `scikit-learn` — regression models (`RidgeCV`, `RandomForestRegressor`, etc.), TF-IDF, preprocessing, and K-Fold cross-validation
- `surprise` — algorithm base classes (`AlgoBase`)
- `constants`, `loaders` — local project modules

---

## Hackathon Submission Module (`hackathon_make_predictions.ipynb`)

This notebook is built around the hackathon workflow for the course dataset in `data/hackathon`.
It provides a reproducible submission pipeline using `ContentBased` models and a helper to send predictions to the hackathon API.

### Hackathon workflow

- Configure the repository root `.env` file with `HACKATHON_URL` and `HACKATHON_TOKEN`.
- Use `load_ratings(surprise_format=True)` after setting `C.DATA_PATH = Path('data/hackathon')` to load the hackathon training data.
- Train a `ContentBased(feature_method, regressor_method)` model on the full hackathon train set.
- Produce the submission file with `make_hackathon_prediction(...)`, which returns a DataFrame ordered as required by the hidden test set.
- Submit with `submit_predictions(df_predictions)` and optionally query remaining quota with `check_quota()`.

### Hackathon helper

- `python_helper/hackathon_submit.py` provides `submit_predictions(df)` and `check_quota()`.
- It auto-loads environment variables from `.env` when available, so you can keep credentials out of version control.
- The helper validates that the submission DataFrame has exactly the columns `['userId', 'movieId', 'rating']` in that order.

### Best current hackathon configuration

The notebook records a top current submission strategy based on content-rich features and cross-validated Ridge regression:

- `feature_method='all_content_tmdb_tags2000'`
- `regressor_method='ridge_cv'`

This combination is the current recommended starting point for hackathon predictions.
