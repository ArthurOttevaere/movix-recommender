# standard library imports
from collections import defaultdict

# third parties imports
import numpy as np
import random as rd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD, SVDpp
from surprise.prediction_algorithms.predictions import PredictionImpossible  # new import
from constants import Constant as C  # new import
from loaders import load_items  # new import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold

import heapq



def get_top_n(predictions, n):
    """Return the top-N recommendation for each user from a set of predictions.
    Source: inspired by https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    and modified by cvandekerckh for random tie breaking

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    rd.seed(0)

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        rd.shuffle(user_ratings)
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First algorithm
class ModelBaseline1(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        return 2


# Second algorithm
class ModelBaseline2(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        rd.seed(0)

    def estimate(self, u, i):
        return rd.uniform(self.trainset.rating_scale[0], self.trainset.rating_scale[1])


# Third algorithm
class ModelBaseline3(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

        return self

    def estimate(self, u, i):
        return self.the_mean


# Fourth Model
class ModelBaseline4(SVD):
    def __init__(self, random_state=1):
        SVD.__init__(self, n_factors=100, random_state=random_state)

# Latent Factor — hyperparameters tuned via GridSearchCV (latent_factor.ipynb)
class LatentFactor(SVD):
    def __init__(self, random_state=1):
        SVD.__init__(self, n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.1, random_state=random_state)

# SVD++ — extends SVD by integrating implicit feedback (all rated items, regardless of rating value)
# Hyperparameters: Koren, Y. (2008). Factorization meets the neighborhood. KDD '08, pp. 426-434.
class LatentFactorPP(SVDpp):
    def __init__(self, random_state=1):
        SVDpp.__init__(self, n_factors=20, n_epochs=20, lr_all=0.007, reg_all=0.02, random_state=random_state)

# SVD ranking v2 — hyperparameters from ranking-oriented GridSearchCV (latent_factor.ipynb §4c)
# n_factors=20 + reg_all=0.01: best RMSE(0.828)/coverage trade-off — reg halved vs v1.
class LatentFactorRanking2(SVD):
    def __init__(self, random_state=1):
        SVD.__init__(self, n_factors=20, n_epochs=30, lr_all=0.005, reg_all=0.01, random_state=random_state)


# iALS — Weighted Regularized Matrix Factorization (WRMF)
# Hu Y. et al. (2008). "Collaborative Filtering for Implicit Feedback Datasets." ICDM, pp. 263-272.
# Uses explicit ratings as confidence weights: c_ui = 1 + alpha * r_ui
# A 5-star film contributes more to the optimisation than a 1-star film.
# ALS training — very fast (~30 sec on ML-100K).
class ModeliALS(AlgoBase):
    def __init__(self, factors=50, iterations=20, regularization=0.01,
                 alpha=40, random_state=42):
        AlgoBase.__init__(self)
        self.factors        = factors
        self.iterations     = iterations
        self.regularization = regularization
        self.alpha          = alpha   # confidence scaling — Hu et al. (2008) §3: c_ui = 1 + alpha * r_ui
        self.random_state   = random_state
        self._als           = None

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        from implicit.als import AlternatingLeastSquares
        from scipy.sparse import csr_matrix

        rows, cols, vals = [], [], []
        for u in range(trainset.n_users):
            for iid, r in trainset.ur[u]:
                rows.append(u)
                cols.append(iid)
                vals.append(1.0 + self.alpha * r)  # confidence weight c_ui = 1 + alpha * r_ui

        user_items = csr_matrix(
            (vals, (rows, cols)),
            shape=(trainset.n_users, trainset.n_items),
        )
        self._als = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
        )
        self._als.fit(user_items)
        return self

    def estimate(self, u, i):
        if self._als is None:
            raise PredictionImpossible("Model not fitted.")
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User or item unknown.")
        return float(np.dot(self._als.user_factors[u], self._als.item_factors[i]))

# BPR-MF — Bayesian Personalized Ranking with Matrix Factorisation
# Rendle et al. (2009) "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI '09.
# Optimises pairwise ranking loss: rated items ranked above unrated items.
# RMSE/MAE are not meaningful for this model — only LOO and full-catalogue metrics matter.
class ModelBPR(AlgoBase):
    def __init__(self, factors=64, learning_rate=0.01, regularization=0.01, iterations=100, random_state=42):
        AlgoBase.__init__(self)
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self._bpr = None

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        from implicit.bpr import BayesianPersonalizedRanking
        from scipy.sparse import csr_matrix
        rows, cols, vals = [], [], []
        for u in range(trainset.n_users):
            for iid, _ in trainset.ur[u]:
                rows.append(u)
                cols.append(iid)
                vals.append(1.0)
        user_items = csr_matrix((vals, (rows, cols)), shape=(trainset.n_users, trainset.n_items))
        self._bpr = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        self._bpr.fit(user_items)
        return self

    def estimate(self, u, i):
        if self._bpr is None:
            raise PredictionImpossible("Model not fitted.")
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User or item unknown.")
        return float(np.dot(self._bpr.user_factors[u], self._bpr.item_factors[i]))


# BPR-MF + Popularity Penalty for novelty — suited for "Discover Something New".
# Overrides test() to apply a log-popularity penalty to BPR scores before ranking.
# β controls the relevance/novelty trade-off: 0.0 = pure BPR, 1.0 = pure novelty.
# Source: Abdollahpouri et al. (2019). "Managing Popularity Bias in Recommender
#         Systems with Personalized Re-ranking." FLAIRS'19.
class ModelBPRNovelty(ModelBPR):
    def __init__(self, beta=0.2,
                 factors=64, learning_rate=0.01, regularization=0.01,
                 iterations=100, random_state=42):
        ModelBPR.__init__(self, factors=factors, learning_rate=learning_rate,
                          regularization=regularization, iterations=iterations,
                          random_state=random_state)
        self.beta = beta          # novelty weight: 0 = pure BPR, 1 = pure novelty
        self._log_pop = None      # log-popularity vector indexed by inner item id

    def fit(self, trainset):
        ModelBPR.fit(self, trainset)
        # popularity(i) = number of users who rated item i in the trainset
        pop = np.array([len(trainset.ir[i]) for i in range(trainset.n_items)], dtype=np.float64)
        self._log_pop = np.log1p(pop)   # log(1 + pop) — stable at 0
        return self

    def test(self, testset):
        from collections import defaultdict
        from surprise.prediction_algorithms.predictions import Prediction

        raw_preds = ModelBPR.test(self, testset)

        # Group predictions by user
        user_preds = defaultdict(list)
        for pred in raw_preds:
            user_preds[pred.uid].append(pred)

        reranked = []
        log_pop_max = float(self._log_pop.max()) if self._log_pop.max() > 0 else 1.0

        for uid, preds in user_preds.items():
            scores = np.array([p.est for p in preds])
            s_min, s_max = scores.min(), scores.max()
            norm_scores = (scores - s_min) / (s_max - s_min) if s_max > s_min else np.full(len(preds), 0.5)

            # Popularity penalty: penalise popular items proportionally
            # Items absent from the trainset (can occur in split evaluation) get penalty 0
            def _pop_penalty(raw_iid):
                try:
                    return self._log_pop[self.trainset.to_inner_iid(raw_iid)] / log_pop_max
                except ValueError:
                    return 0.0

            adjusted = np.array([
                (1 - self.beta) * norm_scores[k] - self.beta * _pop_penalty(p.iid)
                for k, p in enumerate(preds)
            ])

            for k, pred in enumerate(preds):
                reranked.append(Prediction(pred.uid, pred.iid, pred.r_ui,
                                           float(adjusted[k]), pred.details))

        return reranked


# KNN with means, using msd baseline similarity measure and user-based collaborative filtering
class ModelBaseline5(KNNWithMeans):
    def __init__(self, random_state=1):
        KNNWithMeans.__init__(self, k=3, min_k = 2, sim_options={'name': 'msd', 'user_based': True}, random_state=random_state)

# User-based model
class UserBased(AlgoBase):
    def __init__(self, k=3, min_k=1, sim_options={}, **kwargs):
        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        # 1. Computing the ratings matrix
        self.compute_rating_matrix()
        
        # 2. Compute the similarity matrix
        self.compute_similarity_matrix()
        
        # 3. Computing the mean rating of every user
        self.mean_ratings = []
        for u in range(self.trainset.n_users):
            user_ratings = [r for (_, r) in self.trainset.ur[u]]
            self.mean_ratings.append(np.mean(user_ratings))
        
        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        
        # The estimate is by default set to the user average rating
        estimate = self.mean_ratings[u]
        
        # Step 1: Create the peer group of user u for item i
        # Potential neighbor: (neighbor_inner_id, similarity_value, rating)
        potential_neighbors = []
        
        # Access ratings of item i with self.trainset.ir[i]
        for (v, r_vi) in self.trainset.ir[i]:
            if v == u:
                continue
            
            sim_uv = self.sim[u, v]
            if sim_uv > 0:
                potential_neighbors.append((v, sim_uv, r_vi))
        
        # Pick top neighbors efficiently using heapq
        # Since we want to sort by similarity_value (index 1 of the tuple), 
        # we use a lambda function for the key.
        top_neighbors = heapq.nlargest(self.k, potential_neighbors, key=lambda x: x[1])
        
        # Step 2: Compute the weighted average
        actual_k = len(top_neighbors)
        
        # If actual_k is above min_k, we add the weighted average component
        if actual_k >= self.min_k:
            weighted_sum = 0
            sum_sim = 0
            
            for (v, sim_uv, r_vi) in top_neighbors:
                # Weighted average calculation: sim * (rating - neighbor_mean)
                weighted_sum += sim_uv * (r_vi - self.mean_ratings[v])
                sum_sim += abs(sim_uv)
            
            if sum_sim > 0:
                estimate += (weighted_sum / sum_sim)
        
        return estimate

    def compute_rating_matrix(self):
        m = self.trainset.n_users
        n = self.trainset.n_items
        
        # Preallocate an mxn numpy array with NaN
        self.ratings_matrix = np.empty((m, n))
        self.ratings_matrix[:] = np.nan
        
        # Access ratings of a specific user with self.trainset.ur[uiid]
        for uiid in range(m):
            for (iiid, rating) in self.trainset.ur[uiid]:
                self.ratings_matrix[uiid, iiid] = rating

    def compute_similarity_matrix(self):
        m = self.trainset.n_users
        self.sim = np.eye(m)
        min_support = self.sim_options.get('min_support', 1)
        
        # Retrieve the similarity metric name from sim_options (default: msd)
        sim_name = self.sim_options.get('name', 'msd').lower()
        
        for i in range(m):
            for j in range(i + 1, m):
                row_i = self.ratings_matrix[i]
                row_j = self.ratings_matrix[j]
                
                # Intersection: items rated by BOTH users
                mask_intersection = ~np.isnan(row_i - row_j)
                support = np.sum(mask_intersection)
                
                if support >= min_support:
                    if sim_name == 'msd':
                        # Ta logique MSD actuelle
                        sq_diff = np.sum((row_i[mask_intersection] - row_j[mask_intersection])**2)
                        msd = sq_diff / support
                        similarity = 1 / (msd + 1)
                    
                    elif sim_name == 'jacard':
                        # Logique Jaccard : intersection / union
                        mask_union = ~np.isnan(row_i) | ~np.isnan(row_j)
                        union_count = np.sum(mask_union)
                        similarity = support / union_count if union_count > 0 else 0
                    
                    self.sim[i, j] = similarity
                    self.sim[j, i] = similarity
        
# Content-based model        
class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method, knn_k=30):
        AlgoBase.__init__(self)
        self.regressor_method = regressor_method
        self.knn_k = knn_k
        self.feature_groups = None
        self.content_features = self.create_content_features(features_method)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize_rows(matrix):
        """Row-wise L2 normalisation (zero-norm rows kept as zero)."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    def _load_genome(self):
        df_genome = pd.read_csv(C.CONTENT_PATH / 'genome-scores.csv')
        df_genome = df_genome.pivot(index='movieId', columns='tagId', values='relevance')
        return df_genome.fillna(0)

    def _load_genome_scaled(self):
        from sklearn.preprocessing import StandardScaler
        df_genome = self._load_genome()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_genome)
        return pd.DataFrame(scaled, index=df_genome.index, columns=df_genome.columns)

    def _load_genome_pruned(self, var_threshold=0.01):
        from sklearn.preprocessing import StandardScaler
        df_genome = self._load_genome()
        keep = df_genome.var(axis=0) > var_threshold
        df_genome = df_genome.loc[:, keep]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_genome)
        return pd.DataFrame(scaled, index=df_genome.index,
                            columns=[f'g_{c}' for c in df_genome.columns])

    def _load_overview_embeddings(self):
        """Load pre-computed sentence-transformer embeddings of (title + tagline + overview)."""
        emb_path = C.CONTENT_PATH / 'tmdb_embeddings.npz'
        if not emb_path.exists():
            print(f"[TMDB-EMB] {emb_path} not found. Run 'python encode_overviews.py' first.")
            return None
        data = np.load(emb_path)
        movie_ids = data['movie_ids']
        emb = data['embeddings']
        cols = [f'tmdb_emb_{i}' for i in range(emb.shape[1])]
        return pd.DataFrame(emb, index=movie_ids, columns=cols)

    def _load_visuals_scaled(self):
        from sklearn.preprocessing import StandardScaler
        visual_path = C.CONTENT_PATH / 'visuals' / 'LLVisualFeatures13K_Log.csv'
        df_visuals = pd.read_csv(visual_path)
        df_visuals = df_visuals.set_index('ML_Id')
        df_visuals = df_visuals.fillna(df_visuals.mean())
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_visuals)
        return pd.DataFrame(scaled, index=df_visuals.index, columns=df_visuals.columns)

    def _load_tmdb_features(self,
                            n_lang=30, n_country=30, n_spoken=20,
                            n_studio=100, n_director=200, n_cast=600,
                            n_keyword=300, n_collection=50,
                            n_writer=80, n_overview=500):
        """Load TMDB-derived features from local cache (run fetch_tmdb.py once)."""
        import json
        import re
        from collections import Counter

        cache_path = C.CONTENT_PATH / 'tmdb_cache.json'
        if not cache_path.exists():
            print(f"[TMDB] Cache not found at {cache_path}. Run 'python fetch_tmdb.py' first.")
            return None

        with open(cache_path) as f:
            cache = json.load(f)
        rows = [{'movieId': int(k), **v} for k, v in cache.items() if v is not None]
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index('movieId')

        blocks = []

        # 1) runtime z-scored
        runtime = pd.to_numeric(df['runtime'], errors='coerce')
        runtime = runtime.where(runtime > 0)
        runtime = runtime.fillna(runtime.mean())
        std = runtime.std() if runtime.std() > 0 else 1.0
        blocks.append(pd.DataFrame({'tmdb_runtime': (runtime - runtime.mean()) / std}))

        def safe(name):
            return re.sub(r'[^a-zA-Z0-9_]+', '_', str(name)).strip('_').lower()

        def multi_hot_topk(series, top_k, prefix):
            counts = Counter()
            for val in series:
                if isinstance(val, list):
                    counts.update(v for v in val if v)
                elif pd.notna(val) and val:
                    counts[val] += 1
            top_items = [it for it, _ in counts.most_common(top_k)]
            idx_map = {it: i for i, it in enumerate(top_items)}

            mat = np.zeros((len(series), len(top_items)), dtype=np.float32)
            for i, val in enumerate(series):
                if isinstance(val, list):
                    for v in val:
                        j = idx_map.get(v)
                        if j is not None:
                            mat[i, j] = 1.0
                elif pd.notna(val) and val:
                    j = idx_map.get(val)
                    if j is not None:
                        mat[i, j] = 1.0
            cols = [f'tmdb_{prefix}_{safe(it)}' for it in top_items]
            return pd.DataFrame(mat, index=series.index, columns=cols)

        # Standard blocks
        blocks.append(multi_hot_topk(df['original_language'], n_lang, 'lang'))
        blocks.append(multi_hot_topk(df['production_countries'], n_country, 'country'))
        blocks.append(multi_hot_topk(df['spoken_languages'], n_spoken, 'spoken'))
        blocks.append(multi_hot_topk(df['production_companies'], n_studio, 'studio'))
        blocks.append(multi_hot_topk(df['directors'], n_director, 'director'))
        blocks.append(multi_hot_topk(df['cast_top5'], n_cast, 'cast'))

        # NEW blocks (graceful if cache from old schema)
        if 'keywords' in df.columns:
            blocks.append(multi_hot_topk(df['keywords'], n_keyword, 'keyword'))
        if 'collection' in df.columns:
            blocks.append(multi_hot_topk(df['collection'], n_collection, 'collection'))
        if 'budget' in df.columns:
            budget = pd.to_numeric(df['budget'], errors='coerce')
            budget = budget.where(budget > 0)
            if budget.notna().any():
                budget_log = np.log1p(budget.fillna(budget.median()))
                std_b = budget_log.std() if budget_log.std() > 0 else 1.0
                blocks.append(pd.DataFrame({
                    'tmdb_budget_log': (budget_log - budget_log.mean()) / std_b
                }))
        if 'writers' in df.columns:
            blocks.append(multi_hot_topk(df['writers'], n_writer, 'writer'))
        if 'release_date' in df.columns:
            rd_series = pd.to_datetime(df['release_date'], errors='coerce')
            base = pd.Timestamp('2000-01-01')
            days = (rd_series - base).dt.days.astype(float)
            if days.notna().any():
                days = days.fillna(days.median())
                std_d = days.std() if days.std() > 0 else 1.0
                blocks.append(pd.DataFrame({
                    'tmdb_release_days': (days - days.mean()) / std_d
                }))
        if 'overview' in df.columns:
            overviews = df['overview'].fillna('').astype(str)
            if (overviews.str.len() > 0).sum() > 100:  # need enough text
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf_ovr = TfidfVectorizer(
                    max_features=n_overview,
                    min_df=5,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    stop_words='english'
                )
                tfidf_mat = tfidf_ovr.fit_transform(overviews)
                df_overview = pd.DataFrame(
                    tfidf_mat.toarray(),
                    index=df.index,
                    columns=[f'tmdb_ovr_{c}' for c in tfidf_ovr.get_feature_names_out()]
                )
                blocks.append(df_overview)

        result = pd.concat(blocks, axis=1)
        result = result[~result.index.duplicated(keep='first')]
        return result

    def _load_tags(self, max_features=200, ngrams=(1, 1), sublinear=False, min_df=2):
        df_tags = pd.read_csv(C.CONTENT_PATH / 'tags.csv')
        df_tags = df_tags.dropna(subset=['tag'])
        df_tags['tag'] = df_tags.tag.astype(str).str.lower()
        df_tags_grouped = df_tags.groupby('movieId')['tag'].apply(' '.join).reset_index().set_index('movieId')

        tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngrams,
            sublinear_tf=sublinear,
            stop_words='english'
        )
        tfidf_matrix = tfidf.fit_transform(df_tags_grouped['tag'])
        return pd.DataFrame(
            tfidf_matrix.toarray(),
            index=df_tags_grouped.index,
            columns=[f'tag_{c}' for c in tfidf.get_feature_names_out()]
        )

    def _load_year_genres(self, df_items, with_decades=False):
        years = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)').astype(float)
        years.columns = ['release_year']
        years = years.fillna(years.mean())
        df_year = (years - years.mean()) / years.std()

        extras = []
        if with_decades:
            bins = pd.cut(
                years['release_year'],
                bins=[0, 1970, 1980, 1990, 2000, 2010, 3000],
                labels=['era_pre70', 'era_70s', 'era_80s', 'era_90s', 'era_00s', 'era_10s']
            )
            df_decade = pd.get_dummies(bins, prefix='', prefix_sep='').astype(float)
            df_decade.index = df_items.index
            extras.append(df_decade)

            n_genres = df_items[C.GENRES_COL].fillna('').str.count(r'\|') + 1
            n_genres = (n_genres - n_genres.mean()) / n_genres.std()
            extras.append(pd.DataFrame({'n_genres': n_genres.values}, index=df_items.index))

        tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
        tfidf_matrix = tfidf.fit_transform(df_items[C.GENRES_COL].fillna(''))
        df_genres = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=df_items.index,
            columns=[f'genre_{c}' for c in tfidf.get_feature_names_out()]
        )
        if extras:
            df_year = pd.concat([df_year] + extras, axis=1)
        return df_year, df_genres

    @staticmethod
    def _scale_block(df, target_norm=1.0):
        values = df.values
        f_norm = np.linalg.norm(values, ord='fro')
        if f_norm < 1e-9:
            return df
        scale = target_norm * np.sqrt(values.shape[1]) / f_norm
        return df * scale

    # ── Content Analyzer ──────────────────────────────────────────────────

    def create_content_features(self, features_method):
        df_items = load_items()

        if features_method is None:
            return None
        
        elif features_method == "title_length":
            return df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')

        elif features_method == "genome_scaled":
            return self._load_genome_scaled()

        elif features_method == "genome_scaled_tags":
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags()
            return df_genome_scaled.join(df_tags, how='left').fillna(0)

        elif features_method == "genome_scaled_visuals_scaled":
            df_genome_scaled = self._load_genome_scaled()
            df_visuals_scaled = self._load_visuals_scaled()
            return df_genome_scaled.join(df_visuals_scaled, how='left').fillna(0)

        elif features_method == "genome_scaled_tags_visuals_scaled":
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags()
            df_visuals_scaled = self._load_visuals_scaled()
            df_features = df_genome_scaled.join(df_tags, how='left')
            df_features = df_features.join(df_visuals_scaled, how='left')
            return df_features.fillna(0)

        elif features_method == "all_content":
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags()
            df_year, df_genres = self._load_year_genres(df_items)
            df_features = df_genome_scaled.join(df_tags, how='outer')
            df_features = df_features.join(df_year, how='outer')
            df_features = df_features.join(df_genres, how='outer')
            return df_features.fillna(0)

        elif features_method == "all_content_rich_tags":
            # Same as all_content but with richer TF-IDF on tags
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags(max_features=500, ngrams=(1, 2), sublinear=True, min_df=3)
            df_year, df_genres = self._load_year_genres(df_items)
            df_features = df_genome_scaled.join(df_tags, how='outer')
            df_features = df_features.join(df_year, how='outer')
            df_features = df_features.join(df_genres, how='outer')
            return df_features.fillna(0)

        elif features_method == "all_content_decade":
            # all_content + decade one-hot ADDITIVE (normalised year kept)
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags()
            df_year, df_genres = self._load_year_genres(df_items, with_decades=True)
            df_features = df_genome_scaled.join(df_tags, how='outer')
            df_features = df_features.join(df_year, how='outer')
            df_features = df_features.join(df_genres, how='outer')
            return df_features.fillna(0)

        elif features_method == "all_content_full":
            # Combinaison : rich tags + decade additif
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags(max_features=1128, ngrams=(1, 2), sublinear=True, min_df=3)
            df_year, df_genres = self._load_year_genres(df_items, with_decades=True)
            df_features = df_genome_scaled.join(df_tags, how='outer')
            df_features = df_features.join(df_year, how='outer')
            df_features = df_features.join(df_genres, how='outer')
            return df_features.fillna(0)

        elif features_method == "all_content_tmdb":
            # all_content_full + TMDB metadata (runtime, lang, country, studio, director, cast)
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags(max_features=500, ngrams=(1, 2), sublinear=True, min_df=3)
            df_year, df_genres = self._load_year_genres(df_items, with_decades=True)
            df_tmdb = self._load_tmdb_features()
            df_features = df_genome_scaled.join(df_tags, how='outer')
            df_features = df_features.join(df_year, how='outer')
            df_features = df_features.join(df_genres, how='outer')
            if df_tmdb is not None:
                df_features = df_features.join(df_tmdb, how='outer')
            else:
                print("[all_content_tmdb] Warning: TMDB cache missing, falling back to all_content_full")
            return df_features.fillna(0)

        elif features_method == "all_content_tmdb_tags1128":
            # Pareil que all_content_tmdb mais tags TF-IDF max_features=1128 (= dim genome).
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags(max_features=1128, ngrams=(1, 2), sublinear=True, min_df=3)
            df_year, df_genres = self._load_year_genres(df_items, with_decades=True)
            df_tmdb = self._load_tmdb_features()
            df_features = df_genome_scaled.join(df_tags, how='outer')
            df_features = df_features.join(df_year, how='outer')
            df_features = df_features.join(df_genres, how='outer')
            if df_tmdb is not None:
                df_features = df_features.join(df_tmdb, how='outer')
            return df_features.fillna(0)

        elif features_method == "all_content_tags1128":
            # genome + tags(1128) + year/decade/genres, SANS TMDB (pour isoler l'effet 1128).
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags(max_features=1128, ngrams=(1, 2), sublinear=True, min_df=3)
            df_year, df_genres = self._load_year_genres(df_items, with_decades=True)
            df_features = df_genome_scaled.join(df_tags, how='outer')
            df_features = df_features.join(df_year, how='outer')
            df_features = df_features.join(df_genres, how='outer')
            return df_features.fillna(0)

        elif features_method == "genome_tags1128_only":
            # Minimal hypothesis: just genome (1128) + tags(1128). Exact match de l'indice ami.
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags(max_features=1128, ngrams=(1, 2), sublinear=True, min_df=3)
            df_features = df_genome_scaled.join(df_tags, how='outer')
            return df_features.fillna(0)

        elif features_method == "all_content_tmdb_emb_tags1128":
            # Combo : tags1128 + TMDB + SBERT embeddings. Stack des 2 gains.
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags(max_features=1128, ngrams=(1, 2), sublinear=True, min_df=3)
            df_year, df_genres = self._load_year_genres(df_items, with_decades=True)
            df_tmdb = self._load_tmdb_features()
            df_emb = self._load_overview_embeddings()
            df_features = df_genome_scaled.join(df_tags, how='outer')
            df_features = df_features.join(df_year, how='outer')
            df_features = df_features.join(df_genres, how='outer')
            if df_tmdb is not None:
                df_features = df_features.join(df_tmdb, how='outer')
            if df_emb is not None:
                df_features = df_features.join(df_emb, how='outer')
            return df_features.fillna(0)

        elif features_method == "all_content_tmdb_tags2000":
            # Pushes further: tags max=2000 (beyond genome dimension).
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags(max_features=2000, ngrams=(1, 2), sublinear=True, min_df=3)
            df_year, df_genres = self._load_year_genres(df_items, with_decades=True)
            df_tmdb = self._load_tmdb_features()
            df_features = df_genome_scaled.join(df_tags, how='outer')
            df_features = df_features.join(df_year, how='outer')
            df_features = df_features.join(df_genres, how='outer')
            if df_tmdb is not None:
                df_features = df_features.join(df_tmdb, how='outer')
            return df_features.fillna(0)

        elif features_method == "all_content_tmdb_emb":
            # all_content_tmdb + 384-d Sentence-Transformer embeddings on (title+tagline+overview)
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags(max_features=500, ngrams=(1, 2), sublinear=True, min_df=3)
            df_year, df_genres = self._load_year_genres(df_items, with_decades=True)
            df_tmdb = self._load_tmdb_features()
            df_emb = self._load_overview_embeddings()
            df_features = df_genome_scaled.join(df_tags, how='outer')
            df_features = df_features.join(df_year, how='outer')
            df_features = df_features.join(df_genres, how='outer')
            if df_tmdb is not None:
                df_features = df_features.join(df_tmdb, how='outer')
            if df_emb is not None:
                df_features = df_features.join(df_emb, how='outer')
            else:
                print("[all_content_tmdb_emb] Warning: embeddings missing, falling back to all_content_tmdb")
            return df_features.fillna(0)

        elif features_method == "all_content_v2":
            df_genome = self._load_genome_pruned(var_threshold=0.01)
            df_tags = self._load_tags(max_features=1000, ngrams=(1, 2), sublinear=True, min_df=3)
            df_year, df_genres = self._load_year_genres(df_items, with_decades=True)

            df_genome = self._scale_block(df_genome.fillna(0))
            df_tags = self._scale_block(df_tags.fillna(0))
            df_year = self._scale_block(df_year.fillna(0))
            df_genres = self._scale_block(df_genres.fillna(0))

            df_features = df_genome.join(df_tags, how='outer')
            df_features = df_features.join(df_year, how='outer')
            df_features = df_features.join(df_genres, how='outer')
            df_features = df_features.fillna(0)

            cols = list(df_features.columns)
            self.feature_groups = {
                'genome': [c for c in cols if c.startswith('g_')],
                'tags': [c for c in cols if c.startswith('tag_')],
                'year': [c for c in cols if c.startswith('release_year') or c.startswith('era_') or c == 'n_genres'],
                'genres': [c for c in cols if c.startswith('genre_')],
            }
            return df_features

        else:
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')

    # ── Profile Learner ───────────────────────────────────────────────────

    def _build_user_frame(self, u):
        feature_names = list(self.content_features.columns)
        df_user = pd.DataFrame(self.trainset.ur[u], columns=['item_id', 'user_ratings'])
        df_user['item_id'] = df_user['item_id'].map(self.trainset.to_raw_iid)
        df_user = df_user.merge(self.content_features, how='left',
                                left_on='item_id', right_index=True)
        df_user = df_user.dropna(subset=feature_names, how='all').fillna(0)
        if len(df_user) < 2:
            return None, None
        return df_user[feature_names].values, df_user['user_ratings'].values

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.user_profile = {u: None for u in trainset.all_users()}
        # Prepare a per-user explain vector (feature importance percentages)
        self.user_profile_explain = {u: None for u in trainset.all_users()}
        if self.content_features is not None:
            feature_names = list(self.content_features.columns)
            for u in trainset.all_users():
                X, y = self._build_user_frame(u)
                if X is None:
                    self.user_profile_explain[u] = None
                    continue
                weights = np.array(y, dtype=np.float64)
                if weights.sum() <= 1e-9:
                    self.user_profile_explain[u] = None
                    continue
                # weighted average of feature values across items the user rated
                v = weights @ X / weights.sum()
                # keep only non-negative contributions (interpretation: positive importance)
                v_pos = np.maximum(v, 0.0)
                s = v_pos.sum()
                if s <= 1e-9:
                    self.user_profile_explain[u] = None
                    continue
                imp = v_pos / s  # percentages summing to 1
                self.user_profile_explain[u] = pd.Series(imp, index=feature_names)

        self.global_mean = trainset.global_mean
        self.user_means = {
            u: np.mean([r for _, r in trainset.ur[u]])
            for u in trainset.all_users()
        }
        self.user_n_ratings = {
            u: len(trainset.ur[u])
            for u in trainset.all_users()
        }

        knn_methods = ('content_knn_centered', 'ridge_knn_blend')
        if self.regressor_method in knn_methods and self.content_features is not None:
            feat_values = self.content_features.values.astype(np.float32)
            self.content_features_norm = pd.DataFrame(
                self._l2_normalize_rows(feat_values),
                index=self.content_features.index,
                columns=self.content_features.columns
            )
            self.user_knn_cache = {}
            for u in trainset.all_users():
                raw_items = [self.trainset.to_raw_iid(iid) for iid, _ in trainset.ur[u]]
                ratings = np.array([r for _, r in trainset.ur[u]], dtype=np.float32)
                feat_rows = self.content_features_norm.reindex(raw_items)
                mask = feat_rows.notna().any(axis=1).values & (
                    np.linalg.norm(feat_rows.fillna(0).values, axis=1) > 0
                )
                if mask.sum() < 2:
                    self.user_knn_cache[u] = None
                    continue
                self.user_knn_cache[u] = (
                    feat_rows.fillna(0).values[mask].astype(np.float32),
                    ratings[mask]
                )

        if self.regressor_method == 'random_score':
            pass

        elif self.regressor_method == 'random_sample':
            for u in self.user_profile:
                self.user_profile[u] = [rating for _, rating in self.trainset.ur[u]]

        elif self.regressor_method == 'stacking_groups':
            assert self.feature_groups is not None, \
                "stacking_groups requires features_method='all_content_v2'"
            self.user_group_models = {}
            self.user_meta_models = {}
            kf = KFold(n_splits=3, shuffle=True, random_state=0)
            group_alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

            for u in self.user_profile:
                X, y = self._build_user_frame(u)
                if X is None or len(y) < 5:
                    self.user_profile[u] = None
                    continue

                col_idx = {g: [list(self.content_features.columns).index(c) for c in cols]
                           for g, cols in self.feature_groups.items()}

                oof_preds = {g: np.zeros(len(y)) for g in col_idx}
                for tr, va in kf.split(X):
                    for g, idx in col_idx.items():
                        if not idx:
                            continue
                        reg = RidgeCV(alphas=group_alphas, scoring='neg_root_mean_squared_error')
                        reg.fit(X[tr][:, idx], y[tr])
                        oof_preds[g][va] = reg.predict(X[va][:, idx])

                meta_X = np.column_stack([oof_preds[g] for g in col_idx])
                meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                               scoring='neg_root_mean_squared_error',
                               fit_intercept=True)
                meta.fit(meta_X, y)
                self.user_meta_models[u] = (meta, list(col_idx.keys()))

                full_models = {}
                for g, idx in col_idx.items():
                    if not idx:
                        continue
                    reg = RidgeCV(alphas=group_alphas, scoring='neg_root_mean_squared_error')
                    reg.fit(X[:, idx], y)
                    full_models[g] = (reg, idx)
                self.user_group_models[u] = full_models
                self.user_profile[u] = True

        elif self.regressor_method == 'elastic_cv':
            for u in self.user_profile:
                X, y = self._build_user_frame(u)
                if X is None:
                    self.user_profile[u] = None
                    continue
                regressor = ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9],
                    alphas=[0.001, 0.01, 0.1, 1.0],
                    cv=3,
                    max_iter=2000,
                    random_state=0,
                    n_jobs=1
                )
                try:
                    regressor.fit(X, y)
                    self.user_profile[u] = regressor
                except Exception:
                    self.user_profile[u] = None

        elif self.regressor_method in (
            'linear_regression_false', 'linear_regression_true', 'random_forest', 'ridge', 'ridge_cv', 'ridge_cv_bias',
            'ridge_cv_centered', 'ridge_knn_blend',
            'ridge_a10', 'ridge_a100', 'ridge_a500', 'ridge_a1000', 'ridge_a5000', 'ridge_a10000',
            'huber', 'bayesian_ridge'
        ):
            for u in self.user_profile:
                X, y = self._build_user_frame(u)
                if X is None:
                    self.user_profile[u] = None
                    continue

                if self.regressor_method == 'ridge_cv_centered':
                    y = y - self.user_means[u]

                if self.regressor_method == 'linear_regression_false':
                    regressor = LinearRegression(fit_intercept=False)
                elif self.regressor_method == 'linear_regression_true':
                    regressor = LinearRegression(fit_intercept=True)
                elif self.regressor_method == 'random_forest':
                    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
                elif self.regressor_method == 'ridge':
                    regressor = Ridge(alpha=1.0)
                elif self.regressor_method == 'ridge_a20':
                    regressor = Ridge(alpha=20.0)
                elif self.regressor_method == 'ridge_a100':
                    regressor = Ridge(alpha=100.0)
                elif self.regressor_method == 'ridge_a500':
                    regressor = Ridge(alpha=500.0)
                elif self.regressor_method == 'ridge_a1000':
                    regressor = Ridge(alpha=1000.0)
                elif self.regressor_method == 'ridge_a5000':
                    regressor = Ridge(alpha=5000.0)
                elif self.regressor_method == 'ridge_a10000':
                    regressor = Ridge(alpha=10000.0)
                elif self.regressor_method == 'huber':
                    from sklearn.linear_model import HuberRegressor
                    regressor = HuberRegressor(alpha=0.001, max_iter=200)
                elif self.regressor_method == 'bayesian_ridge':
                    from sklearn.linear_model import BayesianRidge
                    regressor = BayesianRidge()
                else:
                    regressor = RidgeCV(
                        alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
                        scoring='neg_root_mean_squared_error'
                    )
                try:
                    regressor.fit(X, y)
                    self.user_profile[u] = regressor
                except Exception:
                    self.user_profile[u] = None

    # ── Scoring component ─────────────────────────────────────────────────

    def _predict_ridge(self, u, raw_item_id):
        if self.user_profile.get(u) is None:
            return self.user_means.get(u, self.global_mean)
        if raw_item_id not in self.content_features.index:
            return self.user_means.get(u, self.global_mean)
        x = self.content_features.loc[[raw_item_id], :].values
        return float(self.user_profile[u].predict(x)[0])

    def _predict_knn(self, u, raw_item_id):
        cache = self.user_knn_cache.get(u)
        if cache is None or raw_item_id not in self.content_features_norm.index:
            return self.user_means.get(u, self.global_mean)
        x_i = self.content_features_norm.loc[raw_item_id].values.astype(np.float32)
        if np.linalg.norm(x_i) == 0:
            return self.user_means.get(u, self.global_mean)
        feat, ratings = cache
        sim = np.clip(feat @ x_i, 0.0, None)
        if sim.sum() <= 1e-9:
            return self.user_means.get(u, self.global_mean)
        k = min(self.knn_k, len(sim))
        if k < len(sim):
            top_idx = np.argpartition(-sim, k - 1)[:k]
            sim = sim[top_idx]
            ratings_used = ratings[top_idx]
        else:
            ratings_used = ratings
        weight_sum = sim.sum()
        if weight_sum <= 1e-9:
            return self.user_means.get(u, self.global_mean)
        user_mean = self.user_means[u]
        return float(user_mean + (sim * (ratings_used - user_mean)).sum() / weight_sum)

    def _predict_stacking(self, u, raw_item_id):
        if self.user_profile.get(u) is None:
            return self.user_means.get(u, self.global_mean)
        if raw_item_id not in self.content_features.index:
            return self.user_means.get(u, self.global_mean)
        x = self.content_features.loc[[raw_item_id], :].values
        group_models = self.user_group_models[u]
        meta, group_order = self.user_meta_models[u]
        sub_preds = []
        for g in group_order:
            reg, idx = group_models[g]
            sub_preds.append(reg.predict(x[:, idx])[0])
        return float(meta.predict(np.array(sub_preds).reshape(1, -1))[0])

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        if self.regressor_method == 'random_score':
            rd.seed()
            return rd.uniform(0.5, 5)
        elif self.regressor_method == 'random_sample':
            rd.seed()
            return rd.choice(self.user_profile[u])

        raw_item_id = self.trainset.to_raw_iid(i)
        lo, hi = self.trainset.rating_scale

        if self.regressor_method in (
            'linear_regression_false', 'linear_regression_true', 'random_forest', 'ridge', 'ridge_cv', 'ridge_cv_bias',
            'ridge_cv_centered', 'elastic_cv',
            'ridge_a10', 'ridge_a100', 'ridge_a500', 'ridge_a1000', 'ridge_a5000', 'ridge_a10000',
            'huber', 'bayesian_ridge'
        ):
            if self.content_features is None or raw_item_id not in self.content_features.index:
                return float(np.clip(self.user_means.get(u, self.global_mean), lo, hi))

            score = self._predict_ridge(u, raw_item_id)

            if self.regressor_method == 'ridge_cv_bias':
                n_ratings = self.user_n_ratings[u]
                weight = min(1.0, n_ratings / 50)
                score = weight * score + (1 - weight) * self.user_means[u]
            elif self.regressor_method == 'ridge_cv_centered':
                score = score + self.user_means[u]

            return float(np.clip(score, lo, hi))

        elif self.regressor_method == 'content_knn_centered':
            return float(np.clip(self._predict_knn(u, raw_item_id), lo, hi))

        elif self.regressor_method == 'ridge_knn_blend':
            return float(np.clip(
                0.5 * self._predict_ridge(u, raw_item_id)
                + 0.5 * self._predict_knn(u, raw_item_id),
                lo, hi
            ))

        elif self.regressor_method == 'stacking_groups':
            return float(np.clip(self._predict_stacking(u, raw_item_id), lo, hi))

        return self.global_mean

    def explain(self, u):
        """Return a dict {feature_name: importance} for user `u`.
        Accepts either an inner user id (int) or a raw user id (string/int).
        Importance values are percentages summing to 1 (or empty dict if unavailable).
        """
        if not hasattr(self, 'user_profile_explain') or self.user_profile_explain is None:
            return {}
        inner = u
        if u not in self.user_profile_explain:
            try:
                inner = self.trainset.to_inner_uid(u)
            except Exception:
                return {}
        series = self.user_profile_explain.get(inner)
        if series is None:
            return {}
        return series.to_dict()
    
class UserBased_tuned(AlgoBase):
    def __init__(self, k=40, min_k=5, sim_options={}, **kwargs):
        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        # 1. Computing the ratings matrix
        self.compute_rating_matrix()
        
        # 2. Compute the similarity matrix
        self.compute_similarity_matrix()
        
        # 3. Computing the mean rating of every user
        self.mean_ratings = []
        for u in range(self.trainset.n_users):
            user_ratings = [r for (_, r) in self.trainset.ur[u]]
            self.mean_ratings.append(np.mean(user_ratings))
        
        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        
        # The estimate is by default set to the user average rating
        estimate = self.mean_ratings[u]
        
        # Step 1: Create the peer group of user u for item i
        # Potential neighbor: (neighbor_inner_id, similarity_value, rating)
        potential_neighbors = []
        
        # Access ratings of item i with self.trainset.ir[i]
        for (v, r_vi) in self.trainset.ir[i]:
            if v == u:
                continue
            
            sim_uv = self.sim[u, v]
            if sim_uv > 0:
                potential_neighbors.append((v, sim_uv, r_vi))
        
        # Pick top neighbors efficiently using heapq
        top_neighbors = heapq.nlargest(self.k, potential_neighbors, key=lambda x: x[1])
        
        # Step 2: Compute the weighted average
        actual_k = len(top_neighbors)
        
        # If actual_k is above min_k, we add the weighted average component
        if actual_k >= self.min_k:
            weighted_sum = 0
            sum_sim = 0
            
            for (v, sim_uv, r_vi) in top_neighbors:
                # Weighted average calculation: sim * (rating - neighbor_mean)
                weighted_sum += sim_uv * (r_vi - self.mean_ratings[v])
                sum_sim += abs(sim_uv)
            
            if sum_sim > 0:
                estimate += (weighted_sum / sum_sim)
        
        return estimate

    def compute_rating_matrix(self):
        m = self.trainset.n_users
        n = self.trainset.n_items
        
        # Preallocate an mxn numpy array with NaN
        self.ratings_matrix = np.empty((m, n))
        self.ratings_matrix[:] = np.nan
        
        # Access ratings of a specific user with self.trainset.ur[uiid]
        for uiid in range(m):
            for (iiid, rating) in self.trainset.ur[uiid]:
                self.ratings_matrix[uiid, iiid] = rating

    def compute_similarity_matrix(self):
        m = self.trainset.n_users
        self.sim = np.eye(m)
        min_support = self.sim_options.get('min_support', 5)
        
        # Retrieve the similarity metric name from sim_options (default: msd)
        sim_name = self.sim_options.get('name', 'msd').lower()
        
        for i in range(m):
            for j in range(i + 1, m):
                row_i = self.ratings_matrix[i]
                row_j = self.ratings_matrix[j]
                
                # Intersection: items rated by BOTH users
                mask_intersection = ~np.isnan(row_i - row_j)
                support = np.sum(mask_intersection)
                
                # FIX CRITIQUE : Assigne une valeur par défaut locale immédiate.
                # Cela élimine définitivement le UnboundLocalError.
                current_similarity = 0.0
                
                if support >= min_support:
                    if sim_name == 'msd':
                        sq_diff = np.sum((row_i[mask_intersection] - row_j[mask_intersection])**2)
                        msd = sq_diff / support
                        current_similarity = 1 / (msd + 1)
                    
                    elif sim_name == 'jacard':
                        mask_union = ~np.isnan(row_i) | ~np.isnan(row_j)
                        union_count = np.sum(mask_union)
                        current_similarity = support / union_count if union_count > 0 else 0.0
                    
                    elif sim_name == 'cosine_jaccard':
                        # 1. Extraction des vecteurs sur l'intersection
                        u_vec = row_i[mask_intersection]
                        v_vec = row_j[mask_intersection]
                        
                        # 2. Calcul du Cosinus de base
                        dot_product = np.sum(u_vec * v_vec)
                        norm_u = np.sqrt(np.sum(u_vec**2))
                        norm_v = np.sqrt(np.sum(v_vec**2))
                        cosine = dot_product / (norm_u * norm_v) if (norm_u * norm_v) > 0 else 0.0
                        
                        # 3. Calcul de l'indice de Jaccard
                        mask_union = ~np.isnan(row_i) | ~np.isnan(row_j)
                        union_count = np.sum(mask_union)
                        jaccard = support / union_count if union_count > 0 else 0.0
                        
                        # 4. Fusion des deux métriques
                        current_similarity = float(cosine * jaccard)
                    
                # Assignation sécurisée dans la matrice globale
                self.sim[i, j] = current_similarity
                self.sim[j, i] = current_similarity
