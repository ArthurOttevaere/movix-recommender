# standard library imports
from collections import defaultdict

# third parties imports
import numpy as np
import random as rd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
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

# Content-Based algorithm
# standard library imports
from collections import defaultdict

# third parties imports
import numpy as np
import random as rd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
from surprise.prediction_algorithms.predictions import PredictionImpossible
from constants import Constant as C
from loaders import load_items
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV


def get_top_n(predictions, n):
    """Return the top-N recommendation for each user from a set of predictions.
    Source: inspired by https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    and modified by cvandekerckh for random tie breaking
    """
    rd.seed(0)
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

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


# Content-Based algorithm
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
            # all_content + decade one-hot ADDITIF (year normalisé conservé)
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
            'linear_regression', 'random_forest', 'ridge', 'ridge_cv', 'ridge_cv_bias',
            'ridge_cv_centered', 'ridge_knn_blend'
        ):
            for u in self.user_profile:
                X, y = self._build_user_frame(u)
                if X is None:
                    self.user_profile[u] = None
                    continue

                if self.regressor_method == 'ridge_cv_centered':
                    y = y - self.user_means[u]

                if self.regressor_method == 'linear_regression':
                    regressor = LinearRegression(fit_intercept=True)
                elif self.regressor_method == 'random_forest':
                    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
                elif self.regressor_method == 'ridge':
                    regressor = Ridge(alpha=1.0)
                else:
                    regressor = RidgeCV(
                        alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
                        scoring='neg_root_mean_squared_error'
                    )
                regressor.fit(X, y)
                self.user_profile[u] = regressor

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
            'linear_regression', 'random_forest', 'ridge', 'ridge_cv', 'ridge_cv_bias',
            'ridge_cv_centered', 'elastic_cv'
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


# ─────────────────────────────────────────────────────────────────────────────
# ContentBasedV2 : TruncatedSVD + non-linear regressors + per-user stacking
# ─────────────────────────────────────────────────────────────────────────────
class ContentBasedV2(AlgoBase):
    """Content-based extension of ContentBased.

    Adds:
    - TruncatedSVD dim reduction on the content feature matrix (svd_components)
    - Non-linear regressors per-user: LightGBM, HistGradientBoosting, RF(n=100)
    - Per-user stacking of multiple regressors (OOF + RidgeCV meta-learner)
    - Finer RidgeCV alpha grids ('fine', 'ultra')
    - Adaptive shrinkage toward user_mean (bias_threshold)
    - Item-cold-start fallback via content-space KNN

    All purely content-based: no signal from other users' ratings of the target item.
    """

    _ALPHA_GRIDS = {
        'default': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
        'fine':    list(np.logspace(-3, 5, 30)),
    }

    def __init__(
        self,
        features_method='all_content_tmdb_emb',
        regressor_method='ridge_cv',  # ridge_cv | rf100 | lgbm | hgb | stack
        svd_components=None,
        alpha_grid='default',         # 'default' | 'fine' | 'ultra'
        stack_models=None,            # list[str] when regressor_method='stack'
        bias_threshold=200,
        knn_fallback=True,
        knn_k=30,
        lgbm_params=None,
        random_state=0,
    ):
        AlgoBase.__init__(self)
        self.features_method = features_method
        self.regressor_method = regressor_method
        self.svd_components = svd_components
        self.alpha_grid = alpha_grid
        self.stack_models = stack_models or ['ridge_cv', 'lgbm']
        self.bias_threshold = bias_threshold
        self.knn_fallback = knn_fallback
        self.knn_k = knn_k
        self.lgbm_params = lgbm_params or {}
        self.random_state = random_state

        # Reuse all the feature loading machinery from ContentBased
        self._cb = ContentBased(features_method, regressor_method='ridge_cv')

        # Filled by fit()
        self.X = None               # post-SVD content DataFrame indexed by raw movieId
        self.X_norm = None          # L2-normalized rows for KNN fallback
        self.user_profile = {}      # u -> regressor or dict for stacking
        self.user_means = {}
        self.user_n_ratings = {}
        self.global_mean = None
        self.user_knn_cache = {}    # u -> (feat_rows, ratings)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _apply_svd(self, X_base):
        """Optional TruncatedSVD + StandardScaler post-reduction."""
        if self.svd_components is None:
            return X_base
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import StandardScaler
        n_comp = min(self.svd_components, X_base.shape[1] - 1, X_base.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        reduced = svd.fit_transform(X_base.values)
        scaled = StandardScaler().fit_transform(reduced)
        return pd.DataFrame(
            scaled, index=X_base.index,
            columns=[f'svd_{i}' for i in range(n_comp)]
        )

    def _build_user_frame(self, u, X):
        """Per-user (X_train, y_train) from feature matrix X."""
        feature_names = list(X.columns)
        df_user = pd.DataFrame(self.trainset.ur[u], columns=['item_id', 'user_ratings'])
        df_user['item_id'] = df_user['item_id'].map(self.trainset.to_raw_iid)
        df_user = df_user.merge(X, how='left', left_on='item_id', right_index=True)
        df_user = df_user.dropna(subset=feature_names, how='all').fillna(0)
        if len(df_user) < 2:
            return None, None
        return df_user[feature_names].values, df_user['user_ratings'].values

    def _make_regressor(self, method):
        """Build a fresh sklearn regressor for the given method."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        if method == 'ridge_cv':
            if self.alpha_grid == 'ultra':
                # 2-stage : coarse first, then fine around best — handled in _fit_ridge_ultra
                return None
            alphas = self._ALPHA_GRIDS.get(self.alpha_grid, self._ALPHA_GRIDS['default'])
            return RidgeCV(alphas=alphas, scoring='neg_root_mean_squared_error')

        if method == 'rf100':
            return RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=3,
                max_features='sqrt', n_jobs=1, random_state=self.random_state,
            )

        if method == 'hgb':
            return HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=5, l2_regularization=0.5,
                random_state=self.random_state,
            )

        if method == 'lgbm':
            try:
                from lightgbm import LGBMRegressor
            except ImportError:
                # Fallback to HGB if lightgbm missing
                return self._make_regressor('hgb')
            params = dict(
                n_estimators=300, learning_rate=0.05, num_leaves=15,
                max_depth=4, min_child_samples=5, subsample=0.8,
                colsample_bytree=0.6, reg_lambda=0.5, n_jobs=1,
                random_state=self.random_state, verbose=-1,
            )
            params.update(self.lgbm_params)
            return LGBMRegressor(**params)

        raise ValueError(f"Unknown base regressor method: {method}")

    def _fit_ridge_ultra(self, X, y):
        """Two-stage RidgeCV: coarse log-spaced search, then fine refit around best."""
        coarse = RidgeCV(
            alphas=np.logspace(-3, 5, 15),
            scoring='neg_root_mean_squared_error',
        )
        coarse.fit(X, y)
        center = np.log10(coarse.alpha_)
        fine_alphas = np.logspace(center - 0.5, center + 0.5, 20)
        fine = RidgeCV(alphas=fine_alphas, scoring='neg_root_mean_squared_error')
        fine.fit(X, y)
        return fine

    def _fit_one_user(self, X_user, y_user, method, min_rows_lgbm=80):
        """Fit a single sklearn regressor on (X_user, y_user) where y is centered."""
        if method == 'lgbm' and len(y_user) < min_rows_lgbm:
            method = 'ridge_cv'

        if method == 'ridge_cv' and self.alpha_grid == 'ultra':
            return self._fit_ridge_ultra(X_user, y_user)

        reg = self._make_regressor(method)
        reg.fit(X_user, y_user)
        return reg

    # ── Stacking per-user (OOF base preds → RidgeCV meta) ─────────────────

    def _fit_user_stack(self, X_user, y_user):
        """OOF stack of self.stack_models for one user. y_user is centered.

        Returns {'meta': RidgeCV, 'base': dict[name -> fitted regressor], 'order': list}
        or None if not enough rows.
        """
        n = len(y_user)
        if n < 5:
            return None
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        oof = {m: np.zeros(n) for m in self.stack_models}
        for tr, va in kf.split(X_user):
            for m in self.stack_models:
                reg = self._fit_one_user(X_user[tr], y_user[tr], m)
                oof[m][va] = reg.predict(X_user[va])

        meta_X = np.column_stack([oof[m] for m in self.stack_models])
        meta = RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
            scoring='neg_root_mean_squared_error',
            fit_intercept=True,
        )
        meta.fit(meta_X, y_user)

        full = {m: self._fit_one_user(X_user, y_user, m) for m in self.stack_models}
        return {'meta': meta, 'base': full, 'order': list(self.stack_models)}

    # ── KNN content-space fallback (cold-item handling) ───────────────────

    def _build_knn_cache(self):
        feat = self.X.values.astype(np.float32)
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        feat_norm = feat / norms
        self.X_norm = pd.DataFrame(feat_norm, index=self.X.index, columns=self.X.columns)

        for u in self.trainset.all_users():
            raw_items = [self.trainset.to_raw_iid(iid) for iid, _ in self.trainset.ur[u]]
            ratings = np.array([r for _, r in self.trainset.ur[u]], dtype=np.float32)
            feat_rows = self.X_norm.reindex(raw_items)
            mask = feat_rows.notna().any(axis=1).values & (
                np.linalg.norm(feat_rows.fillna(0).values, axis=1) > 0
            )
            if mask.sum() < 2:
                self.user_knn_cache[u] = None
                continue
            self.user_knn_cache[u] = (
                feat_rows.fillna(0).values[mask].astype(np.float32),
                ratings[mask],
            )

    def _predict_knn(self, u, raw_item_id):
        cache = self.user_knn_cache.get(u)
        if cache is None or raw_item_id not in self.X_norm.index:
            return self.user_means.get(u, self.global_mean)
        x_i = self.X_norm.loc[raw_item_id].values.astype(np.float32)
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
        if sim.sum() <= 1e-9:
            return self.user_means.get(u, self.global_mean)
        user_mean = self.user_means[u]
        return float(user_mean + (sim * (ratings_used - user_mean)).sum() / sim.sum())

    # ── Fit / Estimate ────────────────────────────────────────────────────

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.global_mean = trainset.global_mean
        self.user_means = {u: np.mean([r for _, r in trainset.ur[u]]) for u in trainset.all_users()}
        self.user_n_ratings = {u: len(trainset.ur[u]) for u in trainset.all_users()}

        # 1. Get base content features and (optional) reduce via TruncatedSVD
        X_base = self._cb.content_features
        self.X = self._apply_svd(X_base)

        # 2. Build KNN cache for cold-item fallback
        if self.knn_fallback:
            self._build_knn_cache()

        # 3. Per-user training (y centered by user_mean)
        n_users = len(list(trainset.all_users()))
        progress_every = max(1, n_users // 10)
        for k, u in enumerate(trainset.all_users()):
            res = self._build_user_frame(u, self.X)
            if res[0] is None:
                self.user_profile[u] = None
                continue
            X_user, y_user = res
            y_centered = y_user - self.user_means[u]

            try:
                if self.regressor_method == 'stack':
                    self.user_profile[u] = self._fit_user_stack(X_user, y_centered)
                else:
                    self.user_profile[u] = self._fit_one_user(X_user, y_centered, self.regressor_method)
            except Exception as e:
                self.user_profile[u] = None

            if (k + 1) % progress_every == 0:
                print(f'  [V2 fit] {k + 1}/{n_users} users', flush=True)

        return self

    def _predict_one(self, u, raw_item_id):
        profile = self.user_profile.get(u)
        if profile is None:
            if self.knn_fallback:
                return self._predict_knn(u, raw_item_id)
            return self.user_means.get(u, self.global_mean)
        if raw_item_id not in self.X.index:
            if self.knn_fallback:
                return self._predict_knn(u, raw_item_id)
            return self.user_means.get(u, self.global_mean)

        x = self.X.loc[[raw_item_id], :].values
        if isinstance(profile, dict) and 'meta' in profile:
            # Stacking
            base_preds = np.array([profile['base'][m].predict(x)[0] for m in profile['order']])
            pred_centered = float(profile['meta'].predict(base_preds.reshape(1, -1))[0])
        else:
            pred_centered = float(profile.predict(x)[0])

        # De-center + adaptive shrinkage toward user_mean
        user_mean = self.user_means[u]
        raw_pred = pred_centered + user_mean
        w = min(1.0, self.user_n_ratings[u] / self.bias_threshold)
        return w * raw_pred + (1 - w) * user_mean

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        raw_item_id = self.trainset.to_raw_iid(i)
        lo, hi = self.trainset.rating_scale
        score = self._predict_one(u, raw_item_id)
        return float(np.clip(score, lo, hi))