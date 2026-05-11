# standard library imports
from collections import defaultdict
import heapq

# third parties imports
import numpy as np
import random as rd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
from surprise import PredictionImpossible
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

# KNN with means, using msd baseline similarity measure and user-based collaborative filtering
class ModelBaseline5(KNNWithMeans):
    def __init__(self, random_state=1):
        KNNWithMeans.__init__(self, k=3, min_k = 2, sim_options={'name': 'msd', 'user_based': True}, random_state=random_state)

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


# Content-based model with various content features and regression methods
class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method):
        AlgoBase.__init__(self)
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _load_genome(self):
        """Genome : 1128 tags sémantiques, relevance 0-1 pour chaque film."""
        df_genome = pd.read_csv(C.CONTENT_PATH / 'genome-scores.csv')
        df_genome = df_genome.pivot(index='movieId', columns='tagId', values='relevance')
        return df_genome.fillna(0)

    def _load_genome_scaled(self):
        """Genome normalisé avec StandardScaler."""
        from sklearn.preprocessing import StandardScaler
        df_genome = self._load_genome()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_genome)
        return pd.DataFrame(scaled, index=df_genome.index, columns=df_genome.columns)

    def _load_visuals_scaled(self):
        """Visuals normalisés avec StandardScaler."""
        from sklearn.preprocessing import StandardScaler
        visual_path = C.CONTENT_PATH / 'visuals' / 'LLVisualFeatures13K_Log.csv'
        df_visuals = pd.read_csv(visual_path)
        df_visuals = df_visuals.set_index('ML_Id')
        df_visuals = df_visuals.fillna(df_visuals.mean())
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_visuals)
        return pd.DataFrame(scaled, index=df_visuals.index, columns=df_visuals.columns)

    def _load_tags(self):
        """Tags libres des utilisateurs agrégés par film, vectorisés TF-IDF."""
        df_tags = pd.read_csv(C.CONTENT_PATH / 'tags.csv')
        # Aggregate all tags per movie
        df_tags_grouped = df_tags.groupby('movieId')['tag'].apply(
            lambda x: ' '.join(x.astype(str))
        ).reset_index()
        df_tags_grouped = df_tags_grouped.set_index('movieId')

        tfidf = TfidfVectorizer(max_features=200, min_df=2)
        tfidf_matrix = tfidf.fit_transform(df_tags_grouped['tag'])
        df_features = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=df_tags_grouped.index,
            columns=[f'tag_{c}' for c in tfidf.get_feature_names_out()]
        )
        return df_features

    def _load_year_genres(self, df_items):
        """Année normalisée + genres TF-IDF."""
        df_year = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)').astype(float)
        df_year.columns = ['release_year']
        df_year = df_year.fillna(df_year.mean())
        df_year = (df_year - df_year.mean()) / df_year.std()

        tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
        tfidf_matrix = tfidf.fit_transform(df_items[C.GENRES_COL].fillna(''))
        df_genres = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=df_items.index,
            columns=tfidf.get_feature_names_out()
        )
        return df_year, df_genres
    

    # ── Content Analyzer ──────────────────────────────────────────────────

    def create_content_features(self, features_method):
        """Content Analyzer"""
        df_items = load_items()

        if features_method is None:
            return None

        elif features_method == "genome_scaled":
            # Best result so far (0.7451)
            return self._load_genome_scaled()

        elif features_method == "genome_scaled_tags":
            # Genome normalisé + tags libres TF-IDF — best combination (0.7451)
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags()
            return df_genome_scaled.join(df_tags, how='left').fillna(0)

        elif features_method == "genome_scaled_visuals_scaled":
            # Genome normalisé + visuals normalisés
            df_genome_scaled = self._load_genome_scaled()
            df_visuals_scaled = self._load_visuals_scaled()
            return df_genome_scaled.join(df_visuals_scaled, how='left').fillna(0)

        elif features_method == "genome_scaled_tags_visuals_scaled":
            # Genome normalisé + tags libres + visuals normalisés
            df_genome_scaled = self._load_genome_scaled()
            df_tags = self._load_tags()
            df_visuals_scaled = self._load_visuals_scaled()
            df_features = df_genome_scaled.join(df_tags, how='left')
            df_features = df_features.join(df_visuals_scaled, how='left')
            return df_features.fillna(0)

        else:
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')

    # ── Profile Learner ───────────────────────────────────────────────────

    def fit(self, trainset):
        """Profile Learner"""
        AlgoBase.fit(self, trainset)
        self.user_profile = {u: None for u in trainset.all_users()}

        # Store global mean and user means for bias correction
        self.global_mean = trainset.global_mean
        self.user_means = {
            u: np.mean([r for _, r in trainset.ur[u]])
            for u in trainset.all_users()
        }
        self.user_n_ratings = {
            u: len(trainset.ur[u])
            for u in trainset.all_users()
        }

        if self.regressor_method == 'random_score':
            pass

        elif self.regressor_method == 'random_sample':
            for u in self.user_profile:
                self.user_profile[u] = [rating for _, rating in self.trainset.ur[u]]

        elif self.regressor_method in (
            'linear_regression', 'random_forest', 'ridge', 'ridge_cv', 'ridge_cv_bias'
        ):
            feature_names = list(self.content_features.columns)
            for u in self.user_profile:
                df_user = pd.DataFrame(self.trainset.ur[u], columns=['item_id', 'user_ratings'])
                df_user['item_id'] = df_user['item_id'].map(self.trainset.to_raw_iid)

                df_user = df_user.merge(
                    self.content_features,
                    how='left',
                    left_on='item_id',
                    right_index=True
                )

                # Remove items without any feature
                df_user = df_user.dropna(subset=feature_names, how='all')
                df_user = df_user.fillna(0)

                # Guard : not enough examples to fit
                if len(df_user) < 2:
                    self.user_profile[u] = None
                    continue

                X = df_user[feature_names].values
                y = df_user['user_ratings'].values

                if self.regressor_method == 'linear_regression':
                    regressor = LinearRegression(fit_intercept=True)
                elif self.regressor_method == 'random_forest':
                    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
                elif self.regressor_method == 'ridge':
                    regressor = Ridge(alpha=1.0)
                elif self.regressor_method in ('ridge_cv', 'ridge_cv_bias'):
                    # Optimized directly on RMSE with wide alpha grid
                    regressor = RidgeCV(
                        alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
                        scoring='neg_root_mean_squared_error'
                    )

                regressor.fit(X, y)
                self.user_profile[u] = regressor

    # ── Scoring component ─────────────────────────────────────────────────

    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        if self.regressor_method == 'random_score':
            rd.seed()
            return rd.uniform(0.5, 5)

        elif self.regressor_method == 'random_sample':
            rd.seed()
            return rd.choice(self.user_profile[u])

        elif self.regressor_method in (
            'linear_regression', 'random_forest', 'ridge', 'ridge_cv', 'ridge_cv_bias'
        ):
            raw_item_id = self.trainset.to_raw_iid(i)

            # Fallback for cold-start items
            if self.content_features is None or raw_item_id not in self.content_features.index:
                return self.trainset.global_mean

            x = self.content_features.loc[[raw_item_id], :].values

            if self.user_profile[u] is None or x.shape[0] == 0:
                return self.trainset.global_mean

            # Model prediction
            score = self.user_profile[u].predict(x)[0]

            if self.regressor_method == 'ridge_cv_bias':
                # User bias correction : weighted blend based on number of ratings
                n_ratings = self.user_n_ratings[u]
                user_mean = self.user_means[u]
                global_mean = self.global_mean

                # More ratings = more trust in model, less in global mean
                weight = min(1.0, n_ratings / 50)
                score = weight * score + (1 - weight) * (global_mean + (user_mean - global_mean))

            # Clamp score within rating scale
            lo, hi = self.trainset.rating_scale
            return float(np.clip(score, lo, hi))

        else:
            return self.trainset.global_mean
        