# standard library imports
from collections import defaultdict
import heapq

# third parties imports
import numpy as np
import random as rd
import pandas as pd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
from surprise import PredictionImpossible

from loaders import load_items
from constants import Constant as C
from sklearn.linear_model import LinearRegression



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
    def __init__(self, features_method, regressor_method):
        AlgoBase.__init__(self)
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)

    def create_content_features(self, features_method):
        """Content Analyzer"""
        df_items = load_items()
        if features_method is None:
            df_features = None
        elif features_method == "title_length": # a naive method that creates only 1 feature based on title length
            df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')
        else: # (implement other feature creations here)
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        return df_features
    

    def fit(self, trainset):
        """Profile Learner"""
        AlgoBase.fit(self, trainset)
        
        # Preallocate user profiles
        self.user_profile = {u: None for u in trainset.all_users()}

        if self.regressor_method == 'random_score':
            pass
        
        elif self.regressor_method == 'random_sample':
            for u in self.user_profile:
                self.user_profile[u] = [rating for _, rating in self.trainset.ur[u]]
        # Linear regression 
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
            
                regressor.fit(X, y)
                self.user_profile[u] = regressor 
        
    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        # First, handle cases for unknown users and items
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')


        if self.regressor_method == 'random_score':
            rd.seed()
            score = rd.uniform(0.5,5)

        elif self.regressor_method == 'random_sample':
            rd.seed()
            score = rd.choice(self.user_profile[u])
        
        # (implement here the regressor prediction)
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

            # Clamp score within rating scale
            lo, hi = self.trainset.rating_scale
            return float(np.clip(score, lo, hi))

        else:
            return self.trainset.global_mean


        #return score
