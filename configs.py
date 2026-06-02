# local imports
from models import *
from surprise import KNNBaseline

class EvalConfig:
    
    models = [
        # The commented models are not included in the evaluation.
        # ("baseline_1", ModelBaseline1, {}),  # model_name, model class, model parameters (dict)
        # ("baseline_2", ModelBaseline2, {}),
        # ("baseline_3", ModelBaseline3, {}),
        # ("baseline_4", ModelBaseline4, {"random_state": 1}),
        # ("KNNwithMeans", ModelBaseline5, {"random_state": 1}),
        # ("UserBased_Manual", UserBased, {"k": 3, "min_k": 2, "sim_options": {'name': 'msd', 'min_support': 3, 'user_based': True}}),
        # ("RandomSample", ContentBased, {"features_method": "title_length", "regressor_method": "random_sample"}),
        # ("RandomScore", ContentBased, {"features_method": "title_length", "regressor_method": "random_score"}),
        # ("LinearRegression_Intercept_False", ContentBased, {"features_method": "title_length", "regressor_method": "linear_regression_false"}),
        # ("LinearRegression_Intercept_True", ContentBased, {"features_method": "title_length", "regressor_method": "linear_regression_true"}),
        # ("ContentBased_ridge_cv", ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_cv"}),
        # ("ContentBased_ridge", ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge"}),
        # ("LatentFactor", LatentFactor, {}),          # results already available
        # ("LatentFactorPP", LatentFactorPP, {}),       # results already available
        # ("LatentFactorRanking2", LatentFactorRanking2, {}),  # results already available

        # For the models below, no reranking applied 
        ("ContentBased_ridge_cv", ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_cv", "rerank_popularity": False, "rerank_alpha": 0.1}),
        #("ContentBased_Ridge_Fixed_rerank_a005", ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_fixed", "rerank_popularity": True, "rerank_alpha": 0.05 }),

        ("BPR",         ModelBPR,        {"factors": 64, "learning_rate": 0.01, "regularization": 0.01, "iterations": 100, "rerank_popularity": False, "rerank_alpha": 0.1}),
        ("BPR_Novelty", ModelBPRNovelty, {"factors": 64, "learning_rate": 0.01, "regularization": 0.01, "iterations": 100, "beta": 0.2, "rerank_popularity": False, "rerank_alpha": 0.1}),
        ("iALS",        ModeliALS,       {"factors": 50, "iterations": 20, "regularization": 0.01, "alpha": 40, "rerank_popularity": False, "rerank_alpha": 0.1}),

        # For the models below, reranking is applied with alpga = 0.5
        #("UserBased_tuned_cosine_jaccard", UserBased_tuned, {"k": 40, "min_k": 3, "sim_options": {'name': 'cosine_jaccard', 'min_support': 5}}),
        #("UserBased_hidden_new", UserBased_tuned, {"k": 40, "min_k": 2, "sim_options": {'name': 'pearson_baseline', 'min_support': 3}}),
        ("UserBased_Pearson_Natif", KNNBaseline, {"k": 40, "min_k": 2, "sim_options": {'name': 'pearson_baseline', 'user_based': True, 'min_support': 3}, "rerank_popularity": True, "rerank_alpha": 0.5}),
        ("UserBased_Tuned_Jaccard", UserBased_tuned, {"k": 40, "min_k": 2, "sim_options": {'name': 'jacard', 'user_based': True, 'min_support': 3}, "rerank_popularity": True, "rerank_alpha": 0.5}),
    ]

    # Evaluation metrics — aligned with RecSys literature (NCF, SASRec, LightGCN, Vargas & Castells 2011)
    split_metrics = ["rmse", "mae"]
    loo_metrics   = ["hit_rate@5", "hit_rate@10", "hit_rate@20",
                     "ndcg@5",     "ndcg@10",     "ndcg@20"]
    full_metrics  = ["coverage", "miuf", "ild"]

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --

    # Negative-sampling LOO (protocol He et al. 2017 NCF, WWW'17)
    # 1 positive item + 99 random negatives = 100 candidates per user.
    # Only models in neg_sampling_model_names are evaluated (faster).
    # Columns added to the report with suffix [ns] (negative sampling).
    neg_sampling_metrics = ["hit_rate@5",  "hit_rate@10",  "hit_rate@20",
                            "ndcg@5",      "ndcg@10",      "ndcg@20"]
    neg_sampling_model_names = {"BPR", "BPR_Novelty", "iALS", "ContentBased_ridge_cv", "UserBased_Pearson_Natif", "UserBased_Tuned_Jaccard"}

