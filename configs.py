# local imports
from models import *

# 1. On prépare ton modèle "Hidden Gems" optimisé (k=15, min_k=2)
algo_hidden_gems_base = UserBased_tuned(k=15, min_k=2, sim_options={'name': 'msd', 'min_support': 3})

class EvalConfig:
    
    models = [
        #("baseline_1", ModelBaseline1, {}),  # model_name, model class, model parameters (dict)
        #("baseline_2", ModelBaseline2, {}),
        #("baseline_3", ModelBaseline3, {}),
        #("baseline_4", ModelBaseline4, {"random_state": 1}),

        # ("KNNwithMeans", ModelBaseline5, {"random_state": 1}),
        #("UserBased_Manual", UserBased, {"k": 3, "min_k": 2, "sim_options": {'name': 'msd', 'min_support': 3, 'user_based': True}}),
        # ("RandomSample", ContentBased, {"features_method": "title_length", "regressor_method": "random_sample"}),
        # ("RandomScore", ContentBased, {"features_method": "title_length", "regressor_method": "random_score"}),
        # ("LinearRegression_Intercept_False", ContentBased, {"features_method": "title_length", "regressor_method": "linear_regression_false"}),
        # ("LinearRegression_Intercept_True", ContentBased, {"features_method": "title_length", "regressor_method": "linear_regression_true"}),
        # ("ContentBased_ridge_cv", ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_cv"}),
        # ("ContentBased_ridge", ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge"}),
        #("LatentFactor", LatentFactor, {}),

        ("UserBased_tuned_cosine_jaccard", UserBased_tuned, {"k": 40, "min_k": 3, "sim_options": {'name': 'cosine_jaccard', 'min_support': 5}}),
        ("UserBased_tuned_msd", UserBased_tuned, {"k": 40, "min_k": 3, "sim_options": {'name': 'msd', 'min_support': 5}}),
        ("UserBased_hidden_gems", UserBased_tuned, {"k": 15, "min_k": 2, "sim_options": {'name': 'msd', 'min_support': 3}}),
        #("UserBased_Filtered_20pct", HiddenGemsFilterWrapper, {"base_algo": algo_hidden_gems_base, "exclude_top_pct": 0.20}),
        
    ]

    # Evaluation metrics — alignées avec la littérature RecSys (NCF, SASRec, LightGCN, Vargas & Castells 2011)
    split_metrics = ["rmse", "mae"]
    loo_metrics   = ["hit_rate@5", "hit_rate@10", "hit_rate@20",
                     "ndcg@5",     "ndcg@10",     "ndcg@20"]
    full_metrics  = ["coverage", "miuf", "ild"]

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --

    # Negative-sampling LOO (protocole He et al. 2017 NCF, WWW'17)
    # 1 item positif + 99 negatifs aleatoires = 100 candidats par utilisateur.
    # Seuls les modeles dans neg_sampling_model_names sont evalues (plus rapide).
    # Colonnes ajoutees au rapport avec le suffixe [ns] (negative sampling).
    neg_sampling_metrics = ["hit_rate@5",  "hit_rate@10",  "hit_rate@20",
                            "ndcg@5",      "ndcg@10",      "ndcg@20"]
    neg_sampling_model_names = {"LatentFactor", "LatentFactorPP", "LatentFactorRanking2", "BPR"}
