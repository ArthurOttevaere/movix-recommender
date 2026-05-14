# local imports
from models import *


class EvalConfig:
    
    models = [
        ("baseline_1", ModelBaseline1, {}),  # model_name, model class, model parameters (dict)
        ("baseline_2", ModelBaseline2, {}),
        ("baseline_3", ModelBaseline3, {}),
        ("baseline_4", ModelBaseline4, {"random_state": 1}),
        ("KNNwithMeans", ModelBaseline5, {"random_state": 1}),
        #("UserBased", UserBased, {"k": 3, "min_k": 2, "sim_options": {'name': 'msd', 'user_based': True}, "random_state": 1})
        ("ContentBased", ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_cv"})
    ]
    split_metrics = ["rmse"]
    loo_metrics = []
    full_metrics = []

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --
