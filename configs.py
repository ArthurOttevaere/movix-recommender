# local imports
from models import *


class EvalConfig:
    
    models = [
        #("baseline_1", ModelBaseline1, {}),  # model_name, model class, model parameters (dict)
        #("baseline_2", ModelBaseline2, {}),
        #("baseline_3", ModelBaseline3, {}),
        #("baseline_4", ModelBaseline4, {"random_state": 1}),
        ("KNNwithMeans", ModelBaseline5, {"random_state": 1}),
        ("UserBased_Manual", UserBased, {"k": 3, "min_k": 2, "sim_options": {'name': 'msd', 'min_support': 3, 'user_based': True}}),
        ("UserBased_tuned", UserBased_tuned, {"k": 40, "min_k": 5, "sim_options": {'name': 'msd', 'min_support': 10, 'user_based': True}}),

        #("RandomSample", ContentBased, {"features_method": "title_length", "regressor_method": "random_sample"}),
        #("RandomScore", ContentBased, {"features_method": "title_length", "regressor_method": "random_score"}),

        #("LinearRegression_Intercept_False", ContentBased, {"features_method": "title_length", "regressor_method": "linear_regression_false"}),
        #("LinearRegression_Intercept_True", ContentBased, {"features_method": "title_length", "regressor_method": "linear_regression_true"}),
        
        #("ContentBased_ridge_cv", ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_cv"}), # best so far
        #("ContentBased_ridge", ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge"}) 

    ]

    # Evaluation metrics. Only rmse is activated here for computational reasons, but you can activate more metrics if you want.
    split_metrics = ["rmse"] # = "mae"
    loo_metrics = [] # hit_rate
    full_metrics = [] # novelty

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --
