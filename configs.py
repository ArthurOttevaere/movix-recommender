# local imports
from models import *


class EvalConfig:

    models = [
        ("baseline_1", ModelBaseline1, {}),
        ("baseline_2", ModelBaseline2, {}),
        ("baseline_3", ModelBaseline3, {}),
        ("baseline_4", ModelBaseline4, {"random_state": 1}),
        ("genome_scaled_ridgecv", ContentBased, {"features_method": "genome_scaled", "regressor_method": "ridge_cv"}),
        ("genome_scaled_tags_ridgecv", ContentBased, {"features_method": "genome_scaled_tags", "regressor_method": "ridge_cv"}),
        ("genome_scaled_tags_ridgecv_bias", ContentBased, {"features_method": "genome_scaled_tags", "regressor_method": "ridge_cv_bias"}),
        ("genome_scaled_tags_visuals_ridgecv_bias", ContentBased, {"features_method": "genome_scaled_tags_visuals_scaled", "regressor_method": "ridge_cv_bias"}),
        ("genome_scaled_tags_ridgecv_centered", ContentBased, {"features_method": "genome_scaled_tags", "regressor_method": "ridge_cv_centered"}),
        ("all_content_ridgecv", ContentBased, {"features_method": "all_content", "regressor_method": "ridge_cv"}),

        # Incréments additifs sur all_content (mêmes scaling, juste plus de signal)
        ("all_content_rich_tags", ContentBased, {"features_method": "all_content_rich_tags", "regressor_method": "ridge_cv"}),
        ("all_content_decade", ContentBased, {"features_method": "all_content_decade", "regressor_method": "ridge_cv"}),
        ("all_content_full", ContentBased, {"features_method": "all_content_full", "regressor_method": "ridge_cv"}),

        # all_content_full + TMDB metadata (runtime, lang, country, studio, director, cast)
        ("all_content_tmdb", ContentBased, {"features_method": "all_content_tmdb", "regressor_method": "ridge_cv"}),

        # all_content_tmdb + 384-d Sentence-Transformer embeddings on (title + tagline + overview)
        ("all_content_tmdb_emb", ContentBased, {"features_method": "all_content_tmdb_emb", "regressor_method": "ridge_cv"}),
    ]

    split_metrics = ["rmse"]
    loo_metrics   = []
    full_metrics  = []

    # Split parameters
    test_size = 0.25

    # Loo parameters
    top_n_value = 40