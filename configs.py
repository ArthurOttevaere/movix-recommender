# local imports
from models import *


class EvalConfig:

    models = [
        #("baseline_1", ModelBaseline1, {}),
        #("baseline_2", ModelBaseline2, {}),
        #("baseline_3", ModelBaseline3, {}),
        #("baseline_4", ModelBaseline4, {"random_state": 1}),
        #("genome_scaled_ridgecv", ContentBased, {"features_method": "genome_scaled", "regressor_method": "ridge_cv"}),
        #("genome_scaled_tags_ridgecv", ContentBased, {"features_method": "genome_scaled_tags", "regressor_method": "ridge_cv"}),
        #("genome_scaled_tags_ridgecv_bias", ContentBased, {"features_method": "genome_scaled_tags", "regressor_method": "ridge_cv_bias"}),
        #("genome_scaled_tags_visuals_ridgecv_bias", ContentBased, {"features_method": "genome_scaled_tags_visuals_scaled", "regressor_method": "ridge_cv_bias"}),
        #("genome_scaled_tags_ridgecv_centered", ContentBased, {"features_method": "genome_scaled_tags", "regressor_method": "ridge_cv_centered"}),
        #("all_content_ridgecv", ContentBased, {"features_method": "all_content", "regressor_method": "ridge_cv"}),

        # Incréments additifs sur all_content (mêmes scaling, juste plus de signal)
        #("all_content_rich_tags", ContentBased, {"features_method": "all_content_rich_tags", "regressor_method": "ridge_cv"}),
        #("all_content_decade", ContentBased, {"features_method": "all_content_decade", "regressor_method": "ridge_cv"}),
        #("all_content_full", ContentBased, {"features_method": "genome_scaled", "regressor_method": "ridge"}),

        # all_content_full + TMDB metadata (runtime, lang, country, studio, director, cast)
        ("all_content_tmdb", ContentBased, {"features_method": "all_content_tmdb", "regressor_method": "ridge_cv"}),

        # ── Test piste "ami" : tags TF-IDF max_features=1128 au lieu de 500 ──
        # Hypothèse minimale : genome(1128) + tags(1128) seul
        #("genome_tags1128_only_ridgecv", ContentBased, {"features_method": "genome_tags1128_only", "regressor_method": "ridge_cv"}),
        # Pareil mais avec year/decade/genres
        #("all_content_tags1128_ridgecv", ContentBased, {"features_method": "all_content_tags1128", "regressor_method": "ridge_cv"}),
        # Avec TMDB en plus
        #("all_content_tmdb_tags1128_ridgecv", ContentBased, {"features_method": "all_content_tmdb_tags1128", "regressor_method": "ridge_cv"}),
        # Variante simple Ridge (pas CV) sur genome pur — l'indice "ridge" littéral
        #("genome_only_ridge_simple", ContentBased, {"features_method": "genome_scaled", "regressor_method": "ridge"}),

        # Compounding : tags1128 + SBERT embeddings + TMDB
        #("all_content_tmdb_emb_tags1128", ContentBased, {"features_method": "all_content_tmdb_emb_tags1128", "regressor_method": "ridge_cv"}),
        # Test si pousser au-delà de 1128 aide
        #("all_content_tmdb_tags2000", ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_cv"}),

        
        # ── Test "Ridge sans CV" sur le meilleur feature set actuel ──────────
        # Hypothèse : RidgeCV overfit la distribution eval. Alpha fixe = plus stable, peut mieux généraliser au hackathon.
        ("tmdb_tags2000_ridge_a20",    ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_a20"}),
        #("tmdb_tags2000_ridge_a100",   ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_a100"}),
        #("tmdb_tags2000_ridge_a500",   ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_a500"}),
        #("tmdb_tags2000_ridge_a1000",  ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_a1000"}),
        #("tmdb_tags2000_ridge_a5000",  ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_a5000"}),
        #("tmdb_tags2000_ridge_a10000", ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "ridge_a10000"}),
        # Bayesian Ridge : tune alpha automatiquement via ML estimation (différent de CV)
        #("tmdb_tags2000_bayesian",     ContentBased, {"features_method": "all_content_tmdb_tags2000", "regressor_method": "bayesian_ridge"}),

        # all_content_tmdb + 384-d Sentence-Transformer embeddings on (title + tagline + overview)
        #("all_content_tmdb_emb", ContentBased, {"features_method": "all_content_tmdb_emb", "regressor_method": "ridge_cv"}),

        # Quick win : stacking_groups déjà codé mais jamais évalué
        #("all_content_v2_stack_groups", ContentBased, {"features_method": "all_content_v2", "regressor_method": "stacking_groups"}),

        # ── ContentBasedV2 variants (TruncatedSVD + non-linear + stacking) ────────
        #("v2_ridge_svd400_emb",     ContentBasedV2, {"features_method": "all_content_tmdb_emb", "regressor_method": "ridge_cv", "svd_components": 400, "alpha_grid": "fine"}),
        #("v2_lgbm_svd200_emb",      #ContentBasedV2, {"features_method": "all_content_tmdb_emb", "regressor_method": "lgbm", "svd_components": 200}),
        #("v2_rf100_svd200_emb",     ContentBasedV2, {"features_method": "all_content_tmdb_emb", "regressor_method": "rf100", "svd_components": 200}),
        #("v2_stack_ridge_lgbm_emb", ContentBasedV2, {"features_method": "all_content_tmdb_emb", "regressor_method": "stack", "stack_models": ["ridge_cv", "lgbm"], "svd_components": 300}),
        #("v2_ridge_fine_alphas",    ContentBasedV2, {"features_method": "all_content_tmdb_emb", "regressor_method": "ridge_cv", "alpha_grid": "ultra"}),
    ]

    split_metrics = ["rmse"]
    loo_metrics   = []
    full_metrics  = []

    # Split parameters
    test_size = 0.25

    # Loo parameters
    top_n_value = 40