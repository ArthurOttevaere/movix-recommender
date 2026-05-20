# third parties imports
from pathlib import Path


class Constant:

    DATA_PATH = Path('data/small')  # -- fill here the dataset size to use
    EVALUATION_PATH = Path('evaluation')
    # Content
    CONTENT_PATH = DATA_PATH / 'content'
    # - item
    ITEMS_FILENAME = 'movies.csv'
    ITEM_ID_COL = 'movieId'
    LABEL_COL = 'title'
    GENRES_COL = 'genres'

    # Evidence
    EVIDENCE_PATH = DATA_PATH / 'evidence'
    # - ratings
    RATINGS_FILENAME = 'ratings.csv'
    USER_ID_COL = 'userId'
    RATING_COL = 'rating'
    TIMESTAMP_COL = 'timestamp'
    USER_ITEM_RATINGS = [USER_ID_COL, ITEM_ID_COL, RATING_COL]

    # Rating scale
    RATINGS_SCALE = (0.5, 5.0)  # -- fill in here the ratings scale as a tuple (min_value, max_value)

    # === User-Based Model Evaluation ===
    EVAL_K_VALUES = [5, 10, 20]   # values of k for Precision@k, Recall@k, NDCG@k, HitRate@k
    EVAL_THRESHOLD = 4.0          # minimum rating to consider an item "relevant"

    # Test users for evaluate_userbased.py
    # Each entry:
    #   "name"         : label for display / CSV export
    #   "input"        : {movie_id: rating} — ratings the model sees (simulates a new user)
    #   "ground_truth" : {movie_id: rating} — held-out ratings used to evaluate recommendations
    #
    # Constraint: ground_truth keys must NOT appear in input
    # (the model never recommends already-rated items)
    TEST_USERS_USERBASED = [
        {
            "name": "scifi_fan",
            "input": {
                260: 5.0,    # Star Wars: Episode IV
                2571: 4.5,   # The Matrix
                480: 4.0,    # Jurassic Park
                79132: 4.5,  # Inception
            },
            "ground_truth": {
                1196: 5.0,   # Star Wars: Episode V
                1210: 4.5,   # Star Wars: Episode VI
                4993: 3.5,   # LOTR: The Fellowship of the Ring
            },
        },
        {
            "name": "drama_fan",
            "input": {
                318: 5.0,    # The Shawshank Redemption
                527: 5.0,    # Schindler's List
                356: 4.0,    # Forrest Gump
            },
            "ground_truth": {
                593: 4.5,    # The Silence of the Lambs
                296: 4.0,    # Pulp Fiction
            },
        },
        {
            "name": "mixed_fan",
            "input": {
                1: 4.5,      # Toy Story
                58559: 5.0,  # The Dark Knight
                296: 4.0,    # Pulp Fiction
                356: 3.5,    # Forrest Gump
            },
            "ground_truth": {
                79132: 4.5,  # Inception
                4993: 3.0,   # LOTR: The Fellowship of the Ring
                318: 4.5,    # The Shawshank Redemption
            },
        },
    ]
