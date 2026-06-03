# third parties imports
from pathlib import Path


class Constant:

    DATA_PATH = Path('data/hackathon')  # -- fill here the dataset size to use
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
        # --- Profils originaux ---
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

        # --- New profiles ---

        {
            "name": "action_blockbuster",
            "input": {
                589: 4.5,    # Terminator 2
                2571: 5.0,   # The Matrix
                3578: 4.5,   # Gladiator
                58559: 5.0,  # The Dark Knight
            },
            "ground_truth": {
                1240: 5.0,   # The Terminator
                377: 3.5,    # Speed
                91529: 4.0,  # The Dark Knight Rises
                59315: 4.5,  # Iron Man
            },
        },

        # Teste si le modèle détecte le signal "animation" malgré les genres variés
        {
            "name": "animation_family",
            "input": {
                1: 5.0,      # Toy Story
                588: 4.5,    # Aladdin
                4886: 5.0,   # Monsters, Inc.
                6377: 5.0,   # Finding Nemo
            },
            "ground_truth": {
                4306: 4.0,   # Shrek
                50872: 4.5,  # Ratatouille
                8961: 5.0,   # The Incredibles
                134853: 4.5, # Inside Out
            },
        },

        # Test the recommendation on a crime/thriller profile that is dark and consistent
        {
            "name": "crime_thriller",
            "input": {
                50: 5.0,     # The Usual Suspects
                2959: 5.0,   # Fight Club
                296: 4.5,    # Pulp Fiction
                593: 4.5,    # The Silence of the Lambs
            },
            "ground_truth": {
                608: 4.5,    # Fargo
                858: 5.0,    # The Godfather
                48516: 4.5,  # The Departed
                4226: 4.0,   # Memento
            },
        },

        # Cold start : only 3 films in input — test the robustness with few data
        {
            "name": "cold_start",
            "input": {
                1721: 5.0,   # Titanic
                356: 4.5,    # Forrest Gump
                2858: 4.0,   # American Beauty
            },
            "ground_truth": {
                527: 5.0,    # Schindler's List
                318: 4.5,    # The Shawshank Redemption
                597: 3.0,    # Pretty Woman (under the threshold, measure the precision of the model)
            },
        },

        # Notes for varied genres — test if the model avoids the "bubble" effect
        {
            "name": "eclectic_casual",
            "input": {
                1: 3.5,      # Toy Story
                260: 4.0,    # Star Wars: Episode IV
                318: 4.0,    # The Shawshank Redemption
                2571: 3.5,   # The Matrix
                1721: 3.5,   # Titanic
                296: 4.0,    # Pulp Fiction
            },
            "ground_truth": {
                2858: 3.5,   # American Beauty
                2959: 4.0,   # Fight Club
                4993: 3.5,   # LOTR: The Fellowship of the Ring
                79132: 4.5,  # Inception
            },
        },

        # Classic adventures / franchises — test the recommendation of consistent sequels
        {
            "name": "classic_adventures",
            "input": {
                1198: 5.0,   # Raiders of the Lost Ark
                1291: 4.5,   # Indiana Jones and the Last Crusade
                260: 5.0,    # Star Wars: Episode IV
                1196: 4.5,   # Star Wars: Episode V
            },
            "ground_truth": {
                1210: 4.5,   # Star Wars: Episode VI
                4993: 4.0,   # LOTR: The Fellowship of the Ring
                5952: 4.0,   # LOTR: The Two Towers
                480: 3.5,    # Jurassic Park
            },
        },
    ]
