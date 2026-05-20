"""Template d'entraînement pour le modèle hybride.

Le partenaire :
1. Importe sa sous-classe de HybridRecommender depuis backend.app.recommender.impl_<nom>
2. Adapte la section MARQUÉE ci-dessous (instanciation + fit)
3. Lance :
     python -m backend.scripts.train_hybrid
4. Vérifie que backend/artifacts/hybrid_model.pkl est créé
5. Relance le serveur : `uvicorn backend.app.main:app --reload`
   /healthz doit reporter ton `name` / `version`.

Important :
  - n'importe rien du frontend ici
  - ne dépend pas de chemins absolus à l'exécution
  - la classe doit être importable par le serveur (pas définie inline ici)
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from backend.app.config import get_settings
from backend.app.recommender.base import HybridRecommender

# ----------------------------------------------------------------------------
# TODO PARTENAIRE : importe ta classe ici
# from backend.app.recommender.impl_myhybrid import MyHybrid
# ----------------------------------------------------------------------------

log = logging.getLogger("train_hybrid")


def load_data(settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(settings.ratings_path)
    movies = pd.read_csv(settings.movies_path)
    return ratings, movies


def build_model() -> HybridRecommender:
    # TODO PARTENAIRE : instancie ta classe (avec ses hyperparams)
    # return MyHybrid(n_factors=50, alpha=0.5, ...)
    raise NotImplementedError(
        "Edit backend/scripts/train_hybrid.py and instantiate your HybridRecommender subclass."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Override output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = get_settings()
    out_path: Path = args.out or settings.artifact_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading data…")
    ratings, movies = load_data(settings)
    log.info("Ratings: %s rows | Movies: %s rows", len(ratings), len(movies))

    log.info("Instantiating model…")
    model = build_model()
    if not isinstance(model, HybridRecommender):
        raise TypeError("build_model() must return a HybridRecommender instance")

    log.info("Fitting %s v%s…", model.name, model.version)
    model.fit(ratings, movies)

    log.info("Pickling to %s", out_path)
    with open(out_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Done. Restart uvicorn to load the new artifact.")


if __name__ == "__main__":
    main()
