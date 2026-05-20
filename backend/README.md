# Movix backend (FastAPI)

Sert les recommandations au frontend Movix (`frontend/`).

## Lancement local

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Healthcheck : <http://localhost:8000/healthz>

Si `backend/artifacts/hybrid_model.pkl` est absent, le serveur démarre sur un
fallback popularité (`DummyHybrid`) — la liaison frontend fonctionne quand même.

## Lier le frontend

`frontend/config.js` est git-ignoré. Au premier clone, copier le template :

```bash
cp frontend/config.example.js frontend/config.js
```

Puis dans `frontend/config.js` :

```js
API_BASE_URL: 'http://localhost:8000',
USE_MOCK: false,
TMDB_API_KEY: '...',     // ta clé TMDB (gratuite)
GEMINI_API_KEY: '...',   // optionnel
```

Puis servir le frontend (le navigateur refuse certaines requêtes `fetch` en
`file://`) :

```bash
cd frontend && python3 -m http.server 5500
```

Et ouvrir <http://localhost:5500/>.

## Tester

```bash
pytest backend/tests -v
```

## Pour le partenaire qui livre le modèle hybride

Voir **`backend/CONTRIBUTING_HYBRID.md`**.

## Endpoints

| Méthode | Route                          | Description                              |
|---------|--------------------------------|------------------------------------------|
| GET     | `/healthz`                     | Statut + modèle chargé                   |
| GET     | `/onboarding/movies`           | Films pour l'écran d'onboarding          |
| POST    | `/onboarding/submit`           | Enregistre ≥5 notes, retourne un token   |
| GET     | `/recommendations/{token}`     | Hero + 8 carrousels                      |
| POST    | `/rate`                        | Ajoute une note                          |
| POST    | `/watchlist`                   | Ajoute / retire de la watchlist          |
| GET     | `/profile/{token}`             | Stats utilisateur                        |

## Convention IDs

- **`movie_id`** = MovieLens ID (interne, utilisé par le modèle et les CSVs).
- **`tmdb_id`** = TMDB ID (utilisé par le frontend pour les posters).

Le backend les sert tous les deux dans chaque réponse — `frontend/js/tmdb.js`
utilise `tmdb_id` pour résoudre les images.
