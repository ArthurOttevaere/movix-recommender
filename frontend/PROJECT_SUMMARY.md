# MOVIX — Résumé du Projet

## Vue d'ensemble
**Movix** est une application web de recommandation de films personnalisée. Elle utilise un moteur de recommandation multi-modèles pour suggérer des films en fonction du profil utilisateur, avec une interface Netflix-like et un chatbot IA (Gemini) pour assistance.

---

## 🎯 Objectif Principal
Fournir des recommandations de films **hautement personnalisées** en fonction :
- Des évaluations de films de l'utilisateur
- De multiples algorithmes de recommandation
- De préférences par genre et époque
- D'interactions conversationnelles avec un chatbot IA

---

## 🏗️ Architecture Générale

### Frontend
- **Type :** Single Page Application (SPA) vanilla JavaScript
- **Données :** TMDB API (base de données de films)
- **Stockage :** LocalStorage + ProfileStore (pstore) pour données locales
- **Mode mock :** Données statiques en JSON pour développement

### Backend (optionnel)
- **Stack :** FastAPI (Python)
- **Point d'entrée :** `CONFIG.API_BASE_URL` (null = mode mock, `http://localhost:8000` = backend)
- **Endpoints :** Onboarding, recommandations, notation, watchlist

---

## 📂 Structure des Fichiers

### Pages HTML
| Fichier | Rôle |
|---------|------|
| `index.html` | Page principale - accueil avec carousels de recommandations |
| `onboarding.html` | Écran d'onboarding - l'utilisateur note 5+ films pour initialiser son profil |
| `profiles.html` | Gestionnaire de profils multiples (Netflix-style) |
| `profile.html` | Page de profil utilisateur - paramètres et statistiques |
| `play.html` | Lecteur de film (non détaillé) |
| `movie.html` | Page de détails d'un film |

### Dossier CSS
- `main.css` - Styles globaux et variables CSS
- `navbar.css` - Barre de navigation
- `hero.css` - Section héros (film principal)
- `carousel.css` - Carousels de recommandations
- `modal.css` - Modales popup
- `onboarding.css` - Écran d'onboarding
- `profile.css` - Page de profil
- `profiles.css` - Gestionnaire de profils
- `chatbot.css` - Chatbot conversationnel
- `views.css` - Transitions entre vues

### Dossier JS
| Fichier | Responsabilité |
|---------|-----------------|
| `api.js` | Appels API backend/mock pour recommandations, onboarding, notation, watchlist |
| `auth.js` | Gestion de l'authentification et du token utilisateur |
| `carousel.js` | Rendu des carousels, création de cartes films, gestion du scroll |
| `chatbot.js` | Chatbot IA (Gemini) avec fallback keyword matcher |
| `gemini.js` | Appels API Gemini 2.0 Flash Lite |
| `home.js` | Logique de la page d'accueil - EventBus, héros, recommandations |
| `modal.js` | Gestionnaire des modales (détails film, confirmations) |
| `navigation.js` | Navigation entre vues, gestion du routage |
| `profile.js` | Gestion du profil utilisateur, settings |
| `profiles.js` | Système de profils multiples (créer, supprimer, sélectionner) |
| `ratings.js` | Système de notation de films |
| `saga.js` | Gestion des franchises/sagas de films |
| `tmdb.js` | Intégration API TMDB (détails films, images, recherche) |
| `toast.js` | Système de notifications (toasts/snackbars) |
| `watchlist.js` | Gestion de la liste de films à regarder |

### Dossier Mock Data
- `onboarding.json` - Films pour l'écran d'onboarding
- `profile.json` - Données de profil utilisateur
- `recommendations.json` - Recommandations et carousels (multiples modèles)

---

## 🔄 Flux Utilisateur Principal

### 1️⃣ Sélection/Création de Profil (`profiles.html`)
- Écran d'accueil - sélectionner un profil existant ou en créer un nouveau
- Avatars colorés personnalisables
- Données stockées dans localStorage

### 2️⃣ Onboarding (`onboarding.html`)
- Affiche une grille de films
- Utilisateur note au minimum **5 films** (échelle 1-10)
- Les notes sont enregistrées localement
- Génère un token d'authentification unique

### 3️⃣ Accueil (`index.html`)
- **Section Héros** - affiche un film "vedette" avec détails
- **Carousels horizontaux** de recommandations :
  - Content-based (features similaires)
  - Collaborative (KNN utilisateurs)
  - SVD (factorisation matricielle)
  - Ensemble (diversité)
  - Trending (populaire)
  - Genre-spécifiques (Drama, Thriller, Sci-Fi, etc.)
- **Barre de recherche** - recherche live de films
- **Chatbot CineBot** - assistant IA conversationnel

### 4️⃣ Interactions Utilisateur
- **Notation** - cliquer sur les étoiles pour évaluer
- **Watchlist** - ajouter/retirer films
- **Saga** - explorer les franchises (ex: Marvel, DC)
- **Surprise** - bouton pour recommandation aléatoire
- **Profile** - voir stats personnelles, gérer paramètres

---

## 🎨 Système de Recommandation

### Modèles Implémentés
| Modèle | Description |
|--------|-------------|
| **Content-based** | Basé sur les features du film (genre, acteurs, synopsis) |
| **User-based (KNN)** | Recommandations d'utilisateurs similaires |
| **SVD (Latent factors)** | Factorisation matricielle - découvrir patterns cachés |
| **Ensemble** | Combine plusieurs modèles pour diversité |
| **Trending** | Films populaires en temps réel |
| **History** | Basé sur l'historique de visionnage |
| **Genre-specific** | Recommandations par genre (Drama, Thriller, Sci-Fi, Animation, Crime) |

Chaque carousel affiche son modèle via un badge : *"Content-based · Genome features"*

### Chatbot CineBot
- Intégration **Gemini 2.0 Flash Lite** pour conversations naturelles
- Comprend : genres, acteurs, ambiances, décennies, exclusions
- Fallback sur **keyword matcher** (patterns regex en FR + EN) si pas de clé API
- Peut filtrer par genre, année, mood, etc.

**Exemples de requêtes:**
- "Je veux un film d'horreur des années 80"
- "Quelque chose de drôle et léger"
- "Un truc scientifique avec tension"

---

## 💾 Système de Stockage

### LocalStorage
```
cm_profiles           → Liste de tous les profils
cm_active_profile     → ID du profil actif
user_token            → Token d'authentification
p<profile_id>:ratings → Notes de films par profil
p<profile_id>:watchlist → Liste à regarder par profil
```

### ProfileStore (pstore)
- Wrapper local-first pour données structurées
- Préfixe clé avec `p<profile_id>:` pour multi-profils
- Fallback sur localStorage si pstore indisponible

---

## 🔐 Authentification

### Flux
1. **Pas d'authentification externe** - système basé sur localStorage
2. Chaque profil génère un `user_token` unique lors du premier onboarding
3. Le token est stocké localement et utilisé pour récupérer les recommandations
4. `auth.requireAuth()` redirige vers `onboarding.html` si non connecté

---

## 🎬 Intégration TMDB

### Clé API
- Stockée dans `config.js`
- Endpoints utilisés :
  - Recherche de films
  - Détails (synopsis, casting, images)
  - Images de posters/backdrops
  - Données de genres

### Images
- Base URL : `https://image.tmdb.org/t/p/w500`
- Format : Placeholder SVG en fallback

---

## 🌐 Configuration

### `config.js`
```javascript
{
  TMDB_API_KEY: 'xxx',           // Clé TMDB
  TMDB_BASE_URL: 'https://...',  // URL API TMDB
  TMDB_IMAGE_BASE: 'https://...', // CDN images
  API_BASE_URL: null,             // Backend (null = mock)
  USE_MOCK: true,                 // Utiliser données statiques
  GEMINI_API_KEY: 'xxx',          // Clé Gemini pour chatbot
}
```

---

## 🎯 Genres Supportés

18 genres TMDB + 1 catégorie custom :
- Action, Aventure, Animation
- Baby & Toddler (custom)
- Comédie, Crime
- Documentaire, Drame
- Famille, Fantastique
- Histoire, Horreur
- Mystère, Romance
- Science-fiction, Thriller
- Guerre, Western

---

## 📊 Données Utilisateur par Profil

### Profil
```json
{
  "id": "p1234abc",
  "name": "Arthur",
  "avatar": "blue",
  "onboarded": true,
  "created_at": "2024-XX-XX",
  "age": 25,
  "gender": "M"
}
```

### Recommandations
```json
{
  "hero": { "movie_id": 1, "tmdb_id": 550, "title": "Fight Club", ... },
  "carousels": [
    {
      "id": "content-based-1",
      "label": "Because You Liked...",
      "model": "content_based",
      "movies": [ { "movie_id": 2, "title": "...", "score": 0.87, ... }, ... ]
    },
    ...
  ]
}
```

---

## 🚀 Fonctionnalités Clés

✅ **Profils multiples** - Netflix-style avec avatars personnalisables  
✅ **Onboarding interactif** - notation minimale 5 films  
✅ **Recommandations multi-modèles** - 7+ algorithmes différents  
✅ **Recherche en temps réel** - overlay modal avec résultats instantanés  
✅ **Watchlist** - gérer films à regarder  
✅ **Carousels infinies** - scroll horizontal fluide  
✅ **Chatbot IA** - Gemini + fallback keyword  
✅ **Notation dynamique** - 1-10 étoiles, influence recommandations  
✅ **Système de streak** - tracker continuation viewing  
✅ **Détails films complets** - synopsis, casting, images  
✅ **Mode mock/backend** - flexible pour développement/production  

---

## 📱 Compatibilité

- **Navigateurs** : Modern browsers (ES6 support)
- **Responsive** : Mobile-first design
- **Offline** : Fonctionne en offline via localStorage

---

## 🔌 Points d'Intégration

### API Backend Potentiel
- `/onboarding/movies` - Films pour onboarding
- `/onboarding/submit` - Soumettre notes
- `/recommendations/{token}` - Récupérer recommandations
- `/rate` - Enregistrer notation
- `/watchlist` - Gérer liste

### Services Externes
- **TMDB** - Base de données films
- **Gemini 2.0** - Chatbot IA
- **LocalStorage** - Stockage client

---

## 🎓 Stack Technique Résumé

| Technologie | Rôle |
|-------------|------|
| **HTML5** | Structure pages |
| **CSS3** | Styling + animations |
| **Vanilla JavaScript** | Logique application (ES6) |
| **TMDB API** | Base de données films |
| **Gemini 2.0 Flash Lite** | Chatbot IA |
| **LocalStorage** | Persistance client |
| **FastAPI** (optionnel) | Backend recommandations |

---

## 📝 Résumé pour Autres IA

**Movix est une plateforme SPA de recommandation de films avec :**
1. Système de profils multiples stockés localement
2. Moteur de recommandation multi-algorithmes
3. Chatbot IA conversationnel (Gemini)
4. Interface Netflix-like avec carousels responsifs
5. Gestion complète du cycle utilisateur (onboarding → recommandations → notation)
6. Architecture flexible backend/mock
7. Support multilingue (FR + EN)
8. Système de watchlist et rating

**Cas d'usage principal :** Aider utilisateurs à découvrir films correspondant à leurs goûts via recommandations intelligentes et assistance conversationnelle.
