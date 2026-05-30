// ─── EventBus ────────────────────────────────────────────────────────────────
const eventBus = {
  listeners: {},
  on(event, cb) {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(cb);
  },
  emit(event, data) {
    (this.listeners[event] || []).forEach(cb => cb(data));
  }
};

// Genres for the Discover dropdown.
//   tmdb : TMDB genre id (for the live "Top 8 in <genre>" carousel)
//   ml   : matching MovieLens genre name(s) used to filter the model carousels
//          (our model movies carry MovieLens genres, e.g. "Sci-Fi", "Children")
const DISCOVER_GENRES = [
  { name: 'Action',      tmdb: 28,    ml: ['Action'] },
  { name: 'Adventure',   tmdb: 12,    ml: ['Adventure'] },
  { name: 'Animation',   tmdb: 16,    ml: ['Animation'] },
  { name: 'Children',    tmdb: 10751, ml: ['Children'] },
  { name: 'Comedy',      tmdb: 35,    ml: ['Comedy'] },
  { name: 'Crime',       tmdb: 80,    ml: ['Crime'] },
  { name: 'Documentary', tmdb: 99,    ml: ['Documentary'] },
  { name: 'Drama',       tmdb: 18,    ml: ['Drama'] },
  { name: 'Fantasy',     tmdb: 14,    ml: ['Fantasy'] },
  { name: 'Horror',      tmdb: 27,    ml: ['Horror'] },
  { name: 'Musical',     tmdb: 10402, ml: ['Musical'] },
  { name: 'Mystery',     tmdb: 9648,  ml: ['Mystery'] },
  { name: 'Romance',     tmdb: 10749, ml: ['Romance'] },
  { name: 'Sci-Fi',      tmdb: 878,   ml: ['Sci-Fi'] },
  { name: 'Thriller',    tmdb: 53,    ml: ['Thriller'] },
  { name: 'War',         tmdb: 10752, ml: ['War'] },
  { name: 'Western',     tmdb: 37,    ml: ['Western'] },
];

// A model movie (carrying MovieLens genres) matches a Discover genre?
function movieMatchesGenre(movie, genre) {
  const gs = movie.genres || [];
  return genre.ml.some(name => gs.includes(name));
}

// Home page rows after the hero, in display order.
// Each is backed by one of the 4 backend models (matched by `model`).
const HOME_CAROUSELS = [
  { model: 'ials',          label: 'Top Picks For You' },
  { model: 'content_based', label: 'Because You Like' },
  { model: 'user_based',    label: 'Viewers Like You Also Watched' },
  { model: 'bpr',           label: 'Discover Something New' },
];

// Pick the model-backed carousels we want to display, in the right order,
// and apply the canonical label for each.
function selectHomeCarousels(carousels) {
  const byModel = {};
  carousels.forEach(c => { byModel[c.model] = c; });
  return HOME_CAROUSELS
    .filter(spec => byModel[spec.model])
    .map(spec => ({ ...byModel[spec.model], label: spec.label }));
}

let heroMovies = [];
let heroIndex = 0;
let heroTimer = null;
let allMoviesPool = [];
let homeModelCarousels = [];       // the 4 model carousels, FULL movie lists (for genre filtering in Discover)
let discoverLoaded = false;
let surpriseInit = false;
let lastSurpriseId = null;     // pour éviter de retomber sur le même film au reroll
let surpriseCandidatesData = [];  // films non vus + score = VRAI match (note prédite)
let selectedDiscoverGenre = DISCOVER_GENRES[0].tmdb;  // default = first genre

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  auth.requireAuth();
  updateStreak();

  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => navbar.classList.toggle('scrolled', window.scrollY > 10));

  initModal();
  initSagaModal();
  navigation.init([]);

  let data;
  try {
    data = await loadRecommendations();
  } catch (err) {
    console.error('Failed to load recommendations:', err);
    return;
  }
  if (!data) return;  // une reprise de session a redirigé (onboarding)

  allMoviesPool = [data.hero, ...data.carousels.flatMap(c => c.movies)].filter(Boolean);
  navigation.setMoviePool(allMoviesPool);

  // Vivier « Surprise me » : UNIQUEMENT les modèles qui prédisent une NOTE sur
  // l'échelle 0.5–5 (content/svd/user_based) → leur `score` normalisé est un vrai
  // % de match avec les ratings de l'utilisateur. On affiche systématiquement le
  // score content-based (le modèle « basé sur vos goûts ») quand le film y figure,
  // et on n'agrège JAMAIS via un max inter-modèles ni via les carrousels de genre
  // (score moyenné) → jamais de % « sorti de nulle part ».
  const RATING_MODELS = new Set(['content_based', 'svd', 'user_based']);
  const contentScore = new Map(
    (data.carousels.find(c => c.model === 'content_based')?.movies || [])
      .filter(m => m && typeof m.score === 'number')
      .map(m => [m.movie_id, m.score])
  );
  const surpriseSeen = new Map();
  for (const c of data.carousels) {
    if (!RATING_MODELS.has(c.model)) continue;
    for (const m of (c.movies || [])) {
      if (!m || m.movie_id == null) continue;
      const genuine = contentScore.has(m.movie_id)
        ? contentScore.get(m.movie_id)                                  // match content-based (canonique)
        : (typeof m.score === 'number' ? m.score : null);              // sinon note prédite svd/user_based
      if (genuine == null) continue;
      if (!surpriseSeen.has(m.movie_id) || contentScore.has(m.movie_id)) {
        surpriseSeen.set(m.movie_id, { ...m, score: genuine, _genuineMatch: true });
      }
    }
  }
  surpriseCandidatesData = [...surpriseSeen.values()];

  const recCarousels = data.carousels.filter(c => !c.model.startsWith('genre:'));
  homeModelCarousels = selectHomeCarousels(recCarousels);   // full lists (kept for Discover genre filtering)
  buildHero(data, recCarousels);
  // Home shows up to 20 per row; Discover filters the full lists by genre.
  await renderHomeCarousels(homeModelCarousels.map(c => ({ ...c, movies: c.movies.slice(0, 20) })), allMoviesPool);
});

// Charge les recommandations en survivant à un token périmé. Le store backend est
// en mémoire : à chaque redémarrage d'uvicorn (ou reload sur modif de code), les
// tokens créés à l'onboarding disparaissent → /recommendations/{token} renvoie 404.
// Plutôt que de planter (page qui charge à l'infini → on devait vider le localStorage
// à la main), on RECONSTRUIT la session à partir des ratings déjà stockés en
// localStorage : on les re-soumet pour obtenir un token frais, puis on réessaie.
// Aucun re-onboarding, aucun vidage de cache. (Le profil démo Lenny, lui, a un token
// fixe reconstruit côté serveur → il ne tombe jamais en 404.)
async function loadRecommendations() {
  const token = auth.getToken();
  try {
    return await api.getRecommendations(token);
  } catch (err) {
    if (!err || err.status !== 404) throw err;  // vraie erreur réseau → on remonte

    const stored = JSON.parse(pstore.get('ratings') || '{}');
    const arr = Object.entries(stored).map(([id, r]) => ({ movie_id: parseInt(id), rating: r }));
    if (arr.length >= 5) {
      const res = await api.submitOnboarding(arr);  // pose un nouveau token dans pstore
      if (res?.user_token) return await api.getRecommendations(res.user_token);
    }

    // Pas assez de ratings pour reconstruire → repartir proprement par l'onboarding
    pstore.remove('user_token');
    window.location.href = 'onboarding.html';
    return null;
  }
}

function updateStreak() {
  const today = new Date().toDateString();
  const lastActive = pstore.get('last_active');
  let streak = parseInt(pstore.get('streak_days') || '0');

  if (!lastActive) {
    streak = 1;
  } else if (lastActive === today) {
    return; // already counted today
  } else {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    streak = (lastActive === yesterday.toDateString()) ? streak + 1 : 1;
  }
  pstore.set('streak_days', String(streak));
  pstore.set('last_active', today);
}

// ─── Hero ─────────────────────────────────────────────────────────────────────

/**
 * Derive the user's favourite genre(s) from their stored ratings,
 * then pick the top hero movies accordingly.
 */
function getPersonalisedHeroMovies(data, recCarousels) {
  const ratings = JSON.parse(pstore.get('ratings') || '{}');
  const allPool = [data.hero, ...data.carousels.flatMap(c => c.movies)].filter(Boolean);

  // If the user has rated films, bias the hero toward their favourite genres
  if (Object.keys(ratings).length > 0) {
    // Count genre scores weighted by user rating
    const genreScore = {};
    for (const [id, rating] of Object.entries(ratings)) {
      const movie = allPool.find(m => String(m.movie_id) === String(id));
      if (!movie) continue;
      for (const g of (movie.genres || [])) {
        genreScore[g] = (genreScore[g] || 0) + rating;
      }
    }

    // Sort genres by cumulative score
    const topGenres = Object.entries(genreScore)
      .sort((a, b) => b[1] - a[1])
      .map(([g]) => g)
      .slice(0, 3);

    if (topGenres.length) {
      // Pick highly-scored candidates that match top genres (exclude already-rated films)
      const ratedIds = new Set(Object.keys(ratings).map(Number));
      const candidates = allPool
        .filter(m => !ratedIds.has(m.movie_id))
        .filter(m => (m.genres || []).some(g => topGenres.includes(g)))
        .sort((a, b) => (b.score || 0) - (a.score || 0));

      if (candidates.length >= 2) {
        // Use top 3 personalised candidates as the hero rotation
        return candidates.slice(0, 3);
      }
    }
  }

  // Fallback: original behaviour (hero + first 2 rec movies)
  return [data.hero, ...(recCarousels[0]?.movies?.slice(0, 2) || [])].filter(Boolean).slice(0, 3);
}

function buildHero(data, recCarousels) {
  heroMovies = getPersonalisedHeroMovies(data, recCarousels);
  heroIndex = 0;

  applyHero(heroMovies[0]);
  renderHeroDots();
  startHeroTimer();

  heroMovies.forEach((m, i) => {
    if (!m?.tmdb_id) return;
    getMovieDetails(m.tmdb_id).then(d => {
      m._details = d;
      if (i === heroIndex) {
        crossfadeHeroBg(d.backdrop_url);
        _animateHeroContent(() => fillHeroDetails(d, m));
      }
    }).catch(() => { });
  });
}

function applyHero(movie) {
  if (!movie) return;
  const url = movie._details?.backdrop_url || movie.backdrop_url || 'img/placeholder_backdrop.svg';
  const heroEl = document.getElementById('hero');
  heroEl.style.backgroundImage = `url('${url}')`;
  heroEl.style.cursor = 'pointer';
  document.getElementById('hero-title').textContent = movie.title || '';
  document.getElementById('hero-overview').textContent = '';
  document.getElementById('hero-badge').innerHTML = '';
  document.getElementById('hero-meta').innerHTML = '';
  document.getElementById('hero-info-btn').onclick = (e) => { e.stopPropagation(); openMovieModal(movie); };
  document.getElementById('hero-play-btn').onclick = (e) => { e.stopPropagation(); window.location.href = 'play.html'; };
  heroEl.onclick = (e) => {
    // Open modal when clicking the hero background, but ignore clicks on the dots/buttons
    if (e.target.closest('.hero-actions') || e.target.closest('.hero-dots')) return;
    openMovieModal(movie);
  };
  if (movie._details) fillHeroDetails(movie._details, movie);
}

function fillHeroDetails(details, meta) {
  document.getElementById('hero-title').textContent = details.title;
  document.getElementById('hero-overview').textContent = details.overview || '';
  document.getElementById('hero-badge').innerHTML = (details.genres || []).slice(0, 3)
    .map(g => `<span class="hero-genre-pill">${g}</span>`).join('');

  const match = meta.score ? Math.round(meta.score * 100) : null;
  document.getElementById('hero-meta').innerHTML = [
    match ? `<span class="hero-meta-score">${match}% Match</span>` : '',
    details.year ? `<span>${details.year}</span>` : '',
    details.runtime_min ? `<span>${details.runtime_min} min</span>` : '',
    details.vote_average ? `<span class="hero-meta-stars"><span class="tmdb-star">★</span> ${details.vote_average.toFixed(1)}<span class="tmdb-outof">/10</span></span>` : ''
  ].filter(Boolean).join('');
}

function _animateHeroContent(updateFn) {
  const content = document.getElementById('hero-content');
  content.style.transition = 'opacity 0.2s ease, transform 0.2s ease';
  content.style.opacity = '0';
  content.style.transform = 'translateY(10px)';
  setTimeout(() => {
    updateFn();
    content.style.opacity = '1';
    content.style.transform = '';
    setTimeout(() => { content.style.transition = ''; }, 300);
  }, 220);
}

function crossfadeHeroBg(newUrl) {
  if (!newUrl || newUrl.includes('placeholder')) return;
  const hero = document.getElementById('hero');

  let overlay = hero.querySelector('.hero-bg-crossfade');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.className = 'hero-bg-crossfade';
    hero.insertBefore(overlay, hero.firstChild);
  }

  overlay.style.backgroundImage = `url('${newUrl}')`;
  overlay.style.opacity = '0';

  requestAnimationFrame(() => requestAnimationFrame(() => {
    overlay.style.opacity = '1';
    setTimeout(() => {
      hero.style.backgroundImage = `url('${newUrl}')`;
      overlay.style.opacity = '0';
    }, 850);
  }));
}

function refreshHeroContent() {
  const m = heroMovies[heroIndex];
  if (!m) return;

  document.getElementById('hero-info-btn').onclick = (e) => { e.stopPropagation(); openMovieModal(m); };
  document.getElementById('hero-play-btn').onclick = (e) => { e.stopPropagation(); window.location.href = 'play.html'; };
  document.getElementById('hero').onclick = (e) => {
    if (e.target.closest('.hero-actions') || e.target.closest('.hero-dots')) return;
    openMovieModal(m);
  };

  if (m._details) {
    crossfadeHeroBg(m._details.backdrop_url);
    _animateHeroContent(() => fillHeroDetails(m._details, m));
  } else {
    _animateHeroContent(() => {
      document.getElementById('hero-title').textContent = m.title || '';
    });
  }
}

function renderHeroDots() {
  const dotsEl = document.getElementById('hero-dots');
  dotsEl.innerHTML = heroMovies.map((_, i) =>
    `<button class="hero-dot${i === 0 ? ' active' : ''}" data-index="${i}" aria-label="Film ${i + 1}"></button>`
  ).join('');
  dotsEl.querySelectorAll('.hero-dot').forEach(dot => {
    dot.addEventListener('click', () => {
      heroIndex = parseInt(dot.dataset.index);
      refreshHeroContent();
      syncHeroDots();
      restartHeroTimer();
    });
  });
}

function syncHeroDots() {
  document.querySelectorAll('.hero-dot').forEach((d, i) => d.classList.toggle('active', i === heroIndex));
}

function startHeroTimer() {
  clearInterval(heroTimer);
  heroTimer = setInterval(() => {
    heroIndex = (heroIndex + 1) % heroMovies.length;
    refreshHeroContent();
    syncHeroDots();
  }, 8000);
}

function restartHeroTimer() { clearInterval(heroTimer); startHeroTimer(); }

// ─── Home carousels ───────────────────────────────────────────────────────────
async function renderHomeCarousels(carousels, allMovies) {
  const container = document.getElementById('carousels-container');
  container.innerHTML = '';

  // ── Carousel 1: live TMDB Trending (Top 8, ranked) — unchanged ─────────────
  const trendingSection = createCarouselSection({
    id: 'tmdb-trending', label: 'Top 8 Today', model: 'trending', showRank: true
  });
  container.appendChild(trendingSection);

  // ── Carousels 2–5: model-backed recommendation rows ────────────────────────
  for (const c of carousels) {
    container.appendChild(createCarouselSection(c));
  }

  if (typeof lucide !== 'undefined') lucide.createIcons();

  for (const c of carousels) await populateCarousel(c);

  // ── Populate the live TMDB carousel (parallel, no await blocking) ──────────
  loadTrendingCarousel();
}

async function loadTrendingCarousel() {
  try {
    const results = await TMDB.trending('day');
    const movies = results.slice(0, 8).map(r => TMDB.parseLite(r));
    populateCarouselWithPosters({ id: 'tmdb-trending', movies, showRank: true });
    if (typeof lucide !== 'undefined') lucide.createIcons();
  } catch (e) {
    const track = document.getElementById('track-tmdb-trending');
    if (track) track.innerHTML = '<p class="empty-state" style="padding:20px">Could not load.</p>';
  }
}

async function refreshCarousels() {
  const data = await loadRecommendations();  // résilient à un token périmé (restart)
  if (!data) return;
  const allMovies = [data.hero, ...data.carousels.flatMap(c => c.movies)].filter(Boolean);
  const recCarousels = data.carousels.filter(c => !c.model.startsWith('genre:'));
  homeModelCarousels = selectHomeCarousels(recCarousels);
  const container = document.getElementById('carousels-container');
  container.style.transition = 'opacity 0.3s';
  container.style.opacity = '0.5';
  await renderHomeCarousels(homeModelCarousels.map(c => ({ ...c, movies: c.movies.slice(0, 20) })), allMovies);
  container.style.opacity = '1';
  setTimeout(() => { container.style.transition = ''; }, 350);
}

// ─── Discover view — custom dropdown (clean streaming-site control) ──────────
async function renderDiscoverView() {
  const dropdown = document.getElementById('genre-dropdown');
  const trigger = document.getElementById('genre-dropdown-trigger');
  const panel = document.getElementById('genre-dropdown-panel');
  const labelEl = document.getElementById('genre-dropdown-label');
  if (!dropdown || !trigger || !panel || !labelEl) return;

  if (!discoverLoaded) {
    // Build options: one per genre (no "For you")
    panel.innerHTML = DISCOVER_GENRES.map(g => `
      <li class="genre-dropdown-item" role="option" data-value="${g.tmdb}">${g.name}</li>
    `).join('');

    const closePanel = () => {
      dropdown.classList.remove('open');
      trigger.setAttribute('aria-expanded', 'false');
    };
    const openPanel = () => {
      dropdown.classList.add('open');
      trigger.setAttribute('aria-expanded', 'true');
    };

    trigger.addEventListener('click', (e) => {
      e.stopPropagation();
      dropdown.classList.contains('open') ? closePanel() : openPanel();
    });

    panel.querySelectorAll('.genre-dropdown-item').forEach(item => {
      item.addEventListener('click', () => {
        const value = item.dataset.value;
        panel.querySelectorAll('.genre-dropdown-item').forEach(it => it.classList.toggle('selected', it === item));
        labelEl.textContent = item.textContent.trim();
        selectedDiscoverGenre = parseInt(value);
        sessionStorage.setItem('discover_genre', value);
        closePanel();
        renderDiscoverContent();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    });

    // Outside click + Escape close the panel
    document.addEventListener('click', e => {
      if (!dropdown.contains(e.target)) closePanel();
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closePanel();
    });

    // Restore previous selection, else default to the first genre
    const saved = sessionStorage.getItem('discover_genre');
    const startItem =
      (saved && panel.querySelector(`.genre-dropdown-item[data-value="${saved}"]`))
      || panel.querySelector('.genre-dropdown-item');
    if (startItem) {
      startItem.classList.add('selected');
      labelEl.textContent = startItem.textContent.trim();
      selectedDiscoverGenre = parseInt(startItem.dataset.value);
    }

    discoverLoaded = true;
  }

  renderDiscoverContent();
}

function renderDiscoverContent() {
  const container = document.getElementById('genre-carousels-container');
  if (!container) return;
  container.innerHTML = '';

  const genre = DISCOVER_GENRES.find(g => g.tmdb === selectedDiscoverGenre);
  if (!genre) return;

  // Same 5 rows as the home page, but adapted to the selected genre.

  // 1. Live TMDB "Top 8 in <genre>" (ranked) — mirrors the home "Top 8 Today".
  const trendingId = `disc-trending-${genre.tmdb}`;
  container.appendChild(createCarouselSection({
    id: trendingId, label: `Top 8 in ${genre.name}`, model: 'trending', showRank: true
  }));

  // 2–5. The four model carousels, filtered to this genre (identical display).
  const modelCarousels = homeModelCarousels.map(c => ({
    ...c,
    id: `disc-${c.model}-${genre.tmdb}`,
    movies: c.movies.filter(m => movieMatchesGenre(m, genre)).slice(0, 20),
  }));
  for (const c of modelCarousels) container.appendChild(createCarouselSection(c));

  if (typeof lucide !== 'undefined') lucide.createIcons();

  // Populate row 1 from TMDB.
  TMDB.discoverByGenre({ genres: [genre.tmdb], sort_by: 'popularity.desc' })
    .then(results => {
      const movies = results.slice(0, 8).map(r => TMDB.parseLite(r));
      populateCarouselWithPosters({ id: trendingId, movies, showRank: true });
      if (typeof lucide !== 'undefined') lucide.createIcons();
    })
    .catch(() => {
      const track = document.getElementById(`track-${trendingId}`);
      if (track) track.innerHTML = '<p class="empty-state" style="padding:20px">Could not load.</p>';
    });

  // Populate rows 2–5 from the genre-filtered model recommendations.
  for (const c of modelCarousels) {
    if (c.movies.length) {
      populateCarousel(c);
    } else {
      const track = document.getElementById(`track-${c.id}`);
      if (track) track.innerHTML = `<p class="empty-state" style="padding:20px">No ${genre.name} picks yet — rate more films.</p>`;
    }
  }
}

// ─── Surprise Me view ────────────────────────────────────────────────────────
async function initSurpriseView() {
  if (surpriseInit) return;
  surpriseInit = true;
  document.getElementById('surprise-btn')?.addEventListener('click', rollSurprise);
  rollSurprise();
}

// Candidats « Surprise me » : films non vus dont le `score` est garanti être un
// VRAI match avec les ratings de l'utilisateur (note prédite content/svd/user_based,
// cf. surpriseCandidatesData construit au chargement). On exclut les films déjà
// notés/vus et on trie par match décroissant.
function buildSurpriseCandidates() {
  const ratings = JSON.parse(pstore.get('ratings') || '{}');
  const ratedIds = new Set(Object.keys(ratings).map(Number));
  return surpriseCandidatesData
    .filter(m => typeof m.score === 'number' && m.user_rating == null && !ratedIds.has(m.movie_id))
    .sort((a, b) => (b.score || 0) - (a.score || 0));
}

// Tirage : on reste DANS les meilleurs matchs (top ~40 %) pour que le film
// proposé plaise vraiment — le but est de faire gagner du temps, pas de sortir
// une curiosité au hasard — mais on pondère par le score et on évite le dernier
// tirage, donc chaque lancer reste une vraie surprise.
function pickSurprise(candidates) {
  if (!candidates.length) return null;
  const windowSize = Math.min(40, Math.max(8, Math.ceil(candidates.length * 0.4)));
  let pool = candidates.slice(0, windowSize);
  if (pool.length > 1 && lastSurpriseId != null) {
    const filtered = pool.filter(m => m.movie_id !== lastSurpriseId);
    if (filtered.length) pool = filtered;
  }
  const weights = pool.map(m => Math.pow((m.score || 0.01) + 0.05, 2));  // ∝ match²
  const total = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  let chosen = pool[0];
  for (let i = 0; i < pool.length; i++) {
    r -= weights[i];
    if (r <= 0) { chosen = pool[i]; break; }
  }
  lastSurpriseId = chosen.movie_id;
  return chosen;
}

async function rollSurprise() {
  const result = document.getElementById('surprise-result');
  const btn = document.getElementById('surprise-btn');
  if (!result) return;

  // Trigger dice roll animation on the button icon
  if (btn) {
    btn.classList.remove('rolling');
    // Force reflow so the animation can replay on rapid re-clicks
    void btn.offsetWidth;
    btn.classList.add('rolling');
    setTimeout(() => btn.classList.remove('rolling'), 760);
  }

  // 1) Cas nominal : tirer parmi les recommandations personnalisées (films non vus
  //    que l'utilisateur va aimer). C'est le rôle attendu du « Surprise me ».
  let movie = pickSurprise(buildSurpriseCandidates());

  // 2) Repli cold-start (aucune reco perso dispo) : pépite TMDB bien notée.
  if (!movie) {
    try {
      const randomGenre = DISCOVER_GENRES[Math.floor(Math.random() * DISCOVER_GENRES.length)];
      const page = 1 + Math.floor(Math.random() * 5);
      const results = await TMDB.discoverByGenre({
        genres: [randomGenre.tmdb],
        sort_by: 'vote_average.desc',
        vote_count_gte: 300,
        vote_count_lte: 5000,
        page,
      });
      const pool = results.map(r => TMDB.parseLite(r));
      if (pool.length) movie = pool[Math.floor(Math.random() * pool.length)];
    } catch { /* ignore */ }
  }

  if (!movie) {
    result.innerHTML = '<p class="empty-state">Nothing to surprise you with right now.</p>';
    return;
  }

  result.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
  result.style.opacity = '0';
  result.style.transform = 'translateY(10px)';

  // Sync the result reveal with the mid-point of the dice animation
  setTimeout(async () => {
    let details;
    try { details = await getMovieDetails(movie.tmdb_id); } catch { details = null; }
    const poster = details?.poster_url || movie.poster_url || 'img/placeholder_poster.svg';
    const backdrop = details?.backdrop_url || movie.backdrop_url || 'img/placeholder_backdrop.svg';
    // % Match affiché UNIQUEMENT pour un vrai match basé sur les ratings (note
    // prédite). Le repli TMDB (vote_average) n'est PAS un match utilisateur → pas
    // de badge, jamais de % « sorti de nulle part ».
    const match = (movie._genuineMatch && typeof movie.score === 'number')
      ? Math.round(movie.score * 100)
      : null;
    const matchBadge = match != null
      ? `<span class="hero-meta-score" style="margin-right:8px">${match}% Match</span>`
      : '';
    result.innerHTML = `
      <div class="surprise-card" data-movie-id="${movie.movie_id}" style="background-image:url('${backdrop}')">
        <div class="surprise-card-gradient"></div>
        <img class="surprise-card-poster" src="${poster}" alt="${escapeHtml(movie.title)}">
        <div class="surprise-card-info">
          <h2>${escapeHtml(details?.title || movie.title)}</h2>
          <p class="surprise-meta">${matchBadge}${(details?.genres || movie.genres || []).slice(0, 3).join(' · ') || ''} ${details?.year ? '· ' + details.year : ''}</p>
          <p class="surprise-overview">${escapeHtml((details?.overview || '').slice(0, 280))}</p>
          <div class="surprise-actions">
            <button class="btn-primary" id="surprise-open"><i data-lucide="play"></i> Play</button>
            <button class="btn-secondary" id="surprise-reroll"><i data-lucide="dices"></i> Try another</button>
          </div>
        </div>
      </div>`;
    if (typeof lucide !== 'undefined') lucide.createIcons();

    document.getElementById('surprise-open')?.addEventListener('click', () => openMovieModal(movie));
    document.querySelector('.surprise-card')?.addEventListener('click', e => {
      if (e.target.closest('button')) return;
      openMovieModal(movie);
    });
    document.getElementById('surprise-reroll')?.addEventListener('click', rollSurprise);

    result.style.opacity = '1';
    result.style.transform = '';
  }, 380);
}

// ─── Watchlist view ───────────────────────────────────────────────────────────
async function renderWatchlistView() {
  const container = document.getElementById('watchlist-full-grid');
  if (!container) return;

  const ids = JSON.parse(pstore.get('watchlist') || '[]');
  const countEl = document.getElementById('watchlist-count-header');

  if (!ids.length) {
    container.innerHTML = `
      <div class="empty-watchlist">
        <div class="empty-watchlist-icon"><i data-lucide="bookmark" style="width:40px;height:40px"></i></div>
        <h3>Your list is empty</h3>
        <p>Add films from the carousels by clicking <strong>+</strong> on any card.</p>
        <button class="btn-primary" onclick="navigation.showView('home')">
          <i data-lucide="home"></i> Browse Recommendations
        </button>
      </div>`;
    if (countEl) countEl.textContent = '';
    if (typeof lucide !== 'undefined') lucide.createIcons();
    return;
  }

  if (countEl) countEl.textContent = `${ids.length} film${ids.length > 1 ? 's' : ''}`;

  // Fetch movies: try local pool first, then TMDB
  const poolMovies = ids
    .map(id => allMoviesPool.find(m => m.movie_id === id))
    .filter(Boolean);

  // For IDs not in the local pool, load from TMDB
  const poolIds = new Set(poolMovies.map(m => m.movie_id));
  const tmdbIds = ids.filter(id => !poolIds.has(id));
  const tmdbMovies = await Promise.allSettled(tmdbIds.map(id => getMovieDetails(id)));
  const extra = tmdbMovies.filter(r => r.status === 'fulfilled').map(r => r.value);

  const movies = [...poolMovies, ...extra];

  if (!movies.length) {
    container.innerHTML = '<p class="empty-state">Films not found in the catalog.</p>';
    return;
  }

  container.innerHTML = '';

  for (const movie of movies) {
    const card = document.createElement('div');
    card.className = 'watchlist-full-card';
    card.style.animation = 'cardReveal 0.35s ease both';
    card.innerHTML = `
      <div class="wfc-poster-wrap">
        <img src="img/placeholder_poster.svg" alt="${escapeHtml(movie.title)}" loading="lazy">
        <div class="card-overlay">
          <div class="card-title">${escapeHtml(movie.title)}</div>
          <div class="card-meta">${movie.year || ''}${movie.genres?.length ? ' · ' + movie.genres.slice(0, 2).join(', ') : ''}</div>
          <div class="card-actions">
            <button class="btn-watchlist-card in-list wfc-remove" data-movie-id="${movie.movie_id}" title="Remove from list">
              <i data-lucide="check" class="wl-icon-default"></i><i data-lucide="minus" class="wl-icon-hover"></i>
            </button>
            ${movie.score ? `<span class="card-score">${Math.round(movie.score * 100)}%</span>` : ''}
          </div>
        </div>
      </div>`;

    // Set poster (already loaded if extra, else lazy-load)
    if (movie.poster_url && movie.poster_url.includes('image.tmdb')) {
      card.querySelector('img').src = movie.poster_url;
    } else {
      getMovieDetails(movie.tmdb_id).then(d => {
        const img = card.querySelector('img');
        if (img) img.src = d.poster_url;
      }).catch(() => { });
    }

    card.querySelector('.wfc-poster-wrap').addEventListener('click', e => {
      if (!e.target.closest('.wfc-remove')) openMovieModal(movie);
    });

    card.querySelector('.wfc-remove').addEventListener('click', async e => {
      e.stopPropagation();
      await watchlist.toggle(movie.movie_id);
      card.style.cssText = 'opacity:0;transform:scale(0.88);transition:all 0.22s ease';
      setTimeout(() => {
        card.remove();
        const rem = container.querySelectorAll('.watchlist-full-card').length;
        if (countEl) countEl.textContent = rem ? `${rem} film${rem > 1 ? 's' : ''}` : '';
        if (!rem) renderWatchlistView();
      }, 230);
    });

    container.appendChild(card);
  }

  if (typeof lucide !== 'undefined') lucide.createIcons();
}

// ─── EventBus ─────────────────────────────────────────────────────────────────
eventBus.on('rating:updated', async ({ movieId, rating }) => {
  updateStreak();
  if (!CONFIG.USE_MOCK) await refreshCarousels();
});
