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

const DISCOVER_GENRES = [
  { id: 28, name: 'Action' },
  { id: 12, name: 'Adventure' },
  { id: 16, name: 'Animation' },
  { id: -1, name: 'Baby & Toddler', _baby: true },
  { id: 35, name: 'Comedy' },
  { id: 80, name: 'Crime' },
  { id: 99, name: 'Documentary' },
  { id: 18, name: 'Drama' },
  { id: 10751, name: 'Family' },
  { id: 14, name: 'Fantasy' },
  { id: 36, name: 'History' },
  { id: 27, name: 'Horror' },
  { id: 9648, name: 'Mystery' },
  { id: 10749, name: 'Romance' },
  { id: 878, name: 'Science Fiction' },
  { id: 53, name: 'Thriller' },
  { id: 10752, name: 'War' },
  { id: 37, name: 'Western' },
];

let heroMovies = [];
let heroIndex = 0;
let heroTimer = null;
let allMoviesPool = [];
let discoverLoaded = false;
let surpriseInit = false;
let selectedDiscoverGenre = null;  // null = "For you" (default)

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  auth.requireAuth();
  updateStreak();

  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => navbar.classList.toggle('scrolled', window.scrollY > 10));

  initModal();
  initSagaModal();
  navigation.init([]);

  const token = auth.getToken();
  let data;
  try {
    data = await api.getRecommendations(token);
  } catch (err) {
    console.error('Failed to load recommendations:', err);
    return;
  }

  allMoviesPool = [data.hero, ...data.carousels.flatMap(c => c.movies)].filter(Boolean);
  navigation.setMoviePool(allMoviesPool);

  patchRecommendationLabels(data.carousels);

  const recCarousels = data.carousels.filter(c => !c.model.startsWith('genre:'));
  buildHero(data, recCarousels);
  await renderHomeCarousels(recCarousels, allMoviesPool);
});

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

// ─── Dynamic labels ───────────────────────────────────────────────────────────
function patchRecommendationLabels(carousels) {
  const ratings = JSON.parse(pstore.get('ratings') || '{}');
  if (!Object.keys(ratings).length) return;

  const [topId] = Object.entries(ratings).sort((a, b) => b[1] - a[1])[0];
  const topMovie = allMoviesPool.find(m => m.movie_id === parseInt(topId));
  if (!topMovie) return;

  const c = carousels.find(c => c.model === 'content_based');
  if (c) c.label = `Because You Watched ${topMovie.title}`;
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

  const cwCarousel = buildContinueWatching(allMovies);

  // Promote the trending/top-10 carousel so it appears early on the page
  const orderedRecs = [...carousels].sort((a, b) => {
    const aTop = a.model === 'trending' ? -1 : 0;
    const bTop = b.model === 'trending' ? -1 : 0;
    return aTop - bTop;
  });
  // Give a clearer "Top 10" label to the trending carousel so users recognize it,
  // and flag it as the ONLY carousel allowed to display the giant rank badge.
  orderedRecs.forEach(c => {
    if (c.model === 'trending') {
      c.label = 'Top 8 Today';
      c.showRank = true;
      // Ensure at most 10 movies are shown in this carousel
      if (c.movies && c.movies.length > 10) c.movies = c.movies.slice(0, 10);
    }
  });

  const allCarousels = cwCarousel ? [cwCarousel, ...orderedRecs] : orderedRecs;

  for (const c of allCarousels) {
    container.appendChild(createCarouselSection(c));
  }

  // ── Add live TMDB carousels: Top Rated + Hidden Gems ───────────────────────
  const topRatedSection = createCarouselSection({
    id: 'tmdb-top-rated', label: 'Top Rated of All Time', model: 'top_rated'
  });
  container.appendChild(topRatedSection);

  const hiddenGemsSection = createCarouselSection({
    id: 'tmdb-hidden-gems', label: 'Hidden Gems for You', model: 'hidden_gems'
  });
  container.appendChild(hiddenGemsSection);

  if (typeof lucide !== 'undefined') lucide.createIcons();

  for (const c of allCarousels) await populateCarousel(c);

  // ── Populate live carousels (parallel, no await blocking) ──────────────────
  loadTopRatedCarousel();
  loadHiddenGemsCarousel();
}

async function loadTopRatedCarousel() {
  try {
    const results = await TMDB.topRated(1);
    const movies = results.slice(0, 20).map(r => TMDB.parseLite(r));
    populateCarouselWithPosters({ id: 'tmdb-top-rated', movies });
    if (typeof lucide !== 'undefined') lucide.createIcons();
  } catch (e) {
    const track = document.getElementById('track-tmdb-top-rated');
    if (track) track.innerHTML = '<p class="empty-state" style="padding:20px">Could not load.</p>';
  }
}

async function loadHiddenGemsCarousel() {
  try {
    // High-rated but moderate vote count → niche-but-loved films
    // Bias toward user's favorite genres if we can infer them
    const ratings = JSON.parse(pstore.get('ratings') || '{}');
    const ratedIds = Object.keys(ratings).map(Number);
    let favGenreIds = [];
    if (ratedIds.length) {
      const details = await Promise.allSettled(ratedIds.slice(0, 12).map(id => getMovieDetails(id)));
      const counts = {};
      details.forEach(r => {
        if (r.status !== 'fulfilled') return;
        (r.value.genres || []).forEach(g => {
          // Map name back to TMDB ID
          const id = Object.entries(TMDB_GENRE_MAP).find(([, name]) => name === g)?.[0];
          if (id) counts[id] = (counts[id] || 0) + 1;
        });
      });
      favGenreIds = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 3).map(e => parseInt(e[0]));
    }

    const results = await TMDB.discoverByGenre({
      genres: favGenreIds,
      sort_by: 'vote_average.desc',
      vote_count_gte: 200,
      vote_count_lte: 2500,
    });
    const movies = results.slice(0, 20).map(r => TMDB.parseLite(r));
    populateCarouselWithPosters({ id: 'tmdb-hidden-gems', movies });
    if (typeof lucide !== 'undefined') lucide.createIcons();
  } catch (e) {
    const track = document.getElementById('track-tmdb-hidden-gems');
    if (track) track.innerHTML = '<p class="empty-state" style="padding:20px">Could not load.</p>';
  }
}

function buildContinueWatching(allMovies) {
  const ratings = JSON.parse(pstore.get('ratings') || '{}');
  const ratedIds = Object.keys(ratings).slice(-6).map(Number).reverse();
  if (ratedIds.length < 2) return null;

  const movies = ratedIds
    .map(id => allMovies.find(m => m.movie_id === id))
    .filter(Boolean)
    .map(m => ({ ...m, progress: Math.floor(Math.random() * 55) + 20 }));

  if (!movies.length) return null;
  return { id: 'continue_watching', label: 'Continue Watching', model: 'history', movies };
}

async function refreshCarousels() {
  const token = auth.getToken();
  const data = await api.getRecommendations(token);
  const allMovies = [data.hero, ...data.carousels.flatMap(c => c.movies)].filter(Boolean);
  patchRecommendationLabels(data.carousels);
  const recCarousels = data.carousels.filter(c => !c.model.startsWith('genre:'));
  const container = document.getElementById('carousels-container');
  container.style.transition = 'opacity 0.3s';
  container.style.opacity = '0.5';
  await renderHomeCarousels(recCarousels, allMovies);
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
    // Build options: "For you" first, then all genres alphabetical
    const options = [
      { value: 'all', label: 'For you' },
      ...DISCOVER_GENRES.map(g => ({ value: String(g.id), label: g.name }))
    ];
    panel.innerHTML = options.map(o => `
      <li class="genre-dropdown-item" role="option" data-value="${o.value}">${o.label}</li>
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
        selectedDiscoverGenre = (value === 'all') ? null : parseInt(value);
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

    // Restore previous selection if any
    const saved = sessionStorage.getItem('discover_genre');
    if (saved) {
      const item = panel.querySelector(`.genre-dropdown-item[data-value="${saved}"]`);
      if (item) {
        panel.querySelectorAll('.genre-dropdown-item').forEach(it => it.classList.toggle('selected', it === item));
        labelEl.textContent = item.textContent.trim();
        selectedDiscoverGenre = (saved === 'all') ? null : parseInt(saved);
      }
    } else {
      panel.querySelector('.genre-dropdown-item[data-value="all"]')?.classList.add('selected');
    }

    discoverLoaded = true;
  }

  renderDiscoverContent();
}

function renderDiscoverContent() {
  const container = document.getElementById('genre-carousels-container');
  if (!container) return;
  container.innerHTML = '';

  if (selectedDiscoverGenre == null) {
    // Default "For you" — recommendation carousels (Picked for you, Top Rated, Hidden Gems)
    renderForYouCarousels(container);
    return;
  }

  // Selected genre — show multiple curated rows for that genre
  const genre = DISCOVER_GENRES.find(g => g.id === selectedDiscoverGenre);
  if (!genre) return;

  // Special case: Baby & Toddler uses Animation + Family genre IDs
  if (genre._baby) {
    const babyGenres = [16, 10751]; // Animation + Family
    const rows = [
      { id: 'disc-baby-pop', label: 'Popular for Little Ones', sort_by: 'popularity.desc', vote_count_gte: 50 },
      { id: 'disc-baby-top', label: 'Best Rated Baby & Toddler Films', sort_by: 'vote_average.desc', vote_count_gte: 100 },
      { id: 'disc-baby-new', label: 'New Baby & Toddler Films', sort_by: 'primary_release_date.desc', year_gte: new Date().getFullYear() - 3, vote_count_gte: 20 },
    ];
    for (const r of rows) {
      container.appendChild(createCarouselSection({ id: r.id, label: r.label, model: 'genre:Animation' }));
    }
    if (typeof lucide !== 'undefined') lucide.createIcons();
    rows.forEach(r => {
      TMDB.discoverByGenre({ genres: babyGenres, sort_by: r.sort_by, vote_count_gte: r.vote_count_gte, year_gte: r.year_gte })
        .then(results => {
          const movies = results.slice(0, 20).map(x => TMDB.parseLite(x));
          populateCarouselWithPosters({ id: r.id, movies });
          if (typeof lucide !== 'undefined') lucide.createIcons();
        })
        .catch(() => {
          const track = document.getElementById(`track-${r.id}`);
          if (track) track.innerHTML = `<p class="empty-state" style="padding:20px">Failed to load.</p>`;
        });
    });
    return;
  }

  const rows = [
    { id: `disc-pop-${genre.id}`, label: `Popular in ${genre.name}`, sort_by: 'popularity.desc' },
    { id: `disc-top-${genre.id}`, label: `Top Rated ${genre.name}`, sort_by: 'vote_average.desc', vote_count_gte: 500 },
    { id: `disc-new-${genre.id}`, label: `Recent ${genre.name} Films`, sort_by: 'primary_release_date.desc', year_gte: new Date().getFullYear() - 3 },
  ];

  for (const r of rows) {
    container.appendChild(createCarouselSection({ id: r.id, label: r.label, model: `genre:${genre.name}` }));
  }
  if (typeof lucide !== 'undefined') lucide.createIcons();

  rows.forEach(r => {
    TMDB.discoverByGenre({ genres: [genre.id], sort_by: r.sort_by, vote_count_gte: r.vote_count_gte, year_gte: r.year_gte })
      .then(results => {
        const movies = results.slice(0, 20).map(x => TMDB.parseLite(x));
        populateCarouselWithPosters({ id: r.id, movies });
        if (typeof lucide !== 'undefined') lucide.createIcons();
      })
      .catch(() => {
        const track = document.getElementById(`track-${r.id}`);
        if (track) track.innerHTML = `<p class="empty-state" style="padding:20px">Failed to load.</p>`;
      });
  });
}


function renderForYouCarousels(container) {
  // 1. Picked for you — reuse the existing recommendation pool
  const picked = {
    id: 'disc-foryou-picked', label: 'Picked for You', model: 'ensemble',
    movies: allMoviesPool.slice(0, 20)
  };
  // 2. Top rated (live TMDB)
  const topRatedSection = createCarouselSection({ id: 'disc-foryou-top', label: 'Top Rated of All Time', model: 'top_rated' });
  // 3. Hidden gems (live TMDB)
  const gemsSection = createCarouselSection({ id: 'disc-foryou-gems', label: 'Hidden Gems for You', model: 'hidden_gems' });

  if (picked.movies.length) {
    container.appendChild(createCarouselSection(picked));
  }
  container.appendChild(topRatedSection);
  container.appendChild(gemsSection);
  if (typeof lucide !== 'undefined') lucide.createIcons();

  if (picked.movies.length) populateCarousel(picked);

  // Top rated
  TMDB.topRated(1).then(results => {
    const movies = results.slice(0, 20).map(r => TMDB.parseLite(r));
    populateCarouselWithPosters({ id: 'disc-foryou-top', movies });
    if (typeof lucide !== 'undefined') lucide.createIcons();
  }).catch(() => { });

  // Hidden gems
  TMDB.discoverByGenre({ sort_by: 'vote_average.desc', vote_count_gte: 200, vote_count_lte: 2500 })
    .then(results => {
      const movies = results.slice(0, 20).map(r => TMDB.parseLite(r));
      populateCarouselWithPosters({ id: 'disc-foryou-gems', movies });
      if (typeof lucide !== 'undefined') lucide.createIcons();
    }).catch(() => { });
}

// ─── Surprise Me view ────────────────────────────────────────────────────────
async function initSurpriseView() {
  if (surpriseInit) return;
  surpriseInit = true;
  document.getElementById('surprise-btn')?.addEventListener('click', rollSurprise);
  rollSurprise();
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

  // Prefer films from user's recommendation pool (highest probability of being liked)
  const pool = allMoviesPool.length
    ? [...allMoviesPool]
    : await TMDB.topRated(Math.ceil(Math.random() * 5)).then(rs => rs.map(r => TMDB.parseLite(r))).catch(() => []);

  if (!pool.length) {
    result.innerHTML = '<p class="empty-state">Nothing to surprise you with right now.</p>';
    return;
  }

  const movie = pool[Math.floor(Math.random() * pool.length)];

  result.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
  result.style.opacity = '0';
  result.style.transform = 'translateY(10px)';

  // Sync the result reveal with the mid-point of the dice animation
  setTimeout(async () => {
    let details;
    try { details = await getMovieDetails(movie.tmdb_id); } catch { details = null; }
    const poster = details?.poster_url || movie.poster_url || 'img/placeholder_poster.svg';
    const backdrop = details?.backdrop_url || movie.backdrop_url || 'img/placeholder_backdrop.svg';
    result.innerHTML = `
      <div class="surprise-card" data-movie-id="${movie.movie_id}" style="background-image:url('${backdrop}')">
        <div class="surprise-card-gradient"></div>
        <img class="surprise-card-poster" src="${poster}" alt="${escapeHtml(movie.title)}">
        <div class="surprise-card-info">
          <h2>${escapeHtml(details?.title || movie.title)}</h2>
          <p class="surprise-meta">${(details?.genres || movie.genres || []).slice(0, 3).join(' · ') || ''} ${details?.year ? '· ' + details.year : ''}</p>
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
