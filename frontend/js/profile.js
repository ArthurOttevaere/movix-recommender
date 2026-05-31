// Lightweight pub/sub for the profile page: home.js defines its own eventBus but
// it isn't loaded here, so star ratings would otherwise never refresh the history.
window.eventBus = window.eventBus || {
  listeners: {},
  on(e, cb) { (this.listeners[e] = this.listeners[e] || []).push(cb); },
  emit(e, d) { (this.listeners[e] || []).forEach(cb => cb(d)); },
};

// Hours watched derived from the CSV (total films viewed) for preloaded profiles.
// When set, it stays the source of truth for the Hours Watched card (local rating
// estimates never override it).
let csvHoursWatched = null;

// Preloaded demo profile (Lenny): the genre ranking + badge come from the rich
// server-side profile and must stay put even after the user rates a few films.
let isPreloadedProfile = false;

document.addEventListener('DOMContentLoaded', async () => {
  auth.requireAuth();

  window.addEventListener('scroll', () => {
    document.getElementById('navbar')?.classList.toggle('scrolled', window.scrollY > 10);
  });

  // Ensure member_since is tracked
  const hasMember = typeof pstore !== 'undefined' ? pstore.get('member_since') : localStorage.getItem('member_since');
  if (!hasMember) {
    const today = new Date().toISOString().split('T')[0];
    if (typeof pstore !== 'undefined') pstore.set('member_since', today);
    else localStorage.setItem('member_since', today);
  }

  // 1. Compute sync stats from localStorage immediately
  const local = computeLocalStats();
  renderStatsGrid(local);

  // 2. Load profile data + render watchlist
  const token = auth.getToken();
  let profile;
  try {
    profile = await api.getProfile(token);
  } catch (e) {
    profile = { rating_history: [], watchlist: [] };
  }

  isPreloadedProfile = !!(profile.most_watched && profile.most_watched.length);

  // Hours watched for preloaded profiles (Lenny): the server figure is derived
  // from the CSV's total films watched. Keep it as the source of truth.
  if (isPreloadedProfile && profile.hours_watched) {
    csvHoursWatched = profile.hours_watched;
    const hEl = document.getElementById('stat-hours-watched');
    if (hEl) hEl.textContent = csvHoursWatched;
  }

  // Most Watched + Recently Watched — données issues de la bibliothèque CSV
  // (renseignées côté backend pour le profil démo Lenny ; vides sinon → masquées).
  renderMostWatched(profile.most_watched || []);
  renderRecentlyWatched(profile.recently_watched || []);
  renderFavoriteSagas(profile.most_watched || []);
  await renderWatchlistSection(profile.watchlist_movies || []);
  renderRatingHistory(local);

  // 3. Async: compute genre stats from TMDB (needs network). Preloaded demo
  // profiles keep their richer server-computed genres instead.
  if (local.ratingIds.length > 0 && !isPreloadedProfile) {
    computeGenreStats(local.ratingIds, local.ratingsMap).then(stats => {
      renderGenreList(stats.topGenres);

      // Update hours with real runtime data (unless a CSV figure is in charge)
      if (!csvHoursWatched && stats.hoursWatched > 0) {
        animateStatUpdate('stat-hours-watched', stats.hoursWatched);
      }

      // Compute and show badge from actual top genre
      const badge = computeBadge(stats.topGenres, local.totalRatings);
      if (badge) renderBadge(badge);
    });
  } else {
    // No local ratings (e.g. Lenny démo) — use the genres computed server-side
    renderGenreList(profile.top_genres || []);
    const badge = computeBadge(profile.top_genres || [], profile.total_ratings || 0);
    if (badge) renderBadge(badge);
  }

  if (typeof lucide !== 'undefined') lucide.createIcons();

  // 4. Live update when a rating is changed
  if (typeof eventBus !== 'undefined') {
    eventBus.on('rating:updated', async () => {
      const updatedLocal = computeLocalStats();
      renderStatsGrid(updatedLocal);
      renderRatingHistory(updatedLocal);

      if (updatedLocal.ratingIds.length > 0 && !isPreloadedProfile) {
        computeGenreStats(updatedLocal.ratingIds, updatedLocal.ratingsMap).then(stats => {
          renderGenreList(stats.topGenres);
          if (!csvHoursWatched && stats.hoursWatched > 0) {
            animateStatUpdate('stat-hours-watched', stats.hoursWatched);
          }
          const badge = computeBadge(stats.topGenres, updatedLocal.totalRatings);
          if (badge) renderBadge(badge);
        });
      }
    });
  }
});

// ─── Local stats (synchronous) ────────────────────────────────────────────────
function computeLocalStats() {
  const ratingsMap = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('ratings') : localStorage.getItem('ratings')) || '{}');
  const watchlistIds = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('watchlist') : localStorage.getItem('watchlist')) || '[]');
  const values = Object.values(ratingsMap).map(Number);

  const totalRatings = values.length;
  const meanRating = totalRatings
    ? (values.reduce((a, b) => a + b, 0) / totalRatings).toFixed(1)
    : null;

  const distribution = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
  values.forEach(r => {
    const b = Math.round(r);
    if (b >= 1 && b <= 5) distribution[b]++;
  });

  const memberSince = (typeof pstore !== 'undefined' ? pstore.get('member_since') : localStorage.getItem('member_since')) || new Date().toISOString().split('T')[0];
  const streak = parseInt((typeof pstore !== 'undefined' ? pstore.get('streak_days') : localStorage.getItem('streak_days')) || '1');

  return {
    totalRatings,
    meanRating,
    distribution,
    watchlistCount: watchlistIds.length,
    memberSince,
    streak,
    hoursWatched: Math.round(totalRatings * 1.75), // estimate, updated async
    completionRate: totalRatings > 0 ? Math.min(95, 60 + Math.round(totalRatings * 2)) : 0,
    ratingIds: Object.keys(ratingsMap).map(Number),
    ratingsMap,
  };
}

// ─── Async genre stats (needs TMDB) ───────────────────────────────────────────
async function computeGenreStats(ratingIds, ratingsMap) {
  // Ratings are keyed by MovieLens movie_id; resolve to the real TMDB id via the
  // meta captured at rating time so genres come from the right films.
  const metaMap = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('rating_meta') : localStorage.getItem('rating_meta')) || '{}');
  const results = await Promise.allSettled(
    ratingIds.slice(0, 25).map(id => getMovieDetails(metaMap[id]?.tmdb_id || id))
  );

  const genreAccum = {};
  let totalMinutes = 0;

  results.forEach(r => {
    if (r.status !== 'fulfilled') return;
    const movie = r.value;
    const rating = Number(ratingsMap[movie.tmdb_id] || ratingsMap[movie.movie_id]);
    if (!rating) return;
    if (movie.runtime_min) totalMinutes += movie.runtime_min;

    (movie.genres || []).forEach(genre => {
      if (!genreAccum[genre]) genreAccum[genre] = { count: 0, total: 0 };
      genreAccum[genre].count++;
      genreAccum[genre].total += rating;
    });
  });

  const topGenres = Object.entries(genreAccum)
    .map(([genre, s]) => ({ genre, count: s.count, avg_rating: s.total / s.count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 7);

  const featureProfile = {};
  const maxCount = topGenres[0]?.count || 1;
  topGenres.forEach(g => {
    featureProfile[g.genre] = +(g.count / maxCount).toFixed(2);
  });

  const hoursWatched = Math.round(totalMinutes / 60);

  return { topGenres, featureProfile, hoursWatched };
}

function computeBadge(topGenres, totalRatings) {
  if (!topGenres.length) return null;
  const top = topGenres[0];
  const iconMap = {
    'Drama': '🎭', 'Science Fiction': '🚀', 'Horror': '👻', 'Action': '💥',
    'Animation': '✨', 'Thriller': '😱', 'Crime': '🔍', 'Comedy': '😄',
    'Romance': '❤️', 'Fantasy': '🪄', 'Adventure': '🗺️', 'Mystery': '🕵️',
    'Family': '👨‍👩‍👧', 'History': '📜',
  };
  const titleMap = {
    'Drama': 'Drama Expert', 'Science Fiction': 'Sci-Fi Explorer', 'Horror': 'Horror Aficionado',
    'Action': 'Action Junkie', 'Animation': 'Animation Fan', 'Thriller': 'Thrill Seeker',
    'Crime': 'Crime Buff', 'Comedy': 'Comedy Lover', 'Romance': 'Romantic at Heart',
    'Fantasy': 'Fantasy Dreamer', 'Adventure': 'Explorer', 'Mystery': 'Mystery Detective',
    'Family': 'Family Friendly', 'History': 'History Buff',
  };
  return {
    icon: iconMap[top.genre] || '🎬',
    title: titleMap[top.genre] || 'Cinephile',
    description: `${top.count} ${top.genre} film${top.count > 1 ? 's' : ''} rated · Avg ${top.avg_rating.toFixed(1)}★`,
  };
}

// ─── Render: Stats grid ───────────────────────────────────────────────────────
function renderStatsGrid(local) {
  const el = id => document.getElementById(id);

  if (el('stat-total-ratings')) el('stat-total-ratings').textContent = local.totalRatings;
  if (el('stat-hours-watched')) el('stat-hours-watched').textContent = csvHoursWatched ?? (local.hoursWatched || '—');
  if (el('stat-mean-rating')) el('stat-mean-rating').textContent = local.meanRating || '—';
  if (el('stat-streak')) el('stat-streak').textContent = local.streak;
  if (el('stat-watchlist')) el('stat-watchlist').textContent = local.watchlistCount;
  if (el('stat-completion')) el('stat-completion').textContent = local.completionRate + '%';

  if (el('profile-member-since')) {
    const d = new Date(local.memberSince);
    el('profile-member-since').textContent = `Member since ${d.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}`;
  }
}

function animateStatUpdate(elId, newValue) {
  const el = document.getElementById(elId);
  if (!el || el.textContent === String(newValue)) return;
  el.style.transition = 'opacity 0.25s ease';
  el.style.opacity = '0';
  setTimeout(() => {
    el.textContent = newValue;
    el.style.opacity = '1';
    setTimeout(() => { el.style.transition = ''; }, 280);
  }, 260);
}

function renderBadge(badge) {
  const card = document.getElementById('stat-badge-card');
  if (!card) return;
  card.style.display = '';
  card.style.animation = 'fadeIn 0.4s ease';
  const icon = document.getElementById('badge-icon');
  const title = document.getElementById('badge-title');
  const desc = document.getElementById('badge-desc');
  if (icon) icon.textContent = badge.icon;
  if (title) title.textContent = badge.title;
  if (desc) desc.textContent = badge.description;
}

// ─── Render: Top genres — ranked list (top 5) ────────────────────────────────
function renderGenreList(genres) {
  const el = document.getElementById('genre-list');
  if (!el) return;

  const top5 = (genres || []).filter(g => g && g.genre).slice(0, 5);
  if (!top5.length) {
    el.innerHTML = '<p class="empty-state">Rate some films to reveal your top genres.</p>';
    return;
  }

  // Bar width = relative weight of each genre (visual ranking only — no raw
  // counts shown, since the preloaded demo profile has no precise per-film ratings).
  const max = Math.max(...top5.map(g => g.count), 1);
  el.innerHTML = top5.map((g, i) => `
      <li class="genre-rank-row" style="animation:cardReveal 0.3s ease both;animation-delay:${i * 40}ms">
        <span class="genre-rank-num">${i + 1}</span>
        <div class="genre-rank-body">
          <span class="genre-rank-name">${escapeHtml(g.genre)}</span>
          <div class="genre-rank-bar"><div class="genre-rank-fill" style="width:${Math.round(g.count / max * 100)}%"></div></div>
        </div>
      </li>`).join('');
}

// Standard poster card (same markup/behaviour as the home carousels: title overlay
// fades in on hover). `badge` is optional HTML pinned on the poster.
function _profilePosterCard(m, { badge = '', extraClass = '' } = {}) {
  const meta = `${m.year || ''}${m.genres?.length ? ' · ' + m.genres.slice(0, 2).join(', ') : ''}`;
  return `
    <div class="movie-card ${extraClass}" data-movie-id="${m.movie_id}" data-tmdb="${m.tmdb_id || ''}">
      <div class="movie-card-inner">
        <img src="img/placeholder_poster.svg" alt="${escapeHtml(m.title || '')}" loading="lazy">
        ${badge}
        <div class="card-overlay">
          <div class="card-title">${escapeHtml(m.title || '')}</div>
          <div class="card-meta">${meta}</div>
        </div>
      </div>
    </div>`;
}

// ─── Render: Most Watched — poster podium (top 3) ─────────────────────────────
function renderMostWatched(movies) {
  const section = document.getElementById('most-watched-section');
  const podium = document.getElementById('most-watched-podium');
  if (!section || !podium) return;

  if (!movies?.length) { section.style.display = 'none'; return; }
  section.style.display = '';

  const places = ['first', 'second', 'third'];
  const top3 = movies.slice(0, 3).map((m, i) => ({ ...m, rank: i }));
  // Classic podium layout: 2nd · 1st · 3rd
  const order = [top3[1], top3[0], top3[2]].filter(Boolean);

  podium.innerHTML = order.map(m => {
    const badge = `<span class="watch-badge"><i data-lucide="eye"></i> ${m.n_watched}×</span>`;
    return `
      <div class="poster-podium-col podium-${places[m.rank]}" style="animation:cardReveal 0.4s ease both">
        <div class="podium-figure">
          <span class="podium-num">${m.rank + 1}</span>
          ${_profilePosterCard(m, { badge })}
        </div>
      </div>`;
  }).join('');

  _wireProfilePosters(podium, movies);
}

// ─── Render: Recently Watched — horizontal rail of poster cards ───────────────
function renderRecentlyWatched(movies) {
  const section = document.getElementById('recently-watched-section');
  const rail = document.getElementById('recently-watched-rail');
  if (!section || !rail) return;

  if (!movies?.length) { section.style.display = 'none'; return; }
  section.style.display = '';

  rail.innerHTML = movies.map(m => _profilePosterCard(m)).join('');
  _wireProfilePosters(rail, movies);
}

// ─── Render: Favorite Sagas — poster podium (top 3) ───────────────────────────
// Ranks the user's franchises by total views: for each known saga (TMDB
// collection), sum n_watched over the films the user has actually watched.
async function renderFavoriteSagas(mostWatched) {
  const section = document.getElementById('fav-sagas-section');
  const podium = document.getElementById('fav-sagas-podium');
  if (!section || !podium) return;

  const canCompute = mostWatched?.length && typeof SAGAS !== 'undefined' && typeof TMDB !== 'undefined';
  if (!canCompute) { section.style.display = 'none'; return; }

  // Fetch every saga collection, then map each member film (tmdb_id) → saga.
  const collections = await Promise.allSettled(SAGAS.map(s => TMDB.getCollection(s.id)));
  const tmdbToSaga = new Map();
  SAGAS.forEach((s, i) => {
    if (collections[i].status !== 'fulfilled') return;
    (collections[i].value.parts || []).forEach(p => { if (p.id) tmdbToSaga.set(p.id, s); });
  });

  // Accumulate the user's watch counts per saga + remember the most-watched film
  // (its poster represents the saga on the podium).
  const acc = new Map();
  mostWatched.forEach(m => {
    const saga = tmdbToSaga.get(m.tmdb_id);
    if (!saga) return;
    const cur = acc.get(saga.id) || { saga, views: 0, films: 0, topFilm: null, topN: -1 };
    cur.views += m.n_watched || 0;
    cur.films += 1;
    if ((m.n_watched || 0) > cur.topN) { cur.topN = m.n_watched || 0; cur.topFilm = m; }
    acc.set(saga.id, cur);
  });

  // Ranked by number of DISTINCT films watched within the saga (views break ties).
  const top3 = [...acc.values()].sort((a, b) => b.films - a.films || b.views - a.views).slice(0, 3);
  if (!top3.length) { section.style.display = 'none'; return; }
  section.style.display = '';

  const places = ['first', 'second', 'third'];
  const ranked = top3.map((e, i) => ({ ...e, rank: i }));
  const order = [ranked[1], ranked[0], ranked[2]].filter(Boolean);

  podium.innerHTML = order.map(e => {
    const f = e.topFilm || {};
    const badge = `<span class="watch-badge"><i data-lucide="film"></i> ${e.films}</span>`;
    return `
      <div class="poster-podium-col podium-${places[e.rank]}" data-saga-id="${e.saga.id}"
           style="animation:cardReveal 0.4s ease both">
        <div class="podium-figure">
          <span class="podium-num">${e.rank + 1}</span>
          <div class="movie-card" data-tmdb="${f.tmdb_id || ''}">
            <div class="movie-card-inner">
              <img src="img/placeholder_poster.svg" alt="${escapeHtml(e.saga.name)}" loading="lazy">
              ${badge}
              <div class="card-overlay">
                <div class="card-title">${escapeHtml(e.saga.name)}</div>
                <div class="card-meta">${e.films} film${e.films > 1 ? 's' : ''} watched</div>
              </div>
            </div>
          </div>
        </div>
      </div>`;
  }).join('');

  // Lazy-load the representative poster + open the saga modal on click.
  podium.querySelectorAll('.poster-podium-col').forEach(col => {
    const sagaId = parseInt(col.dataset.sagaId);
    const tmdbId = parseInt(col.querySelector('.movie-card')?.dataset.tmdb);
    const img = col.querySelector('img');
    if (tmdbId) getMovieDetails(tmdbId).then(d => { if (img) img.src = d.poster_url; }).catch(() => { });
    col.addEventListener('click', () => {
      const saga = SAGAS.find(s => s.id === sagaId);
      if (saga && typeof openSagaModal === 'function') openSagaModal(saga);
    });
  });

  if (typeof lucide !== 'undefined') lucide.createIcons({ root: podium });
}

// Lazy-load posters + open the shared movie modal on click.
function _wireProfilePosters(container, movies) {
  container.querySelectorAll('.movie-card').forEach(card => {
    const id = parseInt(card.dataset.movieId);
    const tmdbId = parseInt(card.dataset.tmdb) || id;
    const img = card.querySelector('img');
    if (tmdbId) getMovieDetails(tmdbId).then(d => { if (img) img.src = d.poster_url; }).catch(() => { });

    card.addEventListener('click', () => {
      if (typeof openMovieModal !== 'function') return;
      const movie = movies.find(m => m.movie_id === id)
        || { movie_id: id, tmdb_id: tmdbId, title: card.querySelector('.card-title')?.textContent || '' };
      openMovieModal(movie);
    });
  });

  if (typeof lucide !== 'undefined') lucide.createIcons({ root: container });
}

// ─── Render: Watchlist section ────────────────────────────────────────────────
async function renderWatchlistSection(watchlistMovies) {
  const container = document.getElementById('watchlist-grid');
  if (!container) return;

  const ids = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('watchlist') : localStorage.getItem('watchlist')) || '[]');

  if (!ids.length) {
    container.innerHTML = '<p class="empty-state">Your list is empty.</p>';
    return;
  }

  // The watchlist stores MovieLens movie_ids; getMovieDetails() needs the TMDB id.
  // The backend resolves the mapping for us → index it by movie_id so posters/titles
  // are the RIGHT films (e.g. Lenny's wishlist), not whatever TMDB id collides.
  const byId = new Map((watchlistMovies || []).map(m => [m.movie_id, m]));

  container.innerHTML = '';

  for (const id of ids) {
    const meta = byId.get(id) || {};
    const tmdbId = meta.tmdb_id || id;

    const card = document.createElement('div');
    card.className = 'watchlist-card';
    card.style.cursor = 'pointer';

    const details = await getMovieDetails(tmdbId).catch(() => null);
    const title = details?.title || meta.title || '';
    const year = details?.year || meta.year || '';
    const genres = details?.genres?.length ? details.genres : (meta.genres || []);
    card.innerHTML = `
      <div class="watchlist-card-inner" style="height:100%; position:relative;">
        <img src="${details?.poster_url || 'img/placeholder_poster.svg'}" alt="${escapeHtml(title)}" loading="lazy" style="width:100%;height:100%;object-fit:cover;">
        <div class="card-overlay">
          <div class="card-title">${escapeHtml(title)}</div>
          <div class="card-meta">${year}${genres.length ? ' · ' + genres.slice(0, 2).join(', ') : ''}</div>
          <div class="card-actions">
            <button class="btn-watchlist-card in-list btn-remove-watchlist" data-movie-id="${id}" title="Remove from list">
              <i data-lucide="check" class="wl-icon-default"></i><i data-lucide="minus" class="wl-icon-hover"></i>
            </button>
            ${details?.vote_average ? `<span class="card-score">${Math.round(details.vote_average * 10)}%</span>` : ''}
          </div>
        </div>
      </div>`;

    card.querySelector('.btn-remove-watchlist').addEventListener('click', async (e) => {
      e.stopPropagation();
      await watchlist.toggle(id);
      card.style.cssText = 'opacity:0;transform:scale(0.88);transition:all 0.22s ease';
      setTimeout(() => card.remove(), 230);
    });

    card.addEventListener('click', e => {
      if (e.target.closest('.btn-remove-watchlist')) return;
      if (typeof openMovieModal === 'function') {
        openMovieModal({ movie_id: id, tmdb_id: tmdbId, title });
      }
    });

    container.appendChild(card);
  }
}

// ─── Render: Rating History ───────────────────────────────────────────────────
// Explicit star ratings made in the app (NOT the preloaded implicit ratings),
// newest first. Updates live as the user rates. Posters/titles resolve via the
// `rating_meta` captured at rating time (tmdb_id + title).
function renderRatingHistory(local) {
  const container = document.getElementById('rating-history');
  if (!container) return;

  const tRaw = typeof pstore !== 'undefined' ? pstore.get('rating_timestamps') : localStorage.getItem('rating_timestamps');
  const timestamps = JSON.parse(tRaw || '{}');
  const mRaw = typeof pstore !== 'undefined' ? pstore.get('rating_meta') : localStorage.getItem('rating_meta');
  const metaMap = JSON.parse(mRaw || '{}');

  // Only films rated through the app's star UI count here (those record a
  // tmdb_id + title). Preloaded/implicit or legacy ratings have no meta, so the
  // history stays empty until the user actually rates a film on the site.
  const ratedIds = (local.ratingIds || []).filter(id => metaMap[id]);
  if (!ratedIds.length) {
    container.innerHTML = '<p class="empty-state">No ratings yet — rate a film and it\'ll show up here, newest first.</p>';
    return;
  }

  const items = ratedIds
    .map(id => ({
      movie_id: id,
      rating: Number(local.ratingsMap[id]),
      timestamp: timestamps[id] || 0,
      tmdb_id: metaMap[id]?.tmdb_id || null,
      title: metaMap[id]?.title || '',
    }))
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

  container.innerHTML = items.map(item => {
    const stars = Math.round(item.rating);
    const dateStr = item.timestamp
      ? new Date(item.timestamp).toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' })
      : '';
    return `
      <div class="history-item" data-movie-id="${item.movie_id}" data-tmdb="${item.tmdb_id || ''}" style="animation:cardReveal 0.3s ease both">
        <img class="history-poster" src="img/placeholder_poster.svg" alt="${escapeHtml(item.title)}" loading="lazy">
        <div class="history-info">
          <span class="history-title">${escapeHtml(item.title || `Film #${item.movie_id}`)}</span>
          <span class="history-date">${dateStr}</span>
        </div>
        <div class="history-rating" title="${item.rating}/5">
          ${[1, 2, 3, 4, 5].map(v => `<span style="color:${v <= stars ? 'var(--star-color)' : 'var(--text-muted)'}">★</span>`).join('')}
        </div>
      </div>`;
  }).join('');

  // Lazy-load posters (+ backfill any missing title) and wire the modal.
  container.querySelectorAll('.history-item').forEach(el => {
    const tmdbId = parseInt(el.dataset.tmdb);
    const movieId = parseInt(el.dataset.movieId);
    const img = el.querySelector('.history-poster');
    if (tmdbId) {
      getMovieDetails(tmdbId).then(d => {
        if (img) img.src = d.poster_url;
        const titleEl = el.querySelector('.history-title');
        if (titleEl && (!titleEl.textContent.trim() || titleEl.textContent.startsWith('Film #'))) titleEl.textContent = d.title;
      }).catch(() => { });
    }
    el.style.cursor = 'pointer';
    el.addEventListener('click', () => {
      if (typeof openMovieModal === 'function') {
        openMovieModal({ movie_id: movieId, tmdb_id: tmdbId || movieId, title: el.querySelector('.history-title')?.textContent || '' });
      }
    });
  });
}
