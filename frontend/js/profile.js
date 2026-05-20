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
  renderRatingDistribution(local.distribution);

  // 2. Load profile data + render history and watchlist
  const token = auth.getToken();
  let profile;
  try {
    profile = await api.getProfile(token);
  } catch (e) {
    profile = { rating_history: [], watchlist: [] };
  }

  await renderRatingHistory(profile.rating_history, local);
  await renderWatchlistSection();

  // 3. Async: compute genre stats from TMDB (needs network)
  if (local.ratingIds.length > 0) {
    computeGenreStats(local.ratingIds, local.ratingsMap).then(stats => {
      renderTopGenres(stats.topGenres);
      renderFeatureProfile(stats.featureProfile);

      // Update hours with real runtime data
      if (stats.hoursWatched > 0) {
        animateStatUpdate('stat-hours-watched', stats.hoursWatched);
      }

      // Compute and show badge from actual top genre
      const badge = computeBadge(stats.topGenres, local.totalRatings);
      if (badge) renderBadge(badge);
    });
  } else {
    // No local ratings — fall back to mock
    renderTopGenres(profile.top_genres || []);
    renderFeatureProfile(profile.feature_profile || {});
  }

  if (typeof lucide !== 'undefined') lucide.createIcons();

  // 4. Live update when a rating is changed
  if (typeof eventBus !== 'undefined') {
    eventBus.on('rating:updated', async () => {
      const updatedLocal = computeLocalStats();
      renderStatsGrid(updatedLocal);
      renderRatingDistribution(updatedLocal.distribution);
      await renderRatingHistory(profile.rating_history, updatedLocal);
      
      if (updatedLocal.ratingIds.length > 0) {
        computeGenreStats(updatedLocal.ratingIds, updatedLocal.ratingsMap).then(stats => {
          renderTopGenres(stats.topGenres);
          renderFeatureProfile(stats.featureProfile);
          if (stats.hoursWatched > 0) {
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
  const results = await Promise.allSettled(
    ratingIds.slice(0, 25).map(id => getMovieDetails(id))
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
  if (el('stat-hours-watched')) el('stat-hours-watched').textContent = local.hoursWatched || '—';
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

// ─── Render: Top genres ───────────────────────────────────────────────────────
function renderTopGenres(genres) {
  const container = document.getElementById('top-genres');
  if (!container) return;

  if (!genres?.length) {
    container.innerHTML = '<p class="empty-state">Rate some films to see your genre breakdown.</p>';
    return;
  }

  const maxCount = Math.max(...genres.map(g => g.count), 1);
  container.innerHTML = genres.map(g => `
    <div class="genre-row">
      <span class="genre-name">${g.genre}</span>
      <div class="feature-bar">
        <div class="feature-fill" data-target="${Math.round((g.count / maxCount) * 100)}" style="width:0%"></div>
      </div>
      <span class="genre-avg">${g.avg_rating.toFixed(1)}★</span>
    </div>`).join('');

  animateBars(container, '.feature-fill');
}

// ─── Render: Rating distribution ─────────────────────────────────────────────
function renderRatingDistribution(dist) {
  const container = document.getElementById('rating-distribution');
  if (!container || !dist) return;

  const maxCount = Math.max(...Object.values(dist).map(Number), 1);
  container.innerHTML = Object.entries(dist).map(([stars, count]) => `
    <div class="dist-bar-wrapper">
      <div class="dist-bar-container">
        <div class="dist-bar" data-target="${Math.round((count / maxCount) * 100)}" style="height:0%"></div>
      </div>
      <span class="dist-label">${'★'.repeat(parseInt(stars))}</span>
      <span class="dist-count">${count}</span>
    </div>`).join('');

  requestAnimationFrame(() => requestAnimationFrame(() => {
    container.querySelectorAll('.dist-bar[data-target]').forEach(bar => {
      bar.style.transition = 'height 0.6s ease';
      bar.style.height = bar.dataset.target + '%';
    });
  }));
}

// ─── Render: Feature profile ──────────────────────────────────────────────────
function renderFeatureProfile(profile) {
  const container = document.getElementById('feature-profile');
  if (!container) return;

  const entries = typeof profile === 'object' ? Object.entries(profile) : [];
  if (!entries.length) {
    container.innerHTML = '<p class="empty-state">Rate more films to build your taste profile.</p>';
    return;
  }

  const sorted = entries.sort((a, b) => b[1] - a[1]);
  container.innerHTML = sorted.map(([feature, weight]) => `
    <div class="feature-row">
      <span class="feature-name">${feature}</span>
      <div class="feature-bar">
        <div class="feature-fill" data-target="${Math.round(weight * 100)}" style="width:0%"></div>
      </div>
      <span class="feature-pct">${Math.round(weight * 100)}%</span>
    </div>`).join('');

  animateBars(container, '.feature-fill');
}

function animateBars(container, selector) {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    container.querySelectorAll(selector + '[data-target]').forEach(bar => {
      bar.style.transition = 'width 0.7s ease';
      bar.style.width = bar.dataset.target + '%';
    });
  }));
}

// ─── Render: Rating history ───────────────────────────────────────────────────
async function renderRatingHistory(mockHistory, local) {
  const container = document.getElementById('rating-history');
  if (!container) return;

  const tRaw = typeof pstore !== 'undefined' ? pstore.get('rating_timestamps') : localStorage.getItem('rating_timestamps');
  const localTimestamps = JSON.parse(tRaw || '{}');

  // Backfill missing timestamps to ensure stable sorting
  let needsSave = false;
  const fallbackBase = new Date(new Date().setHours(0,0,0,0)).getTime(); // Start of today
  local.ratingIds.forEach((id, index) => {
    if (!localTimestamps[id]) {
      localTimestamps[id] = new Date(fallbackBase + index * 1000).toISOString();
      needsSave = true;
    }
  });
  if (needsSave) {
    if (typeof pstore !== 'undefined') pstore.set('rating_timestamps', JSON.stringify(localTimestamps));
    else localStorage.setItem('rating_timestamps', JSON.stringify(localTimestamps));
  }

  // Merge mock history with localStorage
  const merged = new Map();
  (mockHistory || []).forEach(item => merged.set(item.movie_id, item));
  local.ratingIds.forEach(id => {
    const rating = local.ratingsMap[id];
    const ts = localTimestamps[id];
    if (!merged.has(id)) {
      merged.set(id, {
        movie_id: id, tmdb_id: id, title: null,
        rating: Number(rating),
        timestamp: ts,
      });
    } else {
      merged.get(id).rating = Number(rating); // localStorage overrides mock
      if (localTimestamps[id]) {
        merged.get(id).timestamp = localTimestamps[id]; // local timestamp overrides mock
      }
    }
  });

  const sorted = [...merged.values()].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

  if (!sorted.length) {
    container.innerHTML = '<p class="empty-state">No ratings yet.</p>';
    return;
  }

  container.innerHTML = sorted.map(item => {
    const stars = Math.round(item.rating);
    const dateStr = new Date(item.timestamp).toLocaleDateString('en-US', {
      day: 'numeric', month: 'short', year: 'numeric'
    });
    return `
      <div class="history-item" data-movie-id="${item.movie_id}" style="animation:cardReveal 0.3s ease both">
        <img class="history-poster"
             src="img/placeholder_poster.svg"
             alt="${escapeHtml(item.title || '')}"
             data-tmdb="${item.tmdb_id || item.movie_id}">
        <div class="history-info">
          <span class="history-title">${escapeHtml(item.title || `Film #${item.movie_id}`)}</span>
          <span class="history-date">${dateStr}</span>
        </div>
        <div class="history-rating" title="${item.rating}/5">
          ${[1, 2, 3, 4, 5].map(v =>
      `<span style="color:${v <= stars ? 'var(--star-color)' : 'var(--text-muted)'}">★</span>`
    ).join('')}
          <span class="history-score">${Number(item.rating).toFixed(1)}</span>
        </div>
      </div>`;
  }).join('');

  // Lazy-load TMDB posters + fix placeholder titles
  container.querySelectorAll('[data-tmdb]').forEach(img => {
    const tmdbId = parseInt(img.dataset.tmdb);
    if (!tmdbId) return;
    getMovieDetails(tmdbId).then(d => {
      img.src = d.poster_url;
      const titleEl = img.closest('.history-item')?.querySelector('.history-title');
      if (titleEl && (titleEl.textContent.startsWith('Film #') || !titleEl.textContent.trim())) {
        titleEl.textContent = d.title;
        img.alt = d.title;
      }
    }).catch(() => { });
  });

  // Click → open movie modal (clickable everywhere)
  container.querySelectorAll('.history-item').forEach(item => {
    item.style.cursor = 'pointer';
    item.addEventListener('click', () => {
      const id = parseInt(item.dataset.movieId);
      if (!id || typeof openMovieModal !== 'function') return;
      const tmdbId = parseInt(item.querySelector('[data-tmdb]')?.dataset.tmdb) || id;
      openMovieModal({ movie_id: id, tmdb_id: tmdbId, title: item.querySelector('.history-title')?.textContent || '' });
    });
  });
}

// ─── Render: Watchlist section ────────────────────────────────────────────────
async function renderWatchlistSection() {
  const container = document.getElementById('watchlist-grid');
  if (!container) return;

  const ids = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('watchlist') : localStorage.getItem('watchlist')) || '[]');

  if (!ids.length) {
    container.innerHTML = '<p class="empty-state">Your list is empty.</p>';
    return;
  }

  container.innerHTML = '';

  for (const id of ids) {
    const card = document.createElement('div');
    card.className = 'watchlist-card';
    card.style.cursor = 'pointer';

    const details = await getMovieDetails(id).catch(() => null);
    card.innerHTML = `
      <div class="watchlist-card-inner" style="height:100%; position:relative;">
        <img src="${details?.poster_url || 'img/placeholder_poster.svg'}" alt="${escapeHtml(details?.title || '')}" loading="lazy" style="width:100%;height:100%;object-fit:cover;">
        <div class="card-overlay">
          <div class="card-title">${escapeHtml(details?.title || '')}</div>
          <div class="card-meta">${details?.year || ''}${(details?.genres || []).length ? ' · ' + details.genres.slice(0, 2).join(', ') : ''}</div>
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
        openMovieModal({ movie_id: id, tmdb_id: id, title: details?.title || '' });
      }
    });

    container.appendChild(card);
  }
}
