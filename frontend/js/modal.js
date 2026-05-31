let _currentMovieMeta = null;

function openMovieModal(movieMeta) {
  _currentMovieMeta = movieMeta;

  const overlay = document.getElementById('movie-modal');
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';

  // Reset
  document.getElementById('modal-backdrop-img').src = 'img/placeholder_backdrop.svg';
  document.getElementById('modal-title').textContent = movieMeta.title || '';
  document.getElementById('modal-overview').textContent = '';
  document.getElementById('modal-metadata').innerHTML = '';
  document.getElementById('modal-quick-meta').innerHTML = '';
  document.getElementById('modal-more-grid').innerHTML = '';

  // Watchlist button
  const wlBtn = document.getElementById('modal-watchlist-btn');
  wlBtn.dataset.movieId = movieMeta.movie_id;
  _renderWatchlistBtn(wlBtn, watchlist.isInWatchlist(movieMeta.movie_id));

  // Star rating
  const starContainer = document.getElementById('modal-star-rating');
  starContainer.dataset.movieId = movieMeta.movie_id;
  starContainer.innerHTML = [1, 2, 3, 4, 5].map(v => `<span class="star" data-value="${v}">★</span>`).join('');
  initStarRating(starContainer, movieMeta.movie_id, getRating(movieMeta.movie_id), { tmdb_id: movieMeta.tmdb_id, title: movieMeta.title });

  // Load TMDB details
  getMovieDetails(movieMeta.tmdb_id)
    .then(details => {
      renderModalDetails(details, movieMeta);
      if (typeof lucide !== 'undefined') lucide.createIcons();
    })
    .catch(() => renderModalFallback(movieMeta));

  // Load "More Like This"
  loadMoreLikeThis(movieMeta);

  // Load "Next in Saga" (if this movie is part of a franchise)
  if (typeof loadSagaForMovie === 'function') loadSagaForMovie(movieMeta);
}

function _renderWatchlistBtn(btn, inList) {
  btn.classList.toggle('in-list', inList);
  btn.innerHTML = inList
    ? `<i data-lucide="check" class="wl-modal-icon-default"></i><i data-lucide="minus" class="wl-modal-icon-hover"></i><span class="wl-modal-label">Remove from list</span>`
    : `<i data-lucide="plus" class="wl-modal-icon-default"></i><span class="wl-modal-label">Add to my list</span>`;
  if (typeof lucide !== 'undefined') lucide.createIcons();
}

function renderModalDetails(details, meta) {
  document.getElementById('modal-backdrop-img').src = details.backdrop_url;
  document.getElementById('modal-title').textContent = details.title;
  document.getElementById('modal-overview').textContent = details.overview || '';

  const matchPct = meta.score ? Math.round(meta.score * 100) : null;
  const runtimeStr = details.runtime_min ? `${details.runtime_min} min` : '';

  document.getElementById('modal-quick-meta').innerHTML = [
    matchPct ? `<span class="match-score">${matchPct}% Match</span>` : '',
    details.year ? `<span>${details.year}</span>` : '',
    runtimeStr ? `<span>${runtimeStr}</span>` : '',
    details.vote_average ? `<span class="modal-tmdb-rating"><span class="tmdb-star">★</span> ${details.vote_average.toFixed(1)}<span class="tmdb-outof">/10</span></span>` : ''
  ].filter(Boolean).join('');

  document.getElementById('modal-metadata').innerHTML = `
    <p><span class="meta-label">Genres:</span> ${(details.genres || []).join(', ') || '—'}</p>
    <p><span class="meta-label">Director:</span> ${details.director || '—'}</p>
    <p><span class="meta-label">Cast:</span> ${(details.cast || []).slice(0, 3).map(c => c.name).join(', ') || '—'}</p>
    <p><span class="meta-label">Year:</span> ${details.year || '—'}</p>
    <p><span class="meta-label">Runtime:</span> ${runtimeStr || '—'}</p>
    <p id="modal-saga-membership" style="display:none"></p>
  `;
}

function renderModalFallback(meta) {
  document.getElementById('modal-quick-meta').innerHTML = [
    meta.score ? `<span class="match-score">${Math.round(meta.score * 100)}% Match</span>` : '',
    meta.year ? `<span>${meta.year}</span>` : ''
  ].filter(Boolean).join('');

  document.getElementById('modal-metadata').innerHTML = `
    <p><span class="meta-label">Genres:</span> ${(meta.genres || []).join(', ') || '—'}</p>
    <p><span class="meta-label">Year:</span> ${meta.year || '—'}</p>
    <p id="modal-saga-membership" style="display:none"></p>
  `;
}

async function loadMoreLikeThis(movieMeta) {
  const section = document.getElementById('modal-more-section');
  if (!section) return;

  try {
    // Content-based item-item similarity (films closest to THIS one in feature space)
    const data = await api.getSimilar(movieMeta.movie_id);
    const similar = (data.movies || [])
      .filter(m => m.movie_id !== movieMeta.movie_id)
      .slice(0, 6);

    if (!similar.length) { section.style.display = 'none'; return; }
    section.style.display = '';

    const grid = document.getElementById('modal-more-grid');
    grid.innerHTML = '';

    for (const movie of similar) {
      const card = document.createElement('div');
      card.className = 'modal-more-card';
      const inList = watchlist.isInWatchlist(movie.movie_id);
      const scorePct = movie.score ? Math.round(movie.score * 100) : null;

      card.innerHTML = `
        <div class="modal-more-poster-wrap">
          <img src="img/placeholder_poster.svg" alt="${escapeHtml(movie.title)}" loading="lazy">
          <div class="card-overlay">
            <div class="card-title">${escapeHtml(movie.title)}</div>
            <div class="card-meta">${movie.year || ''}${movie.genres?.length ? ' · ' + movie.genres.slice(0, 2).join(', ') : ''}</div>
            <div class="card-actions">
              <button class="btn-watchlist-card ${inList ? 'in-list' : ''}" data-movie-id="${movie.movie_id}">
                ${inList ? '<i data-lucide="check" class="wl-icon-default"></i><i data-lucide="minus" class="wl-icon-hover"></i>' : '<i data-lucide="plus" class="wl-icon-default"></i>'}
              </button>
              ${scorePct ? `<span class="card-score">${scorePct}%</span>` : ''}
            </div>
          </div>
        </div>`;

      // Wire watchlist button
      const wlBtn = card.querySelector('.btn-watchlist-card');
      wlBtn.addEventListener('click', async e => {
        e.stopPropagation();
        await watchlist.toggle(movie.movie_id, e.currentTarget);
      });

      // Load poster
      getMovieDetails(movie.tmdb_id).then(d => {
        const img = card.querySelector('img');
        if (img) img.src = d.poster_url;
      }).catch(() => { });

      card.addEventListener('click', e => {
        if (!e.target.closest('.btn-watchlist-card')) {
          closeMovieModal();
          setTimeout(() => openMovieModal(movie), 280);
        }
      });

      grid.appendChild(card);
    }

    if (typeof lucide !== 'undefined') lucide.createIcons();
  } catch (err) {
    section.style.display = 'none';
    console.warn('More like this failed:', err);
  }
}

function closeMovieModal() {
  document.getElementById('movie-modal')?.classList.remove('open');
  document.body.style.overflow = '';
  _currentMovieMeta = null;
}

function initModal() {
  document.getElementById('modal-close')?.addEventListener('click', closeMovieModal);
  document.getElementById('movie-modal')?.addEventListener('click', e => {
    if (e.target.id === 'movie-modal') closeMovieModal();

    const chip = e.target.closest('.meta-chip-saga[data-action="saga"]');
    if (!chip) return;
    const id = parseInt(chip.dataset.sagaId);
    const saga = typeof SAGAS !== 'undefined' ? SAGAS.find(s => s.id === id) : null;
    if (saga) { closeMovieModal(); setTimeout(() => openSagaModal(saga), 320); }
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMovieModal(); });

  document.getElementById('modal-watchlist-btn')?.addEventListener('click', async () => {
    if (!_currentMovieMeta) return;
    await watchlist.toggle(_currentMovieMeta.movie_id);
    const btn = document.getElementById('modal-watchlist-btn');
    _renderWatchlistBtn(btn, watchlist.isInWatchlist(_currentMovieMeta.movie_id));
  });

  document.getElementById('modal-play-btn')?.addEventListener('click', () => {
    window.location.href = 'play.html';
  });
}
