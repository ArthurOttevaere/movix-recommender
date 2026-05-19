// ─── Saga view (TMDB collections) ────────────────────────────────────────────
// Curated list of well-known franchise collections. The numeric `id` is the
// TMDB collection ID — used by TMDB.getCollection() to fetch every film inside.

const SAGAS = [
  { id: 10, name: 'Star Wars' },
  { id: 86311, name: 'The Avengers' },
  { id: 1241, name: 'Harry Potter' },
  { id: 119, name: 'The Lord of the Rings' },
  { id: 263, name: 'The Dark Knight Trilogy' },
  { id: 87359, name: 'Mission: Impossible' },
  { id: 9485, name: 'The Fast and the Furious' },
  { id: 295, name: 'Pirates of the Caribbean' },
  { id: 645, name: 'James Bond' },
  { id: 2980, name: 'Ghostbusters' },
  { id: 230, name: 'The Godfather' },
  { id: 8945, name: 'Mad Max' },
  { id: 528, name: 'The Terminator' },
  { id: 1709, name: 'Rocky' },
  { id: 8650, name: 'Transformers' },
  { id: 131296, name: 'Thor' },
  { id: 304378, name: 'John Wick' },
  { id: 87096, name: 'Avatar' },
  { id: 1570, name: 'Die Hard' },
  { id: 33514, name: 'The Bourne' },
];

let sagaLoaded = false;
const sagaCache = new Map();  // collectionId → fetched collection

async function renderSagaView() {
  if (sagaLoaded) return;
  sagaLoaded = true;

  const grid = document.getElementById('saga-grid');
  if (!grid) return;
  grid.innerHTML = '';

  // Pre-render skeleton tiles for instant feedback
  SAGAS.forEach(saga => {
    const card = document.createElement('button');
    card.className = 'saga-card';
    card.type = 'button';
    card.dataset.collectionId = saga.id;
    card.innerHTML = `
      <div class="saga-card-img-wrap">
        <img src="img/placeholder_backdrop.svg" alt="${escapeHtml(saga.name)}" loading="lazy">
        <div class="saga-card-gradient"></div>
      </div>
      <div class="saga-card-name">${escapeHtml(saga.name)}</div>`;
    card.addEventListener('click', () => openSagaModal(saga));
    grid.appendChild(card);
  });
  if (typeof lucide !== 'undefined') lucide.createIcons();

  // Lazily fetch each collection's backdrop in parallel
  SAGAS.forEach(saga => {
    TMDB.getCollection(saga.id)
      .then(data => {
        sagaCache.set(saga.id, data);
        const card = grid.querySelector(`.saga-card[data-collection-id="${saga.id}"]`);
        if (!card) return;
        const img = card.querySelector('.saga-card-img-wrap img');
        const bd = data.backdrop_path
          ? TMDB.backdropUrl(data.backdrop_path)
          : (data.parts?.[0]?.backdrop_path ? TMDB.backdropUrl(data.parts[0].backdrop_path) : 'img/placeholder_backdrop.svg');
        if (img) img.src = bd;
      })
      .catch(() => { });
  });

}

async function openSagaModal(saga) {
  const modal = document.getElementById('saga-modal');
  const titleEl = document.getElementById('saga-modal-title');
  const overviewEl = document.getElementById('saga-modal-overview');
  const backdropEl = document.getElementById('saga-modal-backdrop');
  const grid = document.getElementById('saga-modal-grid');
  if (!modal) return;

  modal.classList.add('open');
  document.body.style.overflow = 'hidden';
  titleEl.textContent = saga.name;
  overviewEl.textContent = '';
  grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div> Loading films…</div>';

  let data = sagaCache.get(saga.id);
  if (!data) {
    try { data = await TMDB.getCollection(saga.id); sagaCache.set(saga.id, data); }
    catch (e) {
      grid.innerHTML = '<p class="empty-state" style="padding:40px 0">Could not load this saga.</p>';
      return;
    }
  }

  if (data.backdrop_path) backdropEl.src = TMDB.backdropUrl(data.backdrop_path);
  else if (data.parts?.[0]?.backdrop_path) backdropEl.src = TMDB.backdropUrl(data.parts[0].backdrop_path);
  overviewEl.textContent = data.overview || `${data.parts?.length || 0} films in this saga.`;

  // Films inside the saga, sorted by release date asc
  const parts = (data.parts || [])
    .filter(p => p.poster_path)
    .sort((a, b) => {
      const dateA = a.release_date ? new Date(a.release_date).getTime() : 0;
      const dateB = b.release_date ? new Date(b.release_date).getTime() : 0;
      return dateA - dateB;
    });

  grid.innerHTML = '';
  parts.forEach((p, i) => {
    const movie = TMDB.parseLite(p);
    const card = document.createElement('div');
    card.className = 'modal-more-card';
    card.style.animationDelay = `${Math.min(i * 30, 400)}ms`;
    const inList = watchlist.isInWatchlist(movie.movie_id);

    card.innerHTML = `
      <div class="modal-more-poster-wrap">
        <img src="${TMDB.posterUrl(p.poster_path)}" alt="${escapeHtml(movie.title)}" loading="lazy">
        <div class="card-overlay">
          <div class="card-title">${escapeHtml(movie.title)}</div>
          <div class="card-meta">${movie.year || ''}${movie.genres?.length ? ' · ' + movie.genres.slice(0, 2).join(', ') : ''}</div>
          <div class="card-actions">
            <button class="btn-watchlist-card ${inList ? 'in-list' : ''}" data-movie-id="${movie.movie_id}">
              ${inList ? '<i data-lucide="check" class="wl-icon-default"></i><i data-lucide="minus" class="wl-icon-hover"></i>' : '<i data-lucide="plus" class="wl-icon-default"></i>'}
            </button>
          </div>
        </div>
      </div>`;

    // Wire watchlist button
    const wlBtn = card.querySelector('.btn-watchlist-card');
    wlBtn.addEventListener('click', async e => {
      e.stopPropagation();
      await watchlist.toggle(movie.movie_id, e.currentTarget);
    });

    card.addEventListener('click', e => {
      if (!e.target.closest('.btn-watchlist-card')) {
        closeSagaModal();
        setTimeout(() => openMovieModal(movie), 200);
      }
    });
    grid.appendChild(card);
  });

  if (typeof lucide !== 'undefined') lucide.createIcons();
}

function closeSagaModal() {
  document.getElementById('saga-modal')?.classList.remove('open');
  document.body.style.overflow = '';
}

function initSagaModal() {
  document.getElementById('saga-modal-close')?.addEventListener('click', closeSagaModal);
  document.getElementById('saga-modal')?.addEventListener('click', e => {
    if (e.target.id === 'saga-modal') closeSagaModal();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeSagaModal();
  });
}

/**
 * Called when a movie modal opens — checks if the movie belongs to any known
 * saga (using the cached collection data), then populates the "Next in Saga"
 * section with the saga films that come AFTER the current one (in release order).
 */
async function loadSagaForMovie(movieMeta) {
  const section = document.getElementById('modal-saga-section');
  const titleEl = document.getElementById('modal-saga-title');
  const grid = document.getElementById('modal-saga-grid');
  if (!section || !grid) return;

  section.style.display = 'none';
  grid.innerHTML = '';

  const tmdbId = movieMeta.tmdb_id;
  if (!tmdbId) return;

  // Ensure all sagas are loaded into the cache
  await Promise.allSettled(
    SAGAS
      .filter(s => !sagaCache.has(s.id))
      .map(s => TMDB.getCollection(s.id).then(d => sagaCache.set(s.id, d)).catch(() => {}))
  );

  // Find which saga this movie belongs to
  let matchedSaga = null;
  let matchedParts = null;
  for (const [, data] of sagaCache) {
    const parts = (data.parts || [])
      .filter(p => p.poster_path)
      .sort((a, b) => (a.release_date || '').localeCompare(b.release_date || ''));
    const idx = parts.findIndex(p => p.id === tmdbId);
    if (idx !== -1) {
      matchedSaga = SAGAS.find(s => sagaCache.get(s.id) === data);
      matchedParts = parts.slice(idx + 1); // films that come AFTER the current one
      break;
    }
  }

  // Always show "Part of [Saga]" in metadata when a saga is found
  const memberEl = document.getElementById('modal-saga-membership');
  if (memberEl && matchedSaga) {
    memberEl.style.display = '';
    memberEl.innerHTML = `<span class="meta-label">Part of</span> <button class="meta-chip meta-chip-saga" data-action="saga" data-saga-id="${matchedSaga.id}">${escapeHtml(matchedSaga.name)}</button>`;
  }

  if (!matchedParts || matchedParts.length === 0) return;

  section.style.display = '';
  if (titleEl && matchedSaga) {
    titleEl.textContent = `Next in ${matchedSaga.name}`;
  }

  matchedParts.forEach((p, i) => {
    const movie = TMDB.parseLite(p);
    const card = document.createElement('div');
    card.className = 'modal-more-card';
    card.style.animationDelay = `${Math.min(i * 30, 400)}ms`;
    const inList = watchlist.isInWatchlist(movie.movie_id);

    card.innerHTML = `
      <div class="modal-more-poster-wrap">
        <img src="${TMDB.posterUrl(p.poster_path)}" alt="${escapeHtml(movie.title)}" loading="lazy">
        <div class="card-overlay">
          <div class="card-title">${escapeHtml(movie.title)}</div>
          <div class="card-meta">${movie.year || ''}</div>
          <div class="card-actions">
            <button class="btn-watchlist-card ${inList ? 'in-list' : ''}" data-movie-id="${movie.movie_id}">
              ${inList ? '<i data-lucide="check" class="wl-icon-default"></i><i data-lucide="minus" class="wl-icon-hover"></i>' : '<i data-lucide="plus" class="wl-icon-default"></i>'}
            </button>
          </div>
        </div>
      </div>`;

    const wlBtn = card.querySelector('.btn-watchlist-card');
    wlBtn.addEventListener('click', async e => {
      e.stopPropagation();
      await watchlist.toggle(movie.movie_id, e.currentTarget);
    });

    card.addEventListener('click', e => {
      if (!e.target.closest('.btn-watchlist-card')) {
        closeMovieModal();
        setTimeout(() => openMovieModal(movie), 280);
      }
    });

    grid.appendChild(card);
  });

  if (typeof lucide !== 'undefined') lucide.createIcons();
}

