// escapeHtml is defined globally in api.js

const navigation = {
  currentView: 'home',
  _allMovies: [],
  _searchTimer: null,
  _lastQuery: '',
  _recommendedIds: new Set(),  // populated by setMoviePool

  init(allMovies) {
    this._allMovies = allMovies || [];
    this._recommendedIds = new Set(this._allMovies.map(m => m.movie_id));
    this._bindNavLinks();
    this._bindSearch();
    this._updateWatchlistBadge();

    if (typeof eventBus !== 'undefined') {
      eventBus.on('watchlist:updated', () => this._updateWatchlistBadge());
    }
  },

  setMoviePool(movies) {
    this._allMovies = movies;
    this._recommendedIds = new Set(movies.map(m => m.movie_id));
  },

  showView(name) {
    if (this.currentView === name) return;
    this.currentView = name;

    document.querySelectorAll('.view-section').forEach(s => {
      s.classList.toggle('active', s.id === `view-${name}`);
    });

    document.querySelectorAll('[data-view]').forEach(el => {
      el.classList.toggle('active', el.dataset.view === name);
    });

    window.scrollTo({ top: 0, behavior: 'smooth' });

    if (name === 'watchlist' && typeof renderWatchlistView === 'function') renderWatchlistView();
    if (name === 'discover'  && typeof renderDiscoverView  === 'function') renderDiscoverView();
    if (name === 'saga'      && typeof renderSagaView      === 'function') renderSagaView();
    if (name === 'surprise'  && typeof initSurpriseView    === 'function') initSurpriseView();
  },

  _bindNavLinks() {
    document.querySelectorAll('[data-view]').forEach(el => {
      el.addEventListener('click', e => {
        e.preventDefault();
        this.showView(el.dataset.view);
      });
    });
  },

  _updateWatchlistBadge() {
    const count = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('watchlist') : localStorage.getItem('watchlist')) || '[]').length;
    const badge = document.getElementById('watchlist-nav-badge');
    if (!badge) return;
    badge.textContent = count || '';
    badge.style.display = count ? 'inline-flex' : 'none';
  },

  // ─── Search ──────────────────────────────────────────────────────────────────

  openSearch() {
    const overlay = document.getElementById('search-overlay');
    overlay?.classList.add('open');
    document.body.style.overflow = 'hidden';
    setTimeout(() => document.getElementById('search-input')?.focus(), 60);
  },

  closeSearch() {
    document.getElementById('search-overlay')?.classList.remove('open');
    document.body.style.overflow = '';
    const input = document.getElementById('search-input');
    if (input) {
      input.value = '';
      this._renderSearchPayload({ movies: [], persons: [] }, false, false);
    }
    this._lastQuery = '';
    clearTimeout(this._searchTimer);
  },

  _bindSearch() {
    document.getElementById('navbar-search-btn')?.addEventListener('click', () => this.openSearch());
    document.getElementById('search-close-btn')?.addEventListener('click', () => this.closeSearch());

    document.getElementById('search-overlay')?.addEventListener('click', e => {
      if (e.target.id === 'search-overlay') this.closeSearch();
    });

    const input = document.getElementById('search-input');
    if (input) {
      input.addEventListener('input', () => this._handleSearch(input.value));
      input.addEventListener('keydown', e => { if (e.key === 'Escape') this.closeSearch(); });
    }

    document.addEventListener('keydown', e => {
      const tag = document.activeElement?.tagName;
      if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault();
        this.openSearch();
      }
    });
  },

  _handleSearch(query) {
    const q = query.trim();
    clearTimeout(this._searchTimer);

    if (q.length < 1) {
      this._renderSearchPayload({ movies: [], persons: [] }, false, false);
      this._lastQuery = '';
      return;
    }

    const ql = q.toLowerCase();

    // Local search — instantaneous
    const unique = new Map();
    this._allMovies.forEach(m => {
      if (m?.title?.toLowerCase().includes(ql) && !unique.has(m.movie_id)) {
        unique.set(m.movie_id, m);
      }
    });
    const local = [...unique.values()];
    this._renderSearchPayload({ movies: local, persons: [] }, q.length >= 2, false);
    this._lastQuery = q;

    // TMDB search (parallel: movies + persons) — debounced
    if (q.length >= 2) {
      this._searchTimer = setTimeout(async () => {
        try {
          const [tmdbMoviesRaw, persons] = await Promise.all([
            TMDB.searchMovies(q),
            TMDB.searchPerson(q),
          ]);
          const tmdbMovies = (tmdbMoviesRaw || []).map(r => TMDB.parseLite(r));
          const localIds = new Set(local.map(m => m.movie_id));
          const combined = [...local, ...tmdbMovies.filter(m => !localIds.has(m.movie_id))];

          // Boost: recommended films first
          combined.sort((a, b) => {
            const ra = this._recommendedIds.has(a.movie_id) ? 1 : 0;
            const rb = this._recommendedIds.has(b.movie_id) ? 1 : 0;
            return rb - ra;
          });

          if (document.getElementById('search-input')?.value?.trim() === this._lastQuery) {
            const topPersons = (persons || []).filter(p => p.popularity > 5 && p.profile_path).slice(0, 3);
            this._renderSearchPayload({ movies: combined, persons: topPersons }, false, combined.length === 0 && !topPersons.length);
          }
        } catch (err) {
          console.warn('TMDB search failed:', err);
          this._renderSearchPayload({ movies: local, persons: [] }, false, local.length === 0);
        }
      }, 320);
    }
  },

  // ─── Render the combined search payload ─────────────────────────────────────
  _renderSearchPayload(payload, searching, noResults) {
    const container = document.getElementById('search-results');
    const hint = document.querySelector('.search-hint');
    if (!container) return;

    const { movies = [], persons = [] } = payload;
    const hasAny = movies.length || persons.length;

    if (hint) hint.style.display = (hasAny || searching || noResults) ? 'none' : '';

    if (!hasAny && !searching && !noResults) {
      container.innerHTML = '';
      return;
    }

    if (noResults) {
      container.innerHTML = `<p class="search-empty">No results found — try a different title or person.</p>`;
      return;
    }

    const loadingHtml = searching
      ? `<div class="search-status"><div class="spinner" style="width:14px;height:14px;border-width:2px"></div> Searching TMDB…</div>`
      : `<div class="search-status">${movies.length} film${movies.length !== 1 ? 's' : ''}${persons.length ? ` · ${persons.length} person${persons.length > 1 ? 's' : ''}` : ''}</div>`;

    const personsHtml = persons.length ? `
      <div class="search-section-title">People</div>
      <div class="search-persons">
        ${persons.map(p => `
          <button class="search-person-card" data-person-id="${p.id}" data-person-name="${escapeHtml(p.name)}">
            <img class="search-person-photo" src="${TMDB.profileUrl(p.profile_path)}" alt="${escapeHtml(p.name)}" loading="lazy">
            <div class="search-person-info">
              <div class="search-person-name">${escapeHtml(p.name)}</div>
              <div class="search-person-known">${escapeHtml(p.known_for_department || 'Actor')}</div>
            </div>
            <i data-lucide="chevron-right" class="search-person-chev"></i>
          </button>`).join('')}
      </div>
    ` : '';

    const moviesHtml = movies.length ? `
      ${persons.length ? `<div class="search-section-title">Films</div>` : ''}
      <div class="search-results-grid">
        ${movies.map((m, i) => this._movieCardHtml(m, i)).join('')}
      </div>
    ` : '';

    container.innerHTML = `${loadingHtml}${personsHtml}${moviesHtml}`;
    this._wireMovieCards(container, movies);

    // Person cards — open filmography
    container.querySelectorAll('.search-person-card').forEach(card => {
      card.addEventListener('click', async () => {
        const personId = parseInt(card.dataset.personId);
        const name = card.dataset.personName;
        await this._renderActorFilmography(personId, name);
      });
    });

    // Lazy poster fetch
    container.querySelectorAll('[data-tmdb]').forEach(img => {
      if (img.src.includes('placeholder')) {
        const id = parseInt(img.dataset.tmdb);
        if (id) getMovieDetails(id).then(d => { img.src = d.poster_url; }).catch(() => {});
      }
    });

    if (typeof lucide !== 'undefined') lucide.createIcons();
  },

  _movieCardHtml(m, i) {
    const isRec = this._recommendedIds.has(m.movie_id);
    const poster = (m.poster_url && m.poster_url.includes('image.tmdb')) ? m.poster_url : 'img/placeholder_poster.svg';
    return `
      <div class="search-grid-card ${isRec ? 'is-recommended' : ''}" data-movie-id="${m.movie_id}" style="animation-delay:${Math.min(i * 18, 400)}ms">
        <div class="search-grid-poster-wrap">
          ${isRec ? '<span class="recommended-badge">For you</span>' : ''}
          <img class="search-grid-poster"
               src="${poster}"
               alt="${escapeHtml(m.title)}"
               data-tmdb="${m.tmdb_id}"
               loading="lazy">
          <div class="search-grid-overlay">
            <div class="search-grid-play"><i data-lucide="play"></i></div>
          </div>
        </div>
        <div class="search-grid-info">
          <div class="search-grid-title">${escapeHtml(m.title)}</div>
          <div class="search-grid-meta">${m.year || ''}${m.genres?.length ? ' · ' + m.genres.slice(0, 2).join(', ') : ''}</div>
        </div>
      </div>`;
  },

  _wireMovieCards(container, movies) {
    container.querySelectorAll('.search-grid-card').forEach(card => {
      card.addEventListener('click', () => {
        const id = parseInt(card.dataset.movieId);
        const movie = movies.find(m => m.movie_id === id);
        if (movie) { this.closeSearch(); openMovieModal(movie); }
      });
    });
  },

  async _renderActorFilmography(personId, name) {
    const container = document.getElementById('search-results');
    if (!container) return;
    container.innerHTML = `
      <div class="search-status">
        <button class="search-back-btn" id="search-back-btn">
          <i data-lucide="arrow-left"></i> Back to search
        </button>
        <span style="margin-left:auto">Films with <strong style="color:var(--text-primary)">${escapeHtml(name)}</strong></span>
      </div>
      <div class="loading-spinner"><div class="spinner"></div> Loading filmography…</div>`;
    if (typeof lucide !== 'undefined') lucide.createIcons();

    document.getElementById('search-back-btn')?.addEventListener('click', () => {
      const input = document.getElementById('search-input');
      if (input) this._handleSearch(input.value);
    });

    try {
      const raw = await TMDB.personMovies(personId);
      // Sort by popularity desc, dedupe by id
      const seen = new Set();
      const movies = raw
        .filter(m => m.poster_path && !seen.has(m.id) && seen.add(m.id))
        .sort((a, b) => (b.popularity || 0) - (a.popularity || 0))
        .map(r => TMDB.parseLite(r));

      if (!movies.length) {
        container.innerHTML = container.innerHTML.replace(
          '<div class="loading-spinner"',
          '<p class="search-empty" style="padding:40px 0">No films found.</p><div style="display:none" '
        );
        return;
      }

      // Boost recommended ones
      movies.sort((a, b) => {
        const ra = this._recommendedIds.has(a.movie_id) ? 1 : 0;
        const rb = this._recommendedIds.has(b.movie_id) ? 1 : 0;
        return rb - ra;
      });

      container.innerHTML = `
        <div class="search-status">
          <button class="search-back-btn" id="search-back-btn">
            <i data-lucide="arrow-left"></i> Back to search
          </button>
          <span style="margin-left:auto">${movies.length} film${movies.length > 1 ? 's' : ''} with <strong style="color:var(--text-primary)">${escapeHtml(name)}</strong></span>
        </div>
        <div class="search-results-grid">
          ${movies.map((m, i) => this._movieCardHtml(m, i)).join('')}
        </div>`;
      this._wireMovieCards(container, movies);
      document.getElementById('search-back-btn')?.addEventListener('click', () => {
        const input = document.getElementById('search-input');
        if (input) this._handleSearch(input.value);
      });
      if (typeof lucide !== 'undefined') lucide.createIcons();
    } catch (e) {
      console.warn('Filmography load failed:', e);
      container.innerHTML = `<p class="search-empty">Couldn't load this person's films.</p>`;
    }
  },
};
