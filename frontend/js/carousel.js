// escapeHtml is defined globally in api.js

const MODEL_LABELS = {
  content_based: 'Content-based · Genome features',
  user_based: 'Collaborative · User KNN',
  svd: 'Latent factor · SVD matrix',
  ensemble: 'Ensemble · Diversity boost',
  trending: 'Real-time · Popularity signal',
  history: 'Your watch history',
  'genre:Drama': 'Genre · Drama',
  'genre:Thriller': 'Genre · Thriller',
  'genre:Sci-Fi': 'Genre · Sci-Fi',
  'genre:Animation': 'Genre · Animation',
  'genre:Crime': 'Genre · Crime',
};

function createCarouselSection(carouselData) {
  const section = document.createElement('section');
  section.className = 'carousel';
  section.id = carouselData.id;

  const modelLabel = MODEL_LABELS[carouselData.model] || '';

  section.innerHTML = `
    <div class="carousel-header">
      <h2 class="carousel-title">${escapeHtml(carouselData.label)}</h2>
      ${modelLabel ? `<span class="carousel-model-badge">${escapeHtml(modelLabel)}</span>` : ''}
    </div>
    <div class="carousel-wrapper">
      <button class="carousel-arrow carousel-arrow-left" aria-label="Previous" data-dir="-1">
        <i data-lucide="chevron-left"></i>
      </button>
      <div class="carousel-track" id="track-${carouselData.id}"></div>
      <button class="carousel-arrow carousel-arrow-right" aria-label="Next" data-dir="1">
        <i data-lucide="chevron-right"></i>
      </button>
    </div>`;

  // Skeleton cards
  const track = section.querySelector('.carousel-track');
  for (let i = 0; i < 8; i++) {
    const sk = document.createElement('div');
    sk.className = 'movie-card movie-card-skeleton';
    sk.innerHTML = '<div class="movie-card-inner"></div>';
    track.appendChild(sk);
  }

  // Arrow nav — scroll by ~85% of the track's visible width for a snappy paginate feel
  section.querySelectorAll('.carousel-arrow').forEach(btn => {
    btn.addEventListener('click', () => {
      const dir = parseInt(btn.dataset.dir);
      const step = Math.max(track.clientWidth * 0.85, 600);
      track.scrollBy({ left: dir * step, behavior: 'smooth' });
    });
  });

  return section;
}

function createMovieCard(movie, opts = {}) {
  const card = document.createElement('div');
  card.className = 'movie-card';
  card.dataset.movieId = movie.movie_id;

  const inList = watchlist.isInWatchlist(movie.movie_id);
  const scorePct = movie.score ? Math.round(movie.score * 100) : null;
  let badgeLabel = '';
  const now = new Date();
  if (movie.release_date) {
    const [y, m] = movie.release_date.split('-');
    if (parseInt(y) === now.getFullYear() && parseInt(m) === now.getMonth() + 1) {
      badgeLabel = 'New this month';
    }
  }
  if (!badgeLabel && parseInt(movie.year) >= now.getFullYear() - 1) {
    badgeLabel = 'Recently added';
  }
  // The rank badge only appears in carousels explicitly flagged as showRank.
  const isTopRanked = opts.showRank && movie.rank != null && movie.rank <= 20;
  const hasProgress = movie.progress != null;

  if (isTopRanked) card.classList.add('is-top10');

  card.innerHTML = `
    ${isTopRanked ? `<span class="top10-rank">${movie.rank}</span>` : ''}
    <div class="movie-card-inner">
      <img
        src="img/placeholder_poster.svg"
        alt="${escapeHtml(movie.title)}"
        loading="lazy"
      >
      ${badgeLabel && !isTopRanked ? `<span class="badge-new">${badgeLabel}</span>` : ''}
      ${hasProgress ? `
        <div class="card-progress-bar">
          <div class="card-progress-fill" style="width:${movie.progress}%"></div>
        </div>` : ''}
      <div class="card-overlay">
        <div class="card-title">${escapeHtml(movie.title)}</div>
        <div class="card-meta">${movie.year || ''} · ${(movie.genres || []).slice(0, 2).join(', ')}</div>
        <div class="card-actions">
          <button class="btn-watchlist-card ${inList ? 'in-list' : ''}"
                  data-movie-id="${movie.movie_id}"
                  title="${inList ? 'Remove from My List' : 'Add to My List'}">
              ${inList ? '<i data-lucide="check" class="wl-icon-default"></i><i data-lucide="minus" class="wl-icon-hover"></i>' : '<i data-lucide="plus" class="wl-icon-default"></i>'}
          </button>
          ${scorePct ? `<span class="card-score">${scorePct}%</span>` : ''}
          ${hasProgress ? `<span class="card-progress-label">${movie.progress}%</span>` : ''}
        </div>
      </div>
    </div>`;

  card.addEventListener('click', e => {
    if (!e.target.closest('.btn-watchlist-card')) {
      if (typeof openMovieModal === 'function') openMovieModal(movie);
    }
  });

  card.querySelector('.btn-watchlist-card').addEventListener('click', async e => {
    e.stopPropagation();
    await watchlist.toggle(movie.movie_id, e.currentTarget);
  });

  return card;
}

// Populate from mock/API data — fetches TMDB poster lazily
async function populateCarousel(carouselData) {
  const track = document.getElementById(`track-${carouselData.id}`);
  if (!track) return;
  track.innerHTML = '';

  const cardOpts = { showRank: !!carouselData.showRank };

  carouselData.movies.forEach((movie, i) => {
    // Assign rank based on position when this carousel shows rank badges
    if (cardOpts.showRank) movie.rank = i + 1;

    const card = createMovieCard(movie, cardOpts);
    _staggerCard(card, i);
    track.appendChild(card);

    getMovieDetails(movie.tmdb_id).then(details => {
      const img = card.querySelector('img');
      if (img) img.src = details.poster_url;
      card._tmdbDetails = details;
    }).catch(() => { });
  });

  if (typeof lucide !== 'undefined') lucide.createIcons();
}

// Populate from TMDB parseLite results — poster already known, no extra fetch
function populateCarouselWithPosters(carouselData) {
  const track = document.getElementById(`track-${carouselData.id}`);
  if (!track) return;
  track.innerHTML = '';

  const cardOpts = { showRank: !!carouselData.showRank };

  carouselData.movies.forEach((movie, i) => {
    // Assign rank based on position when this carousel shows rank badges
    if (cardOpts.showRank) movie.rank = i + 1;

    const card = createMovieCard(movie, cardOpts);

    // Set poster immediately if it's a real TMDB URL
    if (movie.poster_url && movie.poster_url.includes('image.tmdb')) {
      const img = card.querySelector('img');
      if (img) img.src = movie.poster_url;
    }

    _staggerCard(card, i);
    track.appendChild(card);
  });

  if (typeof lucide !== 'undefined') lucide.createIcons();
}

function _staggerCard(card, index) {
  const delay = Math.min(index * 30, 500);
  card.style.opacity = '0';
  card.style.transform = 'translateY(14px)';
  setTimeout(() => {
    card.style.transition = 'opacity 0.35s ease, transform 0.35s ease';
    card.style.opacity = '1';
    card.style.transform = '';
    setTimeout(() => {
      card.style.transition = '';
      card.style.opacity = '';
      card.style.transform = '';
    }, 400);
  }, delay);
}
