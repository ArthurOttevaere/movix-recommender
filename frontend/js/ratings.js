const RATING_SCALE = { 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0 };

function getRating(movieId) {
  const raw = typeof pstore !== 'undefined' ? pstore.get('ratings') : localStorage.getItem('ratings');
  const stored = JSON.parse(raw || '{}');
  return stored[movieId] || null;
}

function initStarRating(container, movieId, currentRating = null, meta = null) {
  const stars = container.querySelectorAll('.star');

  const savedValue = currentRating ? Math.round(currentRating) : 0;
  container.dataset.rating = savedValue;

  if (savedValue) {
    highlightStars(stars, savedValue);
    updateRatingLabel(container, savedValue);
  }

  stars.forEach(star => {
    const value = parseInt(star.dataset.value);

    star.addEventListener('mouseenter', () => {
      highlightStars(stars, value);
    });

    star.addEventListener('mouseleave', () => {
      const saved = parseInt(container.dataset.rating || 0);
      highlightStars(stars, saved);
    });

    star.addEventListener('click', () => {
      const rating = RATING_SCALE[value];
      container.dataset.rating = value;
      highlightStars(stars, value);
      updateRatingLabel(container, value);
      saveRating(movieId, rating, meta);
    });
  });
}

function highlightStars(stars, count) {
  stars.forEach((star, i) => {
    star.style.color = i < count ? 'var(--star-color)' : 'var(--text-muted)';
  });
}

function updateRatingLabel(container, value) {
  const label = container.closest('.rating-section')?.querySelector('.rating-label');
  if (label) {
    label.textContent = value ? `${value}/5 stars` : '';
  }
}

async function saveRating(movieId, rating, meta = null) {
  const raw = typeof pstore !== 'undefined' ? pstore.get('ratings') : localStorage.getItem('ratings');
  const stored = JSON.parse(raw || '{}');
  stored[movieId] = rating;
  if (typeof pstore !== 'undefined') pstore.set('ratings', JSON.stringify(stored));
  else localStorage.setItem('ratings', JSON.stringify(stored));

  const tRaw = typeof pstore !== 'undefined' ? pstore.get('rating_timestamps') : localStorage.getItem('rating_timestamps');
  const timestamps = JSON.parse(tRaw || '{}');
  timestamps[movieId] = new Date().toISOString();
  if (typeof pstore !== 'undefined') pstore.set('rating_timestamps', JSON.stringify(timestamps));
  else localStorage.setItem('rating_timestamps', JSON.stringify(timestamps));

  // Remember the TMDB id + title so the rating history can show the right poster
  // (ratings are keyed by MovieLens movie_id, which getMovieDetails can't resolve).
  if (meta && (meta.tmdb_id || meta.title)) {
    const mRaw = typeof pstore !== 'undefined' ? pstore.get('rating_meta') : localStorage.getItem('rating_meta');
    const metaMap = JSON.parse(mRaw || '{}');
    metaMap[movieId] = { tmdb_id: meta.tmdb_id || null, title: meta.title || '' };
    if (typeof pstore !== 'undefined') pstore.set('rating_meta', JSON.stringify(metaMap));
    else localStorage.setItem('rating_meta', JSON.stringify(metaMap));
  }

  // Refresh the UI immediately from the local save (don't wait on the network).
  if (typeof eventBus !== 'undefined') {
    eventBus.emit('rating:updated', { movieId, rating });
  }

  try {
    await api.rateMovie(auth.getToken(), movieId, rating);
  } catch (err) {
    console.warn('Rating saved locally only:', err);
  }
}
