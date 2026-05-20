const RATING_SCALE = { 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0 };

function getRating(movieId) {
  const raw = typeof pstore !== 'undefined' ? pstore.get('ratings') : localStorage.getItem('ratings');
  const stored = JSON.parse(raw || '{}');
  return stored[movieId] || null;
}

function initStarRating(container, movieId, currentRating = null) {
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
      saveRating(movieId, rating);
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

async function saveRating(movieId, rating) {
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

  try {
    const userToken = auth.getToken();
    await api.rateMovie(userToken, movieId, rating);
    if (typeof eventBus !== 'undefined') {
      eventBus.emit('rating:updated', { movieId, rating });
    }
  } catch (err) {
    console.warn('Rating saved locally only:', err);
  }
}
