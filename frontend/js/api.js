// Global utility — available to all scripts loaded after api.js
function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str || '';
  return d.innerHTML;
}

const api = {
  async getOnboardingMovies() {
    if (CONFIG.USE_MOCK) {
      return fetch('mock/onboarding.json').then(r => r.json());
    }
    return fetch(`${CONFIG.API_BASE_URL}/onboarding/movies`).then(r => r.json());
  },

  async submitOnboarding(ratings) {
    if (CONFIG.USE_MOCK) {
      const token = 'mock_session_' + Date.now();
      if (typeof pstore !== 'undefined') pstore.set('user_token', token);
      else localStorage.setItem('user_token', token);
      return { user_token: token, status: 'ok' };
    }
    const res = await fetch(`${CONFIG.API_BASE_URL}/onboarding/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ratings })
    });
    if (!res.ok) throw new Error(`Onboarding failed: ${res.status}`);
    const data = await res.json();
    if (data.user_token) {
      if (typeof pstore !== 'undefined') pstore.set('user_token', data.user_token);
      else localStorage.setItem('user_token', data.user_token);
    }
    return data;
  },

  async getRecommendations(userToken) {
    if (CONFIG.USE_MOCK) {
      return fetch('mock/recommendations.json').then(r => r.json());
    }
    return fetch(`${CONFIG.API_BASE_URL}/recommendations/${userToken}`).then(r => r.json());
  },

  async getMovie(movieId, userToken) {
    const recs = await this.getRecommendations(userToken);
    const allMovies = [
      recs.hero,
      ...recs.carousels.flatMap(c => c.movies)
    ];
    const found = allMovies.find(m => m.movie_id === movieId);
    const tmdbId = found?.tmdb_id || movieId;
    return getMovieDetails(tmdbId);
  },

  async rateMovie(userToken, movieId, rating) {
    if (CONFIG.USE_MOCK) {
      console.log(`[MOCK] Rate movie ${movieId}: ${rating}`);
      return { status: 'ok', profile_updated: true };
    }
    return fetch(`${CONFIG.API_BASE_URL}/rate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_token: userToken, movie_id: movieId, rating })
    }).then(r => r.json());
  },

  async toggleWatchlist(userToken, movieId, action) {
    const wl = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('watchlist') : localStorage.getItem('watchlist')) || '[]');
    if (action === 'add' && !wl.includes(movieId)) wl.push(movieId);
    if (action === 'remove') {
      const idx = wl.indexOf(movieId);
      if (idx > -1) wl.splice(idx, 1);
    }
    if (typeof pstore !== 'undefined') pstore.set('watchlist', JSON.stringify(wl));
    else localStorage.setItem('watchlist', JSON.stringify(wl));

    if (!CONFIG.USE_MOCK) {
      return fetch(`${CONFIG.API_BASE_URL}/watchlist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_token: userToken, movie_id: movieId, action })
      }).then(r => r.json());
    }
    return { status: 'ok' };
  },

  async getProfile(userToken) {
    if (CONFIG.USE_MOCK) {
      const ratings = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('ratings') : localStorage.getItem('ratings')) || '{}');
      const watchlistIds = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('watchlist') : localStorage.getItem('watchlist')) || '[]');
      const base = await fetch('mock/profile.json').then(r => r.json());

      // Merge localStorage ratings into history
      const localHistory = Object.entries(ratings).map(([id, r]) => ({
        movie_id: parseInt(id),
        title: `Film #${id}`,
        rating: r,
        timestamp: new Date().toISOString()
      }));

      return {
        ...base,
        total_ratings: Object.keys(ratings).length || base.total_ratings,
        watchlist: watchlistIds,
        rating_history: base.rating_history.length ? base.rating_history : localHistory
      };
    }
    return fetch(`${CONFIG.API_BASE_URL}/profile/${userToken}`).then(r => r.json());
  }
};
