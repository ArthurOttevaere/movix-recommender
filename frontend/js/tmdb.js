// Genre ID → name mapping (TMDB genre IDs)
const TMDB_GENRE_MAP = {
  28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
  80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
  14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
  9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
  53: 'Thriller', 10752: 'War', 37: 'Western'
};

const TMDB = {
  async getMovie(tmdbId) {
    const url = `${CONFIG.TMDB_BASE_URL}/movie/${tmdbId}?api_key=${CONFIG.TMDB_API_KEY}&language=en-US&append_to_response=credits`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`TMDB error: ${res.status}`);
    return res.json();
  },

  async searchMovie(title, year) {
    const query = encodeURIComponent(title);
    const url = `${CONFIG.TMDB_BASE_URL}/search/movie?api_key=${CONFIG.TMDB_API_KEY}&query=${query}&year=${year}&language=en-US`;
    const res = await fetch(url);
    const data = await res.json();
    return data.results?.[0] || null;
  },

  async searchMovies(query, page = 1) {
    const url = `${CONFIG.TMDB_BASE_URL}/search/movie?api_key=${CONFIG.TMDB_API_KEY}&query=${encodeURIComponent(query)}&language=en-US&page=${page}&include_adult=false`;
    const res = await fetch(url);
    if (!res.ok) return [];
    const data = await res.json();
    return data.results || [];
  },

  async discoverByGenre(genreOrOpts, page = 1, sortBy = 'popularity.desc') {
    // Legacy signature: discoverByGenre(123) or discoverByGenre(123, 2, 'vote_average.desc')
    // New signature:    discoverByGenre({ genres: [123,456], sort_by, vote_count_gte, vote_count_lte, year_gte, year_lte, page })
    const params = new URLSearchParams({
      api_key: CONFIG.TMDB_API_KEY,
      language: 'en-US',
      include_adult: 'false',
    });
    if (typeof genreOrOpts === 'object' && genreOrOpts !== null) {
      const o = genreOrOpts;
      if (o.genres?.length) params.set('with_genres', o.genres.join(','));
      params.set('sort_by', o.sort_by || sortBy);
      params.set('page', String(o.page || page));
      if (o.vote_count_gte != null) params.set('vote_count.gte', String(o.vote_count_gte));
      else params.set('vote_count.gte', '80');
      if (o.vote_count_lte != null) params.set('vote_count.lte', String(o.vote_count_lte));
      if (o.without_genres?.length) params.set('without_genres', o.without_genres.join(','));
      if (o.year_gte != null) params.set('primary_release_date.gte', `${o.year_gte}-01-01`);
      if (o.year_lte != null) params.set('primary_release_date.lte', `${o.year_lte}-12-31`);
      if (o.region) params.set('region', o.region);
    } else {
      params.set('with_genres', String(genreOrOpts));
      params.set('sort_by', sortBy);
      params.set('page', String(page));
      params.set('vote_count.gte', '80');
    }
    const url = `${CONFIG.TMDB_BASE_URL}/discover/movie?${params.toString()}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`TMDB ${res.status}`);
    const data = await res.json();
    return data.results || [];
  },

  async topRated(page = 1) {
    const url = `${CONFIG.TMDB_BASE_URL}/movie/top_rated?api_key=${CONFIG.TMDB_API_KEY}&language=en-US&page=${page}`;
    const res = await fetch(url);
    if (!res.ok) return [];
    const data = await res.json();
    return data.results || [];
  },

  async trending(timeWindow = 'day') {
    const url = `${CONFIG.TMDB_BASE_URL}/trending/movie/${timeWindow}?api_key=${CONFIG.TMDB_API_KEY}&language=en-US`;
    const res = await fetch(url);
    if (!res.ok) return [];
    const data = await res.json();
    return data.results || [];
  },

  async searchPerson(query) {
    const url = `${CONFIG.TMDB_BASE_URL}/search/person?api_key=${CONFIG.TMDB_API_KEY}&query=${encodeURIComponent(query)}&include_adult=false&language=en-US`;
    const res = await fetch(url);
    if (!res.ok) return [];
    const data = await res.json();
    return data.results || [];
  },

  async personMovies(personId) {
    const url = `${CONFIG.TMDB_BASE_URL}/person/${personId}/movie_credits?api_key=${CONFIG.TMDB_API_KEY}&language=en-US`;
    const res = await fetch(url);
    if (!res.ok) return [];
    const data = await res.json();
    return data.cast || [];
  },

  async getCollection(collectionId) {
    const url = `${CONFIG.TMDB_BASE_URL}/collection/${collectionId}?api_key=${CONFIG.TMDB_API_KEY}&language=en-US`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`TMDB ${res.status}`);
    return res.json();
  },

  // Lightweight parse for discover/search results (no credits)
  parseLite(r) {
    return {
      movie_id: r.id,
      tmdb_id: r.id,
      title: r.title || r.original_title || '',
      release_date: r.release_date || '',
      year: r.release_date?.split('-')[0] || '',
      genres: (r.genre_ids || []).map(id => TMDB_GENRE_MAP[id]).filter(Boolean),
      overview: r.overview || '',
      poster_url: TMDB.posterUrl(r.poster_path),
      backdrop_url: TMDB.backdropUrl(r.backdrop_path),
      vote_average: r.vote_average,
      score: r.vote_average ? Math.min(r.vote_average / 10, 1) : null,
    };
  },

  posterUrl(path, size = 'w342') {
    if (!path) return 'img/placeholder_poster.svg';
    return `${CONFIG.TMDB_IMAGE_BASE}/${size}${path}`;
  },

  backdropUrl(path, size = 'w1280') {
    if (!path) return 'img/placeholder_backdrop.svg';
    return `${CONFIG.TMDB_IMAGE_BASE}/${size}${path}`;
  },

  profileUrl(path, size = 'w185') {
    if (!path) return 'img/placeholder_person.svg';
    return `${CONFIG.TMDB_IMAGE_BASE}/${size}${path}`;
  },

  parseMovie(tmdbData) {
    const directors = tmdbData.credits?.crew
      ?.filter(p => p.job === 'Director')
      ?.map(p => p.name) || [];

    const cast = tmdbData.credits?.cast
      ?.slice(0, 5)
      ?.map(p => ({
        name: p.name,
        character: p.character,
        profile_url: TMDB.profileUrl(p.profile_path)
      })) || [];

    return {
      tmdb_id: tmdbData.id,
      movie_id: tmdbData.id,
      title: tmdbData.title,
      original_title: tmdbData.original_title,
      release_date: tmdbData.release_date || '',
      year: tmdbData.release_date?.split('-')[0] || '—',
      runtime_min: tmdbData.runtime,
      genres: tmdbData.genres?.map(g => g.name) || [],
      overview: tmdbData.overview,
      tagline: tmdbData.tagline,
      poster_url: TMDB.posterUrl(tmdbData.poster_path),
      backdrop_url: TMDB.backdropUrl(tmdbData.backdrop_path),
      director: directors[0] || '—',
      cast,
      vote_average: tmdbData.vote_average,
      popularity: tmdbData.popularity,
    };
  }
};

const tmdbCache = new Map();

async function getMovieDetails(tmdbId) {
  if (tmdbCache.has(tmdbId)) return tmdbCache.get(tmdbId);
  const raw = await TMDB.getMovie(tmdbId);
  const parsed = TMDB.parseMovie(raw);
  tmdbCache.set(tmdbId, parsed);
  return parsed;
}
