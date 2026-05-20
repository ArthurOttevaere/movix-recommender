const CONFIG = {
  TMDB_API_KEY: 'cb99611cfdad019191fcad3b9540b57d',
  TMDB_BASE_URL: 'https://api.themoviedb.org/3',
  TMDB_IMAGE_BASE: 'https://image.tmdb.org/t/p',
  API_BASE_URL: 'http://localhost:8000' ,  // null = mode mock | 'http://localhost:8000' = backend FastAPI
  USE_MOCK: false,

  // Free Gemini key from https://aistudio.google.com/apikey
  // Leave empty '' to use the built-in keyword matcher as fallback.
  GEMINI_API_KEY: 'AIzaSyDtxS9eCRGlrlFh_AZYKq6SH5Ywrzbev9s',
};
