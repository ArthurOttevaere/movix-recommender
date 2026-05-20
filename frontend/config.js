// Public defaults — DO NOT put secrets here (this file is committed to git).
// Put real API keys in `config.local.js` (gitignored). It's loaded after
// this file and overrides any value defined below.
const CONFIG = {
  TMDB_API_KEY: '',
  TMDB_BASE_URL: 'https://api.themoviedb.org/3',
  TMDB_IMAGE_BASE: 'https://image.tmdb.org/t/p',
  API_BASE_URL: '',  // vide = URLs relatives (fonctionne avec localhost ET 127.0.0.1)
  USE_MOCK: false,

  // Free Gemini key from https://aistudio.google.com/apikey
  // Leave empty '' to use the built-in keyword matcher as fallback.
  GEMINI_API_KEY: '',
};
