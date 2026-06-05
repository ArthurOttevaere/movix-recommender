// Public defaults — committed to git. Leave all keys empty here.
// To add your real API keys:
//   1. Copy `config.local.js.example` → `config.local.js`
//   2. Fill in your keys in `config.local.js`
// `config.local.js` is gitignored and will never be committed.
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
