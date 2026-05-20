// Copy this file to config.js and fill in your credentials.
// config.js is git-ignored so your real keys stay local.
const CONFIG = {
  // TMDB — free key from https://www.themoviedb.org/settings/api
  TMDB_API_KEY: 'YOUR_TMDB_API_KEY_HERE',
  TMDB_BASE_URL: 'https://api.themoviedb.org/3',
  TMDB_IMAGE_BASE: 'https://image.tmdb.org/t/p',

  // Backend FastAPI (set to null + USE_MOCK:true to run on static mock data)
  API_BASE_URL: 'http://localhost:8000',
  USE_MOCK: false,

  // Free Gemini key from https://aistudio.google.com/apikey
  // Leave empty '' to use the built-in keyword matcher as fallback.
  GEMINI_API_KEY: '',
};
