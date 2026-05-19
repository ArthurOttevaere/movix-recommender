const CONFIG = {
  TMDB_API_KEY: 'b86d4f0cde9c85cb9d47e7f4b4d3d6eb',
  TMDB_BASE_URL: 'https://api.themoviedb.org/3',
  TMDB_IMAGE_BASE: 'https://image.tmdb.org/t/p',
  API_BASE_URL: 'http://localhost:8000',  // null = mode mock | 'http://localhost:8000' = backend FastAPI
  USE_MOCK: false,

  // Free Gemini key from https://aistudio.google.com/apikey
  // Leave empty '' to use the built-in keyword matcher as fallback.
  GEMINI_API_KEY: 'AIzaSyAPo46awM87P7E9G6rsBMm81i_HLVvSG24',
};
