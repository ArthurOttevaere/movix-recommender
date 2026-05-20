// ─── Gemini Flash — semantic movie search engine for CineBot ──────────────────
// Gemini acts as a film expert: given any description, it names the actual movies
// that match, which are then fetched directly from TMDB.
//
// Key: get a free API key at https://aistudio.google.com/apikey
// Set CONFIG.GEMINI_API_KEY in config.js — leave empty to use keyword fallback.

const GEMINI_MODEL = 'gemini-2.0-flash';
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

const TMDB_GENRE_IDS = {
  'Action': 28, 'Adventure': 12, 'Animation': 16, 'Comedy': 35,
  'Crime': 80, 'Documentary': 99, 'Drama': 18, 'Family': 10751,
  'Fantasy': 14, 'History': 36, 'Horror': 27, 'Music': 10402,
  'Mystery': 9648, 'Romance': 10749, 'Science Fiction': 878,
  'Thriller': 53, 'War': 10752, 'Western': 37,
};

function _buildSystemPrompt(profile, topGenres, recentTitles) {
  const profileCtx = [
    profile?.name ? `Name: ${profile.name}` : null,
    profile?.age ? `Age: ${profile.age}` : null,
    profile?.gender ? `Gender: ${profile.gender}` : null,
    topGenres?.length ? `Favourite genres: ${topGenres.join(', ')}` : null,
    recentTitles?.length ? `Recently rated: ${recentTitles.join(', ')}` : null,
  ].filter(Boolean).join('\n');

  const profileGenreIds = topGenres?.length
    ? topGenres.map(g => TMDB_GENRE_IDS[g]).filter(Boolean)
    : [18, 28, 53];

  return `You are CineBot — a film expert AI powering the semantic search of a streaming app.
Your job: given any user message, identify the real movie titles that best match and return them.

${profileCtx ? `USER PROFILE:\n${profileCtx}\n` : ''}
AVAILABLE TMDB GENRE IDs: ${Object.entries(TMDB_GENRE_IDS).map(([n, id]) => `${id}=${n}`).join(', ')}

━━━ HOW TO RESPOND ━━━

CASE A — The user describes a film (plot, characters, theme, actor, director, era, atmosphere):
  → Fill "candidates" with up to 8 real movie titles (in English, as listed on TMDB) that best match.
  → Rank from best match to weakest.
  → Leave "discover" as null.
  → "response": short friendly text in the user's language acknowledging what was understood.

  Examples:
  • "un film avec des voitures pour les enfants"
    candidates: ["Cars", "Cars 2", "Cars 3", "Turbo", "Speed Racer"]
  • "le film avec le lion et son père"
    candidates: ["The Lion King", "The Lion King (2019)"]
  • "un thriller psychologique des années 90"
    candidates: ["The Silence of the Lambs", "Se7en", "Fight Club", "The Usual Suspects", "Memento", "Primal Fear"]
  • "un film de Nolan avec un magicien"
    candidates: ["The Prestige"]
  • "un film d'amour avec Audrey Hepburn"
    candidates: ["Roman Holiday", "Breakfast at Tiffany's", "Sabrina", "Funny Face", "Wait Until Dark"]
  • "un film de guerre réaliste"
    candidates: ["Saving Private Ryan", "Dunkirk", "1917", "Hacksaw Ridge", "Full Metal Jacket", "Apocalypse Now"]

CASE B — The user is vague, wants a surprise, or only says a mood/feeling with no identifiable film:
  → Leave "candidates" as [].
  → Fill "discover" with genre IDs and sort preference for a TMDB discover query.
  → Use the user's favourite genre IDs (${profileGenreIds.join(', ')}) when vague.

  Examples:
  • "surprends-moi" → candidates: [], discover: { genres: [${profileGenreIds.slice(0, 2).join(', ')}], sort: "varied" }
  • "bonsoir", "je sais pas" → same as above
  • "j'ai envie de pleurer" → candidates: [], discover: { genres: [18, 10749], sort: "quality" }
  • "quelque chose de fun" → candidates: [], discover: { genres: [35, 12], sort: "popular" }

━━━ RULES ━━━
- Prefer CASE A whenever possible. Even "un thriller des années 90" should yield real titles, not a discover.
- "candidates" titles must be exact English titles as on TMDB (e.g. "The Dark Knight", not "Batman").
- "response" must be 1-2 sentences max, in the user's language (French if they write French).
- NEVER output markdown, backticks, or explanation — ONLY valid JSON.

JSON SCHEMA:
{
  "candidates": ["<English title>", ...],
  "discover": { "genres": [<TMDB_ids>], "sort": "quality|popular|recent|varied", "decade": "<1990s|null>", "exclude_genres": [<TMDB_ids>] } | null,
  "response": "<friendly reply in user language>"
}`;
}

/**
 * Call Gemini and return the parsed intent, or null on failure.
 */
async function geminiParseIntent(userMessage, history = [], profile = null, topGenres = [], recentTitles = []) {
  const key = CONFIG.GEMINI_API_KEY;
  if (!key) return null;

  const systemPrompt = _buildSystemPrompt(profile, topGenres, recentTitles);
  const contents = [];

  contents.push({ role: 'user', parts: [{ text: systemPrompt }] });
  contents.push({ role: 'model', parts: [{ text: '{"candidates":[],"discover":null,"response":"Understood."}' }] });

  for (const msg of history.slice(-8)) {
    if (msg.role === 'user') {
      contents.push({ role: 'user', parts: [{ text: msg.text }] });
    } else if (msg.role === 'assistant' && msg.text) {
      contents.push({ role: 'model', parts: [{ text: `{"candidates":[],"discover":null,"response":"${msg.text.replace(/"/g, '\\"')}"}` }] });
    }
  }
  contents.push({ role: 'user', parts: [{ text: userMessage }] });

  const body = {
    contents,
    generationConfig: {
      temperature: 0.2,   // low temperature = precise, factual movie titles
      topP: 0.9,
      maxOutputTokens: 600,
      responseMimeType: 'application/json',
    },
  };

  try {
    const res = await fetch(`${GEMINI_URL}?key=${key}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const errText = await res.text();
      console.error(`[CineBot/Gemini] HTTP ${res.status}:`, errText);
      return null;
    }
    const data = await res.json();
    const raw = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    if (!raw) {
      console.error('[CineBot/Gemini] Empty response from API. Full data:', JSON.stringify(data));
      return null;
    }
    const cleaned = raw.replace(/^```json?\n?/i, '').replace(/\n?```$/i, '').trim();
    const parsed = JSON.parse(cleaned);
    console.debug('[CineBot/Gemini] OK →', parsed);
    return parsed;
  } catch (err) {
    console.error('[CineBot/Gemini] Parse error:', err);
    return null;
  }
}

/**
 * Quick diagnostic — run geminiTest() in the browser console to check the API.
 * Example: await geminiTest("un film avec des voitures pour les enfants")
 */
async function geminiTest(msg = 'le film avec le lion et son père') {
  console.group('🎬 CineBot Gemini diagnostic');
  console.log('Model   :', GEMINI_MODEL);
  console.log('Key     :', CONFIG.GEMINI_API_KEY ? CONFIG.GEMINI_API_KEY.slice(0, 8) + '…' : '⚠️  MISSING');
  console.log('Message :', msg);
  const result = await geminiParseIntent(msg, [], null, [], []);
  if (result) {
    console.log('✅ Success →', result);
  } else {
    console.error('❌ Gemini returned null — check the errors above');
  }
  console.groupEnd();
  return result;
}

/**
 * Convert a Gemini "discover" fallback object to TMDB discoverByGenre options.
 */
function geminiDiscoverOpts(discover, topGenres = []) {
  const sortMap = {
    quality: 'vote_average.desc',
    popular: 'popularity.desc',
    recent: 'primary_release_date.desc',
    varied: 'vote_count.desc',
  };
  const decadeMap = {
    '1970s': [1970, 1979], '1980s': [1980, 1989], '1990s': [1990, 1999],
    '2000s': [2000, 2009], '2010s': [2010, 2019], '2020s': [2020, new Date().getFullYear()],
  };

  const opts = {
    sort_by: sortMap[discover?.sort] || 'popularity.desc',
    vote_count_gte: 150,
    page: 1 + Math.floor(Math.random() * 4),
  };

  let genres = discover?.genres?.length ? discover.genres : [];
  if (!genres.length && topGenres.length) {
    genres = topGenres.map(g => TMDB_GENRE_IDS[g]).filter(Boolean);
  }
  if (genres.length) opts.genres = genres;
  if (discover?.exclude_genres?.length) opts.without_genres = discover.exclude_genres;

  if (discover?.decade) {
    const [gte, lte] = decadeMap[discover.decade] || [];
    if (gte) opts.year_gte = gte;
    if (lte) opts.year_lte = lte;
  }

  return opts;
}
