// ─── CineBot — AI-powered recommender (Gemini Flash + keyword fallback) ───────
// If CONFIG.GEMINI_API_KEY is set, all messages go through Gemini 2.0 Flash Lite
// which understands any language, context, actors, moods, exclusions, etc.
// Without a key, it degrades gracefully to the local keyword matcher.

// ─── Keyword fallback lexicon ─────────────────────────────────────────────────
const CHATBOT_LEXICON = [
  // English + French triggers → TMDB genre IDs
  { match: /(scary|horror|effrayant|peur|frisson)/i, genres: [27] },
  { match: /(funny|comedy|comédie|rire|drôle|hilarant)/i, genres: [35] },
  { match: /(romance|love|amour|romantic|romantique)/i, genres: [10749] },
  { match: /(family|kids|enfant|famille)/i, genres: [10751] },
  { match: /(war|guerre)/i, genres: [10752] },
  { match: /(action|fight|combat|bagarre)/i, genres: [28] },
  { match: /(sci[-\s]?fi|science fiction|space|futur|robot|alien|cyber)/i, genres: [878] },
  { match: /(thriller|suspense|haletant|tension)/i, genres: [53] },
  { match: /(animation|cartoon|animé|animation|anime)/i, genres: [16] },
  { match: /(crime|mafia|gangster|policier|heist)/i, genres: [80] },
  { match: /(adventure|aventure|épopée|epic|quest)/i, genres: [12] },
  { match: /(fantasy|magic|magique|fantastique|wizard)/i, genres: [14] },
  { match: /(mystery|mystère|enquête|whodunit)/i, genres: [9648] },
  { match: /(history|historique|historical|ancient|époque)/i, genres: [36] },
  { match: /(drama|drame|émotion|emotional|character)/i, genres: [18] },
  { match: /(documentary|documentaire|doc)/i, genres: [99] },
  { match: /(music|musical|musique|musicien|concert)/i, genres: [10402] },
  { match: /(western|cowboy|far\s?west)/i, genres: [37] },
  { match: /(dark|sombre|noir|grim|gritty)/i, genres: [80, 53] },
  { match: /(feel[-\s]?good|uplifting|heart\s?warming|réconfort|wholesome)/i, genres: [35, 10751] },
  { match: /(mindbend|brain|cerveau|psycho|psychological|cerebral)/i, genres: [9648, 53, 878] },
  { match: /(emotional|tearjerker|sad|triste|larmes)/i, genres: [18, 10749] },
  { match: /(epic|grand|sweeping)/i, genres: [12, 36] },
  { match: /(quirky|indie|independent|offbeat)/i, genres: [18, 35] },
  { match: /(zombie|undead|cannibal)/i, genres: [27, 28] },
  { match: /(superhero|marvel|dc|comic)/i, genres: [28, 878, 12] },
  { match: /(post[-\s]?apocalyptic|dystop|apocalypse)/i, genres: [878, 28] },
  { match: /(coming[-\s]?of[-\s]?age|teen|adolescent)/i, genres: [18] },
  { match: /(spy|espionnage|agent)/i, genres: [53, 28] },
];

const CHATBOT_DECADES = [
  { match: /(70s|seventies|années 70)/i, year_gte: 1970, year_lte: 1979 },
  { match: /(80s|eighties|années 80)/i, year_gte: 1980, year_lte: 1989 },
  { match: /(90s|nineties|années 90)/i, year_gte: 1990, year_lte: 1999 },
  { match: /(2000s|noughties|années 2000)/i, year_gte: 2000, year_lte: 2009 },
  { match: /(2010s|années 2010)/i, year_gte: 2010, year_lte: 2019 },
  { match: /(2020s|recent|récent|nouveau|new)/i, year_gte: 2020, year_lte: new Date().getFullYear() },
  { match: /(classic|classique|vieux|old)/i, year_lte: 1979 },
];

function _keywordParse(text) {
  const t = text.trim();
  const genreSet = new Set();
  CHATBOT_LEXICON.forEach(rule => {
    if (rule.match.test(t)) rule.genres.forEach(g => genreSet.add(g));
  });
  let yearRange = {};
  CHATBOT_DECADES.forEach(d => {
    if (d.match.test(t)) {
      if (d.year_gte != null) yearRange.year_gte = d.year_gte;
      if (d.year_lte != null) yearRange.year_lte = d.year_lte;
    }
  });
  return { genres: [...genreSet], ...yearRange };
}

// ─── User taste profile helpers ───────────────────────────────────────────────
function _getTopGenres(limit = 3) {
  try {
    const ratings = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('ratings') : null) || '{}');
    if (!Object.keys(ratings).length) return [];
    const genreScore = {};
    // Scan allMoviesPool (available from home.js) for genre weights
    const pool = (typeof allMoviesPool !== 'undefined') ? allMoviesPool : [];
    for (const [id, rating] of Object.entries(ratings)) {
      const movie = pool.find(m => String(m.movie_id) === String(id));
      if (!movie) continue;
      for (const g of (movie.genres || [])) {
        genreScore[g] = (genreScore[g] || 0) + rating;
      }
    }
    return Object.entries(genreScore)
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([name]) => name);
  } catch { return []; }
}

function _getRecentTitles(limit = 3) {
  try {
    const ratings = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('ratings') : null) || '{}');
    const ids = Object.keys(ratings).map(Number).slice(-limit * 2).reverse();
    const pool = (typeof allMoviesPool !== 'undefined') ? allMoviesPool : [];
    return ids
      .map(id => pool.find(m => m.movie_id === id)?.title)
      .filter(Boolean)
      .slice(0, limit);
  } catch { return []; }
}

// ─── Seen-film memory (avoid repeats) ─────────────────────────────────────────
function _readSeen() {
  try {
    const raw = (typeof pstore !== 'undefined') ? pstore.get('chatbot_seen') : null;
    const arr = JSON.parse(raw || '[]');
    return new Set(Array.isArray(arr) ? arr : []);
  } catch { return new Set(); }
}

function _writeSeen(setOrArr) {
  const arr = Array.isArray(setOrArr) ? setOrArr : [...setOrArr];
  const clipped = arr.slice(-120);
  if (typeof pstore !== 'undefined') pstore.set('chatbot_seen', JSON.stringify(clipped));
}

function _shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ─── Main recommend function ───────────────────────────────────────────────────
async function chatbotRecommend(text, history = []) {
  const seen = _readSeen();
  const profile = (typeof profiles !== 'undefined') ? profiles.active() : null;
  const topGenres = _getTopGenres(3);
  const recentTitles = _getRecentTitles(3);

  let geminiIntent = null;
  let geminiResponse = null;

  // ── 1. Try Gemini AI ──────────────────────────────────────────────────────
  if (typeof geminiParseIntent === 'function' && CONFIG.GEMINI_API_KEY) {
    const raw = await geminiParseIntent(text, history, profile, topGenres, recentTitles);
    if (raw) {
      geminiIntent = raw;
      geminiResponse = raw.response || null;
    }
  }

  // ── 2. Fetch from TMDB ────────────────────────────────────────────────────
  let results = [];
  let mode = 'discover'; // 'search' | 'discover' | 'keyword'

  try {
    if (geminiIntent) {
      const hasCandidates = Array.isArray(geminiIntent.candidates) && geminiIntent.candidates.length > 0;

      if (hasCandidates) {
        // ── A. Semantic search: fetch each candidate title from TMDB in parallel ──
        mode = 'search';
        const searches = await Promise.all(
          geminiIntent.candidates.map(title =>
            TMDB.searchMovies(title).then(r => r?.[0] || null).catch(() => null)
          )
        );
        // Deduplicate by TMDB id, preserve Gemini's ranking order
        const seen_ids = new Set();
        results = searches.filter(m => {
          if (!m || seen_ids.has(m.id)) return false;
          seen_ids.add(m.id);
          return true;
        });
      } else {
        // ── B. Discovery mode: genre/mood-based discover ───────────────────────
        mode = 'discover';
        const opts = geminiDiscoverOpts(geminiIntent.discover, topGenres);
        results = await TMDB.discoverByGenre(opts);
      }
    } else {
      // ── C. Keyword fallback (no Gemini) ───────────────────────────────────
      mode = 'keyword';
      const fb = _keywordParse(text);
      const want_best = /(top|best|meilleur|cult)/i.test(text);
      const sort = want_best ? 'vote_average.desc' : 'popularity.desc';
      const opts = {
        sort_by: sort,
        vote_count_gte: want_best ? 5000 : 200,
        page: 1 + Math.floor(Math.random() * 5),
      };
      if (fb.genres?.length) opts.genres = fb.genres;
      if (fb.year_gte != null) opts.year_gte = fb.year_gte;
      if (fb.year_lte != null) opts.year_lte = fb.year_lte;
      results = await TMDB.discoverByGenre(opts);
    }
  } catch (e) {
    console.warn('[CineBot] TMDB fetch failed:', e);
    return { mode, picks: [], geminiResponse };
  }

  // ── 3. Parse, filter, pick ────────────────────────────────────────────────
  const parsed = results
    .map(r => TMDB.parseLite(r))
    .filter(m => m.poster_url && !m.poster_url.includes('placeholder'));

  let picks;
  if (mode === 'search') {
    // Preserve Gemini's ranking — no shuffle, no seen filter (user searched explicitly)
    picks = parsed.slice(0, 8);
  } else {
    // Discovery: prioritise unseen, shuffle for variety
    const fresh = parsed.filter(m => !seen.has(m.movie_id));
    const stale = parsed.filter(m => seen.has(m.movie_id));
    const pool = fresh.length >= 4 ? fresh : [...fresh, ..._shuffle(stale)];
    picks = _shuffle(pool.slice(0, 18)).slice(0, 8);
  }

  if (picks.length) {
    _writeSeen([...seen, ...picks.map(p => p.movie_id)]);
  }

  return { mode, picks, geminiResponse };
}

// ─── Markdown renderer (bold, italic, line breaks) ────────────────────────────
function _parseMarkdown(html) {
  return html
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
}

// ─── UI ────────────────────────────────────────────────────────────────────────
const chatbot = {
  _open: false,
  _history: [],

  init() {
    document.getElementById('chatbot-fab')?.addEventListener('click', () => this.toggle());
    document.getElementById('chatbot-close')?.addEventListener('click', () => this.close());
    document.getElementById('chatbot-form')?.addEventListener('submit', e => {
      e.preventDefault();
      this._submit();
    });

    const plusBtn = document.getElementById('chatbot-plus-btn');
    const moodTray = document.getElementById('chatbot-mood-tray');
    if (plusBtn && moodTray) {
      plusBtn.addEventListener('click', () => {
        const isHidden = moodTray.style.display === 'none';
        moodTray.style.display = isHidden ? 'flex' : 'none';
        plusBtn.classList.toggle('active', isHidden);
      });

      document.querySelectorAll('.mood-chip').forEach(btn => {
        btn.addEventListener('click', () => {
          const mood = btn.getAttribute('data-mood');
          const input = document.getElementById('chatbot-input');
          if (input && mood) {
            input.value = mood;
            moodTray.style.display = 'none';
            plusBtn.classList.remove('active');
            this._submit();
          }
        });
      });
    }

    // Show AI badge in header if Gemini key is set
    if (CONFIG.GEMINI_API_KEY) {
      const sub = document.querySelector('.chatbot-sub');
      if (sub) sub.innerHTML = 'Powered by <span style="color:#4ade80;font-weight:700">Gemini AI</span> · Ask me anything';
    }

    this._initResize();

    // Restore conversation from per-profile storage
    try {
      const stored = JSON.parse((typeof pstore !== 'undefined' ? pstore.get('chatbot_history') : null) || '[]');
      this._history = Array.isArray(stored) ? stored : [];
    } catch { this._history = []; }
    this._renderHistory();

    if (!this._history.length) {
      const isAI = !!CONFIG.GEMINI_API_KEY;
      this._renderAssistantMessage({
        text: isAI
          ? "Hey! Tell me what you're in the mood for — a genre, vibe, actor, decade… I understand anything!"
          : "Hey — tell me what you're in the mood for. Try \"a dark thriller from the 90s\" or \"feel-good animated\".",
        picks: [],
      });
    }
  },

  _initResize() {
    const panel = document.getElementById('chatbot-panel');
    if (!panel) return;

    const savedW = localStorage.getItem('chatbot_width');
    const savedH = localStorage.getItem('chatbot_height');
    if (savedW) panel.style.width = savedW + 'px';
    if (savedH) panel.style.height = savedH + 'px';

    const attach = (id, resizeH, resizeW) => {
      const handle = document.getElementById(id);
      if (!handle) return;
      handle.addEventListener('mousedown', e => {
        if (window.innerWidth <= 600) return;
        e.preventDefault();
        const startX = e.clientX, startY = e.clientY;
        const startW = panel.offsetWidth, startH = panel.offsetHeight;
        panel.classList.add('resizing');
        const onDrag = e => {
          if (resizeH) panel.style.height = Math.min(window.innerHeight - 120, Math.max(380, startH + (startY - e.clientY))) + 'px';
          if (resizeW) panel.style.width = Math.min(700, Math.max(280, startW + (startX - e.clientX))) + 'px';
        };
        const onUp = () => {
          panel.classList.remove('resizing');
          localStorage.setItem('chatbot_width', panel.offsetWidth);
          localStorage.setItem('chatbot_height', panel.offsetHeight);
          document.removeEventListener('mousemove', onDrag);
          document.removeEventListener('mouseup', onUp);
        };
        document.addEventListener('mousemove', onDrag);
        document.addEventListener('mouseup', onUp);
      });
    };

    attach('chatbot-resize-n', true, false);
    attach('chatbot-resize-w', false, true);
  },

  toggle() { if (this._open) this.close(); else this.open(); },

  open() {
    document.getElementById('chatbot-panel')?.classList.add('open');
    document.getElementById('chatbot-fab')?.classList.add('hidden');
    this._open = true;
    setTimeout(() => document.getElementById('chatbot-input')?.focus(), 250);
    setTimeout(() => {
      this._outsideHandler = e => {
        const panel = document.getElementById('chatbot-panel');
        const fab = document.getElementById('chatbot-fab');
        if (!panel?.contains(e.target) && !fab?.contains(e.target)) this.close();
      };
      document.addEventListener('click', this._outsideHandler);
    }, 0);
  },

  close() {
    document.getElementById('chatbot-panel')?.classList.remove('open');
    document.getElementById('chatbot-fab')?.classList.remove('hidden');
    this._open = false;
    if (this._outsideHandler) {
      document.removeEventListener('click', this._outsideHandler);
      this._outsideHandler = null;
    }
  },

  async _submit() {
    const input = document.getElementById('chatbot-input');
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;
    input.value = '';

    this._renderUserMessage(text);
    this._history.push({ role: 'user', text });
    this._persistHistory();

    const typing = this._renderTyping();

    // Pass recent history for Gemini context (last 6 messages)
    const { mode, picks, geminiResponse } = await chatbotRecommend(text, this._history.slice(-6));
    typing.remove();

    // ── Build reply text ──────────────────────────────────────────────────
    let reply;
    if (geminiResponse) {
      reply = geminiResponse;
    } else if (!picks.length) {
      reply = "Hmm, nothing found. Try describing a genre, actor, or mood…";
    } else if (mode === 'search') {
      reply = "Here's what I found for your search:";
    } else {
      reply = "Here are some films you might enjoy:";
    }

    this._renderAssistantMessage({ text: reply, picks });
    this._history.push({
      role: 'assistant', text: reply,
      picks: picks.map(p => ({
        movie_id: p.movie_id, tmdb_id: p.tmdb_id, title: p.title, year: p.year,
        poster_url: p.poster_url, genres: p.genres,
      })),
    });
    this._persistHistory();
  },

  _persistHistory() {
    if (this._history.length > 60) this._history = this._history.slice(-60);
    if (typeof pstore !== 'undefined') pstore.set('chatbot_history', JSON.stringify(this._history));
  },

  _renderHistory() {
    const wrap = document.getElementById('chatbot-messages');
    if (!wrap) return;
    wrap.innerHTML = '';
    this._history.forEach(msg => {
      if (msg.role === 'user') this._renderUserMessage(msg.text);
      else this._renderAssistantMessage({ text: msg.text, picks: msg.picks || [] });
    });
  },

  _renderUserMessage(text) {
    const wrap = document.getElementById('chatbot-messages');
    if (!wrap) return;
    const div = document.createElement('div');
    div.className = 'chatbot-msg chatbot-msg-user';
    div.textContent = text;
    wrap.appendChild(div);
    wrap.scrollTop = wrap.scrollHeight;
  },

  _renderAssistantMessage({ text, picks }) {
    const wrap = document.getElementById('chatbot-messages');
    if (!wrap) return;
    const div = document.createElement('div');
    div.className = 'chatbot-msg chatbot-msg-bot';

    const reelHeader = picks.length
      ? `<div class="chatbot-reel-header">
           🎬 <strong>${picks.length} movie${picks.length > 1 ? 's' : ''}</strong>&nbsp;· slide left/right to explore
         </div>`
      : '';

    const reel = picks.length
      ? `<div class="chatbot-reel">
           ${picks.map(p => `
             <button class="chatbot-pick" data-movie-id="${p.movie_id}" data-tmdb-id="${p.tmdb_id}" aria-label="${escapeHtml(p.title)}">
               <div class="chatbot-pick-poster">
                 <img src="${p.poster_url || 'img/placeholder_poster.svg'}" alt="${escapeHtml(p.title)}" loading="lazy">
                 <div class="chatbot-pick-overlay">
                   <span class="chatbot-pick-overlay-title">${escapeHtml(p.title)}</span>
                   ${p.year ? `<span class="chatbot-pick-overlay-year">${p.year}</span>` : ''}
                 </div>
                 ${p.vote_average ? `<span class="chatbot-pick-badge">★ ${Number(p.vote_average).toFixed(1)}</span>` : ''}
               </div>
             </button>`).join('')}
         </div>`
      : '';

    div.innerHTML = `<div class="chatbot-msg-text">${_parseMarkdown(escapeHtml(text))}</div>${reelHeader}${reel}`;
    wrap.appendChild(div);

    div.querySelectorAll('.chatbot-pick').forEach(btn => {
      btn.addEventListener('click', () => {
        const movie = (picks || []).find(p => p.movie_id === parseInt(btn.dataset.movieId));
        if (movie && typeof openMovieModal === 'function') openMovieModal(movie);
      });
    });

    wrap.scrollTop = wrap.scrollHeight;
  },

  _renderTyping() {
    const wrap = document.getElementById('chatbot-messages');
    const div = document.createElement('div');
    div.className = 'chatbot-msg chatbot-msg-bot chatbot-typing';
    div.innerHTML = `<span></span><span></span><span></span>`;
    wrap.appendChild(div);
    wrap.scrollTop = wrap.scrollHeight;
    return div;
  },
};

document.addEventListener('DOMContentLoaded', () => chatbot.init());
