// ─── Multi-profile management (Netflix-style) ───────────────────────────────
// Profiles are stored in localStorage under "cm_profiles" and the active id under
// "cm_active_profile". All per-user data (ratings, watchlist, streak, etc.) is
// namespaced via pstore.get/set, which prefixes the key with `p<id>:`.

const PROFILE_AVATARS = [
  // Color + initial-only SVG avatars (data URIs) — no external network needed
  { id: 'red', bg: '#e50914', fg: '#ffffff', label: 'Red' },
  { id: 'blue', bg: '#0073e6', fg: '#ffffff', label: 'Blue' },
  { id: 'green', bg: '#1db954', fg: '#ffffff', label: 'Green' },
  { id: 'yellow', bg: '#f5c518', fg: '#1a1a1a', label: 'Gold' },
  { id: 'purple', bg: '#7e57c2', fg: '#ffffff', label: 'Violet' },
  { id: 'orange', bg: '#ff6f3c', fg: '#ffffff', label: 'Orange' },
  { id: 'teal', bg: '#26a69a', fg: '#ffffff', label: 'Teal' },
  { id: 'pink', bg: '#ec407a', fg: '#ffffff', label: 'Pink' },
];

const profiles = {
  // ─── Storage primitives ─────────────────────────────────────────────────────
  _readAll() {
    try { return JSON.parse(localStorage.getItem('cm_profiles') || '[]'); }
    catch { return []; }
  },

  _writeAll(list) {
    localStorage.setItem('cm_profiles', JSON.stringify(list));
  },

  // ─── Public API ─────────────────────────────────────────────────────────────
  list() {
    return this._readAll();
  },

  active() {
    const id = localStorage.getItem('cm_active_profile');
    if (!id) return null;
    return this._readAll().find(p => p.id === id) || null;
  },

  activeId() {
    return localStorage.getItem('cm_active_profile') || null;
  },

  create(name, avatarId, extras = {}) {
    const list = this._readAll();
    const id = 'p' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    const profile = {
      id,
      name: (name || '').trim() || 'New Profile',
      avatar: avatarId || PROFILE_AVATARS[list.length % PROFILE_AVATARS.length].id,
      onboarded: false,
      created_at: new Date().toISOString(),
      age: extras.age || null,
      gender: extras.gender || null,
    };
    list.push(profile);
    this._writeAll(list);
    return profile;
  },

  // Ensure a fixed-id, pre-onboarded profile exists (e.g. the built-in "Lenny"
  // demo profile). Idempotent: created once, then left as-is so the user can't
  // lose it. Returns the profile.
  ensureBuiltIn(id, name, avatarId, extras = {}) {
    const list = this._readAll();
    let p = list.find(pr => pr.id === id);
    if (!p) {
      p = {
        id,
        name: name || 'Demo',
        avatar: avatarId || PROFILE_AVATARS[0].id,
        onboarded: true,
        builtIn: true,
        created_at: new Date().toISOString(),
        age: extras.age || null,
        gender: extras.gender || null,
      };
      list.unshift(p);
      this._writeAll(list);
    }
    return p;
  },

  delete(id) {
    const list = this._readAll().filter(p => p.id !== id);
    this._writeAll(list);
    // Wipe scoped data for the deleted profile
    Object.keys(localStorage).forEach(k => {
      if (k.startsWith(`p${id}:`) || k.startsWith(`${id}:`)) {
        localStorage.removeItem(k);
      }
    });
    if (this.activeId() === id) localStorage.removeItem('cm_active_profile');
  },

  select(id) {
    localStorage.setItem('cm_active_profile', id);
  },

  markOnboarded(id) {
    const list = this._readAll();
    const p = list.find(pr => pr.id === id);
    if (p) { p.onboarded = true; this._writeAll(list); }
  },

  update(id, fields) {
    const list = this._readAll();
    const p = list.find(pr => pr.id === id);
    if (!p) return;
    if (fields.name != null) p.name = fields.name.trim() || p.name;
    if (fields.avatar != null) p.avatar = fields.avatar;
    if (fields.age != null) p.age = fields.age;
    if (fields.gender != null) p.gender = fields.gender;
    this._writeAll(list);
  },

  scopedKey(key) {
    const id = this.activeId();
    return id ? `${id}:${key}` : key;
  },

  avatarSvg(avatarId, initial) {
    const av = PROFILE_AVATARS.find(a => a.id === avatarId) || PROFILE_AVATARS[0];
    const ch = (initial || '').trim().charAt(0).toUpperCase();
    const inner = ch
      ? `<text x='50' y='50' text-anchor='middle' dominant-baseline='central'
               font-family='Bebas Neue, Impact, sans-serif' font-size='56'
               fill='${av.fg}'>${ch}</text>`
      : `<circle cx='50' cy='32' r='15' fill='${av.fg}' fill-opacity='0.88'/>
         <ellipse cx='50' cy='74' rx='28' ry='22' fill='${av.fg}' fill-opacity='0.88'/>`;
    const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100' overflow='hidden'>
      <rect width='100' height='100' rx='12' fill='${av.bg}'/>
      ${inner}
    </svg>`;
    return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
  },

  // ─── One-time migration from pre-profile localStorage ──────────────────────
  // If the user had ratings/watchlist saved before profiles existed, fold them
  // into a "default" profile so they don't lose their data.
  migrateLegacyIfNeeded() {
    if (localStorage.getItem('cm_migrated_v1') === '1') return;
    const list = this._readAll();
    const hasLegacyRatings = !!localStorage.getItem('ratings');
    const hasLegacyToken = !!localStorage.getItem('user_token');
    if (!list.length && (hasLegacyRatings || hasLegacyToken)) {
      const p = this.create('Me', PROFILE_AVATARS[0].id);
      p.onboarded = hasLegacyRatings;
      this._writeAll([p]);
      this.select(p.id);
      // Move legacy keys into scoped keys
      ['ratings', 'watchlist', 'user_token', 'member_since', 'streak_days', 'last_active'].forEach(k => {
        const v = localStorage.getItem(k);
        if (v != null) {
          localStorage.setItem(`${p.id}:${k}`, v);
          localStorage.removeItem(k);
        }
      });
    }
    localStorage.setItem('cm_migrated_v1', '1');
  },
};

// ─── pstore — scoped wrapper around localStorage for the active profile ─────
const pstore = {
  get(key) {
    const id = profiles.activeId();
    return localStorage.getItem(id ? `${id}:${key}` : key);
  },
  set(key, val) {
    const id = profiles.activeId();
    localStorage.setItem(id ? `${id}:${key}` : key, val);
  },
  remove(key) {
    const id = profiles.activeId();
    localStorage.removeItem(id ? `${id}:${key}` : key);
  },
};

// Run migration as soon as this script loads
profiles.migrateLegacyIfNeeded();
