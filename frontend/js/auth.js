const auth = {
  getToken() {
    return (typeof pstore !== 'undefined' ? pstore.get('user_token') : localStorage.getItem('user_token'));
  },

  isLoggedIn() {
    return !!this.getToken();
  },

  logout() {
    if (typeof pstore !== 'undefined') {
      pstore.remove('user_token');
      pstore.remove('ratings');
      pstore.remove('watchlist');
    } else {
      localStorage.removeItem('user_token');
      localStorage.removeItem('ratings');
      localStorage.removeItem('watchlist');
    }
    if (typeof profiles !== 'undefined') {
      localStorage.removeItem('cm_active_profile');
    }
    window.location.href = 'profiles.html';
  },

  requireAuth() {
    // Need both an active profile AND auth token to access the app
    const hasProfile = (typeof profiles !== 'undefined') ? !!profiles.activeId() : true;
    if (!hasProfile) {
      window.location.href = 'profiles.html';
      return;
    }
    if (!this.isLoggedIn()) {
      window.location.href = 'onboarding.html';
    }
  }
};
