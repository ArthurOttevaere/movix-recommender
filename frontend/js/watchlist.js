const watchlist = {
  getList() {
    return JSON.parse((typeof pstore !== 'undefined' ? pstore.get('watchlist') : localStorage.getItem('watchlist')) || '[]');
  },

  isInWatchlist(movieId) {
    return this.getList().includes(movieId);
  },

  async toggle(movieId, triggerEl) {
    const token = auth.getToken();
    const wasInList = this.isInWatchlist(movieId);
    const action = wasInList ? 'remove' : 'add';

    await api.toggleWatchlist(token, movieId, action);

    const nowInList = !wasInList;

    if (typeof toast !== 'undefined') {
      toast.show(
        nowInList ? 'Added to My List' : 'Removed from My List',
        nowInList ? 'success' : 'default',
        nowInList ? () => { if (typeof closeMovieModal === 'function') closeMovieModal(); if (typeof navigation !== 'undefined') navigation.showView('watchlist'); } : null
      );
    }

    if (triggerEl) this._updateBtn(triggerEl, nowInList);

    document.querySelectorAll(`.btn-watchlist-card[data-movie-id="${movieId}"]`).forEach(btn => {
      this._updateBtn(btn, nowInList);
    });

    const modalBtn = document.getElementById('modal-watchlist-btn');
    if (modalBtn && parseInt(modalBtn.dataset.movieId) === movieId) {
      modalBtn.textContent = nowInList ? '✓ My List' : '+ My List';
      modalBtn.classList.toggle('in-list', nowInList);
    }

    if (typeof eventBus !== 'undefined') {
      eventBus.emit('watchlist:updated', { movieId, action });
    }
  },

  _updateBtn(btn, inList) {
    btn.classList.toggle('in-list', inList);
    if (inList) {
      btn.innerHTML = `<i data-lucide="check" class="wl-icon-default"></i><i data-lucide="minus" class="wl-icon-hover"></i>`;
    } else {
      btn.innerHTML = `<i data-lucide="plus" class="wl-icon-default"></i>`;
    }
    if (typeof lucide !== 'undefined') lucide.createIcons({ root: btn });
  }
};
