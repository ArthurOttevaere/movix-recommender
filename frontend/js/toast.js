const toast = {
  _container: null,

  _getContainer() {
    if (!this._container) this._container = document.getElementById('toast-container');
    return this._container;
  },

  show(message, type = 'default', onClick = null) {
    const container = this._getContainer();
    if (!container) return;

    const el = document.createElement('div');
    el.className = `toast${type !== 'default' ? ` toast-${type}` : ''}`;
    if (onClick) el.classList.add('toast-clickable');

    const iconMap = { success: '✓', warning: '⚠', info: 'ℹ', default: '·' };
    el.innerHTML = `
      <span class="toast-icon">${iconMap[type] || iconMap.default}</span>
      <span class="toast-msg">${message}</span>
      ${onClick ? '<span class="toast-arrow">→</span>' : ''}
    `;

    if (onClick) {
      el.style.cursor = 'pointer';
      el.addEventListener('click', () => {
        onClick();
        el.classList.remove('visible');
        el.addEventListener('transitionend', () => el.remove(), { once: true });
      });
    }

    container.appendChild(el);
    // Force reflow before adding visible class
    el.getBoundingClientRect();
    el.classList.add('visible');

    setTimeout(() => {
      el.classList.remove('visible');
      el.addEventListener('transitionend', () => el.remove(), { once: true });
    }, 3000);
  }
};
