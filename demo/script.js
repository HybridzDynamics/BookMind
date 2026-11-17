

'use strict';

const BASE_URL = window.location.protocol === 'file:'
  ? 'http://127.0.0.1:8000'
  : window.location.origin;

const EXAMPLES = {
  popularity: [],
  content: [
    { label: 'Hunger Games · 1', value: '1' },
    { label: 'Harry Potter · 2', value: '2' },
    { label: 'Twilight · 3', value: '3' }
  ],
  collaborative: [
    { label: 'User 1', value: '1' },
    { label: 'User 42', value: '42' },
    { label: 'Unknown user', value: 'new-reader' }
  ],
  hybrid: [
    { label: 'User 1', value: '1' },
    { label: 'User 42', value: '42' },
    { label: 'Unknown user', value: 'new-reader' }
  ]
};

const MODEL_META = {
  popularity: {
    label: 'Popularity',
    icon: '📊',
    inputMode: 'none',
    inputLabel: 'No input required',
    hint: 'The Popularity model recommends the same top-rated books to all users.',
    explanation: 'Recommends globally popular books ranked by Bayesian weighted score. Useful as a cold-start baseline when there is no user history.'
  },
  content: {
    label: 'Content-Based',
    icon: '🔍',
    inputMode: 'book',
    inputLabel: 'Book ID',
    hint: 'Enter a numeric book ID (e.g. 1, 2, 50). The model finds books similar in themes, genres, and metadata.',
    explanation: 'Uses TF-IDF features on book metadata and cosine NearestNeighbors to find books most similar in content to the selected book.'
  },
  collaborative: {
    label: 'Collaborative Filtering',
    icon: '👥',
    inputMode: 'user',
    inputLabel: 'User ID',
    hint: 'Enter a numeric user ID (e.g. 1, 42, 100). If the user is unknown, a popularity fallback is used.',
    explanation: 'Uses TruncatedSVD on the user-item matrix. Finds books liked by users with similar reading patterns. Falls back to popularity for cold-start users.'
  },
  hybrid: {
    label: 'Hybrid Ensemble',
    icon: '⚗️',
    inputMode: 'user',
    inputLabel: 'User ID',
    hint: 'Enter a numeric user ID (e.g. 1, 42, 100). Results show per-component scores: Collaborative, Content, Readability, and Popularity.',
    explanation: 'Combines four signals: 35% Collaborative + 30% Content + 15% Readability + 20% Popularity. Shows individual sub-scores for transparency.'
  }
};

function initTabs() {
  const buttons = document.querySelectorAll('.tab-btn');
  const panels = document.querySelectorAll('.tab-panel');

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      buttons.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('panel-' + target).classList.add('active');

      if (target === 'evaluation') {
        loadEvaluationTable();
      }
      if (target === 'self-account') {
        refreshAccountSelect();
      }
    });
  });

  const goToDemoBtn = document.getElementById('goToDemoBtn');
  if (goToDemoBtn) {
    goToDemoBtn.addEventListener('click', () => {
      document.getElementById('tab-demo').click();
    });
  }
}

async function checkApiHealth() {
  const pill = document.getElementById('apiStatus');
  try {
    const res = await fetch(`${BASE_URL}/health`, { method: 'GET', signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error('not ok');
    const data = await res.json();
    if (data.status === 'healthy') {
      pill.textContent = '✓ API Online';
      pill.classList.add('healthy');
    } else {
      pill.textContent = 'API Status Unknown';
    }
  } catch {
    pill.textContent = '✗ API Offline';
    pill.classList.add('error');
  }
}

async function loadStats() {
  try {
    const res = await fetch(`${BASE_URL}/stats`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return;
    const data = await res.json();
    if (data.books) setStatEl('statBooks', formatNumber(data.books));
    if (data.users) setStatEl('statUsers', formatNumber(data.users));
    if (data.ratings) setStatEl('statRatings', formatNumber(data.ratings));
  } catch {

    setStatEl('statBooks', '10,000');
    setStatEl('statUsers', '53,424');
    setStatEl('statRatings', '5.97M');
  }
}

function setStatEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function formatNumber(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (n >= 1_000) return n.toLocaleString();
  return String(n);
}

function initDemoControls() {
  const modelSelect = document.getElementById('modelSelect');
  const generateBtn = document.getElementById('generateBtn');

  modelSelect.addEventListener('change', updateModelUI);
  generateBtn.addEventListener('click', fetchRecommendations);
  updateModelUI();

  fetchRecommendations();
}

function updateModelUI() {
  const key = document.getElementById('modelSelect').value;
  const meta = MODEL_META[key];

  document.getElementById('explanationIcon').textContent = meta.icon;
  document.getElementById('explanationTitle').textContent = meta.label;
  document.getElementById('explanationText').textContent = meta.explanation;

  const inputGroup = document.getElementById('inputGroup');
  const inputLabel = document.getElementById('inputLabel');
  const inputValue = document.getElementById('inputValue');
  const inputHint = document.getElementById('inputHint');

  if (meta.inputMode === 'none') {
    inputGroup.classList.add('hidden');
  } else {
    inputGroup.classList.remove('hidden');
    inputLabel.textContent = meta.inputLabel;
    inputHint.textContent = meta.hint;
    inputValue.placeholder = meta.inputMode === 'book' ? 'e.g. 1' : 'e.g. 1';
  }

  document.getElementById('resultsTitle').textContent = meta.label + ' Recommendations';
  renderExamples(key);
}

function renderExamples(modelKey) {
  const row = document.getElementById('exampleRow');
  if (!row) return;
  const examples = EXAMPLES[modelKey] || [];
  row.innerHTML = '';
  if (examples.length === 0) {
    row.classList.add('hidden');
    return;
  }
  row.classList.remove('hidden');
  examples.forEach(example => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'example-chip';
    button.textContent = example.label;
    button.addEventListener('click', () => {
      document.getElementById('inputValue').value = example.value;
      fetchRecommendations();
    });
    row.appendChild(button);
  });
}

async function fetchRecommendations() {
  const model = document.getElementById('modelSelect').value;
  const limit = Number(document.getElementById('limitSelect').value);
  const inputText = document.getElementById('inputValue').value.trim() || '1';

  let url;
  switch (model) {
    case 'popularity':
      url = `${BASE_URL}/recommendations/popular?limit=${limit}`;
      break;
    case 'content':
      url = `${BASE_URL}/recommendations/content/${encodeURIComponent(inputText)}?limit=${limit}`;
      break;
    case 'collaborative':
      url = `${BASE_URL}/recommendations/collaborative/${encodeURIComponent(inputText)}?limit=${limit}`;
      break;
    case 'hybrid':
    default:
      url = `${BASE_URL}/recommendations/hybrid/${encodeURIComponent(inputText)}?limit=${limit}`;
  }

  showLoadingState(true);
  clearResults();

  try {
    const res = await fetch(url, { method: 'GET', signal: AbortSignal.timeout(12000) });
    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || `HTTP ${res.status}: Recommendation request failed.`);
      return;
    }

    const items = Array.isArray(data) ? data : [data];
    renderResults(items, model);
  } catch (err) {
    if (err.name === 'TimeoutError') {
      showError('Request timed out. The model may still be loading. Try again in a moment.');
    } else {
      showError('Cannot connect to BookMind API. Make sure it is running on port 8000.');
    }
  } finally {
    showLoadingState(false);
  }
}

function renderResults(items, model, grid = null) {
  const targetGrid = grid || document.getElementById('resultsGrid');
  const countEl = grid ? null : document.getElementById('resultsCount');
  targetGrid.innerHTML = '';

  if (!items || items.length === 0) {
    if (!grid) document.getElementById('resultsEmpty').classList.remove('hidden');
    if (countEl) countEl.textContent = '';
    return;
  }

  if (!grid) document.getElementById('resultsEmpty').classList.add('hidden');
  if (countEl) countEl.textContent = `${items.length} result${items.length !== 1 ? 's' : ''}`;

  items.forEach((item, idx) => {
    const card = document.createElement('article');
    card.className = 'book-card';
    card.innerHTML = buildCardHTML(item, idx + 1, model);
    targetGrid.appendChild(card);
  });
}

function buildCardHTML(item, rank, model) {
  const title = escapeHtml(item.title || 'Unknown Title');
  const author = escapeHtml(item.author || 'Unknown Author');
  const bookId = escapeHtml(String(item.bookId || ''));
  const reason = escapeHtml(item.reason || '');
  const cover = safeCoverUrl(item.image_url);
  const coverHTML = cover
    ? `<div class="book-cover-wrap"><img class="book-cover" src="${escapeHtml(cover)}" alt="" onerror="this.parentElement.style.display='none'"></div>`
    : '';

  let coldStartBadge = '';
  if (item.reason && item.reason.toLowerCase().includes('cold-start')) {
    coldStartBadge = `<div class="cold-start-badge">📋 Cold-start fallback</div>`;
  }

  let scoresHTML = buildScoresHTML(item, model);

  return `
    <div class="book-rank">#${rank}</div>
    ${coverHTML}
    ${coldStartBadge}
    <div class="book-title">${title}</div>
    <div class="book-author">by ${author}</div>
    <div class="book-id-row">ID: ${bookId}</div>
    ${scoresHTML}
    ${reason ? `<div class="book-reason">${reason}</div>` : ''}
  `;
}

function buildScoresHTML(item, model) {
  const scores = [];

  if (item.hybrid_score !== undefined && item.hybrid_score !== null) {
    scores.push(`<span class="score-tag score-hybrid">⚗️ Hybrid: ${item.hybrid_score.toFixed(4)}</span>`);
  }
  if (item.collaborative_score !== undefined && item.collaborative_score !== null) {
    scores.push(`<span class="score-tag score-collab">👥 Collab: ${item.collaborative_score.toFixed(4)}</span>`);
  }
  if (item.content_score !== undefined && item.content_score !== null) {
    scores.push(`<span class="score-tag score-content">🔍 Content: ${item.content_score.toFixed(4)}</span>`);
  }
  if (item.readability_score !== undefined && item.readability_score !== null) {
    scores.push(`<span class="score-tag score-readability">📖 Readability: ${item.readability_score.toFixed(4)}</span>`);
  }
  if (item.popularity_score !== undefined && item.popularity_score !== null) {
    scores.push(`<span class="score-tag score-pop">📊 Popularity: ${item.popularity_score.toFixed(4)}</span>`);
  }
  if (item.similarity_score !== undefined && item.similarity_score !== null && model === 'content') {
    scores.push(`<span class="score-tag score-similarity">∼ Similarity: ${item.similarity_score.toFixed(4)}</span>`);
  }
  if (item.predicted_score !== undefined && item.predicted_score !== null && model === 'collaborative') {
    scores.push(`<span class="score-tag score-predicted">↑ Score: ${item.predicted_score.toFixed(4)}</span>`);
  }
  if (item.rating !== undefined && item.rating !== null && model === 'popularity') {
    scores.push(`<span class="score-tag score-rating">⭐ Rating: ${Number(item.rating).toFixed(2)}</span>`);
  }

  return scores.length > 0 ? `<div class="score-grid">${scores.join('')}</div>` : '';
}

let evalLoaded = false;

async function loadEvaluationTable() {
  if (evalLoaded) return;

  const tbody = document.getElementById('evalTableBody');
  if (!tbody) return;

  try {
    const res = await fetch(`${BASE_URL}/stats`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {

    }
  } catch {  }

  try {
    const evalRes = await fetch(`${BASE_URL}/evaluation`, { signal: AbortSignal.timeout(5000) });
    if (evalRes.ok) {
      const rows = await evalRes.json();
      const metricMap = { "precision@10": "precision", "recall@10": "recall", "ndcg@10": "ndcg", "hit_rate@10": "hitRate", "coverage": "coverage", "users_evaluated": "users" };
      const evalData = rows.map(r => ({
        model: r.model || r.Model || "",
        precision: parseFloat(r["precision@10"] ?? r.precision) || 0,
        recall: parseFloat(r["recall@10"] ?? r.recall) || 0,
        ndcg: parseFloat(r["ndcg@10"] ?? r.ndcg) || 0,
        hitRate: parseFloat(r["hit_rate@10"] ?? r.hitRate) || 0,
        coverage: parseFloat(r.coverage) || 0,
        users: parseInt(r.users_evaluated ?? r.users) || 0
      }));
      tbody.innerHTML = evalData.map(row => `
    <tr>
      <td><strong>${escapeHtml(row.model)}</strong></td>
      <td>${row.precision.toFixed(4)}</td>
      <td>${row.recall.toFixed(4)}</td>
      <td>${row.ndcg.toFixed(4)}</td>
      <td>${row.hitRate.toFixed(3)}</td>
      <td>${row.coverage.toFixed(4)}</td>
      <td>${row.users}</td>
    </tr>
  `).join('');
      evalLoaded = true;
      return;
    }
  } catch {  }

  const evalData = [
    { model: 'Popularity-Based',        precision: 0.0000, recall: 0.0000, ndcg: 0.0000, hitRate: 0.0000, coverage: 0.0033, users: 500 },
    { model: 'Content-Based',           precision: 0.0020, recall: 0.0200, ndcg: 0.0150, hitRate: 0.0200, coverage: 0.1572, users: 500 },
    { model: 'Collaborative Filtering', precision: 0.0056, recall: 0.0560, ndcg: 0.0247, hitRate: 0.0560, coverage: 0.0665, users: 500 },
    { model: 'Hybrid',                  precision: 0.0052, recall: 0.0520, ndcg: 0.0234, hitRate: 0.0520, coverage: 0.0968, users: 500 }
  ];

  tbody.innerHTML = evalData.map(row => `
    <tr>
      <td><strong>${escapeHtml(row.model)}</strong></td>
      <td>${row.precision.toFixed(4)}</td>
      <td>${row.recall.toFixed(4)}</td>
      <td>${row.ndcg.toFixed(4)}</td>
      <td>${row.hitRate.toFixed(3)}</td>
      <td>${row.coverage.toFixed(4)}</td>
      <td>${row.users}</td>
    </tr>
  `).join('');

  evalLoaded = true;
}

function showLoadingState(show) {
  document.getElementById('resultsLoading').classList.toggle('hidden', !show);
  document.getElementById('generateBtn').disabled = show;
}

function showError(msg) {
  const errorEl = document.getElementById('resultsError');
  const msgEl = document.getElementById('resultsErrorMsg');
  errorEl.classList.remove('hidden');
  if (msgEl) msgEl.textContent = msg;
}

function clearResults() {
  document.getElementById('resultsGrid').innerHTML = '';
  document.getElementById('resultsError').classList.add('hidden');
  document.getElementById('resultsEmpty').classList.add('hidden');
  document.getElementById('resultsCount').textContent = '';
}

function safeCoverUrl(value) {
  const url = String(value || '').trim();
  if (!url || url === 'nan' || url === 'None' || url === 'null') return '';
  if (url.startsWith('https://') || url.startsWith('http://')) return url;
  return '';
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

let currentAccountId = null;

async function loadAccounts() {
  try {
    const res = await fetch(`${BASE_URL}/self-accounts`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      return await res.json();
    }
  } catch {}
  return [];
}

async function refreshAccountSelect() {
  const select = document.getElementById('accountSelect');
  const accounts = await loadAccounts();
  select.innerHTML = '<option value="">-- Select or create account --</option>';
  accounts.forEach(acc => {
    const opt = document.createElement('option');
    opt.value = acc.account_id;
    opt.textContent = `${acc.name} (${acc.ratings_count} ratings)`;
    select.appendChild(opt);
  });
  if (currentAccountId && accounts.find(a => a.account_id === currentAccountId)) {
    select.value = currentAccountId;
  }
}

async function fetchAccountDetails(accountId) {
  try {
    const res = await fetch(`${BASE_URL}/self-accounts/${accountId}`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) return await res.json();
  } catch {}
  return null;
}

async function renderRatings(accountId) {
  const container = document.getElementById('ratingsList');
  const account = await fetchAccountDetails(accountId);
  if (!account) {
    container.innerHTML = '<p class="text-muted">Unable to load ratings.</p>';
    return;
  }
  const ratings = account.ratings || {};
  const bidMap = {};
  for (const [bid, rating] of Object.entries(ratings)) {
    bidMap[bid] = rating;
  }
  if (Object.keys(bidMap).length === 0) {
    container.innerHTML = '<p class="text-muted">No ratings yet. Search and rate books above.</p>';
    return;
  }
  container.innerHTML = '';
  for (const [bid, rating] of Object.entries(bidMap)) {
    const div = document.createElement('div');
    div.className = 'rating-item';
    div.innerHTML = `
      <div class="rating-info">
        <div class="rating-title">Book ID: ${bid}</div>
        <div class="rating-author">Rating: ${rating}</div>
      </div>
      <div class="rating-stars">${'★'.repeat(Math.round(rating))}</div>
      <button class="rating-remove" data-book-id="${bid}" title="Remove rating">×</button>
    `;
    container.appendChild(div);
  }
  container.querySelectorAll('.rating-remove').forEach(btn => {
    btn.addEventListener('click', async () => {
      const bookId = btn.dataset.bookId;
      await removeRating(accountId, bookId);
    });
  });
}

async function addRating(accountId, bookId, rating) {
  try {
    const res = await fetch(`${BASE_URL}/self-accounts/${accountId}/ratings?book_id=${encodeURIComponent(bookId)}&rating=${rating}`, {
      method: 'POST',
      signal: AbortSignal.timeout(5000)
    });
    if (res.ok) {
      await renderRatings(accountId);
      await fetchSelfAccountRecommendations(accountId);
      document.getElementById('bookSearch').value = '';
      document.getElementById('searchResults').innerHTML = '';
    } else {
      const err = await res.json();
      alert(err.detail || 'Failed to add rating');
    }
  } catch (e) {
    alert('Failed to add rating: ' + e.message);
  }
}

async function removeRating(accountId, bookId) {
  try {
    const res = await fetch(`${BASE_URL}/self-accounts/${accountId}/ratings/${encodeURIComponent(bookId)}`, {
      method: 'DELETE',
      signal: AbortSignal.timeout(5000)
    });
    if (res.ok) {
      await renderRatings(accountId);
      await fetchSelfAccountRecommendations(accountId);
    } else {
      alert('Failed to remove rating');
    }
  } catch (e) {
    alert('Failed to remove rating: ' + e.message);
  }
}

async function fetchSelfAccountRecommendations(accountId) {
  const grid = document.getElementById('selfRecGrid');
  const countEl = document.getElementById('selfRecCount');
  const loading = document.getElementById('selfRecLoading');
  const error = document.getElementById('selfRecError');
  const empty = document.getElementById('selfRecEmpty');

  loading.classList.remove('hidden');
  error.classList.add('hidden');
  empty.classList.add('hidden');
  grid.innerHTML = '';
  countEl.textContent = '';

  try {
    const res = await fetch(`${BASE_URL}/self-accounts/${accountId}/recommendations?limit=10`, {
      signal: AbortSignal.timeout(12000)
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'Failed to fetch recommendations');
    }

    const items = Array.isArray(data) ? data : [data];
    renderResults(items, 'hybrid', grid);
    countEl.textContent = `${items.length} result${items.length !== 1 ? 's' : ''}`;
  } catch (err) {
    error.classList.remove('hidden');
    document.getElementById('selfRecErrorMsg').textContent = err.message;
  } finally {
    loading.classList.add('hidden');
  }
}

function initSelfAccountControls() {
  const accountSelect = document.getElementById('accountSelect');
  const createBtn = document.getElementById('createAccountBtn');
  const deleteBtn = document.getElementById('deleteAccountBtn');
  const createForm = document.getElementById('createAccountForm');
  const ratingsSection = document.getElementById('ratingsSection');
  const bookSearch = document.getElementById('bookSearch');
  const searchResults = document.getElementById('searchResults');

  refreshAccountSelect();

  accountSelect.addEventListener('change', async () => {
    currentAccountId = accountSelect.value || null;
    if (currentAccountId) {
      deleteBtn.style.display = 'inline-block';
      ratingsSection.classList.remove('hidden');
      await renderRatings(currentAccountId);
      await fetchSelfAccountRecommendations(currentAccountId);
    } else {
      deleteBtn.style.display = 'none';
      ratingsSection.classList.add('hidden');
      document.getElementById('selfRecGrid').innerHTML = '';
      document.getElementById('selfRecCount').textContent = '';
      document.getElementById('selfRecEmpty').classList.remove('hidden');
    }
  });

  createBtn.addEventListener('click', () => {
    createForm.classList.remove('hidden');
  });

  document.getElementById('cancelAccountBtn').addEventListener('click', () => {
    createForm.classList.add('hidden');
  });

  document.getElementById('saveAccountBtn').addEventListener('click', async () => {
    const accountId = document.getElementById('newAccountId').value.trim();
    const name = document.getElementById('newAccountName').value.trim();
    if (!accountId || !name) {
      alert('Please enter both Account ID and Display Name');
      return;
    }
    try {
      const res = await fetch(`${BASE_URL}/self-accounts?account_id=${encodeURIComponent(accountId)}&name=${encodeURIComponent(name)}`, {
        method: 'POST',
        signal: AbortSignal.timeout(5000)
      });
      if (res.ok) {
        createForm.classList.add('hidden');
        document.getElementById('newAccountId').value = '';
        document.getElementById('newAccountName').value = '';
        await refreshAccountSelect();
        accountSelect.value = accountId;
        accountSelect.dispatchEvent(new Event('change'));
      } else {
        const err = await res.json();
        alert(err.detail || 'Failed to create account');
      }
    } catch (e) {
      alert('Failed to create account: ' + e.message);
    }
  });

  deleteBtn.addEventListener('click', async () => {
    if (!currentAccountId) return;
    if (!confirm(`Delete account "${currentAccountId}"? This cannot be undone.`)) return;
    try {
      const res = await fetch(`${BASE_URL}/self-accounts/${currentAccountId}`, {
        method: 'DELETE',
        signal: AbortSignal.timeout(5000)
      });
      if (res.ok) {
        currentAccountId = null;
        accountSelect.value = '';
        accountSelect.dispatchEvent(new Event('change'));
        await refreshAccountSelect();
      } else {
        alert('Failed to delete account');
      }
    } catch (e) {
      alert('Failed to delete account: ' + e.message);
    }
  });

  let searchTimeout = null;
  bookSearch.addEventListener('input', (e) => {
    const query = e.target.value.trim();
    clearTimeout(searchTimeout);
    if (query.length < 2) {
      searchResults.innerHTML = '';
      return;
    }
    searchTimeout = setTimeout(async () => {
      try {
        const res = await fetch(`${BASE_URL}/books/search?query=${encodeURIComponent(query)}&limit=10`, {
          signal: AbortSignal.timeout(5000)
        });
        if (res.ok) {
          const books = await res.json();
          searchResults.innerHTML = '';
          books.forEach(book => {
            const div = document.createElement('div');
            div.className = 'search-result-item';
            div.innerHTML = `
              <div class="search-result-title">${escapeHtml(book.title)}</div>
              <div class="search-result-meta">${escapeHtml(book.author)} · Rating: ${book.rating}</div>
            `;
            div.addEventListener('click', () => {
              if (!currentAccountId) {
                alert('Please select an account first');
                return;
              }
              const rating = prompt(`Rate "${book.title}" (1-5):`, '4');
              if (rating !== null) {
                const r = parseFloat(rating);
                if (r >= 1 && r <= 5) {
                  addRating(currentAccountId, book.bookId, r);
                } else {
                  alert('Rating must be between 1 and 5');
                }
              }
            });
            searchResults.appendChild(div);
          });
        }
      } catch (e) {
        searchResults.innerHTML = '<p class="text-muted">Search failed</p>';
      }
    }, 300);
  });
}

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('a[href*="/docs"]').forEach(link => {
    link.href = `${BASE_URL}/docs`;
  });
  initTabs();
  checkApiHealth();
  loadStats();
  initDemoControls();
  initSelfAccountControls();
});
