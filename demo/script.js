

'use strict';

const BASE_URL = 'http://127.0.0.1:8000';

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

function renderResults(items, model) {
  const grid = document.getElementById('resultsGrid');
  const countEl = document.getElementById('resultsCount');
  grid.innerHTML = '';

  if (!items || items.length === 0) {
    document.getElementById('resultsEmpty').classList.remove('hidden');
    countEl.textContent = '';
    return;
  }

  document.getElementById('resultsEmpty').classList.add('hidden');
  countEl.textContent = `${items.length} result${items.length !== 1 ? 's' : ''}`;

  items.forEach((item, idx) => {
    const card = document.createElement('article');
    card.className = 'book-card';
    card.innerHTML = buildCardHTML(item, idx + 1, model);
    grid.appendChild(card);
  });
}

function buildCardHTML(item, rank, model) {
  const title = escapeHtml(item.title || 'Unknown Title');
  const author = escapeHtml(item.author || 'Unknown Author');
  const bookId = escapeHtml(String(item.bookId || ''));
  const reason = escapeHtml(item.reason || '');

  let coldStartBadge = '';
  if (item.reason && item.reason.toLowerCase().includes('cold-start')) {
    coldStartBadge = `<div class="cold-start-badge">📋 Cold-start fallback</div>`;
  }

  let scoresHTML = buildScoresHTML(item, model);

  return `
    <div class="book-rank">#${rank}</div>
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

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  checkApiHealth();
  loadStats();
  initDemoControls();
});
