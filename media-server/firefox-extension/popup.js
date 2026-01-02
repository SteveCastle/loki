// Lowkey Media Server Firefox Extension
const API_BASE = 'http://localhost:8090';

// State
let eventSource = null;
let jobs = [];
let isConnecting = false;
let authToken = null;
let currentUrl = '';

// DOM Elements
const elements = {
  serverStatus: null,
  urlDisplay: null,
  optTranscript: null,
  optDescription: null,
  optFileMeta: null,
  optAutoTag: null,
  ingestBtn: null,
  feedback: null,
  jobsList: null,
  refreshBtn: null,
  clearAllBtn: null,
  // Login elements
  loginOverlay: null,
  mainContainer: null,
  loginForm: null,
  loginUsername: null,
  loginPassword: null,
  loginBtn: null,
  loginError: null,
  logoutBtn: null,
};

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', init);

async function init() {
  // Cache DOM elements
  elements.serverStatus = document.getElementById('serverStatus');
  elements.urlDisplay = document.getElementById('urlDisplay');
  elements.optTranscript = document.getElementById('optTranscript');
  elements.optDescription = document.getElementById('optDescription');
  elements.optFileMeta = document.getElementById('optFileMeta');
  elements.optAutoTag = document.getElementById('optAutoTag');
  elements.ingestBtn = document.getElementById('ingestBtn');
  elements.feedback = document.getElementById('feedback');
  elements.jobsList = document.getElementById('jobsList');
  elements.refreshBtn = document.getElementById('refreshBtn');
  elements.clearAllBtn = document.getElementById('clearAllBtn');
  // Login elements
  elements.loginOverlay = document.getElementById('loginOverlay');
  elements.mainContainer = document.getElementById('mainContainer');
  elements.loginForm = document.getElementById('loginForm');
  elements.loginUsername = document.getElementById('loginUsername');
  elements.loginPassword = document.getElementById('loginPassword');
  elements.loginBtn = document.getElementById('loginBtn');
  elements.loginError = document.getElementById('loginError');
  elements.logoutBtn = document.getElementById('logoutBtn');

  // Load auth token
  await loadAuthToken();

  // Check authentication status
  const isAuthenticated = await checkAuthStatus();

  if (isAuthenticated) {
    showMainContent();
    await initializeApp();
  } else {
    showLoginForm();
  }

  // Set up login form event listener
  elements.loginForm.addEventListener('submit', handleLogin);
  elements.logoutBtn.addEventListener('click', handleLogout);
}

// Authentication functions
async function loadAuthToken() {
  try {
    const result = await browser.storage.local.get(['authToken']);
    authToken = result.authToken || null;
  } catch (err) {
    console.error('Failed to load auth token:', err);
    authToken = null;
  }
}

function saveAuthToken(token) {
  authToken = token;
  browser.storage.local.set({ authToken: token });
}

function clearAuthToken() {
  authToken = null;
  browser.storage.local.remove('authToken');
}

async function checkAuthStatus() {
  if (!authToken) return false;

  try {
    const response = await fetch(`${API_BASE}/auth/status`, {
      headers: {
        Authorization: `Bearer ${authToken}`,
      },
    });

    if (!response.ok) return false;

    const data = await response.json();
    return data.loggedIn === true;
  } catch (err) {
    console.error('Auth status check failed:', err);
    return false;
  }
}

async function handleLogin(e) {
  e.preventDefault();

  const username = elements.loginUsername.value.trim();
  const password = elements.loginPassword.value;

  if (!username || !password) {
    showLoginError('Please enter username and password');
    return;
  }

  setLoginLoading(true);
  hideLoginError();

  try {
    const response = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      const text = await response.text();
      try {
        const data = JSON.parse(text);
        throw new Error(data.error || 'Invalid credentials');
      } catch {
        throw new Error('Invalid credentials');
      }
    }

    const data = await response.json();

    if (data.token) {
      saveAuthToken(data.token);
      showMainContent();
      await initializeApp();
    } else {
      throw new Error('No token received');
    }
  } catch (err) {
    console.error('Login error:', err);
    showLoginError(err.message || 'Login failed');
  } finally {
    setLoginLoading(false);
  }
}

async function handleLogout() {
  try {
    await fetch(`${API_BASE}/auth/logout`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${authToken}`,
      },
    });
  } catch (err) {
    console.error('Logout request failed:', err);
  }

  // Clear local state regardless of server response
  clearAuthToken();

  // Disconnect SSE
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  // Reset state
  jobs = [];

  // Show login form
  showLoginForm();
}

function showLoginForm() {
  elements.loginOverlay.style.display = 'flex';
  elements.mainContainer.style.display = 'none';
  elements.loginUsername.value = '';
  elements.loginPassword.value = '';
  hideLoginError();
  // Focus username field
  setTimeout(() => elements.loginUsername.focus(), 100);
}

function showMainContent() {
  elements.loginOverlay.style.display = 'none';
  elements.mainContainer.style.display = 'flex';
}

function showLoginError(message) {
  elements.loginError.textContent = message;
  elements.loginError.style.display = 'block';
}

function hideLoginError() {
  elements.loginError.style.display = 'none';
}

function setLoginLoading(loading) {
  elements.loginBtn.disabled = loading;
  elements.loginBtn.querySelector('.btn-text').style.display = loading
    ? 'none'
    : 'inline';
  elements.loginBtn.querySelector('.btn-loading').style.display = loading
    ? 'inline-flex'
    : 'none';
}

// Make authenticated fetch request
async function authFetch(url, options = {}) {
  const headers = {
    ...options.headers,
  };

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }

  const response = await fetch(url, { ...options, headers });

  // Handle 401 Unauthorized - show login form
  if (response.status === 401) {
    clearAuthToken();
    showLoginForm();
    throw new Error('Session expired. Please log in again.');
  }

  return response;
}

// Initialize app after successful authentication
async function initializeApp() {
  // Load saved preferences
  await loadPreferences();

  // Get current tab URL
  await populateCurrentUrl();

  // Set up event listeners
  elements.ingestBtn.addEventListener('click', createIngestTask);
  elements.refreshBtn.addEventListener('click', refreshJobs);
  elements.clearAllBtn.addEventListener('click', clearAllJobs);

  // Save preferences when options change
  elements.optTranscript.addEventListener('change', savePreferences);
  elements.optDescription.addEventListener('change', savePreferences);
  elements.optFileMeta.addEventListener('change', savePreferences);
  elements.optAutoTag.addEventListener('change', savePreferences);

  // Set up event delegation for job actions
  elements.jobsList.addEventListener('click', handleJobClick);

  // Connect to SSE and fetch initial jobs
  connectSSE();
  fetchJobs();
}

// Handle clicks on job items using event delegation
function handleJobClick(e) {
  const target = e.target.closest('[data-action]');
  if (!target) return;

  const action = target.getAttribute('data-action');
  const id = target.getAttribute('data-id');
  if (!id) return;

  e.preventDefault();

  switch (action) {
    case 'cancel':
      cancelJob(id);
      break;
    case 'remove':
      removeJobFromServer(id);
      break;
    case 'open':
      openJobDetail(id);
      break;
  }
}

// Get current tab URL
async function populateCurrentUrl() {
  try {
    const tabs = await browser.tabs.query({
      active: true,
      currentWindow: true,
    });
    if (tabs[0]?.url) {
      currentUrl = tabs[0].url;
      elements.urlDisplay.textContent = truncateUrl(tabs[0].url);
      elements.urlDisplay.title = tabs[0].url;
    } else {
      elements.urlDisplay.textContent = 'No URL available';
      elements.urlDisplay.classList.add('url-error');
    }
  } catch (err) {
    console.error('Failed to get current tab URL:', err);
    elements.urlDisplay.textContent = 'Failed to get URL';
    elements.urlDisplay.classList.add('url-error');
  }
}

// Truncate URL for display
function truncateUrl(url) {
  if (url.length <= 60) return url;
  try {
    const urlObj = new URL(url);
    const path = urlObj.pathname + urlObj.search;
    const truncatedPath =
      path.length > 40 ? path.slice(0, 20) + '...' + path.slice(-17) : path;
    return urlObj.host + truncatedPath;
  } catch {
    return url.slice(0, 30) + '...' + url.slice(-27);
  }
}

// Save/load preferences
function savePreferences() {
  browser.storage.local.set({
    ingestOptions: {
      transcript: elements.optTranscript.checked,
      description: elements.optDescription.checked,
      fileMeta: elements.optFileMeta.checked,
      autoTag: elements.optAutoTag.checked,
    },
  });
  updateCheckboxStyles();
}

// Update checkbox parent styles for Firefox compatibility
function updateCheckboxStyles() {
  const checkboxes = [
    elements.optTranscript,
    elements.optDescription,
    elements.optFileMeta,
    elements.optAutoTag,
  ];
  checkboxes.forEach((checkbox) => {
    const parent = checkbox.closest('.checkbox-item');
    if (parent) {
      parent.classList.toggle('checked', checkbox.checked);
    }
  });
}

async function loadPreferences() {
  try {
    const result = await browser.storage.local.get(['ingestOptions']);
    if (result.ingestOptions) {
      elements.optTranscript.checked = result.ingestOptions.transcript || false;
      elements.optDescription.checked =
        result.ingestOptions.description || false;
      elements.optFileMeta.checked = result.ingestOptions.fileMeta || false;
      elements.optAutoTag.checked = result.ingestOptions.autoTag || false;
    }
    updateCheckboxStyles();
  } catch (err) {
    console.error('Failed to load preferences:', err);
  }
}

// Build arguments array from checkbox selections
function buildIngestArgs() {
  const args = [];
  if (elements.optTranscript.checked) args.push('--transcript');
  if (elements.optDescription.checked) args.push('--description');
  if (elements.optFileMeta.checked) args.push('--filemeta');
  if (elements.optAutoTag.checked) args.push('--autotag');
  return args;
}

// Create ingest task
async function createIngestTask() {
  if (!currentUrl) {
    showFeedback('No URL available', 'error');
    return;
  }

  // Build the input string: ingest [args] url
  const args = buildIngestArgs();
  let input = 'ingest';
  if (args.length > 0) {
    input += ' ' + args.join(' ');
  }
  input += ' ' + currentUrl;

  // Disable button and show loading
  setLoading(true);
  hideFeedback();

  try {
    const response = await authFetch(`${API_BASE}/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    const data = await response.json();
    showFeedback(`Ingesting: ${data.id.slice(0, 8)}...`, 'success');

    // Refresh jobs list
    fetchJobs();
  } catch (err) {
    console.error('Create task error:', err);
    showFeedback(`Failed: ${err.message}`, 'error');
  } finally {
    setLoading(false);
  }
}

// Fetch current jobs
async function fetchJobs() {
  try {
    const response = await authFetch(`${API_BASE}/jobs/list`);
    if (!response.ok) throw new Error('Failed to fetch jobs');

    const fetchedJobs = await response.json();
    updateServerStatus(true);

    if (Array.isArray(fetchedJobs)) {
      fetchedJobs.forEach((job) => {
        const existingIndex = jobs.findIndex((j) => j.id === job.id);
        if (existingIndex >= 0) {
          jobs[existingIndex] = job;
        } else {
          jobs.push(job);
        }
      });

      // Sort by created_at descending (newest first)
      jobs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

      // Keep only last 20 jobs
      if (jobs.length > 20) {
        jobs = jobs.slice(0, 20);
      }

      renderJobs();
    }
  } catch (err) {
    console.error('Fetch jobs error:', err);
    updateServerStatus(false);
  }
}

// Refresh jobs
function refreshJobs() {
  if (eventSource) {
    eventSource.close();
  }
  jobs = [];
  renderJobs();
  connectSSE();
  fetchJobs();
}

// Clear all non-running jobs
async function clearAllJobs() {
  try {
    const response = await authFetch(`${API_BASE}/jobs/clear`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error('Failed to clear jobs');
    }

    fetchJobs();
  } catch (err) {
    console.error('Clear all jobs error:', err);
    showFeedback('Failed to clear jobs', 'error');
  }
}

// SSE Connection
function connectSSE() {
  if (
    isConnecting ||
    (eventSource && eventSource.readyState === EventSource.OPEN)
  ) {
    return;
  }

  isConnecting = true;
  updateServerStatus('connecting');

  try {
    eventSource = new EventSource(`${API_BASE}/stream`);

    eventSource.onopen = () => {
      isConnecting = false;
      updateServerStatus(true);
      console.log('SSE connected');
    };

    eventSource.onerror = (err) => {
      console.error('SSE error:', err);
      isConnecting = false;
      updateServerStatus(false);

      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }

      setTimeout(connectSSE, 5000);
    };

    eventSource.addEventListener('create', (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.job) {
          addOrUpdateJob(data.job);
        }
      } catch (e) {
        console.error('Parse create event:', e);
      }
    });

    eventSource.addEventListener('update', (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.job) {
          addOrUpdateJob(data.job);
        }
      } catch (e) {
        console.error('Parse update event:', e);
      }
    });

    eventSource.addEventListener('delete', (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.job?.id) {
          removeJob(data.job.id);
        }
      } catch (e) {
        console.error('Parse delete event:', e);
      }
    });
  } catch (err) {
    console.error('SSE connect error:', err);
    isConnecting = false;
    updateServerStatus(false);
  }
}

// Job management
function addOrUpdateJob(job) {
  const existingIndex = jobs.findIndex((j) => j.id === job.id);
  if (existingIndex >= 0) {
    jobs[existingIndex] = job;
  } else {
    jobs.unshift(job);
  }

  if (jobs.length > 20) {
    jobs = jobs.slice(0, 20);
  }

  renderJobs();
}

function removeJob(jobId) {
  jobs = jobs.filter((j) => j.id !== jobId);
  renderJobs();
}

function renderJobs() {
  const activeJobs = jobs.filter(
    (j) => j.state === 'pending' || j.state === 'in_progress'
  );
  const recentJobs = jobs
    .filter((j) => j.state !== 'pending' && j.state !== 'in_progress')
    .slice(0, 5);
  const displayJobs = [...activeJobs, ...recentJobs].slice(0, 10);

  if (displayJobs.length === 0) {
    elements.jobsList.innerHTML = `
      <div class="jobs-empty">
        <span>No active jobs</span>
      </div>
    `;
    return;
  }

  elements.jobsList.innerHTML = displayJobs
    .map((job) => {
      const statusClass = getStatusClass(job.state);
      const timeAgo = formatTimeAgo(job.created_at);
      const truncatedInput = truncate(job.input || '', 50);
      const isActive = job.state === 'pending' || job.state === 'in_progress';

      return `
      <div class="job-item" data-id="${job.id}">
        <div class="job-status ${statusClass}"></div>
        <div class="job-details" data-action="open" data-id="${
          job.id
        }" style="cursor: pointer;" title="Open in web UI">
          <div class="job-command">${escapeHtml(job.command || 'Unknown')}</div>
          <div class="job-input" title="${escapeHtml(
            job.input || ''
          )}">${escapeHtml(truncatedInput)}</div>
          <div class="job-time">${timeAgo}</div>
        </div>
        <div class="job-actions">
          ${
            isActive
              ? `<button class="job-action-btn cancel" data-action="cancel" data-id="${job.id}" title="Cancel">✕</button>`
              : `<button class="job-action-btn remove" data-action="remove" data-id="${job.id}" title="Remove">✕</button>`
          }
        </div>
      </div>
    `;
    })
    .join('');
}

// Cancel a job
async function cancelJob(jobId) {
  try {
    const response = await authFetch(`${API_BASE}/job/${jobId}/cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error('Failed to cancel job');
    }
  } catch (err) {
    console.error('Cancel job error:', err);
    showFeedback('Failed to cancel job', 'error');
  }
}

// Remove a job from the server
async function removeJobFromServer(jobId) {
  try {
    const response = await authFetch(`${API_BASE}/job/${jobId}/remove`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error('Failed to remove job');
    }

    removeJob(jobId);
  } catch (err) {
    console.error('Remove job error:', err);
    showFeedback('Failed to remove job', 'error');
  }
}

// Open job detail in web UI
function openJobDetail(jobId) {
  browser.tabs.create({ url: `${API_BASE}/job/${jobId}` });
}

// UI Helpers
function updateServerStatus(connected) {
  if (connected === 'connecting') {
    elements.serverStatus.className = 'status-indicator connecting';
    elements.serverStatus.title = 'Connecting...';
  } else if (connected) {
    elements.serverStatus.className = 'status-indicator connected';
    elements.serverStatus.title = 'Connected to Lowkey Media Server';
  } else {
    elements.serverStatus.className = 'status-indicator';
    elements.serverStatus.title = 'Disconnected from server';
  }
}

function setLoading(loading) {
  elements.ingestBtn.disabled = loading;
  elements.ingestBtn.querySelector('.btn-text').style.display = loading
    ? 'none'
    : 'inline';
  elements.ingestBtn.querySelector('.btn-loading').style.display = loading
    ? 'inline-flex'
    : 'none';
}

function showFeedback(message, type) {
  elements.feedback.textContent = message;
  elements.feedback.className = `feedback ${type}`;
  elements.feedback.style.display = 'flex';

  if (type === 'success') {
    setTimeout(hideFeedback, 5000);
  }
}

function hideFeedback() {
  elements.feedback.style.display = 'none';
}

// Utility functions
function getStatusClass(state) {
  switch (state) {
    case 'pending':
      return 'pending';
    case 'in_progress':
      return 'running';
    case 'completed':
      return 'completed';
    case 'cancelled':
      return 'cancelled';
    case 'error':
      return 'error';
    default:
      return 'pending';
  }
}

function formatTimeAgo(timestamp) {
  if (!timestamp) return '';

  const date = new Date(timestamp);
  const now = new Date();
  const diff = Math.floor((now - date) / 1000);

  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function truncate(str, maxLen) {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen - 3) + '...';
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// Cleanup on popup close
window.addEventListener('unload', () => {
  if (eventSource) {
    eventSource.close();
  }
});
