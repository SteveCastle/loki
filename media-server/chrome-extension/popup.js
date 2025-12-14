// Shrike Chrome Extension
const API_BASE = "http://localhost:8090";

// State
let eventSource = null;
let jobs = [];
let tasks = []; // Available tasks from server
let isConnecting = false;
let argsHistory = {}; // Per-command argument history
let currentArgsPerCommand = {}; // Current/last-used args for each command
let previousCommand = null; // Track command to save args when switching
let selectedDropdownIndex = -1;

// DOM Elements
const elements = {
  serverStatus: null,
  command: null,
  args: null,
  argsWrapper: null,
  argsDropdown: null,
  clearArgsHistory: null,
  url: null,
  createBtn: null,
  feedback: null,
  jobsList: null,
  refreshBtn: null,
  clearAllBtn: null,
};

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", init);

async function init() {
  // Cache DOM elements
  elements.serverStatus = document.getElementById("serverStatus");
  elements.command = document.getElementById("command");
  elements.args = document.getElementById("args");
  elements.argsWrapper = document.querySelector(".args-wrapper");
  elements.argsDropdown = document.getElementById("argsDropdown");
  elements.clearArgsHistory = document.getElementById("clearArgsHistory");
  elements.url = document.getElementById("url");
  elements.createBtn = document.getElementById("createBtn");
  elements.feedback = document.getElementById("feedback");
  elements.jobsList = document.getElementById("jobsList");
  elements.refreshBtn = document.getElementById("refreshBtn");
  elements.clearAllBtn = document.getElementById("clearAllBtn");

  // Fetch available tasks first
  await fetchTasks();

  // Load saved preferences (after tasks are loaded so we can restore selection)
  await loadPreferences();

  // Get current tab URL
  await populateCurrentUrl();

  // Set up event listeners
  elements.createBtn.addEventListener("click", createTask);
  elements.refreshBtn.addEventListener("click", refreshJobs);
  elements.clearAllBtn.addEventListener("click", clearAllJobs);
  elements.command.addEventListener("change", handleCommandChange);
  elements.clearArgsHistory.addEventListener("click", clearCurrentArgsHistory);

  // Args input events for autocomplete
  elements.args.addEventListener("focus", showArgsDropdown);
  elements.args.addEventListener("blur", handleArgsBlur);
  elements.args.addEventListener("input", handleArgsInput);
  elements.args.addEventListener("keydown", handleArgsKeydown);

  // Allow Enter key to submit from URL field
  elements.url.addEventListener("keypress", (e) => {
    if (e.key === "Enter") createTask();
  });

  // Set up event delegation for job actions
  elements.jobsList.addEventListener("click", handleJobClick);

  // Close dropdown when clicking outside
  document.addEventListener("click", (e) => {
    if (
      !elements.args.contains(e.target) &&
      !elements.argsDropdown.contains(e.target)
    ) {
      hideArgsDropdown();
    }
  });

  // Connect to SSE and fetch initial jobs
  connectSSE();
  fetchJobs();
}

// Handle clicks on job items using event delegation
function handleJobClick(e) {
  // Find the closest element with a data-action attribute
  const target = e.target.closest("[data-action]");
  if (!target) return;

  const action = target.getAttribute("data-action");
  const id = target.getAttribute("data-id");
  if (!id) return;

  e.preventDefault();

  switch (action) {
    case "cancel":
      cancelJob(id);
      break;
    case "remove":
      removeJobFromServer(id);
      break;
    case "open":
      openJobDetail(id);
      break;
  }
}

// Get current tab URL
async function populateCurrentUrl() {
  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });
    if (tab?.url) {
      elements.url.value = tab.url;
    }
  } catch (err) {
    console.error("Failed to get current tab URL:", err);
  }
}

// Fetch available tasks from server
async function fetchTasks() {
  try {
    const response = await fetch(`${API_BASE}/tasks`);
    if (!response.ok) throw new Error("Failed to fetch tasks");

    const data = await response.json();
    if (Array.isArray(data.tasks)) {
      tasks = data.tasks;
      populateCommandDropdown();
      updateServerStatus(true);
    }
  } catch (err) {
    console.error("Fetch tasks error:", err);
    // Fall back to a minimal set if server is unavailable
    tasks = [
      { id: "gallery-dl", name: "gallery-dl" },
      { id: "yt-dlp", name: "yt-dlp" },
    ];
    populateCommandDropdown();
    updateServerStatus(false);
  }
}

// Populate the command dropdown with tasks
function populateCommandDropdown() {
  elements.command.innerHTML = tasks
    .map(
      (task) =>
        `<option value="${escapeHtml(task.id)}">${escapeHtml(
          task.name
        )}</option>`
    )
    .join("");
}

// Save/load preferences
function savePreferences() {
  // Save current args for the current command before saving
  const command = elements.command.value;
  currentArgsPerCommand[command] = elements.args.value;

  chrome.storage.local.set({
    lastCommand: command,
    argsHistory: argsHistory,
    currentArgsPerCommand: currentArgsPerCommand,
  });
}

async function loadPreferences() {
  return new Promise((resolve) => {
    chrome.storage.local.get(
      ["lastCommand", "argsHistory", "currentArgsPerCommand"],
      (result) => {
        if (result.argsHistory) {
          argsHistory = result.argsHistory;
        }
        if (result.currentArgsPerCommand) {
          currentArgsPerCommand = result.currentArgsPerCommand;
        }
        if (result.lastCommand) {
          elements.command.value = result.lastCommand;
          // Check if the saved command actually exists in the dropdown
          // (it might have been removed from the server)
          if (elements.command.value !== result.lastCommand) {
            // Command doesn't exist anymore, use first available
            previousCommand = elements.command.value;
          } else {
            // Restore the args for the last command
            elements.args.value =
              currentArgsPerCommand[result.lastCommand] || "";
            previousCommand = result.lastCommand;
          }
        } else {
          // Default to first option if no saved command
          previousCommand = elements.command.value;
        }
        updateClearHistoryButton();
        resolve();
      }
    );
  });
}

// Handle command change
function handleCommandChange() {
  // Save the args for the previous command before switching
  if (previousCommand) {
    currentArgsPerCommand[previousCommand] = elements.args.value;
  }

  const newCommand = elements.command.value;

  // Load args for the new command
  elements.args.value = currentArgsPerCommand[newCommand] || "";

  // Update previous command for next switch
  previousCommand = newCommand;

  updateClearHistoryButton();
  savePreferences();
}

// Args history management
function addArgsToHistory(command, args) {
  if (!args || args.trim() === "") return;

  const trimmedArgs = args.trim();

  if (!argsHistory[command]) {
    argsHistory[command] = [];
  }

  // Remove if already exists (to move to front)
  argsHistory[command] = argsHistory[command].filter((a) => a !== trimmedArgs);

  // Add to front
  argsHistory[command].unshift(trimmedArgs);

  // Keep only last 10 entries per command
  if (argsHistory[command].length > 10) {
    argsHistory[command] = argsHistory[command].slice(0, 10);
  }

  savePreferences();
  updateClearHistoryButton();
}

function removeArgFromHistory(command, arg) {
  if (!argsHistory[command]) return;

  argsHistory[command] = argsHistory[command].filter((a) => a !== arg);

  if (argsHistory[command].length === 0) {
    delete argsHistory[command];
  }

  savePreferences();
  updateClearHistoryButton();
  renderArgsDropdown();
}

function clearCurrentArgsHistory() {
  const command = elements.command.value;
  delete argsHistory[command];
  delete currentArgsPerCommand[command];
  elements.args.value = "";
  savePreferences();
  updateClearHistoryButton();
  hideArgsDropdown();
}

function updateClearHistoryButton() {
  const command = elements.command.value;
  const hasHistory = argsHistory[command] && argsHistory[command].length > 0;
  elements.clearArgsHistory.style.display = hasHistory ? "inline-flex" : "none";
}

// Args dropdown functions
function showArgsDropdown() {
  renderArgsDropdown();
  const command = elements.command.value;
  const history = argsHistory[command] || [];

  if (history.length > 0) {
    elements.argsDropdown.style.display = "block";
    elements.argsWrapper.classList.add("dropdown-open");
    selectedDropdownIndex = -1;
  }
}

function hideArgsDropdown() {
  elements.argsDropdown.style.display = "none";
  elements.argsWrapper.classList.remove("dropdown-open");
  selectedDropdownIndex = -1;
}

function handleArgsBlur(e) {
  // Save args when leaving the field
  savePreferences();

  // Delay hide to allow click on dropdown items
  setTimeout(() => {
    if (!elements.argsDropdown.contains(document.activeElement)) {
      hideArgsDropdown();
    }
  }, 150);
}

function handleArgsInput() {
  renderArgsDropdown();
  const history = getFilteredHistory();

  if (history.length > 0) {
    elements.argsDropdown.style.display = "block";
    elements.argsWrapper.classList.add("dropdown-open");
  } else {
    hideArgsDropdown();
  }
}

function handleArgsKeydown(e) {
  const history = getFilteredHistory();

  if (elements.argsDropdown.style.display === "none" || history.length === 0) {
    if (e.key === "Enter") {
      e.preventDefault();
      createTask();
    }
    return;
  }

  switch (e.key) {
    case "ArrowDown":
      e.preventDefault();
      selectedDropdownIndex = Math.min(
        selectedDropdownIndex + 1,
        history.length - 1
      );
      updateDropdownSelection();
      break;
    case "ArrowUp":
      e.preventDefault();
      selectedDropdownIndex = Math.max(selectedDropdownIndex - 1, -1);
      updateDropdownSelection();
      break;
    case "Enter":
      e.preventDefault();
      if (
        selectedDropdownIndex >= 0 &&
        selectedDropdownIndex < history.length
      ) {
        selectArg(history[selectedDropdownIndex]);
      } else {
        createTask();
      }
      break;
    case "Escape":
      hideArgsDropdown();
      break;
  }
}

function getFilteredHistory() {
  const command = elements.command.value;
  const currentInput = elements.args.value.toLowerCase().trim();
  const history = argsHistory[command] || [];

  if (!currentInput) {
    return history;
  }

  return history.filter((arg) => arg.toLowerCase().includes(currentInput));
}

function renderArgsDropdown() {
  const history = getFilteredHistory();

  if (history.length === 0) {
    elements.argsDropdown.innerHTML =
      '<div class="args-dropdown-empty">No saved arguments</div>';
    return;
  }

  elements.argsDropdown.innerHTML = history
    .map(
      (arg, index) => `
    <div class="args-dropdown-item${
      index === selectedDropdownIndex ? " selected" : ""
    }" data-arg="${escapeHtml(arg)}">
      <span class="arg-text" title="${escapeHtml(arg)}">${escapeHtml(
        arg
      )}</span>
      <span class="remove-arg" data-remove-arg="${escapeHtml(
        arg
      )}" title="Remove">✕</span>
    </div>
  `
    )
    .join("");

  // Add click handlers
  elements.argsDropdown
    .querySelectorAll(".args-dropdown-item")
    .forEach((item) => {
      item.addEventListener("mousedown", (e) => {
        // Prevent the blur event from firing before we can handle the click
        e.preventDefault();

        // Check if clicking the remove button
        if (e.target.classList.contains("remove-arg")) {
          const arg = e.target.getAttribute("data-remove-arg");
          removeArgFromHistory(elements.command.value, arg);
          return;
        }

        const arg = item.getAttribute("data-arg");
        selectArg(arg);
      });
    });
}

function updateDropdownSelection() {
  const items = elements.argsDropdown.querySelectorAll(".args-dropdown-item");
  items.forEach((item, index) => {
    item.classList.toggle("selected", index === selectedDropdownIndex);
  });

  // Scroll selected item into view
  if (selectedDropdownIndex >= 0 && items[selectedDropdownIndex]) {
    items[selectedDropdownIndex].scrollIntoView({ block: "nearest" });
  }
}

function selectArg(arg) {
  elements.args.value = arg;
  hideArgsDropdown();
  elements.args.focus();
}

// Create task
async function createTask() {
  const command = elements.command.value;
  const args = elements.args.value.trim();
  const url = elements.url.value.trim();

  if (!url) {
    showFeedback("Please enter a URL or input", "error");
    return;
  }

  // Build the input string: command [args] url
  let input = command;
  if (args) {
    input += ` ${args}`;
  }
  input += ` ${url}`;

  // Disable button and show loading
  setLoading(true);
  hideFeedback();

  try {
    const response = await fetch(`${API_BASE}/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    const data = await response.json();
    showFeedback(`Task created: ${data.id.slice(0, 8)}...`, "success");

    // Save args to history on successful task creation
    addArgsToHistory(command, args);

    // Refresh jobs list
    fetchJobs();
  } catch (err) {
    console.error("Create task error:", err);
    showFeedback(`Failed: ${err.message}`, "error");
  } finally {
    setLoading(false);
  }
}

// Fetch current jobs
async function fetchJobs() {
  try {
    const response = await fetch(`${API_BASE}/jobs/list`);
    if (!response.ok) throw new Error("Failed to fetch jobs");

    const fetchedJobs = await response.json();
    updateServerStatus(true);

    // Update jobs array with fetched jobs
    if (Array.isArray(fetchedJobs)) {
      // Merge with existing jobs, preferring fetched data
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
    console.error("Fetch jobs error:", err);
    updateServerStatus(false);
  }
}

// Refresh jobs
function refreshJobs() {
  // Reconnect SSE to get fresh state
  if (eventSource) {
    eventSource.close();
  }
  jobs = [];
  renderJobs();
  connectSSE();
  fetchJobs();
  // Also refresh tasks in case new ones were registered
  fetchTasks();
}

// Clear all non-running jobs
async function clearAllJobs() {
  try {
    const response = await fetch(`${API_BASE}/jobs/clear`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    if (!response.ok) {
      throw new Error("Failed to clear jobs");
    }

    // Refresh the jobs list
    fetchJobs();
  } catch (err) {
    console.error("Clear all jobs error:", err);
    showFeedback("Failed to clear jobs", "error");
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
  updateServerStatus("connecting");

  try {
    eventSource = new EventSource(`${API_BASE}/stream`);

    eventSource.onopen = () => {
      isConnecting = false;
      updateServerStatus(true);
      console.log("SSE connected");
    };

    eventSource.onerror = (err) => {
      console.error("SSE error:", err);
      isConnecting = false;
      updateServerStatus(false);

      // Close and attempt reconnect after delay
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }

      setTimeout(connectSSE, 5000);
    };

    // Job creation event
    eventSource.addEventListener("create", (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.job) {
          addOrUpdateJob(data.job);
        }
      } catch (e) {
        console.error("Parse create event:", e);
      }
    });

    // Job update event
    eventSource.addEventListener("update", (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.job) {
          addOrUpdateJob(data.job);
        }
      } catch (e) {
        console.error("Parse update event:", e);
      }
    });

    // Job deletion event
    eventSource.addEventListener("delete", (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.job?.id) {
          removeJob(data.job.id);
        }
      } catch (e) {
        console.error("Parse delete event:", e);
      }
    });
  } catch (err) {
    console.error("SSE connect error:", err);
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

  // Keep only last 20 jobs in the extension
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
  // Filter to show only active/recent jobs
  const activeJobs = jobs.filter((j) => j.state === 0 || j.state === 1);
  const recentJobs = jobs
    .filter((j) => j.state !== 0 && j.state !== 1)
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
      const truncatedInput = truncate(job.input || "", 50);
      const isActive = job.state === 0 || job.state === 1;

      return `
      <div class="job-item" data-id="${job.id}">
        <div class="job-status ${statusClass}"></div>
        <div class="job-details" data-action="open" data-id="${
          job.id
        }" style="cursor: pointer;" title="Open in web UI">
          <div class="job-command">${escapeHtml(job.command || "Unknown")}</div>
          <div class="job-input" title="${escapeHtml(
            job.input || ""
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
    .join("");
}

// Cancel a job
async function cancelJob(jobId) {
  try {
    const response = await fetch(`${API_BASE}/job/${jobId}/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    if (!response.ok) {
      throw new Error("Failed to cancel job");
    }
  } catch (err) {
    console.error("Cancel job error:", err);
    showFeedback("Failed to cancel job", "error");
  }
}

// Remove a job from the server
async function removeJobFromServer(jobId) {
  try {
    const response = await fetch(`${API_BASE}/job/${jobId}/remove`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    if (!response.ok) {
      throw new Error("Failed to remove job");
    }

    // Remove from local list
    removeJob(jobId);
  } catch (err) {
    console.error("Remove job error:", err);
    showFeedback("Failed to remove job", "error");
  }
}

// Open job detail in web UI
function openJobDetail(jobId) {
  chrome.tabs.create({ url: `${API_BASE}/job/${jobId}` });
}

// UI Helpers
function updateServerStatus(connected) {
  if (connected === "connecting") {
    elements.serverStatus.className = "status-indicator connecting";
    elements.serverStatus.title = "Connecting...";
  } else if (connected) {
    elements.serverStatus.className = "status-indicator connected";
    elements.serverStatus.title = "Connected to Shrike server";
  } else {
    elements.serverStatus.className = "status-indicator";
    elements.serverStatus.title = "Disconnected from server";
  }
}

function setLoading(loading) {
  elements.createBtn.disabled = loading;
  elements.createBtn.querySelector(".btn-text").style.display = loading
    ? "none"
    : "inline";
  elements.createBtn.querySelector(".btn-loading").style.display = loading
    ? "inline-flex"
    : "none";
}

function showFeedback(message, type) {
  elements.feedback.textContent = message;
  elements.feedback.className = `feedback ${type}`;
  elements.feedback.style.display = "flex";

  // Auto-hide after 5 seconds for success messages
  if (type === "success") {
    setTimeout(hideFeedback, 5000);
  }
}

function hideFeedback() {
  elements.feedback.style.display = "none";
}

// Utility functions
function getStatusClass(state) {
  switch (state) {
    case 0:
      return "pending";
    case 1:
      return "running";
    case 2:
      return "completed";
    case 3:
      return "cancelled";
    case 4:
      return "error";
    default:
      return "pending";
  }
}

function formatTimeAgo(timestamp) {
  if (!timestamp) return "";

  const date = new Date(timestamp);
  const now = new Date();
  const diff = Math.floor((now - date) / 1000);

  if (diff < 60) return "Just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function truncate(str, maxLen) {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen - 3) + "...";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function debounce(fn, ms) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), ms);
  };
}

// Cleanup on popup close
window.addEventListener("unload", () => {
  if (eventSource) {
    eventSource.close();
  }
});
