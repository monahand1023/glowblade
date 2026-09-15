const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const pickerSection = document.getElementById("picker-section");
const canvas = document.getElementById("picker-canvas");
const ctx = canvas.getContext("2d");
const controlsSection = document.getElementById("controls-section");
const colorSelect = document.getElementById("color-select");
const intensityInput = document.getElementById("intensity-input");
const voiceSelect = document.getElementById("voice-select");
const bladeExtendInput = document.getElementById("blade-extend-input");
const submitButton = document.getElementById("submit-button");
const rerenderButton = document.getElementById("rerender-button");
const progressSection = document.getElementById("progress-section");
const progressFill = document.getElementById("progress-fill");
const progressLabel = document.getElementById("progress-label");
const previewImage = document.getElementById("preview-image");
const resultSection = document.getElementById("result-section");
const resultPlayer = document.getElementById("result-player");
const downloadLink = document.getElementById("download-link");
const errorMessage = document.getElementById("error-message");
const pickerHint = document.getElementById("picker-hint");
const stepItems = document.querySelectorAll("#steps li");

const STEP_ORDER = ["upload", "confirm", "render", "result"];

function setStep(current) {
  const idx = STEP_ORDER.indexOf(current);
  stepItems.forEach((li) => {
    const stepIdx = STEP_ORDER.indexOf(li.dataset.step);
    li.classList.toggle("done", stepIdx < idx);
    li.classList.toggle("active", stepIdx === idx);
  });
}

// Saber color -> UI accent. Purely cosmetic (page chrome), independent of
// the exact BGR values the renderer uses for the glow itself.
const ACCENT_COLORS = {
  red: ["#ff3b3b", "255, 59, 59"],
  blue: ["#3b82f6", "59, 130, 246"],
  green: ["#22c55e", "34, 197, 94"],
};

function updateAccentColor() {
  const [hex, rgb] = ACCENT_COLORS[colorSelect.value] || ACCENT_COLORS.red;
  document.documentElement.style.setProperty("--accent", hex);
  document.documentElement.style.setProperty("--accent-rgb", rgb);
}

const MANUAL_HINT =
  "Click the object to track. Shift-click to exclude a spot (e.g. a hand).";

let jobId = null;
let frameImage = null;
let points = []; // [x, y, label]
let hasRendered = false; // true once this job has completed at least one render
// Which frame the points on screen refer to. Automatic detection reports the
// frame where the object was easiest to find -- usually mid-swing, not frame
// 0 -- so this travels with the points to the tracker, which prompts there
// and propagates both ways.
let promptFrame = 0;
// The detected mask, drawn under the points until the user overrides it.
let detectOverlay = null;
// Throttles /preview polling to roughly 1/s -- the glow stage emits one SSE
// progress event per rendered frame, and fetching a preview image on every
// single one of those would mean one HTTP request per frame.
let lastPreviewFetch = 0;
const PREVIEW_MIN_INTERVAL_MS = 800;

// Mirrors format_duration() in lightsaber_fx/progress.py.
function formatDuration(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

function showError(message) {
  errorMessage.hidden = false;
  errorMessage.textContent = message;
}

function clearError() {
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

function renderRequestBody() {
  return {
    sabers: [{
      points,
      prompt_frame: promptFrame,
      color: colorSelect.value,
      intensity: parseFloat(intensityInput.value),
      voice: voiceSelect.value,
    }],
    blade_extend: bladeExtendInput.checked,
  };
}

async function uploadFile(file) {
  clearError();
  const formData = new FormData();
  formData.append("file", file);
  const resp = await fetch("/api/upload", { method: "POST", body: formData });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    showError(body.detail || "Upload failed");
    return;
  }
  const data = await resp.json();
  jobId = data.job_id;
  points = [];
  hasRendered = false;
  promptFrame = 0;
  detectOverlay = null;
  submitButton.hidden = false;
  submitButton.disabled = true;
  rerenderButton.hidden = true;
  resultSection.hidden = true;

  await showFrame(data.frame0_url, data.width, data.height);
  detect();
}

// Loads `url` into the picker canvas and resolves once it is drawn.
function showFrame(url, width, height) {
  return new Promise((resolve) => {
    frameImage = new Image();
    frameImage.onload = () => {
      canvas.width = width;
      canvas.height = height;
      redrawPoints();
      pickerSection.hidden = false;
      controlsSection.hidden = false;
      setStep("confirm");
      resolve();
    };
    frameImage.onerror = () => resolve();
    frameImage.src = url;
  });
}

// Shows the glow stage's latest in-progress frame, throttled to avoid one
// request per rendered frame. Loads off-DOM first so a 404 (no frame
// written yet) never flashes a broken-image icon in the visible <img>.
function maybeUpdatePreview(stage) {
  if (stage !== "glow") {
    previewImage.hidden = true;
    return;
  }
  const now = Date.now();
  if (now - lastPreviewFetch < PREVIEW_MIN_INTERVAL_MS) return;
  lastPreviewFetch = now;
  const probe = new Image();
  probe.onload = () => {
    previewImage.src = probe.src;
    previewImage.hidden = false;
  };
  probe.src = `/api/jobs/${jobId}/preview?t=${now}`;
}

function loadImage(url) {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => resolve(null);
    image.src = url;
  });
}

// Asks the server to find the swung object, and shows what it found for the
// user to accept or override. A failure here is not an error the user needs
// to see: it just means they click the object themselves, which is the
// behaviour they had before this existed.
async function detect() {
  const startedFor = jobId;
  pickerHint.textContent = "Looking for the swung object...";
  let data;
  try {
    const resp = await fetch(`/api/jobs/${jobId}/detect`, { method: "POST" });
    if (!resp.ok) throw new Error("detect failed");
    data = await resp.json();
  } catch {
    pickerHint.textContent = MANUAL_HINT;
    return;
  }
  // The user may have dropped another file while this was running.
  if (startedFor !== jobId) return;
  if (!data.found) {
    pickerHint.textContent = `Couldn't find it automatically. ${MANUAL_HINT}`;
    return;
  }

  promptFrame = data.frame_index;
  points = data.points;
  detectOverlay = await loadImage(data.mask_url);
  if (startedFor !== jobId) return;
  await showFrame(data.frame_url, data.width, data.height);
  pickerHint.textContent =
    `Found it in frame ${data.frame_index + 1} (elongation ${data.elongation}). ` +
    "Render it, or click the object yourself to override.";
}

function redrawPoints() {
  if (frameImage) ctx.drawImage(frameImage, 0, 0);
  if (detectOverlay) ctx.drawImage(detectOverlay, 0, 0, canvas.width, canvas.height);
  for (const [x, y, label] of points) {
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = label === 1 ? "lime" : "red";
    ctx.fill();
  }
  submitButton.disabled = !points.some(([, , label]) => label === 1);
}

canvas.addEventListener("click", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((event.clientX - rect.left) * (canvas.width / rect.width));
  const y = Math.round((event.clientY - rect.top) * (canvas.height / rect.height));
  const label = event.shiftKey ? 0 : 1;
  // The first click discards the detected proposal rather than adding to it.
  // A user correcting a detection disagrees with it, and keeping it would
  // keep whatever was wrong -- and give SAM2 two contradictory prompts if the
  // mask was on the wrong object. promptFrame stays as it is: the frame on
  // screen is still the frame these coordinates refer to.
  if (detectOverlay) {
    detectOverlay = null;
    points = [];
    pickerHint.textContent = MANUAL_HINT;
  }
  points.push([x, y, label]);
  redrawPoints();
});

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  if (event.dataTransfer.files[0]) uploadFile(event.dataTransfer.files[0]);
});

submitButton.addEventListener("click", async () => {
  clearError();
  const resp = await fetch(`/api/jobs/${jobId}/points`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(renderRequestBody()),
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    showError(body.detail || "Could not start render");
    return;
  }
  pickerSection.hidden = true;
  submitButton.hidden = true;
  rerenderButton.hidden = true;
  progressSection.hidden = false;
  previewImage.hidden = true;
  lastPreviewFetch = 0;
  setStep("render");
  listenForProgress();
});

// Re-renders the same job (glow/audio/mux only -- no re-tracking) with
// whatever color/intensity/voice/blade-extend the controls currently hold.
// No points are sent: the job already has its tracking masks.
rerenderButton.addEventListener("click", async () => {
  clearError();
  const resp = await fetch(`/api/jobs/${jobId}/rerender`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(renderRequestBody()),
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    showError(body.detail || "Could not start re-render");
    return;
  }
  resultSection.hidden = true;
  rerenderButton.hidden = true;
  progressSection.hidden = false;
  previewImage.hidden = true;
  lastPreviewFetch = 0;
  setStep("render");
  listenForProgress();
});

function listenForProgress() {
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.stage === "done") {
      source.close();
      hasRendered = true;
      progressSection.hidden = true;
      previewImage.hidden = true;
      resultSection.hidden = false;
      rerenderButton.hidden = false;
      resultPlayer.src = data.result_url + `?t=${Date.now()}`; // bust the cache: same URL, new content
      downloadLink.href = data.result_url;
      setStep("result");
    } else if (data.stage === "error") {
      source.close();
      progressSection.hidden = true;
      previewImage.hidden = true;
      showError(data.message);
      // A render (first or re-) failing shouldn't force starting over: bring
      // back whichever action was available before this attempt.
      if (hasRendered) {
        resultSection.hidden = false;
        rerenderButton.hidden = false;
        setStep("result");
      } else {
        submitButton.hidden = false;
        setStep("confirm");
      }
    } else {
      progressFill.style.width = `${data.pct}%`;
      let label = `${data.stage}: ${data.message} (${Math.round(data.pct)}%)`;
      if (data.elapsed != null) {
        label += ` — ${formatDuration(data.elapsed)} elapsed`;
        if (data.eta != null) label += `, ~${formatDuration(data.eta)} left`;
      }
      progressLabel.textContent = label;
      maybeUpdatePreview(data.stage);
    }
  };
  source.onerror = () => {
    source.close();
    showError("Lost connection to the server while rendering.");
  };
}

colorSelect.addEventListener("change", updateAccentColor);
updateAccentColor();
setStep("upload");
