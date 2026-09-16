const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const pickerSection = document.getElementById("picker-section");
const canvas = document.getElementById("picker-canvas");
const ctx = canvas.getContext("2d");
const controlsSection = document.getElementById("controls-section");
const saberSlotsEl = document.getElementById("saber-slots");
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
const PREVIEW_READY_HINT =
  "Highlighted area is what will be tracked -- looks right? " + MANUAL_HINT;

// A "saber slot": one tracked object's points plus its own color/intensity/
// voice, matching the backend's per-saber styling (1-4 objects tracked
// together, each with its own look). points is [[x, y, label], ...].
function makeSaberSlot(color) {
  return {
    points: [], color: color || "red", intensity: 0.35, voice: "neutral",
    detectedMask: null, source: "manual",
  };
}

// Default colors offered to each new slot, in order -- cycles once all
// three named colors are used, since the color <select> only offers three
// but up to 4 sabers can be tracked at once. Purely a starting point; the
// user can change any slot's color independently.
const DEFAULT_SABER_COLORS = ["red", "blue", "green", "red"];
const MAX_SABERS = 4;

let jobId = null;
let frameImage = null;
let sabers = [makeSaberSlot(DEFAULT_SABER_COLORS[0])];
let activeSaberIndex = 0;
let hasRendered = false; // true once this job has completed at least one render
// Which frame the points on screen refer to. Automatic detection reports the
// frame where the object was easiest to find -- usually mid-swing, not frame
// 0 -- so this travels with the points to the tracker, which prompts there
// and propagates both ways. Shared across every saber slot: the backend
// requires all sabers in one job to share a prompt_frame.
let promptFrame = 0;
// What SAM2 would actually segment from the active slot's points right
// now, refreshed after every click/drag and after switching slots -- the
// safety net that would have caught a selection landing on the background
// instead of the object before spending a full render on it. null until
// the first preview request for the current point set resolves.
let selectionPreview = null;
// Ignore a /preview_mask response that arrives after a newer gesture (or a
// slot switch) has already moved on from the points it was computed for.
let previewGeneration = 0;
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
    sabers: sabers.map((saber) => ({
      points: saber.points,
      prompt_frame: promptFrame,
      color: saber.color,
      intensity: saber.intensity,
      voice: saber.voice,
      source: saber.source,
    })),
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
  sabers = [makeSaberSlot(DEFAULT_SABER_COLORS[0])];
  activeSaberIndex = 0;
  hasRendered = false;
  promptFrame = 0;
  selectionPreview = null;
  previewGeneration++;
  submitButton.hidden = false;
  submitButton.disabled = true;
  rerenderButton.hidden = true;
  resultSection.hidden = true;
  syncControlsToActiveSaber();
  renderSaberSlots();

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
  const masks = await Promise.all(data.proposals.map((p) => loadImage(p.mask_url)));
  if (startedFor !== jobId) return;

  sabers = data.proposals.map((proposal, i) => ({
    ...makeSaberSlot(DEFAULT_SABER_COLORS[i]),
    points: proposal.points,
    detectedMask: masks[i],
    source: data.source,
  }));
  activeSaberIndex = 0;

  await showFrame(data.frame_url, data.width, data.height);
  syncControlsToActiveSaber();
  renderSaberSlots();

  if (data.source === "vlm") {
    pickerHint.textContent = data.proposals.length > 1
      ? `Found ${data.proposals.length} objects via AI. Render them, or click any object yourself to override.`
      : "Found it via AI. Render it, or click the object yourself to override.";
  } else {
    pickerHint.textContent =
      `Found 1 object via motion detection (AI detection unavailable). ` +
      "Render it, add more sabers manually if there are others, or click to override.";
  }
}

function redrawPoints() {
  if (frameImage) ctx.drawImage(frameImage, 0, 0);
  const activeDetectedMask = sabers[activeSaberIndex].detectedMask;
  if (activeDetectedMask) {
    ctx.drawImage(activeDetectedMask, 0, 0, canvas.width, canvas.height);
  } else if (selectionPreview) {
    ctx.drawImage(selectionPreview, 0, 0, canvas.width, canvas.height);
  }
  sabers.forEach((saber, i) => {
    const active = i === activeSaberIndex;
    for (const [x, y, label] of saber.points) {
      ctx.beginPath();
      ctx.arc(x, y, active ? 6 : 4, 0, 2 * Math.PI);
      ctx.fillStyle = label === 1
        ? (active ? "lime" : "rgba(50, 205, 50, 0.45)")
        : (active ? "red" : "rgba(255, 60, 60, 0.45)");
      ctx.fill();
    }
  });
  if (dragStart && dragCurrent) {
    ctx.beginPath();
    ctx.moveTo(dragStart.x, dragStart.y);
    ctx.lineTo(dragCurrent.x, dragCurrent.y);
    ctx.strokeStyle = dragStart.label === 1 ? "lime" : "red";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  updateSubmitState();
}

function updateSubmitState() {
  submitButton.disabled = !sabers.every((saber) => saber.points.some(([, , label]) => label === 1));
}

// Pushes the active slot's color/intensity/voice into the shared controls
// so they show (and, on change, edit) whichever saber is selected.
function syncControlsToActiveSaber() {
  const saber = sabers[activeSaberIndex];
  colorSelect.value = saber.color;
  intensityInput.value = saber.intensity;
  voiceSelect.value = saber.voice;
  updateAccentColor();
}

function renderSaberSlots() {
  saberSlotsEl.innerHTML = "";
  sabers.forEach((saber, i) => {
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = "saber-pill" + (i === activeSaberIndex ? " active" : "");
    pill.style.setProperty("--pill-color", (ACCENT_COLORS[saber.color] || ACCENT_COLORS.red)[0]);
    pill.textContent = `Saber ${i + 1}`;
    pill.addEventListener("click", () => setActiveSaber(i));

    if (sabers.length > 1 && !hasRendered) {
      const remove = document.createElement("span");
      remove.className = "saber-pill-remove";
      remove.textContent = "×";
      remove.title = "Remove this saber";
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        removeSaber(i);
      });
      pill.appendChild(remove);
    }
    saberSlotsEl.appendChild(pill);
  });

  if (sabers.length < MAX_SABERS && !hasRendered) {
    const addButton = document.createElement("button");
    addButton.type = "button";
    addButton.id = "add-saber-button";
    addButton.textContent = "+ Add saber";
    addButton.addEventListener("click", addSaber);
    saberSlotsEl.appendChild(addButton);
  }
}

function setActiveSaber(i) {
  activeSaberIndex = i;
  syncControlsToActiveSaber();
  renderSaberSlots();
  selectionPreview = null;
  redrawPoints();
  if (sabers[i].points.some(([, , label]) => label === 1)) {
    updateSelectionPreview();
  } else if (!sabers[i].detectedMask) {
    pickerHint.textContent = MANUAL_HINT;
  }
}

function addSaber() {
  if (sabers.length >= MAX_SABERS) return;
  sabers.push(makeSaberSlot(DEFAULT_SABER_COLORS[sabers.length]));
  setActiveSaber(sabers.length - 1);
}

function removeSaber(i) {
  if (sabers.length <= 1) return;
  sabers.splice(i, 1);
  setActiveSaber(Math.min(activeSaberIndex, sabers.length - 1));
}

// How many include/exclude points a drawn line becomes, evenly spread along
// it -- mirrors detect.py's own _points_on_axis technique for spreading
// prompts along an object's length rather than clustering them at one end.
const LINE_POINT_COUNT = 5;
// A press-and-release shorter than this (in canvas pixels) is a plain
// click, not a drawn line -- so "click a dot" keeps working exactly as
// before, and only a real drag produces multiple points.
const CLICK_VS_DRAG_THRESHOLD_PX = 5;

// Re-segments the current points on `promptFrame` and shows the result as
// an overlay, so a selection that would grab the wrong thing (e.g. the
// background instead of a thin object) is visible immediately instead of
// only after a full track-and-render. Fails open: a network/SAM2 error
// here just falls back to the plain hint, it never blocks submitting.
async function updateSelectionPreview() {
  const startedFor = jobId;
  const generation = ++previewGeneration;
  const points = sabers[activeSaberIndex].points;
  if (!points.some(([, , label]) => label === 1)) {
    selectionPreview = null;
    redrawPoints();
    return;
  }
  pickerHint.textContent = "Previewing selection...";
  let data;
  try {
    const resp = await fetch(`/api/jobs/${jobId}/preview_mask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points, prompt_frame: promptFrame }),
    });
    if (!resp.ok) throw new Error("preview failed");
    data = await resp.json();
  } catch {
    if (startedFor === jobId && generation === previewGeneration) pickerHint.textContent = MANUAL_HINT;
    return;
  }
  if (startedFor !== jobId || generation !== previewGeneration) return;
  const image = await loadImage(`${data.mask_url}?t=${Date.now()}`);
  if (startedFor !== jobId || generation !== previewGeneration) return;
  selectionPreview = image;
  redrawPoints();
  pickerHint.textContent = image ? PREVIEW_READY_HINT : MANUAL_HINT;
}

function pointsAlongLine(x1, y1, x2, y2, count) {
  const pts = [];
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1);
    pts.push([Math.round(x1 + (x2 - x1) * t), Math.round(y1 + (y2 - y1) * t)]);
  }
  return pts;
}

function canvasCoords(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.round((event.clientX - rect.left) * (canvas.width / rect.width)),
    y: Math.round((event.clientY - rect.top) * (canvas.height / rect.height)),
  };
}

function clampedCanvasCoords(event) {
  const rect = canvas.getBoundingClientRect();
  const rawX = (event.clientX - rect.left) * (canvas.width / rect.width);
  const rawY = (event.clientY - rect.top) * (canvas.height / rect.height);
  return {
    x: Math.round(Math.min(Math.max(rawX, 0), canvas.width)),
    y: Math.round(Math.min(Math.max(rawY, 0), canvas.height)),
  };
}

// While a gesture is in progress: dragStart carries the press position and
// which kind of point it will place (Shift = exclude, matching a plain
// shift-click); dragCurrent is the live position, redrawn on every move so
// the in-progress line is visible before release.
let dragStart = null;
let dragCurrent = null;

canvas.addEventListener("mousedown", (event) => {
  event.preventDefault(); // don't let the drag select page text
  const { x, y } = canvasCoords(event);
  dragStart = { x, y, label: event.shiftKey ? 0 : 1 };
  dragCurrent = { x, y };
});

canvas.addEventListener("mousemove", (event) => {
  if (!dragStart) return;
  dragCurrent = canvasCoords(event);
  redrawPoints();
});

// On window, not the canvas: a drag that ends outside the canvas (a real
// possibility -- the object may run close to the frame's edge) must still
// finalize, clamped back onto the canvas, rather than leaving the gesture
// stuck open.
window.addEventListener("mouseup", (event) => {
  if (!dragStart) return;
  const end = clampedCanvasCoords(event);
  const saber = sabers[activeSaberIndex];

  // The first gesture on a slot discards a detected proposal rather than
  // adding to it. A user correcting a detection disagrees with it, and
  // keeping it would keep whatever was wrong -- and give SAM2 two
  // contradictory prompts if the mask was on the wrong object. promptFrame
  // stays as it is: the frame on screen is still the frame these
  // coordinates refer to.
  if (saber.detectedMask) {
    saber.detectedMask = null;
    saber.source = "manual";
    saber.points = [];
    pickerHint.textContent = MANUAL_HINT;
  }

  const dragDistance = Math.hypot(end.x - dragStart.x, end.y - dragStart.y);
  if (dragDistance < CLICK_VS_DRAG_THRESHOLD_PX) {
    saber.points.push([dragStart.x, dragStart.y, dragStart.label]);
  } else {
    for (const [x, y] of pointsAlongLine(dragStart.x, dragStart.y, end.x, end.y, LINE_POINT_COUNT)) {
      saber.points.push([x, y, dragStart.label]);
    }
  }

  dragStart = null;
  dragCurrent = null;
  redrawPoints();
  updateSelectionPreview();
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
      renderSaberSlots(); // locks the saber count -- only color/intensity/voice are re-renderable
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

colorSelect.addEventListener("change", () => {
  sabers[activeSaberIndex].color = colorSelect.value;
  updateAccentColor();
  renderSaberSlots();
});
intensityInput.addEventListener("input", () => {
  sabers[activeSaberIndex].intensity = parseFloat(intensityInput.value);
});
voiceSelect.addEventListener("change", () => {
  sabers[activeSaberIndex].voice = voiceSelect.value;
});
updateAccentColor();
setStep("upload");
