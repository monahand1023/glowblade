const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const pickerSection = document.getElementById("picker-section");
const canvas = document.getElementById("picker-canvas");
const ctx = canvas.getContext("2d");
const colorSelect = document.getElementById("color-select");
const intensityInput = document.getElementById("intensity-input");
const submitButton = document.getElementById("submit-button");
const progressSection = document.getElementById("progress-section");
const progressFill = document.getElementById("progress-fill");
const progressLabel = document.getElementById("progress-label");
const resultSection = document.getElementById("result-section");
const resultPlayer = document.getElementById("result-player");
const downloadLink = document.getElementById("download-link");
const errorMessage = document.getElementById("error-message");

let jobId = null;
let frameImage = null;
let points = []; // [x, y, label]

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

  frameImage = new Image();
  frameImage.onload = () => {
    canvas.width = data.width;
    canvas.height = data.height;
    ctx.drawImage(frameImage, 0, 0);
    pickerSection.hidden = false;
  };
  frameImage.src = data.frame0_url;
}

function redrawPoints() {
  ctx.drawImage(frameImage, 0, 0);
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
  points.push([x, y, label]);
  redrawPoints();
});

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});
dropzone.addEventListener("dragover", (event) => event.preventDefault());
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  if (event.dataTransfer.files[0]) uploadFile(event.dataTransfer.files[0]);
});

submitButton.addEventListener("click", async () => {
  clearError();
  const resp = await fetch(`/api/jobs/${jobId}/points`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points,
      color: colorSelect.value,
      intensity: parseFloat(intensityInput.value),
    }),
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    showError(body.detail || "Could not start render");
    return;
  }
  pickerSection.hidden = true;
  progressSection.hidden = false;
  listenForProgress();
});

function listenForProgress() {
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.stage === "done") {
      source.close();
      progressSection.hidden = true;
      resultSection.hidden = false;
      resultPlayer.src = data.result_url;
      downloadLink.href = data.result_url;
    } else if (data.stage === "error") {
      source.close();
      progressSection.hidden = true;
      showError(data.message);
    } else {
      progressFill.style.width = `${data.pct}%`;
      let label = `${data.stage}: ${data.message} (${Math.round(data.pct)}%)`;
      if (data.elapsed != null) {
        label += ` — ${formatDuration(data.elapsed)} elapsed`;
        if (data.eta != null) label += `, ~${formatDuration(data.eta)} left`;
      }
      progressLabel.textContent = label;
    }
  };
  source.onerror = () => {
    source.close();
    showError("Lost connection to the server while rendering.");
  };
}
