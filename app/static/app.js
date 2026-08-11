const $ = (id) => document.getElementById(id);
const state = {
  file: null, tracker: null, running: false, ws: null, result: null,
  ff: null, dets: [], drawn: [], mode: "instance",
};

async function init() {
  const r = await fetch("/api/trackers").then((r) => r.json());
  $("devicePill").textContent = "device: " + r.device;
  const grid = $("trackerGrid");
  r.trackers.forEach((t, i) => {
    const c = document.createElement("div");
    c.className = "tcard";
    c.innerHTML = `<div class="name">${t.name.split(" (")[0]}</div>
      <div class="tag">${t.name.match(/\((.*?)\)/)?.[1] || ""}</div>
      ${t.needs_reid ? '<span class="reid">ReID</span>' : ""}`;
    c.onclick = () => selectTracker(t.key, c);
    grid.appendChild(c);
    if (i === 0) selectTracker(t.key, c);
  });
}

function selectTracker(key, el) {
  if (state.running) return;
  state.tracker = key;
  document.querySelectorAll(".tcard").forEach((c) => c.classList.remove("active"));
  el.classList.add("active");
  $("algoBadge").textContent = el.querySelector(".name").textContent;
  updateRunState();
}

// ---- upload ----
const dz = $("dropzone"), fi = $("fileInput");
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", (e) => {
  e.preventDefault(); dz.classList.remove("drag");
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});
fi.addEventListener("change", () => fi.files[0] && uploadFile(fi.files[0]));

async function uploadFile(file) {
  $("dzText").textContent = "Uploading " + file.name + " …";
  const fd = new FormData(); fd.append("file", file);
  const r = await fetch("/api/upload", { method: "POST", body: fd }).then((r) => r.json());
  state.file = r.file;
  $("dzText").innerHTML = "✓ <u>" + r.name + "</u> — ready";
  $("statusLine").textContent = "Detecting objects in first frame…";
  await loadFirstFrame();
  updateRunState();
}

// ---- first-frame detection + selection canvas ----
const canvas = $("selCanvas");
const ctx = canvas.getContext("2d");

async function loadFirstFrame() {
  const r = await fetch("/api/first-frame", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file: state.file }),
  }).then((r) => r.json());
  if (r.error) { $("statusLine").textContent = "Error: " + r.error; return; }

  const img = new Image();
  await new Promise((res) => { img.onload = res; img.src = r.image; });
  state.ff = { img, w: r.width, h: r.height };
  state.dets = r.detections.map((d) => ({ ...d, selected: false }));
  state.drawn = [];
  canvas.width = r.width; canvas.height = r.height;

  $("placeholder").style.display = "none";
  $("stream").style.display = "none";
  canvas.style.display = "block";
  $("selTools").hidden = false;
  renderSel();
  $("statusLine").textContent =
    `${state.dets.length} objects found. Select some, or run to track everything.`;
  updateSummary();
}

function label(x, y, text, bg) {
  ctx.font = "600 16px Inter, sans-serif";
  const w = ctx.measureText(text).width + 10;
  ctx.fillStyle = bg;
  ctx.fillRect(x, Math.max(0, y - 22), w, 22);
  ctx.fillStyle = "#08101f";
  ctx.fillText(text, x + 5, Math.max(14, y - 6));
}

let drag = null;   // {x0,y0,x1,y1,moved}

function renderSel() {
  if (!state.ff) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.ff.img, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  state.dets.forEach((d) => {
    const [x1, y1, x2, y2] = d.box;
    ctx.strokeStyle = d.selected ? "#34e5c4" : "rgba(120,150,255,0.7)";
    ctx.fillStyle = d.selected ? "rgba(52,229,196,0.20)" : "rgba(120,150,255,0.05)";
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (d.selected) label(x1, y1, d.name, "#34e5c4");
  });
  state.drawn.forEach((b) => {
    const [x1, y1, x2, y2] = b;
    ctx.strokeStyle = "#ffb443"; ctx.fillStyle = "rgba(255,180,67,0.18)";
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    label(x1, y1, "drawn", "#ffb443");
  });
  if (drag && drag.moved) {
    ctx.strokeStyle = "#ffb443"; ctx.setLineDash([6, 4]);
    ctx.strokeRect(drag.x0, drag.y0, drag.x1 - drag.x0, drag.y1 - drag.y0);
    ctx.setLineDash([]);
  }
}

function toNative(e) {
  const rect = canvas.getBoundingClientRect();
  return [(e.clientX - rect.left) * (state.ff.w / rect.width),
          (e.clientY - rect.top) * (state.ff.h / rect.height)];
}

canvas.addEventListener("mousedown", (e) => {
  if (state.running || !state.ff) return;
  const [x, y] = toNative(e);
  drag = { x0: x, y0: y, x1: x, y1: y, moved: false };
});
canvas.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const [x, y] = toNative(e);
  drag.x1 = x; drag.y1 = y;
  if (Math.abs(x - drag.x0) > 6 && Math.abs(y - drag.y0) > 6) drag.moved = true;
  renderSel();
});
canvas.addEventListener("mouseup", (e) => {
  if (!drag) return;
  if (drag.moved) {
    const box = [Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1),
                 Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)];
    state.drawn.push(box.map((v) => Math.round(v)));
  } else {
    const [x, y] = [drag.x0, drag.y0];
    let hit = -1, area = Infinity;
    state.dets.forEach((d, i) => {
      const [x1, y1, x2, y2] = d.box;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        const a = (x2 - x1) * (y2 - y1);
        if (a < area) { area = a; hit = i; }
      }
    });
    if (hit >= 0) state.dets[hit].selected = !state.dets[hit].selected;
  }
  drag = null;
  renderSel(); updateSummary();
});

$("selAllBtn").onclick = () => { state.dets.forEach((d) => d.selected = true); renderSel(); updateSummary(); };
$("clearBtn").onclick = () => { state.dets.forEach((d) => d.selected = false); state.drawn = []; renderSel(); updateSummary(); };

$("modeToggle").querySelectorAll("button").forEach((b) => {
  b.onclick = () => {
    if (state.running) return;
    state.mode = b.dataset.mode;
    $("modeToggle").querySelectorAll("button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    updateSummary();
  };
});

function buildSelection() {
  const selDets = state.dets.filter((d) => d.selected);
  if (state.mode === "class") {
    const classes = [...new Set(selDets.map((d) => d.cls))];
    const seed = state.drawn.slice();
    if (!classes.length && !seed.length) return { mode: "all", classes: [], seed_boxes: [] };
    return { mode: "class", classes, seed_boxes: seed };
  }
  const seed = selDets.map((d) => d.box).concat(state.drawn);
  if (!seed.length) return { mode: "all", classes: [], seed_boxes: [] };
  return { mode: "instance", classes: [], seed_boxes: seed };
}

function updateSummary() {
  const sel = state.dets.filter((d) => d.selected).length, drawn = state.drawn.length;
  if (!sel && !drawn) { $("selSummary").textContent = "Nothing selected — will track everything."; return; }
  const scope = state.mode === "class"
    ? `classes of ${sel} selected object${sel === 1 ? "" : "s"}`
    : `${sel} selected + ${drawn} drawn instance${sel + drawn === 1 ? "" : "s"}`;
  $("selSummary").textContent = "Tracking " + scope + ".";
}

function updateRunState() {
  $("runBtn").disabled = !(state.file && state.tracker) || state.running;
}

// ---- run ----
$("runBtn").addEventListener("click", () => (state.running ? stop() : start()));
$("downloadBtn").addEventListener("click", () => {
  if (state.result) window.location = "/api/result/" + state.result;
});

function start() {
  state.running = true;
  $("runBtn").textContent = "Stop"; $("runBtn").classList.add("running");
  $("downloadBtn").hidden = true;
  canvas.style.display = "none"; $("selTools").hidden = true;
  $("placeholder").style.display = "none";
  $("statusLine").textContent = "Connecting…";

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/track`);
  state.ws = ws;
  ws.onopen = () => ws.send(JSON.stringify({
    file: state.file, tracker: state.tracker, selection: buildSelection(),
  }));
  ws.onmessage = (ev) => handle(JSON.parse(ev.data));
  ws.onclose = () => finish();
  ws.onerror = () => { $("statusLine").textContent = "Connection error."; };
}

function stop() { if (state.ws) state.ws.close(); finish(); }

function finish() {
  state.running = false;
  $("runBtn").textContent = "Start Tracking"; $("runBtn").classList.remove("running");
  updateRunState();
  if (state.result) $("downloadBtn").hidden = false;
}

function handle(m) {
  if (m.type === "status") { $("statusLine").textContent = m.message; return; }
  if (m.type === "start") { state.result = m.result; $("statusLine").textContent = "Tracking…"; return; }
  if (m.type === "error") { $("statusLine").textContent = "Error: " + m.message; return; }
  if (m.type === "done") { $("statusLine").textContent = "Done ✓ result saved."; return; }
  if (m.type !== "frame") return;

  const img = $("stream");
  img.style.display = "block"; img.src = m.image;

  const s = m.stats;
  $("mFps").textContent = s.fps;
  $("mDet").innerHTML = s.detect_ms + "<i>ms</i>";
  $("mTrk").innerHTML = s.track_ms + "<i>ms</i>";
  $("mObj").textContent = m.tracks;
  $("mCpu").innerHTML = s.cpu_percent + "<i>%</i>";
  $("mRam").innerHTML = s.ram_mb + "<i>MB</i>";
  if (s.gpu) {
    $("mGpu").innerHTML = s.gpu.util_percent + '<i>% · ' +
      Math.round(s.gpu.mem_used_mb) + "/" + Math.round(s.gpu.mem_total_mb) + " MB</i>";
    $("gpuBar").style.width = s.gpu.util_percent + "%";
  } else { $("mGpu").textContent = "CPU only"; }

  if (m.total) {
    $("progressFill").style.width = (100 * m.frame / m.total) + "%";
    $("frameCount").textContent = m.frame + " / " + m.total;
  } else { $("frameCount").textContent = m.frame; }
}

init();
