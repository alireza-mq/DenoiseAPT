(() => {
  "use strict";

  const API = {
    health: "/api/health",
    cases: "/api/cases",
    analyze: "/api/analyze",
    intervene: "/api/intervene"
  };

  const DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024;
  const SIGNAL_ORDER = [
    "reference", "observed", "classical", "cgan",
    "denoiseapt", "hybrid", "automatic", "approved"
  ];
  const SCORE_ORDER = [
    "observed", "cgan", "denoiseapt", "hybrid", "automatic", "approved"
  ];

  const COLORS = {
    reference: "#586c75",
    observed: "#c84f53",
    classical: "#df912e",
    cgan: "#7456a6",
    denoiseapt: "#117b79",
    hybrid: "#8b5d13",
    automatic: "#1769a6",
    approved: "#3478b8",
    grid: "#e4ebec",
    axis: "#788a91",
    selection: "rgba(52, 120, 184, 0.15)",
    selectionLine: "#3478b8",
    anomaly: "rgba(200, 79, 83, 0.09)"
  };

  const METHOD_LABELS = {
    reference: "Clean reference",
    observed: "Corrupted observation",
    classical: "Moving-average filter (w=9)",
    cgan: "Matched conditional GAN",
    denoiseapt: "Generative repair candidate",
    hybrid: "Evaluated DenoiseAPT hybrid",
    automatic: "Established DenoiseAPT output",
    approved: "Current session signal"
  };

  const METHOD_CONTEXT = {
    reference: "evaluation only",
    hybrid: "separately evaluated configuration",
    automatic: "established live default",
    approved: "reversible session edit"
  };

  const state = {
    cases: [],
    upload: null,
    result: null,
    selection: null,
    view: null,
    drag: null,
    plotEntries: [],
    actionPending: false,
    maxRequestBytes: DEFAULT_MAX_REQUEST_BYTES
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const dom = {
    healthDot: $("#healthDot"), healthText: $("#healthText"),
    form: $("#analysisForm"), runButton: $("#runButton"), formError: $("#formError"),
    caseSelect: $("#caseSelect"), caseMeta: $("#caseMeta"), catalogFields: $("#catalogFields"),
    uploadFields: $("#uploadFields"), csvFile: $("#csvFile"), fileMeta: $("#fileMeta"),
    start: $("#windowStart"), length: $("#windowLength"), family: $("#corruptionFamily"),
    severity: $("#severity"), severityValue: $("#severityValue"), seed: $("#seed"),
    randomizeSeed: $("#randomizeSeed"), empty: $("#emptyState"), results: $("#results"),
    caseHeading: $("#caseHeading"), signalPlots: $("#signalPlots"), signalLegend: $("#signalLegend"),
    scoreLegend: $("#scoreLegend"), plotReadout: $("#plotReadout"), plotTemplate: $("#plotTemplate"),
    scoreCanvas: $("#scoreCanvas"), scoreTooltip: $("#scoreTooltip"),
    concernCanvas: $("#concernCanvas"), concernTooltip: $("#concernTooltip"),
    intervalLabel: $("#intervalLabel"), concernBadge: $("#concernBadge"),
    scoreCueValue: $("#scoreCueValue"), scoreCueBar: $("#scoreCueBar"),
    morphCueValue: $("#morphCueValue"), morphCueBar: $("#morphCueBar"),
    uncertaintyCueValue: $("#uncertaintyCueValue"), uncertaintyCueBar: $("#uncertaintyCueBar"),
    blendWeight: $("#blendWeight"), blendValue: $("#blendValue"), metricsBody: $("#metricsBody"),
    metricScope: $("#metricScope"), resetView: $("#resetView"), exportButton: $("#exportButton"),
    certificationCard: $("#certificationCard"), certificationBadge: $("#certificationBadge"),
    certificationSummary: $("#certificationSummary"), certificateMode: $("#certificateMode"),
    certificateDecision: $("#certificateDecision"), certificateRepairs: $("#certificateRepairs"),
    certificateLatency: $("#certificateLatency"), certificateScope: $("#certificateScope"),
    certificateRepairDetail: $("#certificateRepairDetail"),
    certificateWitnesses: $("#certificateWitnesses"),
    certificateLimitations: $("#certificateLimitations"), certificateAudit: $("#certificateAudit"),
    toastRegion: $("#toastRegion")
  };

  function setHealth(status, text) {
    dom.healthDot.className = `health-dot is-${status}`;
    dom.healthText.textContent = text;
  }

  async function fetchJSON(url, options = {}, timeoutMs = 120000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: { "Accept": "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) }
      });
      const contentType = response.headers.get("content-type") || "";
      const body = contentType.includes("application/json") ? await response.json() : { message: await response.text() };
      if (!response.ok) throw new Error(body.error || body.message || `Request failed (${response.status})`);
      return body;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The analysis service did not respond in time.");
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  async function initialize() {
    bindEvents();
    updateSeverity();
    updateBlend();
    await Promise.allSettled([checkHealth(), loadCases()]);
  }

  async function checkHealth() {
    try {
      const payload = await fetchJSON(API.health, {}, 6000);
      if (Number.isSafeInteger(payload.max_request_bytes) && payload.max_request_bytes > 0) {
        state.maxRequestBytes = payload.max_request_bytes;
      }
      const suffix = payload.device ? ` · ${payload.device}` : "";
      const automatic = payload.automatic_runtime_ready ? " · automatic controller ready" : "";
      setHealth("online", `Analysis service ready${suffix}${automatic}`);
    } catch (error) {
      setHealth("offline", "Analysis service unavailable");
      showFormError("The local analysis service is not available. Start the packaged server, then reload this page.");
    }
  }

  async function loadCases() {
    try {
      const payload = await fetchJSON(API.cases, {}, 12000);
      if (!payload || !Array.isArray(payload.cases)) throw new Error("The case catalog response is invalid.");
      state.cases = payload.cases;
      dom.caseSelect.replaceChildren();
      if (!state.cases.length) {
        dom.caseSelect.append(new Option("No packaged cases installed", ""));
        dom.caseMeta.textContent = "Prepare the packaged synthetic case or install the optional benchmark case.";
        return;
      }
      for (const item of state.cases) dom.caseSelect.append(new Option(item.name || item.id, item.id));
      dom.caseSelect.disabled = false;
      updateCaseMeta();
    } catch (error) {
      dom.caseSelect.replaceChildren(new Option("Case catalog unavailable", ""));
      dom.caseMeta.textContent = error.message;
    }
  }

  function bindEvents() {
    dom.form.addEventListener("submit", runAnalysis);
    $$('input[name="source"]').forEach(input => input.addEventListener("change", toggleSource));
    dom.caseSelect.addEventListener("change", updateCaseMeta);
    dom.csvFile.addEventListener("change", handleFileSelection);
    const drop = $(".file-drop");
    ["dragenter", "dragover"].forEach(type => drop.addEventListener(type, event => { event.preventDefault(); drop.classList.add("is-dragging"); }));
    ["dragleave", "drop"].forEach(type => drop.addEventListener(type, event => { event.preventDefault(); drop.classList.remove("is-dragging"); }));
    drop.addEventListener("drop", event => {
      const file = event.dataTransfer.files[0];
      if (file) parseCSVFile(file);
    });
    dom.severity.addEventListener("input", updateSeverity);
    dom.family.addEventListener("change", () => {
      const disabled = dom.family.value === "none";
      dom.severity.disabled = disabled;
      dom.severityValue.textContent = disabled ? "—" : Number(dom.severity.value).toFixed(2);
    });
    dom.randomizeSeed.addEventListener("click", () => { dom.seed.value = crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff; });
    dom.blendWeight.addEventListener("input", updateBlend);
    dom.resetView.addEventListener("click", resetView);
    dom.exportButton.addEventListener("click", exportResult);
    $$(".action-button").forEach(button => button.addEventListener("click", () => applyIntervention(button.dataset.action)));
    window.addEventListener("resize", debounce(renderAll, 120));
  }

  function toggleSource() {
    const source = $('input[name="source"]:checked').value;
    dom.catalogFields.hidden = source !== "catalog";
    dom.uploadFields.hidden = source !== "upload";
    hideFormError();
  }

  function updateCaseMeta() {
    const item = state.cases.find(c => String(c.id) === dom.caseSelect.value);
    if (!item) return;
    const details = [item.domain, finite(item.length) ? `${formatInteger(item.length)} samples` : null,
      finite(item.sample_rate) ? `${formatNumber(item.sample_rate)} Hz` : null,
      finite(item.anomaly_count) ? `${item.anomaly_count} labelled event${item.anomaly_count === 1 ? "" : "s"}` : null].filter(Boolean);
    dom.caseMeta.textContent = details.join(" · ") || "Packaged case ready.";
    if (finite(item.length)) {
      dom.start.max = Math.max(0, item.length - 1);
      dom.length.max = Math.min(2048, item.length);
      dom.length.value = Math.min(Number(dom.length.value), item.length);
    }
  }

  async function handleFileSelection() {
    const file = dom.csvFile.files[0];
    if (file) await parseCSVFile(file);
  }

  async function parseCSVFile(file) {
    try {
      if (!file.name.toLowerCase().endsWith(".csv")) throw new Error("Choose a .csv file.");
      if (file.size > state.maxRequestBytes) {
        throw new Error(`CSV files must be smaller than ${formatMiB(state.maxRequestBytes)} MiB.`);
      }
      dom.fileMeta.textContent = "Reading file…";
      const text = await file.text();
      const parsed = parseCSV(text);
      state.upload = { name: file.name, ...parsed };
      dom.fileMeta.textContent = `${file.name} · ${formatInteger(parsed.values.length)} valid samples · value column “${parsed.valueColumn}”`;
      dom.start.max = Math.max(0, parsed.values.length - 1);
      dom.length.max = Math.min(2048, parsed.values.length);
      dom.length.value = Math.min(Number(dom.length.value), parsed.values.length);
      hideFormError();
    } catch (error) {
      state.upload = null;
      dom.fileMeta.textContent = "The file could not be used.";
      showFormError(error.message);
    }
  }

  function parseCSV(text) {
    // Quoted commas and escaped quotes are supported; multiline quoted fields
    // are intentionally outside the demo parser's compact CSV contract.
    const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter(line => line.trim() !== "");
    if (lines.length < 2) throw new Error("The CSV must contain a header and at least one data row.");
    const rows = lines.map(parseCSVRow);
    const headers = rows[0].map((value, index) => value.trim() || `column_${index + 1}`);
    const body = rows.slice(1);
    const numericCounts = headers.map((_, col) => body.reduce((sum, row) => sum + (Number.isFinite(Number(row[col])) && row[col] !== "" ? 1 : 0), 0));
    const timePattern = /^(time|timestamp|date|datetime|index)$/i;
    const labelPattern = /^(label|labels|anomaly|is_anomaly|target)$/i;
    const candidateCols = numericCounts.map((count, index) => ({ count, index })).filter(x => !timePattern.test(headers[x.index]) && !labelPattern.test(headers[x.index])).sort((a, b) => b.count - a.count);
    const valueCol = candidateCols[0]?.count >= 2 ? candidateCols[0].index : numericCounts.indexOf(Math.max(...numericCounts));
    if (valueCol < 0 || numericCounts[valueCol] < 2) throw new Error("No numeric signal column was found.");
    const timeCol = headers.findIndex((header, index) => index !== valueCol && timePattern.test(header));
    const labelCol = headers.findIndex((header, index) => index !== valueCol && labelPattern.test(header));
    const values = [], timestamps = [], labels = [];
    for (let rowIndex = 0; rowIndex < body.length; rowIndex++) {
      const row = body[rowIndex];
      const value = Number(row[valueCol]);
      if (!Number.isFinite(value)) continue;
      values.push(value);
      if (timeCol >= 0) timestamps.push(row[timeCol] ?? String(values.length - 1));
      if (labelCol >= 0) {
        const label = Number(row[labelCol]);
        if (!Number.isFinite(label)) throw new Error(`Label column contains a non-numeric value at data row ${rowIndex + 1}.`);
        labels.push(label > 0 ? 1 : 0);
      }
    }
    if (values.length < 64) throw new Error("At least 64 finite signal values are required.");
    if (values.length > 250000) throw new Error("The CSV contains more than 250,000 samples. Select a smaller file.");
    return { values, timestamps: timeCol >= 0 ? timestamps : undefined, labels: labelCol >= 0 ? labels : undefined, valueColumn: headers[valueCol] };
  }

  function parseCSVRow(line) {
    const cells = []; let cell = ""; let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        if (quoted && line[i + 1] === '"') { cell += '"'; i++; } else quoted = !quoted;
      } else if (char === "," && !quoted) { cells.push(cell); cell = ""; }
      else cell += char;
    }
    cells.push(cell);
    return cells;
  }

  function updateSeverity() { dom.severityValue.textContent = Number(dom.severity.value).toFixed(2); }
  function updateBlend() { dom.blendValue.textContent = Number(dom.blendWeight.value).toFixed(2); }

  async function runAnalysis(event) {
    event.preventDefault();
    hideFormError();
    const source = $('input[name="source"]:checked').value;
    const start = Number(dom.start.value), length = Number(dom.length.value), seed = Number(dom.seed.value);
    if (!Number.isInteger(start) || start < 0) return showFormError("Start index must be a non-negative integer.");
    if (!Number.isInteger(length) || length < 64 || length > 2048) return showFormError("Window length must be between 64 and 2,048 samples.");
    if (!Number.isInteger(seed) || seed < 0) return showFormError("Random seed must be a non-negative integer.");
    if (source === "catalog" && !dom.caseSelect.value) return showFormError("Select an installed packaged case.");
    if (source === "upload" && !state.upload) return showFormError("Choose a valid CSV file first.");
    const sourceLength = source === "upload" ? state.upload.values.length : state.cases.find(c => String(c.id) === dom.caseSelect.value)?.length;
    if (finite(sourceLength) && start + length > sourceLength) return showFormError(`The requested window ends at ${start + length}, beyond the available ${sourceLength} samples.`);

    const payload = {
      case_id: source === "catalog" ? dom.caseSelect.value : undefined,
      upload: source === "upload" ? { name: state.upload.name, values: state.upload.values, timestamps: state.upload.timestamps, labels: state.upload.labels } : undefined,
      corruption: { family: dom.family.value, severity: dom.family.value === "none" ? 0 : Number(dom.severity.value), seed },
      window: { start, length }
    };
    const requestBody = JSON.stringify(payload);
    const requestBytes = new TextEncoder().encode(requestBody).byteLength;
    if (requestBytes > state.maxRequestBytes) {
      return showFormError(
        `The encoded request is ${formatMiB(requestBytes)} MiB; the server limit is ${formatMiB(state.maxRequestBytes)} MiB.`
      );
    }
    setRunning(true);
    try {
      const response = await fetchJSON(API.analyze, { method: "POST", body: requestBody });
      state.result = validateResult(response);
      state.selection = null;
      state.view = { start: 0, end: state.result.series.observed.length - 1 };
      showResults();
      toast("Comparison complete. Inspect the synchronized plots or select a concern interval.");
      setHealth("online", "Analysis service ready");
    } catch (error) {
      showFormError(error.message);
      toast(error.message, true);
    } finally {
      setRunning(false);
    }
  }

  function validateResult(value) {
    if (!value || typeof value !== "object" || !value.session_id) throw new Error("The analysis response is missing a session identifier.");
    if (!value.series || !Array.isArray(value.series.observed) || !Array.isArray(value.series.denoiseapt) || !Array.isArray(value.series.automatic) || !Array.isArray(value.series.hybrid)) throw new Error("The analysis response is missing required signal arrays.");
    const n = value.series.observed.length;
    if (n < 2 || value.series.denoiseapt.length !== n || value.series.automatic.length !== n || value.series.hybrid.length !== n) throw new Error("The analysis response contains incompatible signal lengths.");
    for (const [name, array] of Object.entries(value.series)) {
      if (array != null && (!Array.isArray(array) || array.length !== n)) throw new Error(`Signal “${name}” has an incompatible length.`);
    }
    if (!value.scores || !Array.isArray(value.scores.observed) || !Array.isArray(value.scores.denoiseapt) || !Array.isArray(value.scores.automatic) || !Array.isArray(value.scores.hybrid)) throw new Error("The response is missing anomaly scores.");
    for (const [name, array] of Object.entries(value.scores)) {
      if (array != null && (!Array.isArray(array) || array.length !== n)) throw new Error(`Score “${name}” has an incompatible length.`);
    }
    if (!value.automatic_control || typeof value.automatic_control !== "object") throw new Error("The response is missing automatic-preservation status.");
    if (!value.hybrid_control || typeof value.hybrid_control !== "object") throw new Error("The response is missing evidence-gated hybrid status.");
    if (!value.concern || !Array.isArray(value.concern.values) || value.concern.values.length !== n) throw new Error("The response is missing the preservation-concern timeline.");
    value.time = Array.isArray(value.time) && value.time.length === n ? value.time : Array.from({ length: n }, (_, i) => i);
    value.cues = value.cues || {};
    value.metrics = value.metrics || {};
    value.anomaly_intervals = Array.isArray(value.anomaly_intervals) ? value.anomaly_intervals : [];
    value.history_depth = finite(value.history_depth) ? Number(value.history_depth) : 0;
    value.revision = finite(value.revision) ? Number(value.revision) : 0;
    return value;
  }

  function setRunning(running) {
    dom.runButton.disabled = running;
    dom.runButton.classList.toggle("is-loading", running);
    dom.runButton.setAttribute("aria-busy", String(running));
  }

  function showResults() {
    dom.empty.hidden = true;
    dom.results.hidden = false;
    const meta = state.result.meta || {};
    dom.caseHeading.textContent = [meta.case_name || state.upload?.name || "Analysis window", meta.domain, meta.corruption?.family ? `${capitalize(meta.corruption.family)} corruption` : null].filter(Boolean).join(" · ");
    dom.resetView.disabled = false;
    dom.exportButton.disabled = false;
    buildSignalPlots();
    buildLegends();
    updateActions();
    updateInspector();
    renderCertification();
    renderMetrics();
    requestAnimationFrame(renderAll);
  }

  function buildSignalPlots() {
    dom.signalPlots.replaceChildren();
    state.plotEntries = [];
    for (const key of SIGNAL_ORDER) {
      const values = state.result.series[key];
      if (!Array.isArray(values) || (key === "approved" && arraysEqual(values, state.result.series.automatic))) continue;
      const fragment = dom.plotTemplate.content.cloneNode(true);
      const article = $(".mini-plot", fragment), canvas = $("canvas", fragment), tooltip = $(".plot-tooltip", fragment);
      $("h4", fragment).textContent = METHOD_LABELS[key] || key;
      $(".mini-plot-head span", fragment).textContent = METHOD_CONTEXT[key] || "same observation";
      canvas.setAttribute("aria-label", `${METHOD_LABELS[key] || key} signal over time`);
      const entry = { type: "signal", key, canvas, tooltip, series: [{ key, values, color: COLORS[key] }] };
      state.plotEntries.push(entry);
      attachPlotInteractions(entry);
      dom.signalPlots.append(fragment);
    }
  }

  function buildLegends() {
    dom.signalLegend.replaceChildren(...SIGNAL_ORDER
      .filter(key => Array.isArray(state.result.series[key]) && !(key === "approved" && arraysEqual(state.result.series.approved, state.result.series.automatic)))
      .map(key => legendItem(key)));
    dom.scoreLegend.replaceChildren(...SCORE_ORDER
      .filter(key => Array.isArray(state.result.scores[key]) && !(key === "approved" && arraysEqual(state.result.series.approved, state.result.series.automatic)))
      .map(key => legendItem(key)));
  }

  function legendItem(key) {
    const span = document.createElement("span"), line = document.createElement("i");
    line.style.background = COLORS[key];
    span.append(line, document.createTextNode(METHOD_LABELS[key] || key));
    return span;
  }

  function renderAll() {
    if (!state.result || dom.results.hidden) return;
    for (const entry of state.plotEntries) drawLinePlot(entry);
    drawScorePlot();
    drawConcernPlot();
  }

  function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * ratio)), height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  }

  function drawLinePlot(entry) {
    const { ctx, width, height } = setupCanvas(entry.canvas);
    const pad = { left: 42, right: 10, top: 7, bottom: 20 };
    ctx.clearRect(0, 0, width, height);
    const range = visibleRange();
    const arrays = entry.series.map(s => s.values);
    const [yMin, yMax] = dataExtent(arrays, range.start, range.end);
    drawGrid(ctx, width, height, pad, yMin, yMax, range);
    drawIntervals(ctx, width, height, pad, range);
    drawSelection(ctx, width, height, pad, range);
    for (const item of entry.series) drawLine(ctx, item.values, item.color, width, height, pad, range, yMin, yMax, 1.55);
  }

  function drawScorePlot() {
    const entry = { canvas: dom.scoreCanvas, tooltip: dom.scoreTooltip, type: "score" };
    if (!dom.scoreCanvas.dataset.bound) { attachPlotInteractions(entry); dom.scoreCanvas.dataset.bound = "true"; }
    const { ctx, width, height } = setupCanvas(dom.scoreCanvas);
    const pad = { left: 48, right: 12, top: 10, bottom: 24 }, range = visibleRange();
    const keys = SCORE_ORDER.filter(k => Array.isArray(state.result.scores[k]) && !(k === "approved" && arraysEqual(state.result.series.approved, state.result.series.automatic)));
    const [rawMin, rawMax] = dataExtent(keys.map(k => state.result.scores[k]), range.start, range.end);
    const yMin = Math.min(0, rawMin), yMax = rawMax;
    ctx.clearRect(0, 0, width, height);
    drawGrid(ctx, width, height, pad, yMin, yMax, range);
    drawIntervals(ctx, width, height, pad, range);
    drawSelection(ctx, width, height, pad, range);
    for (const key of keys) drawLine(ctx, state.result.scores[key], COLORS[key], width, height, pad, range, yMin, yMax, key === "approved" || key === "automatic" ? 2.2 : 1.35);
  }

  function drawConcernPlot() {
    const entry = { canvas: dom.concernCanvas, tooltip: dom.concernTooltip, type: "concern" };
    if (!dom.concernCanvas.dataset.bound) { attachPlotInteractions(entry); dom.concernCanvas.dataset.bound = "true"; }
    const { ctx, width, height } = setupCanvas(dom.concernCanvas);
    const pad = { left: 48, right: 12, top: 7, bottom: 21 }, range = visibleRange();
    const values = state.result.concern.values, count = Math.max(1, range.end - range.start);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f4f8f8"; ctx.fillRect(pad.left, pad.top, width - pad.left - pad.right, height - pad.top - pad.bottom);
    const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
    const maxBars = Math.max(1, Math.floor(plotW / 2)), step = Math.max(1, Math.ceil(count / maxBars));
    for (let i = range.start; i <= range.end; i += step) {
      const segment = values.slice(i, Math.min(range.end + 1, i + step));
      const value = clamp(Math.max(...segment.map(v => finite(v) ? Number(v) : 0)), 0, 1);
      const x1 = pad.left + ((i - range.start) / count) * plotW;
      const x2 = pad.left + ((Math.min(range.end + 1, i + step) - range.start) / count) * plotW;
      ctx.fillStyle = concernColor(value); ctx.fillRect(x1, pad.top, Math.max(1, x2 - x1 + .5), plotH);
    }
    drawSelection(ctx, width, height, pad, range);
    drawXAxis(ctx, width, height, pad, range);
  }

  function drawGrid(ctx, width, height, pad, yMin, yMax, range) {
    const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
    ctx.font = "10px system-ui"; ctx.textBaseline = "middle"; ctx.lineWidth = 1;
    for (let j = 0; j <= 3; j++) {
      const y = pad.top + (j / 3) * plotH;
      ctx.strokeStyle = COLORS.grid; ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
      const value = yMax - (j / 3) * (yMax - yMin);
      ctx.fillStyle = COLORS.axis; ctx.textAlign = "right"; ctx.fillText(shortNumber(value), pad.left - 6, y);
    }
    drawXAxis(ctx, width, height, pad, range);
  }

  function drawXAxis(ctx, width, height, pad, range) {
    const plotW = width - pad.left - pad.right;
    ctx.font = "10px system-ui"; ctx.fillStyle = COLORS.axis; ctx.textBaseline = "alphabetic";
    for (let j = 0; j <= 4; j++) {
      const x = pad.left + (j / 4) * plotW, index = Math.round(range.start + (j / 4) * (range.end - range.start));
      ctx.textAlign = j === 0 ? "left" : j === 4 ? "right" : "center";
      ctx.fillText(formatTimeAt(index), x, height - 4);
    }
  }

  function drawLine(ctx, values, color, width, height, pad, range, yMin, yMax, lineWidth) {
    const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
    const count = Math.max(1, range.end - range.start), span = Math.max(yMax - yMin, 1e-9);
    const maxPoints = Math.max(50, Math.floor(plotW * 1.5)), step = Math.max(1, Math.ceil(count / maxPoints));
    ctx.beginPath(); let started = false;
    for (let i = range.start; i <= range.end; i += step) {
      const value = Number(values[i]); if (!Number.isFinite(value)) { started = false; continue; }
      const x = pad.left + ((i - range.start) / count) * plotW;
      const y = pad.top + (1 - (value - yMin) / span) * plotH;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.stroke();
  }

  function drawIntervals(ctx, width, height, pad, range) {
    const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom, count = Math.max(1, range.end - range.start);
    ctx.fillStyle = COLORS.anomaly;
    for (const interval of state.result.anomaly_intervals) {
      if (!finite(interval.start) || !finite(interval.end) || interval.end < range.start || interval.start > range.end) continue;
      const x1 = pad.left + ((clamp(interval.start, range.start, range.end) - range.start) / count) * plotW;
      const x2 = pad.left + ((clamp(interval.end, range.start, range.end) - range.start) / count) * plotW;
      ctx.fillRect(x1, pad.top, Math.max(2, x2 - x1), plotH);
    }
  }

  function drawSelection(ctx, width, height, pad, range) {
    if (!state.selection) return;
    const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom, count = Math.max(1, range.end - range.start);
    const x1 = pad.left + ((clamp(state.selection.start, range.start, range.end) - range.start) / count) * plotW;
    const x2 = pad.left + ((clamp(state.selection.end, range.start, range.end) - range.start) / count) * plotW;
    ctx.fillStyle = COLORS.selection; ctx.fillRect(Math.min(x1, x2), pad.top, Math.max(2, Math.abs(x2 - x1)), plotH);
    ctx.strokeStyle = COLORS.selectionLine; ctx.lineWidth = 1; ctx.strokeRect(Math.min(x1, x2) + .5, pad.top + .5, Math.max(1, Math.abs(x2 - x1) - 1), plotH - 1);
  }

  function attachPlotInteractions(entry) {
    const canvas = entry.canvas;
    canvas.addEventListener("pointerdown", event => {
      if (!state.result) return;
      canvas.setPointerCapture(event.pointerId);
      const index = indexFromEvent(canvas, event);
      state.drag = { entry, start: index, current: index, moved: false };
    });
    canvas.addEventListener("pointermove", event => {
      if (!state.result) return;
      const index = indexFromEvent(canvas, event);
      if (state.drag?.entry === entry) {
        state.drag.current = index; state.drag.moved ||= Math.abs(state.drag.current - state.drag.start) > 1;
        setSelection(state.drag.start, state.drag.current, false);
      }
      showTooltip(entry, event, index);
    });
    canvas.addEventListener("pointerup", event => {
      if (state.drag?.entry !== entry) return;
      const index = indexFromEvent(canvas, event);
      const radius = Math.max(2, Math.round((visibleRange().end - visibleRange().start) * .012));
      if (!state.drag.moved) setSelection(index - radius, index + radius);
      else setSelection(state.drag.start, index);
      state.drag = null;
    });
    canvas.addEventListener("pointercancel", () => {
      if (state.drag?.entry === entry) state.drag = null;
      hideTooltip(entry);
    });
    canvas.addEventListener("pointerleave", () => hideTooltip(entry));
    canvas.addEventListener("wheel", event => { event.preventDefault(); zoomView(indexFromEvent(canvas, event), event.deltaY > 0 ? 1.3 : .75); }, { passive: false });
    canvas.addEventListener("dblclick", resetView);
    canvas.addEventListener("keydown", event => {
      if (!state.result) return;
      if (event.key === "Escape") { state.selection = null; updateInspector(); updateActions(); renderAll(); }
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault(); const range = visibleRange(); const middle = Math.round((range.start + range.end) / 2), radius = Math.max(2, Math.round((range.end - range.start) * .03)); setSelection(middle - radius, middle + radius);
      }
      if (["ArrowLeft", "ArrowRight"].includes(event.key)) {
        event.preventDefault(); const amount = event.key === "ArrowLeft" ? -1 : 1;
        if (!state.selection) { const middle = Math.round((visibleRange().start + visibleRange().end) / 2); setSelection(middle, middle); }
        else setSelection(state.selection.start + amount, state.selection.end + amount);
      }
    });
  }

  function indexFromEvent(canvas, event) {
    const rect = canvas.getBoundingClientRect(), range = visibleRange();
    const left = canvas === dom.scoreCanvas || canvas === dom.concernCanvas ? 48 : 42;
    const right = canvas === dom.scoreCanvas || canvas === dom.concernCanvas ? 12 : 10;
    const fraction = clamp((event.clientX - rect.left - left) / Math.max(1, rect.width - left - right), 0, 1);
    return Math.round(range.start + fraction * (range.end - range.start));
  }

  function setSelection(a, b, final = true) {
    const n = state.result.series.observed.length;
    state.selection = { start: clamp(Math.round(Math.min(a, b)), 0, n - 1), end: clamp(Math.round(Math.max(a, b)), 0, n - 1) };
    if (state.selection.end === state.selection.start) state.selection.end = Math.min(n - 1, state.selection.start + 1);
    updateInspector(); updateActions(); renderAll();
    if (final) dom.intervalLabel.focus?.();
  }

  function showTooltip(entry, event, index) {
    const tooltip = entry.tooltip; if (!tooltip) return;
    let lines = [`${formatTimeAt(index)} · index ${index}`];
    if (entry.type === "signal") lines.push(`${METHOD_LABELS[entry.key]}: ${formatNumber(entry.series[0].values[index], 4)}`);
    if (entry.type === "score") {
      for (const key of SCORE_ORDER) if (Array.isArray(state.result.scores[key])) lines.push(`${METHOD_LABELS[key]}: ${formatNumber(state.result.scores[key][index], 4)}`);
    }
    if (entry.type === "concern") lines.push(`Concern: ${formatNumber(state.result.concern.values[index], 3)}`);
    tooltip.textContent = lines.join(" · "); tooltip.hidden = false;
    const rect = entry.canvas.getBoundingClientRect();
    tooltip.style.left = `${clamp(event.clientX - rect.left + 10, 4, Math.max(4, rect.width - tooltip.offsetWidth - 5))}px`;
    tooltip.style.top = `${clamp(event.clientY - rect.top - tooltip.offsetHeight - 9, 3, rect.height - tooltip.offsetHeight - 3)}px`;
    dom.plotReadout.textContent = lines.join(" · ");
  }

  function hideTooltip(entry) { if (entry.tooltip) entry.tooltip.hidden = true; }

  function updateInspector() {
    if (!state.result || !state.selection) {
      dom.intervalLabel.textContent = "Select a region in a plot or concern timeline.";
      setConcernBadge(null); setCue(dom.scoreCueValue, dom.scoreCueBar, null); setCue(dom.morphCueValue, dom.morphCueBar, null); setCue(dom.uncertaintyCueValue, dom.uncertaintyCueBar, null); return;
    }
    const { start, end } = state.selection;
    dom.intervalLabel.textContent = `${formatTimeAt(start)}–${formatTimeAt(end)} · indices ${start}–${end} · ${end - start + 1} samples`;
    const concern = meanSlice(state.result.concern.values, start, end), level = concern >= .62 ? "high" : concern >= .32 ? "medium" : "low";
    setConcernBadge(level, concern);
    setCue(dom.scoreCueValue, dom.scoreCueBar, cueMean("score_change", start, end));
    setCue(dom.morphCueValue, dom.morphCueBar, cueMean("morphology", start, end));
    setCue(dom.uncertaintyCueValue, dom.uncertaintyCueBar, cueMean("disagreement", start, end));
  }

  function cueMean(key, start, end) { const array = state.result.cues?.[key]; return Array.isArray(array) ? meanSlice(array, start, end) : null; }
  function setConcernBadge(level, value) {
    dom.concernBadge.className = `concern-badge is-${level || "neutral"}`;
    dom.concernBadge.textContent = level ? `${capitalize(level)} · ${formatNumber(value, 2)}` : "None";
  }
  function setCue(output, bar, value) {
    output.textContent = finite(value) ? formatNumber(value, 3) : "—";
    bar.style.width = finite(value) ? `${clamp(Number(value), 0, 1) * 100}%` : "0%";
  }

  function updateActions() {
    const hasResult = Boolean(state.result), hasSelection = Boolean(state.selection), history = Number(state.result?.history_depth || 0);
    for (const button of $$(".action-button")) {
      if (button.dataset.action === "accept" || button.dataset.action === "restore_automatic") button.disabled = !hasResult || state.actionPending;
      else if (button.dataset.action === "revert") button.disabled = !hasResult || state.actionPending || history < 1;
      else button.disabled = !hasResult || !hasSelection || state.actionPending;
    }
    dom.blendWeight.disabled = !hasResult || !hasSelection || state.actionPending;
  }

  async function applyIntervention(action) {
    if (!state.result || state.actionPending) return;
    const n = state.result.series.observed.length, intervalRequired = action === "protect" || action === "blend";
    if (intervalRequired && !state.selection) return toast("Select an interval before applying this action.", true);
    const range = state.selection || { start: 0, end: n - 1 };
    const payload = { session_id: state.result.session_id, action, start: range.start, end: Math.min(n, range.end + 1), beta: Number(dom.blendWeight.value), expected_revision: Number(state.result.revision || 0) };
    state.actionPending = true; updateActions();
    try {
      const response = await fetchJSON(API.intervene, { method: "POST", body: JSON.stringify(payload) }, 60000);
      if (!response.series || !Array.isArray(response.series.approved) || response.series.approved.length !== n) throw new Error("The intervention response is missing the approved signal.");
      state.result.series.approved = response.series.approved;
      if (response.scores?.approved) state.result.scores.approved = response.scores.approved;
      if (response.metrics?.approved) state.result.metrics.approved = response.metrics.approved;
      if (response.automatic_control) state.result.automatic_control = { ...state.result.automatic_control, ...response.automatic_control };
      state.result.history_depth = finite(response.history_depth) ? response.history_depth : state.result.history_depth;
      state.result.revision = finite(response.revision) ? response.revision : state.result.revision;
      buildSignalPlots(); buildLegends(); renderCertification(); renderMetrics(); requestAnimationFrame(renderAll);
      const label = action === "restore_automatic" ? "Automatic baseline restored" : `${capitalize(action)} applied${intervalRequired ? ` to indices ${range.start}–${range.end}` : ""}`;
      toast(`${label}.`);
    } catch (error) { toast(error.message, true); }
    finally { state.actionPending = false; updateActions(); }
  }

  function renderMetrics() {
    dom.metricsBody.replaceChildren();
    const rows = SIGNAL_ORDER
      .filter(key => key !== "reference" && key !== "observed")
      .filter(key => state.result.metrics[key] && !(key === "approved" && arraysEqual(state.result.series.approved, state.result.series.automatic)));
    if (!rows.length) {
      const row = dom.metricsBody.insertRow(), cell = row.insertCell(); cell.colSpan = 8; cell.className = "not-available"; cell.textContent = "No evaluation measures were returned for this case."; return;
    }
    const hasReference = Array.isArray(state.result.series.reference);
    const benchmarkCase = Boolean(state.result.meta?.benchmark_case);
    const syntheticCase = Boolean(state.result.meta?.synthetic);
    dom.metricScope.textContent = benchmarkCase
      ? "Benchmark evaluation"
      : syntheticCase && hasReference
        ? "Synthetic reference evaluation"
        : hasReference ? "Controlled reference evaluation" : "Observation-only case";
    for (const key of rows) {
      const metric = state.result.metrics[key] || {}, row = dom.metricsBody.insertRow();
      if (key === "approved") row.classList.add("is-approved");
      const values = [METHOD_LABELS[key] || key, metric.rmse, metric.snr_improvement, metric.vus_pr_approx ?? metric.vus_pr, metric.event_recall, metric.erasure_rate, metric.false_event_rate, metric.latency_ms];
      values.forEach((value, index) => {
        const cell = row.insertCell();
        if (index === 0) cell.textContent = value;
        else if (!finite(value)) { cell.textContent = "—"; cell.className = "not-available"; }
        else if (index === 7) cell.textContent = `${formatNumber(value, 1)} ms`;
        else if ([4,5,6].includes(index)) cell.textContent = `${formatNumber(Number(value) * 100, 1)}%`;
        else cell.textContent = formatNumber(value, 3);
      });
    }
  }

  function renderCertification() {
    const control = state.result?.automatic_control || {};
    const certificate = control.certificate || {};
    const reviewOnly = control.mode === "review_only" || !control.certification_eligible;
    const status = String(certificate.status || "unverified");
    const passed = certificate.passed === true;
    const certificateWitnessRows = Array.isArray(certificate.witnesses) ? certificate.witnesses : [];
    const retentionEvents = certificateWitnessRows.reduce((total, item) => total + (Array.isArray(item.event_output_peaks) ? item.event_output_peaks.length : 0), 0);
    const currentIsAutomatic = control.current_is_automatic !== false;
    dom.certificationCard.className = `certification-card ${reviewOnly ? "is-review" : passed ? "is-passed" : "is-failed"}`;
    dom.certificationBadge.className = `certificate-badge is-${reviewOnly ? "unverified" : status}`;
    dom.certificationBadge.textContent = reviewOnly
      ? "Review-only"
      : passed
        ? currentIsAutomatic ? "Witness certificate passed" : "Rechecked: passed"
        : status === "overridden" ? "Override outside contract" : "Witness check failed";
    dom.certificationSummary.textContent = reviewOnly
      ? `No automatic certificate is issued. ${control.eligibility_reason || certificate.error || "Threshold provenance is out of scope."}`
      : passed && retentionEvents === 0
        ? "The current signal passes A/B non-emergence checks, but this window has no observation-supported threshold events; the retention clause has zero opportunities."
        : `The current signal ${passed ? "satisfies" : "does not satisfy"} the frozen A/B retention and non-emergence checks. This is not a claim of physical anomaly truth.`;
    dom.certificateMode.textContent = reviewOnly ? "Review-only" : "A/B witness-bound";
    const decision = control.decision || "human override";
    dom.certificateDecision.textContent = `${capitalize(String(decision).replaceAll("_", " "))}${control.auto_committed === false ? " · not auto-committed" : ""}`;
    const repairs = Array.isArray(control.repair_intervals) ? control.repair_intervals : Array.isArray(control.intervals) ? control.intervals : [];
    dom.certificateRepairs.textContent = reviewOnly ? "Not applied" : String(repairs.length);
    dom.certificateLatency.textContent = finite(control.controller_latency_ms) ? `${formatNumber(control.controller_latency_ms, 2)} ms` : "—";
    const scope = control.runtime_provenance?.threshold_scope || control.provenance?.threshold_scope || {};
    dom.certificateScope.textContent = reviewOnly
      ? "Scores and concern cues remain available for inspection, but the automatic output is not certified or auto-committed."
      : `${scope.calibration_domain || "Frozen calibration domain"}; ${scope.window_length || "fixed"}-sample windows; configured witnesses A and B only.`;
    dom.certificateRepairDetail.textContent = reviewOnly
      ? "not applied in review-only mode"
      : repairs.length
        ? repairs.slice(0, 8).map(item => `[${item.start}, ${item.end}) β=${formatNumber(item.beta, 2)} ${item.action || "repair"}`).join("; ") + (repairs.length > 8 ? `; +${repairs.length - 8} more` : "")
        : "no repair was required; the generative repair candidate passed";
    dom.certificateWitnesses.textContent = certificateWitnessRows.length
      ? certificateWitnessRows.map(item => `${item.witness_id}: ${item.preservation_passed && item.fabrication_passed ? "pass" : "fail"} (${Array.isArray(item.event_output_peaks) ? item.event_output_peaks.length : 0} retention events)`).join("; ")
      : "not evaluated";
    dom.certificateLimitations.replaceChildren();
    const limitations = Array.isArray(certificate.limitations) ? certificate.limitations : [];
    for (const item of limitations) { const li = document.createElement("li"); li.textContent = item; dom.certificateLimitations.append(li); }
    dom.certificateAudit.textContent = control.audit?.decision_hash || "not available in review-only mode";
  }

  function zoomView(anchor, factor) {
    const range = visibleRange(), n = state.result.series.observed.length, span = range.end - range.start + 1;
    const nextSpan = clamp(Math.round(span * factor), Math.min(32, n), n);
    const ratio = span > 1 ? (anchor - range.start) / (span - 1) : .5;
    let start = Math.round(anchor - ratio * (nextSpan - 1)); start = clamp(start, 0, Math.max(0, n - nextSpan));
    state.view = { start, end: start + nextSpan - 1 }; renderAll();
  }
  function resetView() { if (!state.result) return; state.view = { start: 0, end: state.result.series.observed.length - 1 }; renderAll(); }
  function visibleRange() { return state.view || { start: 0, end: Math.max(1, state.result.series.observed.length - 1) }; }

  function exportResult() {
    if (!state.result) return;
    const payload = { exported_at: new Date().toISOString(), session_id: state.result.session_id, meta: state.result.meta, selection: state.selection, series: state.result.series, scores: state.result.scores, automatic_control: state.result.automatic_control, hybrid_control: state.result.hybrid_control, concern: state.result.concern, cues: state.result.cues, anomaly_intervals: state.result.anomaly_intervals, metrics: state.result.metrics };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }), link = document.createElement("a");
    link.href = URL.createObjectURL(blob); link.download = `denoiseapt-${safeFilename(state.result.meta?.case_name || "result")}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000); toast("Result exported as JSON.");
  }

  function showFormError(message) { dom.formError.textContent = message; dom.formError.hidden = false; }
  function hideFormError() { dom.formError.hidden = true; dom.formError.textContent = ""; }
  function toast(message, error = false) {
    const item = document.createElement("div"); item.className = `toast${error ? " is-error" : ""}`; item.setAttribute("role", error ? "alert" : "status"); item.textContent = message; dom.toastRegion.append(item);
    setTimeout(() => item.remove(), 5200);
  }

  function dataExtent(arrays, start, end) {
    let min = Infinity, max = -Infinity;
    for (const array of arrays) for (let i = start; i <= end; i++) { const value = Number(array[i]); if (Number.isFinite(value)) { min = Math.min(min, value); max = Math.max(max, value); } }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [-1, 1];
    if (min === max) return [min - 1, max + 1];
    const margin = (max - min) * .08; return [min - margin, max + margin];
  }
  function meanSlice(array, start, end) { let sum = 0, count = 0; for (let i = start; i <= end; i++) { const value = Number(array[i]); if (Number.isFinite(value)) { sum += value; count++; } } return count ? sum / count : null; }
  function concernColor(value) { return value >= .62 ? "#d56366" : value >= .32 ? "#e4a44e" : "#79b8a5"; }
  function formatTimeAt(index) { const value = state.result?.time?.[index]; if (value == null) return String(index); if (typeof value === "number") return formatNumber(value, 2); const text = String(value); return text.length > 18 ? text.slice(0, 18) : text; }
  function formatNumber(value, digits = 2) { return finite(value) ? Number(value).toLocaleString(undefined, { maximumFractionDigits: digits }) : "—"; }
  function shortNumber(value) { const abs = Math.abs(value); if (abs >= 10000 || (abs > 0 && abs < .001)) return Number(value).toExponential(1); return formatNumber(value, abs < 10 ? 2 : 1); }
  function formatInteger(value) { return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 }); }
  function formatMiB(bytes) { return (Number(bytes) / (1024 * 1024)).toFixed(1); }
  function finite(value) { return value !== null && value !== "" && typeof value !== "boolean" && Number.isFinite(Number(value)); }
  function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
  function capitalize(value) { const text = String(value || ""); return text ? text[0].toUpperCase() + text.slice(1) : text; }
  function safeFilename(value) { return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "result"; }
  function arraysEqual(a, b) { if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }
  function debounce(fn, wait) { let timer; return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); }; }

  initialize();
})();
