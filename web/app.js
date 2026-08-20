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
    "reference", "observed", "median", "wavelet",
    "noisereduce", "rins_t", "our_model"
  ];
  const SCORE_ORDER = ["observed", "our_model"];

  const COLORS = {
    reference: "#586c75",
    observed: "#c84f53",
    median: "#df912e",
    wavelet: "#7456a6",
    noisereduce: "#8b5d13",
    rins_t: "#64767d",
    our_model: "#117b79",
    automaticBaseline: "#64767d",
    adaptedPreview: "#3478b8",
    grid: "#e4ebec",
    axis: "#788a91",
    selection: "rgba(52, 120, 184, 0.15)",
    selectionLine: "#3478b8",
    anomaly: "rgba(200, 79, 83, 0.09)"
  };

  const METHOD_LABELS = {
    reference: "Reference before corruption (evaluation only)",
    observed: "Corrupted observation",
    median: "Median filter (w=3)",
    wavelet: "Wavelet thresholding",
    noisereduce: "Noisereduce",
    rins_t: "RINS-T adaptation",
    our_model: "Our Model"
  };

  const METHOD_CONTEXT = {
    reference: "evaluation only",
    observed: "same controlled condition",
    median: "frozen matched output",
    wavelet: "frozen matched output",
    noisereduce: "frozen matched output",
    rins_t: "official-architecture adaptation",
    our_model: "automatic or expert-adapted output"
  };

  const state = {
    cases: [],
    upload: null,
    result: null,
    automaticModel: null,
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
    start: $("#windowStart"), end: $("#windowEnd"), family: $("#corruptionFamily"),
    severity: $("#severity"), severityValue: $("#severityValue"), seed: $("#seed"),
    empty: $("#emptyState"), results: $("#results"),
    caseHeading: $("#caseHeading"), signalPlots: $("#signalPlots"), signalLegend: $("#signalLegend"),
    evidenceDescription: $("#evidenceDescription"), scoreCueLabel: $("#scoreCueLabel"),
    scoreLegend: $("#scoreLegend"), plotReadout: $("#plotReadout"), plotTemplate: $("#plotTemplate"),
    scoreCanvas: $("#scoreCanvas"), scoreTooltip: $("#scoreTooltip"),
    concernTitle: $("#concernTitle"), concernDescription: $("#concernDescription"),
    concernKey: $("#concernKey"), concernCanvas: $("#concernCanvas"), concernTooltip: $("#concernTooltip"),
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
    adaptationCanvas: $("#adaptationCanvas"), adaptationState: $("#adaptationState"),
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
      // The service keeps legacy research fixtures addressable for tests and
      // compatibility.  The main interface selector exposes only the
      // integrity-checked held-out replay presets.
      state.cases = payload.cases.filter(item => item.benchmark_replay === true);
      dom.caseSelect.replaceChildren();
      if (!state.cases.length) {
        dom.caseSelect.append(new Option("No packaged replay installed", ""));
        dom.caseSelect.disabled = true;
        dom.caseMeta.textContent = "No held-out replay is packaged. Use Upload CSV in review-only mode.";
        const catalogSource = $('input[name="source"][value="catalog"]');
        const uploadSource = $('input[name="source"][value="upload"]');
        catalogSource.disabled = true;
        catalogSource.setAttribute("aria-disabled", "true");
        catalogSource.checked = false;
        uploadSource.checked = true;
        toggleSource();
        return;
      }
      const catalogSource = $('input[name="source"][value="catalog"]');
      catalogSource.disabled = false;
      catalogSource.removeAttribute("aria-disabled");
      for (const item of state.cases) dom.caseSelect.append(new Option(item.name || item.id, item.id));
      const defaultCaseId = String(payload.default_case_id || "");
      if (state.cases.some(item => String(item.id) === defaultCaseId)) {
        dom.caseSelect.value = defaultCaseId;
      }
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
    dom.caseSelect.addEventListener("change", () => {
      clearCaseResults();
      updateCaseMeta();
    });
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
    if (source === "upload") {
      dom.family.value = "none";
      dom.family.disabled = false;
      dom.severity.disabled = true;
      dom.severityValue.textContent = "—";
      dom.start.disabled = false;
      dom.end.disabled = false;
    } else {
      dom.family.disabled = false;
      updateCaseMeta();
    }
    hideFormError();
  }

  function updateCaseMeta() {
    const item = state.cases.find(c => String(c.id) === dom.caseSelect.value);
    if (!item) return;
    const role = item.demo_role === "main_workflow"
      ? "main workflow"
      : item.demo_role === "anomaly_spotlight"
        ? "anomaly-preservation spotlight"
        : item.demo_role === "ecg_optional"
          ? "optional ECG case"
          : null;
    const details = [item.domain, role, item.held_out ? "held-out source group" : null, finite(item.length) ? `[0, ${formatInteger(item.length)})` : null,
      finite(item.sample_rate) ? `${formatNumber(item.sample_rate)} Hz` : null,
      finite(item.anomaly_count) ? `${item.anomaly_count} labelled event${item.anomaly_count === 1 ? "" : "s"}` : null].filter(Boolean);
    dom.caseMeta.textContent = details.join(" · ") || "Packaged case ready.";
    if (finite(item.length)) {
      dom.start.max = Math.max(0, item.length - 64);
      dom.end.max = item.length;
      if (item.fixed_window) {
        dom.start.value = 0; dom.end.value = item.length;
        dom.start.disabled = true; dom.end.disabled = true;
      } else {
        dom.start.disabled = false; dom.end.disabled = false;
      }
      if (item.default_family) dom.family.value = item.default_family;
      if (finite(item.default_severity)) { dom.severity.disabled = false; dom.severity.value = item.default_severity; updateSeverity(); }
      if (finite(item.default_replicate)) dom.seed.value = item.default_replicate;
    }
  }

  function clearCaseResults() {
    state.result = null;
    state.automaticModel = null;
    state.selection = null;
    state.view = null;
    state.plotEntries = [];
    dom.results.hidden = true;
    dom.empty.hidden = false;
    dom.resetView.disabled = true;
    dom.exportButton.disabled = true;
    const title = $("h3", dom.empty), copy = $("p", dom.empty);
    if (title) title.textContent = "New case selected";
    if (copy) copy.textContent = "Run comparison to load the selected case.";
    updateActions();
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
      dom.fileMeta.textContent = `${file.name} · ${formatInteger(parsed.values.length)} valid points · value column “${parsed.valueColumn}” · review only`;
      dom.start.max = Math.max(0, parsed.values.length - 1);
      dom.end.max = parsed.values.length;
      dom.end.value = Math.min(Math.max(Number(dom.end.value), 64), parsed.values.length, 2048);
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
    if (values.length > 250000) throw new Error("The CSV contains more than 250,000 points. Select a smaller file.");
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
  function updateBlend() {
    dom.blendValue.textContent = Number(dom.blendWeight.value).toFixed(2);
    if (state.result) requestAnimationFrame(drawAdaptationPreview);
  }

  async function runAnalysis(event) {
    event.preventDefault();
    hideFormError();
    const source = $('input[name="source"]:checked').value;
    const start = Number(dom.start.value), end = Number(dom.end.value), replicate = Number(dom.seed.value);
    const length = end - start;
    if (!Number.isInteger(start) || start < 0) return showFormError("Start must be a non-negative integer.");
    if (!Number.isInteger(end) || end <= start) return showFormError("End must be greater than Start.");
    if (!Number.isInteger(length) || length < 64 || length > 2048) return showFormError("The interval must contain between 64 and 2,048 points.");
    if (!Number.isInteger(replicate) || replicate < 0 || replicate > 1) return showFormError("Replicate must be 0 or 1.");
    if (source === "catalog" && !dom.caseSelect.value) return showFormError("Select an installed packaged case.");
    if (source === "upload" && !state.upload) return showFormError("Choose a valid CSV file first.");
    const sourceLength = source === "upload" ? state.upload.values.length : state.cases.find(c => String(c.id) === dom.caseSelect.value)?.length;
    if (finite(sourceLength) && end > sourceLength) return showFormError(`The requested interval ends at ${end}, beyond the available ${sourceLength} points.`);

    const payload = {
      case_id: source === "catalog" ? dom.caseSelect.value : undefined,
      upload: source === "upload" ? { name: state.upload.name, values: state.upload.values, timestamps: state.upload.timestamps, labels: state.upload.labels } : undefined,
      corruption: { family: dom.family.value, severity: dom.family.value === "none" ? 0 : Number(dom.severity.value), replicate, seed: replicate },
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
      state.automaticModel = [...state.result.series.our_model];
      const suggested = state.result.meta?.suggested_expert_interval;
      state.selection = Array.isArray(suggested) && suggested.length === 2
        ? { start: Number(suggested[0]), end: Number(suggested[1]) - 1 }
        : null;
      state.view = { start: 0, end: state.result.series.observed.length - 1 };
      showResults();
      toast(source === "catalog"
        ? "Held-out comparison ready. Inspect the synchronized plots or adapt the selected interval."
        : "Review-only analysis ready. No calibrated A/B decision applies to this upload.");
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
    if (!value.series || !Array.isArray(value.series.observed)) throw new Error("The analysis response is missing the observed signal.");
    if (!Array.isArray(value.series.our_model)) {
      value.series.our_model = value.series.automatic || value.series.denoiseapt;
    }
    if (!Array.isArray(value.series.our_model)) throw new Error("The analysis response is missing Our Model output.");
    const n = value.series.observed.length;
    if (n < 2 || value.series.our_model.length !== n) throw new Error("The analysis response contains incompatible signal lengths.");
    for (const [name, array] of Object.entries(value.series)) {
      if (array != null && (!Array.isArray(array) || array.length !== n)) throw new Error(`Signal “${name}” has an incompatible length.`);
    }
    if (!value.scores || !Array.isArray(value.scores.observed)) throw new Error("The response is missing observation scores.");
    if (!Array.isArray(value.scores.our_model)) {
      value.scores.our_model = value.scores.automatic || value.scores.denoiseapt;
    }
    if (!Array.isArray(value.scores.our_model)) throw new Error("The response is missing Our Model scores.");
    for (const [name, array] of Object.entries(value.scores)) {
      if (array != null && (!Array.isArray(array) || array.length !== n)) throw new Error(`Score “${name}” has an incompatible length.`);
    }
    if (!value.automatic_control || typeof value.automatic_control !== "object") throw new Error("The response is missing the configured evidence status.");
    if (!value.concern || !Array.isArray(value.concern.values) || value.concern.values.length !== n) throw new Error("The response is missing the local comparison timeline.");
    value.time = Array.isArray(value.time) && value.time.length === n ? value.time : Array.from({ length: n }, (_, i) => i);
    value.cues = value.cues || {};
    value.metrics = value.metrics || {};
    if (!value.metrics.our_model && value.metrics.automatic) value.metrics.our_model = value.metrics.automatic;
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
    configureConcernPresentation();
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
      if (!Array.isArray(values)) continue;
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
      .filter(key => Array.isArray(state.result.series[key]))
      .map(key => legendItem(key)));
    dom.scoreLegend.replaceChildren(...SCORE_ORDER
      .filter(key => Array.isArray(state.result.scores[key]))
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
    drawAdaptationPreview();
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
    const keys = SCORE_ORDER.filter(k => Array.isArray(state.result.scores[k]));
    const [rawMin, rawMax] = dataExtent(keys.map(k => state.result.scores[k]), range.start, range.end);
    const yMin = Math.min(0, rawMin), yMax = rawMax;
    ctx.clearRect(0, 0, width, height);
    drawGrid(ctx, width, height, pad, yMin, yMax, range);
    drawIntervals(ctx, width, height, pad, range);
    drawSelection(ctx, width, height, pad, range);
    for (const key of keys) drawLine(ctx, state.result.scores[key], COLORS[key], width, height, pad, range, yMin, yMax, key === "our_model" ? 2.2 : 1.35);
  }

  function drawAdaptationPreview() {
    if (!dom.adaptationCanvas) return;
    const { ctx, width, height } = setupCanvas(dom.adaptationCanvas);
    ctx.clearRect(0, 0, width, height);
    if (!state.result || !state.selection) {
      dom.adaptationState.textContent = "Select an interval";
      return;
    }
    const start = state.selection.start, end = state.selection.end;
    const range = { start, end }, pad = { left: 48, right: 12, top: 8, bottom: 22 };
    const observed = state.result.series.observed;
    const automatic = Array.isArray(state.automaticModel)
      ? state.automaticModel : state.result.series.our_model;
    const beta = Number(dom.blendWeight.value);
    const preview = [...automatic];
    for (let index = start; index <= end; index++) {
      preview[index] = beta * observed[index] + (1 - beta) * automatic[index];
    }
    const arrays = [observed, automatic, preview].filter(Array.isArray);
    const [yMin, yMax] = dataExtent(arrays, start, end);
    drawGrid(ctx, width, height, pad, yMin, yMax, range);
    drawLine(ctx, automatic, COLORS.automaticBaseline, width, height, pad, range, yMin, yMax, 1.35);
    drawLine(ctx, observed, COLORS.observed, width, height, pad, range, yMin, yMax, 1.35);
    drawLine(ctx, preview, COLORS.adaptedPreview, width, height, pad, range, yMin, yMax, 2.2);
    dom.adaptationState.textContent = state.result.automatic_control?.current_is_automatic === false
      ? `Adapted output · preview β=${beta.toFixed(2)}`
      : `Automatic output · preview β=${beta.toFixed(2)}`;
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
    if (entry.type === "concern") {
      const label = isBenchmarkReplay() ? "Shared evidence" : "Comparison cue";
      lines.push(`${label}: ${formatNumber(state.result.concern.values[index], 3)}`);
    }
    tooltip.textContent = lines.join(" · "); tooltip.hidden = false;
    const rect = entry.canvas.getBoundingClientRect();
    tooltip.style.left = `${clamp(event.clientX - rect.left + 10, 4, Math.max(4, rect.width - tooltip.offsetWidth - 5))}px`;
    tooltip.style.top = `${clamp(event.clientY - rect.top - tooltip.offsetHeight - 9, 3, rect.height - tooltip.offsetHeight - 3)}px`;
    dom.plotReadout.textContent = lines.join(" · ");
  }

  function hideTooltip(entry) { if (entry.tooltip) entry.tooltip.hidden = true; }

  function updateInspector() {
    if (!state.result || !state.selection) {
      dom.intervalLabel.textContent = "Select a time interval in a plot or evidence timeline.";
      setConcernBadge(null); setCue(dom.scoreCueValue, dom.scoreCueBar, null); setCue(dom.morphCueValue, dom.morphCueBar, null); setCue(dom.uncertaintyCueValue, dom.uncertaintyCueBar, null); return;
    }
    const { start, end } = state.selection;
    dom.intervalLabel.textContent = `[${start}, ${end + 1}) · ${end - start + 1} time points`;
    const concern = meanSlice(state.result.concern.values, start, end);
    const level = isBenchmarkReplay()
      ? (concern >= .5 ? "shared" : "none")
      : (concern >= .62 ? "high" : concern >= .32 ? "medium" : "low");
    setConcernBadge(level, concern);
    setCue(dom.scoreCueValue, dom.scoreCueBar, cueMean("score_change", start, end));
    setCue(dom.morphCueValue, dom.morphCueBar, cueMean("morphology", start, end));
    setCue(dom.uncertaintyCueValue, dom.uncertaintyCueBar, cueMean("disagreement", start, end));
  }

  function cueMean(key, start, end) { const array = state.result.cues?.[key]; return Array.isArray(array) ? meanSlice(array, start, end) : null; }
  function setConcernBadge(level, value) {
    dom.concernBadge.className = `concern-badge is-${level || "neutral"}`;
    if (!level) {
      dom.concernBadge.textContent = "None";
      return;
    }
    const label = isBenchmarkReplay()
      ? (level === "shared" ? "Shared support" : "No shared support")
      : `${capitalize(level)} cue`;
    dom.concernBadge.textContent = `${label} · ${formatNumber(value, 2)}`;
  }

  function isBenchmarkReplay() {
    return Boolean(state.result?.meta?.benchmark_replay);
  }

  function concernKeyItem(className, label) {
    const item = document.createElement("span"), swatch = document.createElement("i");
    swatch.className = className;
    item.append(swatch, document.createTextNode(label));
    return item;
  }

  function configureConcernPresentation() {
    if (isBenchmarkReplay()) {
      const witnessLabel = state.result.meta?.display_witness_label || "Scorer A";
      dom.evidenceDescription.textContent = `Configured ${witnessLabel} traces for the corrupted observation and the current model output; the status check uses Scorers A and B.`;
      dom.scoreCueLabel.textContent = `${witnessLabel} difference`;
      dom.concernTitle.textContent = "Local preservation evidence";
      dom.concernDescription.textContent = `Shared configured ${witnessLabel} support between the observation and Our Model. Click or drag to select an interval.`;
      dom.concernKey.setAttribute("aria-label", "Preservation-evidence key");
      dom.concernKey.replaceChildren(
        concernKeyItem("none", "No shared support"),
        concernKeyItem("shared", "Shared support")
      );
      dom.scoreCanvas.setAttribute("aria-label", `${witnessLabel} score comparison over time`);
      dom.concernCanvas.setAttribute("aria-label", `Local configured ${witnessLabel} preservation evidence over time`);
      return;
    }
    dom.evidenceDescription.textContent = "Review-only comparison traces for the observation and current model output; no calibrated scorer decision applies.";
    dom.scoreCueLabel.textContent = "Scorer difference";
    dom.concernTitle.textContent = "Local comparison cue";
    dom.concernDescription.textContent = "Review-only observation-versus-output comparison cue. It is not calibrated preservation evidence. Click or drag to select an interval.";
    dom.concernKey.setAttribute("aria-label", "Review-only comparison-cue key");
    dom.concernKey.replaceChildren(
      concernKeyItem("low", "Low cue"),
      concernKeyItem("medium", "Medium cue"),
      concernKeyItem("high", "High cue")
    );
    dom.concernCanvas.setAttribute("aria-label", "Review-only local comparison cue over time");
  }
  function setCue(output, bar, value) {
    output.textContent = finite(value) ? formatNumber(value, 3) : "—";
    bar.style.width = finite(value) ? `${clamp(Number(value), 0, 1) * 100}%` : "0%";
  }

  function updateActions() {
    const hasResult = Boolean(state.result), hasSelection = Boolean(state.selection), history = Number(state.result?.history_depth || 0);
    for (const button of $$(".action-button")) {
      if (button.dataset.action === "restore_automatic") button.disabled = !hasResult || state.actionPending;
      else if (button.dataset.action === "revert") button.disabled = !hasResult || state.actionPending || history < 1;
      else button.disabled = !hasResult || !hasSelection || state.actionPending;
    }
    dom.blendWeight.disabled = !hasResult || !hasSelection || state.actionPending;
  }

  async function applyIntervention(action) {
    if (!state.result || state.actionPending) return;
    const n = state.result.series.observed.length, intervalRequired = action === "blend";
    if (intervalRequired && !state.selection) return toast("Select an interval before applying this action.", true);
    const range = state.selection || { start: 0, end: n - 1 };
    const payload = { session_id: state.result.session_id, action, start: range.start, end: Math.min(n, range.end + 1), beta: Number(dom.blendWeight.value), expected_revision: Number(state.result.revision || 0) };
    state.actionPending = true; updateActions();
    try {
      const response = await fetchJSON(API.intervene, { method: "POST", body: JSON.stringify(payload) }, 60000);
      const output = response.series?.our_model || response.series?.approved;
      if (!Array.isArray(output) || output.length !== n) throw new Error("The intervention response is missing Our Model output.");
      state.result.series.our_model = output;
      const scores = response.scores?.our_model || response.scores?.approved;
      if (scores) state.result.scores.our_model = scores;
      const metrics = response.metrics?.our_model || response.metrics?.approved;
      if (metrics) state.result.metrics.our_model = metrics;
      if (response.concern) state.result.concern = response.concern;
      if (response.cues) state.result.cues = response.cues;
      if (response.automatic_control) state.result.automatic_control = { ...state.result.automatic_control, ...response.automatic_control };
      state.result.history_depth = finite(response.history_depth) ? response.history_depth : state.result.history_depth;
      state.result.revision = finite(response.revision) ? response.revision : state.result.revision;
      buildSignalPlots(); buildLegends(); renderCertification(); renderMetrics(); requestAnimationFrame(renderAll);
      const label = action === "restore_automatic" ? "Automatic model restored" : action === "revert" ? "Previous state restored" : `Expert weight applied to [${range.start}, ${range.end + 1})`;
      toast(`${label}.`);
    } catch (error) { toast(error.message, true); }
    finally { state.actionPending = false; updateActions(); }
  }

  function renderMetrics() {
    dom.metricsBody.replaceChildren();
    const rows = SIGNAL_ORDER
      .filter(key => key !== "reference")
      .filter(key => {
        const metric = state.result.metrics[key];
        return metric && (finite(metric.overall_os_nrmse) || finite(metric.anomaly_os_nrmse));
      });
    if (!rows.length) {
      const row = dom.metricsBody.insertRow(), cell = row.insertCell(); cell.colSpan = 3; cell.className = "not-available"; cell.textContent = "Matched-window measures are available only for the held-out replay."; return;
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
      if (key === "our_model") row.classList.add("is-approved");
      const values = [METHOD_LABELS[key] || key, metric.overall_os_nrmse, metric.anomaly_os_nrmse];
      values.forEach((value, index) => {
        const cell = row.insertCell();
        if (index === 0) cell.textContent = value;
        else if (!finite(value)) { cell.textContent = "—"; cell.className = "not-available"; }
        else cell.textContent = formatNumber(value, 4);
      });
    }
  }

  function renderCertification() {
    const control = state.result?.automatic_control || {};
    const certificate = control.certificate || {};
    const benchmarkReplay = control.mode === "heldout_benchmark_replay";
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
        ? currentIsAutomatic ? "Configured A/B check passed" : "Fresh A/B recheck passed"
        : "Configured A/B check failed";
    dom.certificationSummary.textContent = reviewOnly
      ? `No configured A/B decision is issued. ${control.eligibility_reason || certificate.error || "Threshold provenance is out of scope."}`
      : passed && retentionEvents === 0
        ? "The current signal passes A/B non-emergence checks; this condition contains no observation-supported threshold component to retain."
        : `The current signal ${passed ? "satisfies" : "does not satisfy"} the configured A/B observation-component checks. This is not a claim of physical anomaly truth.`;
    dom.certificateMode.textContent = reviewOnly ? "Review-only" : benchmarkReplay ? "Held-out replay" : "A/B configured";
    const decision = control.decision || "human override";
    dom.certificateDecision.textContent = `${capitalize(String(decision).replaceAll("_", " "))}${control.auto_committed === false ? " · not auto-committed" : ""}`;
    const repairs = Array.isArray(control.repair_intervals) ? control.repair_intervals : Array.isArray(control.intervals) ? control.intervals : [];
    const expertActions = control.human_intervention?.actions || [];
    dom.certificateRepairs.textContent = reviewOnly ? "Not applied" : String(expertActions.length);
    dom.certificateLatency.textContent = reviewOnly ? "Not available" : passed ? "Pass" : "Fail";
    const scope = control.runtime_provenance?.threshold_scope || control.provenance?.threshold_scope || {};
    dom.certificateScope.textContent = reviewOnly
      ? "Scores and comparison cues remain available for inspection, but no calibrated A/B threshold decision or automatic commitment applies."
      : `${scope.calibration_domain || "Frozen calibration domain"}; [0, ${scope.window_length || 512}) window; configured Scorers A and B only.`;
    dom.certificateRepairDetail.textContent = reviewOnly
      ? "not applied in review-only mode"
      : expertActions.length
        ? expertActions.slice(-4).map(item => `${item.action} [${item.start}, ${item.end})${finite(item.beta) ? ` β=${formatNumber(item.beta, 2)}` : ""}`).join("; ")
        : "no expert adaptation; showing the automatic model output";
    dom.certificateWitnesses.textContent = certificateWitnessRows.length
      ? certificateWitnessRows.map(item => `${item.witness_id}: ${item.preservation_passed && item.fabrication_passed ? "pass" : "fail"} (${Array.isArray(item.event_output_peaks) ? item.event_output_peaks.length : 0} retention events)`).join("; ")
      : "not evaluated";
    dom.certificateLimitations.replaceChildren();
    const limitations = Array.isArray(certificate.limitations) ? certificate.limitations : [];
    for (const item of limitations) { const li = document.createElement("li"); li.textContent = item; dom.certificateLimitations.append(li); }
    dom.certificateAudit.textContent = control.runtime_provenance?.condition_id || control.audit?.decision_hash || "not available in review-only mode";
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
    const payload = { exported_at: new Date().toISOString(), session_id: state.result.session_id, meta: state.result.meta, selection: state.selection, series: state.result.series, scores: state.result.scores, automatic_control: state.result.automatic_control, concern: state.result.concern, cues: state.result.cues, anomaly_intervals: state.result.anomaly_intervals, metrics: state.result.metrics };
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
  function concernColor(value) {
    if (isBenchmarkReplay()) return value >= .5 ? "#15968f" : "#e3ecec";
    return value >= .62 ? "#d56366" : value >= .32 ? "#e4a44e" : "#79b8a5";
  }
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
