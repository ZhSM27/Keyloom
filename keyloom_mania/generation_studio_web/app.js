"use strict";

const $ = (selector) => document.querySelector(selector);

const STYLE_GROUPS = [
  { name: "连段 / 流动", label: "流" , styles: ["Stream", "Chordstream", "Jumpstream", "Handstream", "Quadstream"] },
  { name: "叠键 / 手癖", label: "叠", styles: ["Jacks", "Longjacks", "Chordjacks", "Minijacks", "Gluts"] },
  { name: "轮转 / 颤音", label: "轮", styles: ["Rolls", "Trills", "Minitrills", "Split Trill", "Jumptrill"] },
];

const STYLE_COLORS = {
  Stream: "#8b92f2", Chordstream: "#8aa9fa", Jumpstream: "#9fc1ff", Handstream: "#8be0cc", Quadstream: "#f1bd83",
  Jacks: "#f2b681", Longjacks: "#e99478", Chordjacks: "#ef9e91", Minijacks: "#dfa4ce", Gluts: "#cb9aea",
  Rolls: "#f09bc1", Trills: "#d994eb", Minitrills: "#b8a0ee", "Split Trill": "#b08bdb", Jumptrill: "#aaafea",
};

const state = {
  sessionId: null,
  sourceFile: null,
  sourceUrl: null,
  selectedStyle: "Jumpstream",
  selectedSecondaryStyles: [],
  models: [],
  selectedModelId: null,
  selectedDensityId: "fixed",
  selectedArrangementId: "rule",
  selectedStructureId: "legacy",
  selectedCoordinationId: "legacy",
  selectedBudgetId: "legacy",
  budgetOptions: [],
  modelAvailable: false,
  uploading: false,
  busy: false,
  previousStatus: null,
  taps: [],
  pollHandle: null,
  renderedBars: [],
  autoTiming: null,
};

function setText(selector, value) { $(selector).textContent = value; }

function formatDuration(ms) {
  const total = Math.max(0, Math.floor(Number(ms || 0) / 1000));
  const minutes = Math.floor(total / 60);
  return `${String(minutes).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function setNotice(message) {
  const notice = $("#modelNotice");
  notice.textContent = message;
  notice.classList.toggle("hidden", !message);
}

function selectedModel() {
  return state.models.find((model) => model.id === state.selectedModelId) || null;
}

function updateModelSelector() {
  const model = selectedModel();
  const budgetSelect = $("#budgetSelect");
  if (budgetSelect) {
    for (const option of budgetSelect.options) {
      const item = state.budgetOptions.find((item) => item.id === option.value);
      option.disabled = item?.available === false || Boolean(item?.supported_generator_ids && !item.supported_generator_ids.includes(model?.id)) || Boolean(item?.supported_main_styles && !item.supported_main_styles.includes(state.selectedStyle));
    }
    if (budgetSelect.selectedOptions[0]?.disabled) {
      state.selectedBudgetId = "legacy";
      budgetSelect.value = "legacy";
    }
  }
  const badge = $("#modelStatusBadge");
  if (!model) {
    setText("#modelSummary", "没有可用的生成模型");
    badge.textContent = "不可用";
    badge.dataset.tone = "legacy";
    badge.classList.remove("hidden");
    return;
  }
  const detail = model.notice ? `${model.description} · ${model.notice}` : model.description;
  setText("#modelSummary", model.available ? detail : model.unavailable_reason || detail);
  $("#modelSelect").title = detail;
  badge.textContent = model.status_label || (model.available ? "可选" : "不可用");
  badge.dataset.tone = model.status || "legacy";
  badge.classList.remove("hidden");
}

function renderModelOptions(health) {
  const arrangementRoot = $("#arrangementSelect");
  arrangementRoot.replaceChildren();
  for (const planner of health.arrangement_options || [{id: "rule", label: "音乐分段规则编排", available: true}]) {
    const option = document.createElement("option");
    option.value = planner.id;
    option.textContent = planner.label + (planner.parameter_count ? ` · ${(planner.parameter_count / 1000000).toFixed(1)}M` : "") + (planner.status === "research" ? "（研究）" : "");
    option.disabled = !planner.available;
    arrangementRoot.append(option);
  }
  state.selectedArrangementId = health.default_arrangement_model_id || "rule";
  arrangementRoot.value = state.selectedArrangementId;
  const structureRoot = $("#structureSelect");
  structureRoot.replaceChildren();
  for (const item of health.structure_options || [{id: "legacy", label: "原音乐分段", available: true}]) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label + (item.status === "research" ? "（研究）" : "");
    option.disabled = item.available === false;
    structureRoot.append(option);
  }
  state.selectedStructureId = health.default_structure_id || "legacy";
  structureRoot.value = state.selectedStructureId;
  const coordinationRoot = $("#coordinationSelect");
  coordinationRoot.replaceChildren();
  for (const item of health.coordination_options || [{id: "legacy", label: "原编排与密度执行"}]) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label + (item.status === "research" ? "（研究）" : "");
    coordinationRoot.append(option);
  }
  state.selectedCoordinationId = health.default_coordination_id || "legacy";
  coordinationRoot.value = state.selectedCoordinationId;
  const budgetRoot = $("#budgetSelect");
  budgetRoot.replaceChildren();
  state.budgetOptions = health.budget_options || [];
  for (const item of health.budget_options || [{id: "legacy", label: "原密度条件"}]) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label + (item.status === "research" ? "（研究）" : "");
    option.disabled = item.available === false;
    option.title = item.unavailable_reason || "";
    budgetRoot.append(option);
  }
  state.selectedBudgetId = health.default_budget_id || "legacy";
  budgetRoot.value = state.selectedBudgetId;
  const densityRoot = $("#densitySelect");
  densityRoot.replaceChildren();
  for (const density of health.density_options || [{id: "fixed", label: "原 Planner 密度", available: true}]) {
    const option = document.createElement("option");
    option.value = density.id;
    option.textContent = density.label + (density.status === "research" ? "（研究）" : "");
    option.disabled = !density.available;
    densityRoot.append(option);
  }
  state.selectedDensityId = health.default_density_model_id || "fixed";
  densityRoot.value = state.selectedDensityId;
  const root = $("#modelSelect");
  state.models = Array.isArray(health.generator_options) ? health.generator_options : [];
  root.replaceChildren();
  for (const model of state.models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = `${model.label} · ${model.status_label}${model.available ? "" : "（不可用）"}`;
    option.disabled = !model.available;
    root.append(option);
  }
  const preferred = state.models.find((model) => model.id === health.default_generator_id && model.available)
    || state.models.find((model) => model.recommended && model.available)
    || state.models.find((model) => model.available)
    || null;
  state.selectedModelId = preferred?.id || state.models[0]?.id || null;
  if (state.selectedModelId) root.value = state.selectedModelId;
  root.disabled = !state.models.some((model) => model.available);
  updateModelSelector();
}

function renderStyles() {
  const root = $("#styleGroups");
  root.replaceChildren();
  for (const group of STYLE_GROUPS) {
    const row = document.createElement("div");
    row.className = "style-group";
    const title = document.createElement("span");
    title.className = "style-group-name";
    title.textContent = group.name;
    const choices = document.createElement("div");
    choices.className = "style-options";
    for (const style of group.styles) {
      const color = STYLE_COLORS[style] || "#a897fa";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "style-option";
      button.dataset.style = style;
      button.setAttribute("aria-pressed", String(style === state.selectedStyle));
      button.style.setProperty("--style-color", color);
      button.title = `${style} · 设为主键型`;
      const mark = document.createElement("i");
      mark.textContent = style.slice(0, 1);
      const text = document.createElement("span");
      text.textContent = style;
      button.append(mark, text);
      button.addEventListener("click", () => selectStyle(style));
      choices.append(button);
    }
    row.append(title, choices);
    root.append(row);
  }
  const secondary = $("#secondaryStyleOptions");
  secondary.replaceChildren();
  for (const group of STYLE_GROUPS) {
    const section = document.createElement("div");
    section.className = "secondary-style-group";
    const heading = document.createElement("span");
    heading.textContent = group.name;
    const options = document.createElement("div");
    options.className = "secondary-style-group-options";
    for (const style of group.styles) {
      const label = document.createElement("label");
      label.className = "secondary-style-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = style;
      checkbox.checked = state.selectedSecondaryStyles.includes(style);
      checkbox.addEventListener("change", () => {
        const selected = Array.from(secondary.querySelectorAll("input:checked"), (input) => input.value);
        if (selected.length > 3) {
          checkbox.checked = false;
          return;
        }
        state.selectedSecondaryStyles = selected;
        updateSecondaryStyleControls();
        setText("#styleStat", [state.selectedStyle, ...selected].join(" + "));
        invalidatePreview("已更改副键型参考；重新生成后会更新此规划预览。");
        updateGenerateButton();
      });
      const text = document.createElement("span");
      text.textContent = style;
      label.append(checkbox, text);
      options.append(label);
    }
    section.append(heading, options);
    secondary.append(section);
  }
  updateSelectedStyle();
}

function updateSecondaryStyleControls() {
  const root = $("#secondaryStyleOptions");
  if (!root) return;
  const selected = Array.from(root.querySelectorAll("input:checked"), (input) => input.value);
  state.selectedSecondaryStyles = selected;
  for (const checkbox of root.querySelectorAll("input[type=checkbox]")) {
    checkbox.disabled = checkbox.value === state.selectedStyle || (!checkbox.checked && selected.length >= 3);
  }
  setText("#secondaryStyleCount", `${selected.length} / 3`);
}

function updateSelectedStyle() {
  const color = STYLE_COLORS[state.selectedStyle] || "#a897fa";
  for (const button of document.querySelectorAll(".style-option")) {
    button.setAttribute("aria-pressed", String(button.dataset.style === state.selectedStyle));
  }
  $("#selectionSwatch").textContent = state.selectedStyle.slice(0, 1);
  $("#selectionSwatch").style.setProperty("--style-color", color);
  $("#selectedStyleName").textContent = state.selectedStyle;
  $("#selectedStyleName").style.color = color;
  const secondary = $("#secondaryStyleOptions");
  if (secondary) {
    const matching = Array.from(secondary.querySelectorAll("input")).find((input) => input.value === state.selectedStyle);
    if (matching?.checked) matching.checked = false;
    updateSecondaryStyleControls();
  }
  setText("#styleStat", [state.selectedStyle, ...state.selectedSecondaryStyles].join(" + "));
}

function selectStyle(style) {
  state.selectedStyle = style;
  updateSelectedStyle();
  updateModelSelector();
  invalidatePreview("已更改主要键型；重新生成后会更新此规划预览。 ");
  updateGenerateButton();
}

function invalidatePreview(message = "") {
  if (state.previousStatus === "complete") {
    state.previousStatus = null;
    $("#resultDownload").classList.add("hidden");
    $("#resultStats").classList.add("hidden");
    $("#densityDiagnostic").classList.add("hidden");
    setText("#previewSubtitle", message);
  }
}

function updateGenerateButton() {
  const model = selectedModel();
  const ready = Boolean(state.sessionId) && !state.uploading && !state.busy && state.modelAvailable && Boolean(model?.available);
  $("#generateButton").disabled = !ready;
  $("#measureTempo").disabled = !state.sessionId || state.uploading || state.busy;
  $("#modelSelect").disabled = state.uploading || state.busy || !state.models.some((item) => item.available);
  $("#densitySelect").disabled = state.uploading || state.busy;
  $("#arrangementSelect").disabled = state.uploading || state.busy;
  $("#structureSelect").disabled = state.uploading || state.busy;
  $("#coordinationSelect").disabled = state.uploading || state.busy;
  $("#budgetSelect").disabled = state.uploading || state.busy;
  if (state.busy) {
    $("#generateButton .button-label").textContent = "正在生成谱面…";
  } else if (state.previousStatus === "complete") {
    $("#generateButton .button-label").textContent = "调整设置后重新生成";
  } else {
    $("#generateButton .button-label").textContent = "生成可游玩谱面";
  }
}

function resetAnalysisView() {
  state.autoTiming = null;
  $("#applyTiming").disabled = true;
  $("#songInfo").classList.add("hidden");
  $("#audioPlayer").classList.add("hidden");
  $("#tempoResult").classList.add("hidden");
  $("#tempoAlternatives").classList.add("hidden");
  $("#progressCard").classList.add("hidden");
  $("#resultDownload").classList.add("hidden");
  $("#resultStats").classList.add("hidden");
  $("#generationError").classList.add("hidden");
  $("#timelineEmpty").classList.remove("hidden");
  setText("#previewSubtitle", "上传歌曲并开始生成后，这里会显示逐小节的规划结果。");
  state.previousStatus = null;
  state.renderedBars = [];
  drawTimeline([]);
}

function showSource(file) {
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.sourceFile = file;
  state.sourceUrl = URL.createObjectURL(file);
  $("#audioPlayer").src = state.sourceUrl;
  $("#audioPlayer").classList.remove("hidden");
  $("#songInfo").classList.remove("hidden");
  $("#tempoResult").classList.remove("hidden");
  $("#uploadState").textContent = "检查音频中…";
  $("#durationValue").textContent = "--:--";
  $("#titleInput").value = file.name.replace(/\.[^.]+$/, "").slice(0, 120);
  $("#artistInput").value = "Unknown artist";
  setText("#tempoText", "正在在本机估算拍速…");
  $("#tempoConfidence").textContent = "分析中";
  $("#tapTempo").disabled = true;
  $("#tapHint").textContent = "点击播放器试听后即可校准";
  $("#dropTitle").textContent = "歌曲已选中";
  $("#dropMeta").textContent = file.name;
}

async function uploadFile(file) {
  if (!file) return;
  if (!/\.(mp3|ogg|wav|flac|m4a)$/i.test(file.name)) {
    setNotice("请选择 MP3、OGG、WAV、FLAC 或 M4A 格式的音频。");
    return;
  }
  if (file.size > 180 * 1024 * 1024) {
    setNotice("单个音频文件最多 180 MiB。");
    return;
  }
  clearInterval(state.pollHandle);
  state.sessionId = null;
  state.busy = false;
  state.uploading = true;
  state.taps = [];
  setNotice("");
  resetAnalysisView();
  showSource(file);
  updateGenerateButton();
  $("#uploadProgress").classList.remove("hidden");
  const data = new FormData();
  data.append("audio", file, file.name);
  try {
    const response = await new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("POST", "/api/upload");
      request.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) $("#uploadProgressBar").style.width = `${Math.round(event.loaded / event.total * 100)}%`;
      });
      request.addEventListener("load", () => {
        let result;
        try { result = JSON.parse(request.responseText); } catch { result = {}; }
        if (request.status < 200 || request.status >= 300) reject(new Error(result.error || `上传失败 (${request.status})`));
        else resolve(result);
      });
      request.addEventListener("error", () => reject(new Error("连接本机服务失败。请确认 studio.bat 中的服务窗口仍在运行。")));
      request.send(data);
    });
    state.sessionId = response.session.id;
    state.uploading = false;
    $("#uploadProgress").classList.add("hidden");
    setText("#uploadState", "正在分析拍点…");
    await pollSession(state.sessionId, true);
  } catch (error) {
    state.uploading = false;
    $("#uploadProgress").classList.add("hidden");
    $("#uploadState").textContent = "上传失败";
    setText("#tempoText", error.message);
    setNotice(error.message);
    updateGenerateButton();
  }
}

function drawTimeline(bars) {
  const canvas = $("#timelineCanvas");
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, rect.width);
  const height = 98;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#101319";
  context.fillRect(0, 0, width, height);
  const left = 10, right = width - 10, top = 12, bottom = height - 10;
  for (const row of [1, 2, 3]) {
    const y = top + (bottom - top) * row / 4;
    context.strokeStyle = "#212732";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(right, y);
    context.stroke();
  }
  if (!bars.length) return;
  const gap = Math.min(1.4, width / bars.length * 0.14);
  const stride = (right - left) / bars.length;
  for (let index = 0; index < bars.length; index += 1) {
    const bar = bars[index];
    const x = left + index * stride;
    const rest = bar.quiet || !bar.style;
    if (rest) {
      context.fillStyle = index % 4 === 0 ? "#333945" : "#292f39";
      context.fillRect(x + gap / 2, top + 25, Math.max(1, stride - gap), 17);
      continue;
    }
    const color = STYLE_COLORS[bar.style] || STYLE_COLORS[state.selectedStyle] || "#a897fa";
    const weight = Math.min(1, Math.max(.34, Number(bar.notes_per_beat || 0) / 3));
    const support = bar.role === "support";
    const barHeight = 18 + weight * 36;
    context.globalAlpha = support ? .50 + .30 * weight : .45 + .48 * weight;
    context.fillStyle = color;
    context.fillRect(x + gap / 2, bottom - barHeight, Math.max(1, stride - gap), barHeight);
    context.globalAlpha = 1;
    if (index > 0 && bar.segment_id >= 0 && bar.segment_id !== bars[index - 1].segment_id) {
      context.strokeStyle = "rgb(255 255 255 / 45%)";
      context.beginPath();
      context.moveTo(x, top);
      context.lineTo(x, bottom);
      context.stroke();
    }
    if (bar.section_id >= 0 && (index === 0 || bar.section_id !== bars[index - 1].section_id)) {
      context.strokeStyle = "rgb(154 226 203 / 90%)";
      context.lineWidth = 2;
      context.beginPath(); context.moveTo(x, top); context.lineTo(x, bottom); context.stroke();
      context.lineWidth = 1;
    }
    if (support && stride > 4) {
      context.strokeStyle = "rgb(154 226 203 / 65%)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(x + gap / 2, bottom - barHeight);
      context.lineTo(x + gap / 2 + Math.max(1, stride - gap), bottom - barHeight);
      context.stroke();
    }
    if (barHeight > 36 && stride > 4) {
      context.fillStyle = "rgb(255 255 255 / 17%)";
      context.fillRect(x + gap / 2, bottom - barHeight, Math.max(1, stride - gap), 2);
    }
  }
}

function setResult(job) {
  const manifest = job.manifest;
  if (!manifest) return;
  state.renderedBars = job.bars || [];
  $("#timelineEmpty").classList.add("hidden");
  drawTimeline(state.renderedBars);
  $("#resultStats").classList.remove("hidden");
  $("#resultDownload").classList.remove("hidden");
  $("#generationError").classList.add("hidden");
  setText("#noteStat", Number(manifest.notes).toLocaleString("zh-CN"));
  setText("#barStat", Number(manifest.bars).toLocaleString("zh-CN"));
  setText("#restStat", Number(manifest.quiet_bars).toLocaleString("zh-CN"));
  setText("#actualStarStat", `${Number(manifest.actual_star).toFixed(2)}★`);
  const secondary = (manifest.secondary_styles || (manifest.secondary_style ? [manifest.secondary_style] : [])).join(" + ");
  const styleSummary = [manifest.main_style, secondary].filter(Boolean).join(" + ");
  setText("#styleStat", styleSummary);
  const unused = (manifest.unused_secondary_styles || []).map(item => item.style);
  const unusedDetail = unused.length ? ` · 未独立成段的副型：${unused.join("、")}` : "";
  setText("#previewSubtitle", `${manifest.bpm.toFixed(1)} BPM · ${formatDuration(manifest.audio_duration_ms)} · ${styleSummary} · 主副键型按音乐段落安排${unusedDetail}`);
  if (manifest.structure_id === "hierarchical") {
    setText("#previewSubtitle", `${manifest.bpm.toFixed(1)} BPM · ${styleSummary} · ${manifest.hierarchy_sections} 个段落 / ${manifest.hierarchy_phrases} 个乐句 · 绿线为段落边界，白线为乐句边界${unusedDetail}`);
  }
  setText("#downloadDetail", `${manifest.model_label || "已选模型"} · ${manifest.arrangement_model_label || "规则编排"} · ${manifest.density_model_label || "原 Planner 密度"} · ${manifest.coordination_label || "原执行"} · 实际难度 ${Number(manifest.actual_star).toFixed(2)}★ · 可导入 osu!lazer 游玩`);
  $("#downloadLink").href = `/api/download/${state.sessionId}`;
  $("#downloadLink").download = job.filename || "beatmap.osz";
  const execution = manifest.execution_diagnostics;
  const diagnostic = $("#densityDiagnostic");
  diagnostic.classList.toggle("hidden", !execution);
  if (execution) {
    const budget = manifest.density_budget || {};
    const mae = execution.notes_beat_mae == null ? "—" : Number(execution.notes_beat_mae).toFixed(2);
    diagnostic.textContent = `密度诊断 · 音乐计划 ${execution.music_notes} 音 / 实际导出 ${execution.export_notes} 音 · 排键减键 ${execution.prepare_removed_notes + execution.layout_removed_notes} 音 · 导出减键 ${execution.export_removed_notes} 音 · 完整拍密度误差 ${mae} 音/拍 · 预算调整 ${budget.changed_beats || 0} 拍。完整逐拍记录随谱包提供；误差不代表音乐质量。`;
  }
}

function updateProgress(job) {
  const stageNames = {
    queued: "正在排队", decode: "读取音频", features: "提取歌曲特征", planner: "规划全曲结构",
    generation: "生成音符", packaging: "校验并打包", complete: "谱面已经准备好", failed: "生成失败",
  };
  const stage = job.stage || job.status || "queued";
  const value = Math.max(0, Math.min(1, Number(job.progress || 0)));
  $("#progressCard").classList.remove("hidden");
  $("#generationProgress").style.width = `${Math.round(value * 100)}%`;
  setText("#progressPercent", `${Math.round(value * 100)}%`);
  setText("#stageTitle", stageNames[stage] || "正在生成谱面");
  setText("#stageMessage", job.message || "模型将在本机运行。");
  for (const item of document.querySelectorAll("[data-step]")) {
    const order = ["features", "planner", "generation", "packaging", "complete", "failed"];
    const current = order.indexOf(stage);
    const itemIndex = order.indexOf(item.dataset.step);
    item.classList.toggle("active", item.dataset.step === stage || (stage === "decode" && item.dataset.step === "features"));
    item.classList.toggle("done", current > itemIndex && itemIndex >= 0 && !["failed"].includes(stage));
  }
}

async function pollSession(sessionId, waitForAnalysis = false) {
  clearInterval(state.pollHandle);
  const poll = async () => {
    try {
      const response = await fetch(`/api/session/${sessionId}`, { cache: "no-store" });
      const job = await response.json();
      if (sessionId !== state.sessionId) return true;
      if (!response.ok) throw new Error(job.error || "读取生成任务失败。");
      if (job.status === "analyzing") {
        $("#uploadState").textContent = "正在分析拍点…";
        setText("#tempoText", job.message || "分析本机音频…");
        if (!waitForAnalysis) updateProgress(job);
        return false;
      }
      if (job.status === "ready") {
        clearInterval(state.pollHandle);
        setText("#uploadState", "可以生成谱面");
        setText("#durationValue", formatDuration(job.duration_ms));
        const timingAvailable = job.detected_bpm !== null && job.detected_offset_ms !== null;
        setText("#tempoText", !timingAvailable ? job.message : `Beat This! · ${Number(job.detected_bpm).toFixed(3)} BPM · offset ${Number(job.detected_offset_ms).toFixed(1)} ms`);
        setText("#timingNote", job.tempo_note || job.message);
        $("#tempoConfidence").textContent = Number(job.confidence || 0) >= .3 ? "已自动估算" : "需试听确认";
        state.autoTiming = timingAvailable ? {bpm: job.detected_bpm, offset: job.detected_offset_ms} : null;
        $("#applyTiming").disabled = !timingAvailable;
        if (timingAvailable && !$("#manualTiming").checked) {
          $("#bpmInput").value = Number(job.detected_bpm).toFixed(3);
          $("#offsetInput").value = Number(job.detected_offset_ms).toFixed(1);
        }
        const alternatives = $("#tempoAlternatives");
        alternatives.replaceChildren();
        alternatives.classList.toggle("hidden", !(job.tempo_alternatives || []).length);
        const label = document.createElement("span");
        label.textContent = "候选速度：";
        alternatives.append(label);
        for (const candidate of job.tempo_alternatives || []) {
          const choice = document.createElement("button");
          choice.type = "button";
          choice.textContent = `${candidate.bpm} BPM`;
          choice.title = `${candidate.status === "uncertain" ? "候选网格不稳定；" : ""}使用 offset ${candidate.offset_ms} ms`;
          choice.addEventListener("click", () => {
            $("#bpmInput").value = candidate.bpm;
            $("#offsetInput").value = candidate.offset_ms;
            setText("#tempoText", `${candidate.bpm} BPM · offset ${candidate.offset_ms} ms`);
            invalidatePreview("已切换拍速候选，重新生成后生效。");
          });
          alternatives.append(choice);
        }
        $("#titleInput").value = job.title || $("#titleInput").value;
        $("#artistInput").value = job.artist || "Unknown artist";
        $("#tapTempo").disabled = false;
        $("#tapHint").textContent = "点击歌曲第一拍，再继续点按";
        state.previousStatus = null;
        state.busy = false;
        state.uploading = false;
        updateGenerateButton();
        return true;
      }
      if (job.status === "queued" || job.status === "running") {
        state.busy = true;
        updateProgress(job);
        updateGenerateButton();
        return false;
      }
      if (job.status === "complete") {
        clearInterval(state.pollHandle);
        state.busy = false;
        state.previousStatus = "complete";
        updateProgress(job);
        setResult(job);
        updateGenerateButton();
        return true;
      }
      if (job.status === "failed") {
        clearInterval(state.pollHandle);
        state.busy = false;
        state.uploading = false;
        state.previousStatus = "failed";
        updateProgress(job);
        $("#generationError").classList.remove("hidden");
        $("#generationError").textContent = job.message || "音频读取或谱面生成失败。";
        setText("#uploadState", "请检查错误信息");
        if (waitForAnalysis) setText("#tempoText", job.message || "音频无法读取。");
        updateGenerateButton();
        return true;
      }
      return false;
    } catch (error) {
      clearInterval(state.pollHandle);
      setNotice(error.message);
      state.busy = false;
      state.uploading = false;
      updateGenerateButton();
      return true;
    }
  };
  const done = await poll();
  if (!done) state.pollHandle = window.setInterval(poll, 750);
}

async function generate() {
  if (!state.sessionId || !state.modelAvailable || state.busy) return;
  const model = selectedModel();
  if (!model?.available) {
    setNotice("当前生成模型不可用，请切换检查点。");
    return;
  }
  const bpm = Number($("#bpmInput").value);
  const offset = Number($("#offsetInput").value);
  const stars = Number($("#starInput").value);
  if (!Number.isFinite(bpm) || bpm < 30 || bpm > 400) {
    setNotice("BPM 需介于 30 到 400。请修正后重试。");
    $("#bpmInput").focus();
    return;
  }
  if ($("#offsetInput").value.trim() === "" || !Number.isFinite(offset) || offset < -3000 || offset > 60000) {
    setNotice("Offset 需介于 -3000 到 60000 毫秒。请修正后重试。");
    $("#offsetInput").focus();
    return;
  }
  setNotice("");
  state.busy = true;
  updateGenerateButton();
  $("#resultDownload").classList.add("hidden");
  $("#resultStats").classList.add("hidden");
  $("#generationError").classList.add("hidden");
  updateProgress({ stage: "queued", status: "queued", progress: 0.1, message: "准备启动本地 GPU 模型。" });
  const settings = {
    bpm, first_beat_ms: offset, target_star: stars, main_style: state.selectedStyle, secondary_styles: state.selectedSecondaryStyles,
    density_model_id: state.selectedDensityId,
    arrangement_model_id: state.selectedArrangementId,
    structure_id: state.selectedStructureId,
    coordination_id: state.selectedCoordinationId,
    budget_id: state.selectedBudgetId,
    title: $("#titleInput").value.trim(), artist: $("#artistInput").value.trim(), model_id: model.id,
  };
  try {
    const response = await fetch(`/api/generate/${state.sessionId}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(settings),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `启动生成失败 (${response.status})`);
    state.previousStatus = null;
    await pollSession(state.sessionId, false);
  } catch (error) {
    state.busy = false;
    updateGenerateButton();
    $("#generationError").classList.remove("hidden");
    $("#generationError").textContent = error.message;
  }
}

function tapBeat() {
  $("#manualTiming").checked = true;
  if ($("#audioPlayer").classList.contains("hidden")) return;
  const audio = $("#audioPlayer");
  const tap = performance.now();
  if (!state.taps.length || tap - state.taps[state.taps.length - 1] > 2500) {
    state.taps = [tap];
    $("#offsetInput").value = Math.round(audio.currentTime * 1000);
    $("#tapHint").textContent = "再点 4–7 次稳定拍速";
    setText("#tempoText", `已记录第一拍 ${Math.round(audio.currentTime * 1000)} ms · 继续点按`);
    return;
  }
  state.taps.push(tap);
  if (state.taps.length > 9) state.taps.shift();
  if (state.taps.length >= 4) {
    const intervals = [];
    for (let index = 1; index < state.taps.length; index += 1) intervals.push(state.taps[index] - state.taps[index - 1]);
    const sorted = intervals.toSorted((left, right) => left - right);
    const median = sorted[Math.floor(sorted.length / 2)];
    const bpm = 60_000 / median;
    if (bpm >= 60 && bpm <= 220) {
      $("#bpmInput").value = bpm.toFixed(1);
      $("#tempoConfidence").textContent = "手动校准";
      setText("#tempoText", `${bpm.toFixed(1)} BPM · 首拍 ${$("#offsetInput").value} ms`);
      $("#tapHint").textContent = `已采样 ${intervals.length} 个间隔，可继续提高稳定性`;
      invalidatePreview("已更改拍点设置；重新生成后会更新此规划预览。");
    } else {
      $("#tapHint").textContent = "拍速超出 60–220 BPM，请重新点按";
    }
  }
}

async function initialize() {
  renderStyles();
  $("#modelSelect").addEventListener("change", (event) => {
    state.selectedModelId = event.target.value;
    updateModelSelector();
    const model = selectedModel();
    invalidatePreview(`已切换到 ${model?.label || "其他模型"}；重新生成后会更新谱面。`);
    updateGenerateButton();
  });
  $("#densitySelect").addEventListener("change", (event) => {
    state.selectedDensityId = event.target.value;
    invalidatePreview("已更改密度方案，重新生成后生效。");
    updateGenerateButton();
  });
  $("#arrangementSelect").addEventListener("change", (event) => {
    state.selectedArrangementId = event.target.value;
    invalidatePreview("已更改主副编排模型，重新生成后生效。");
    updateGenerateButton();
  });
  $("#coordinationSelect").addEventListener("change", (event) => {
    state.selectedCoordinationId = event.target.value;
    invalidatePreview("已更改编排与密度协调方案，重新生成后生效。");
    updateGenerateButton();
  });
  $("#budgetSelect").addEventListener("change", (event) => {
    state.selectedBudgetId = event.target.value;
    invalidatePreview("已更改生成器密度预算，重新生成后生效。");
    updateGenerateButton();
  });
  $("#structureSelect").addEventListener("change", (event) => {
    state.selectedStructureId = event.target.value;
    invalidatePreview("已更改段落与乐句规划，重新生成后生效。");
    updateGenerateButton();
  });
  const star = $("#starInput");
  star.addEventListener("input", () => {
    setText("#starValue", Number(star.value).toFixed(1));
    invalidatePreview("已更改目标难度；重新生成后会更新此规划预览。");
  });
  $("#dropzone").addEventListener("click", () => $("#audioFile").click());
  $("#audioFile").addEventListener("change", (event) => uploadFile(event.target.files[0]));
  $("#dropzone").addEventListener("dragover", (event) => { event.preventDefault(); $("#dropzone").classList.add("dragging"); });
  $("#dropzone").addEventListener("dragleave", () => $("#dropzone").classList.remove("dragging"));
  $("#dropzone").addEventListener("drop", (event) => {
    event.preventDefault();
    $("#dropzone").classList.remove("dragging");
    uploadFile(event.dataTransfer.files[0]);
  });
  $("#clearAudio").addEventListener("click", () => {
    clearInterval(state.pollHandle);
    state.sessionId = null;
    state.sourceFile = null;
    state.uploading = false;
    state.taps = [];
    $("#audioFile").value = "";
    $("#audioPlayer").removeAttribute("src");
    if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
    state.sourceUrl = null;
    $("#dropTitle").textContent = "拖入歌曲，或点击选择";
    $("#dropMeta").textContent = "MP3 · OGG · WAV · FLAC · M4A   /   最大 180 MiB";
    resetAnalysisView();
    updateGenerateButton();
  });
  $("#generateButton").addEventListener("click", generate);
  $("#measureTempo").addEventListener("click", async () => {
    if (!state.sessionId || state.busy || state.uploading) return;
    invalidatePreview("重新测量拍点后，请重新生成谱面以应用新的时间网格。");
    state.uploading = true;
    updateGenerateButton();
    try {
      const response = await fetch(`/api/analyze/${state.sessionId}`, {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || "自动测量失败。");
      setText("#tempoText", "正在自动测量 BPM 和 offset…");
      await pollSession(state.sessionId, true);
    } catch (error) {
      state.uploading = false;
      setNotice(error.message);
      updateGenerateButton();
    }
  });
  $("#tapTempo").addEventListener("click", tapBeat);
  $("#applyTiming").addEventListener("click", () => {
    if (!state.autoTiming) return;
    $("#manualTiming").checked = false;
    $("#bpmInput").value = state.autoTiming.bpm;
    $("#offsetInput").value = state.autoTiming.offset;
    invalidatePreview("已应用自动检测建议，重新生成后生效。");
  });
  for (const id of ["#bpmInput", "#offsetInput"]) {
    $(id).addEventListener("input", () => { $("#manualTiming").checked = true; });
  }
  $("#bpmInput").addEventListener("change", () => { invalidatePreview("已更改 BPM；重新生成后会更新此规划预览。"); updateGenerateButton(); });
  $("#offsetInput").addEventListener("change", () => invalidatePreview("已更改第一拍时间；重新生成后会更新此规划预览。"));
  $("#titleInput").addEventListener("input", () => invalidatePreview("已更改歌曲资料；重新生成后会更新此谱面。"));
  $("#artistInput").addEventListener("input", () => invalidatePreview("已更改歌曲资料；重新生成后会更新此谱面。"));
  window.addEventListener("resize", () => drawTimeline(state.renderedBars));
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    const health = await response.json();
    renderModelOptions(health);
    state.modelAvailable = response.ok && health.status === "ok" && health.models_available;
    setText("#runtimeLabel", health.device || "本机运行");
    if (!state.modelAvailable) {
      const detail = (health.missing || []).map((path) => path.replace(`${location.origin}/`, "")).join("、");
      setNotice(`${health.runtime_error || "生成模型或特征资产缺失。"}${detail ? `缺少：${detail}` : ""}`);
    }
  } catch {
    state.modelAvailable = false;
    setNotice("无法连接本机生成服务。请从项目根目录双击 studio.bat 启动。");
  }
  updateGenerateButton();
  drawTimeline([]);
}

initialize();
