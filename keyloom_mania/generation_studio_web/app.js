"use strict";

const $ = (selector) => document.querySelector(selector);

const STYLE_GROUPS = [
  { name: "连段 / 流动", label: "流", color: "#66ccff", styles: ["Stream", "Chordstream", "Jumpstream", "Handstream", "Quadstream"] },
  { name: "叠键 / 手癖", label: "叠", color: "#ff66ab", styles: ["Jacks", "Longjacks", "Chordjacks", "Minijacks", "Gluts"] },
  { name: "轮转 / 颤音", label: "轮", color: "#ffcc22", styles: ["Rolls", "Trills", "Minitrills", "Split Trill", "Jumptrill"] },
];

const STYLE_COLORS = {
  Stream: "#8fdcff", Chordstream: "#66ccff", Jumpstream: "#4fb4f0", Handstream: "#7fe3e6", Quadstream: "#9fb0ff",
  Jacks: "#ff8fc2", Longjacks: "#ff66ab", Chordjacks: "#ff5f73", Minijacks: "#ffa3cf", Gluts: "#d77bdc",
  Rolls: "#ffd95c", Trills: "#ffcc22", Minitrills: "#ffb347", "Split Trill": "#d9e85a", Jumptrill: "#b3d944",
};

// osu!lazer star-rating spectrum, so the target and actual stars read like an in-game difficulty pill
const STAR_SPECTRUM = [
  [0.1, "#aaaaaa"], [0.1, "#4290fb"], [1.25, "#4fc0ff"], [2.0, "#4fffd5"], [2.5, "#7cff4f"], [3.3, "#f6f05c"],
  [4.2, "#ff8068"], [4.9, "#ff4e6f"], [5.8, "#c645b8"], [6.7, "#6563de"], [7.7, "#18158e"], [9.0, "#000000"],
];

function starColor(stars) {
  const value = Math.max(0.1, Number(stars) || 0);
  const hex = (color) => [1, 3, 5].map((index) => parseInt(color.slice(index, index + 2), 16));
  for (let index = 1; index < STAR_SPECTRUM.length; index += 1) {
    const [right, rightColor] = STAR_SPECTRUM[index];
    if (value > right) continue;
    const [left, leftColor] = STAR_SPECTRUM[index - 1];
    const amount = right === left ? 1 : (value - left) / (right - left);
    const [a, b] = [hex(leftColor), hex(rightColor)];
    return `rgb(${a.map((channel, k) => Math.round(channel + (b[k] - channel) * amount)).join(" ")})`;
  }
  return "#000000";
}

function paintStarPill(element, stars, digits = 2) {
  if (!element) return;
  const value = Number(stars);
  element.style.setProperty("--star-bg", starColor(value));
  element.style.setProperty("--star-fg", value >= 6.5 ? "#ffd966" : "rgb(0 0 0 / 75%)");
  if (element.id === "starPill") setText("#starValue", value.toFixed(1));
  else element.textContent = `★ ${value.toFixed(digits)}`;
}

function syncTargetStar() {
  const slider = $("#starInput");
  const value = Number(slider.value);
  slider.style.setProperty("--fill", `${(value - slider.min) / (slider.max - slider.min) * 100}%`);
  paintStarPill($("#starPill"), value, 1);
  paintStarPill($("#wedgeStar"), value);
}

function syncTiming() {
  const bpm = Number($("#bpmInput").value);
  setText("#wedgeBpm", Number.isFinite(bpm) && bpm > 0 ? String(Math.round(bpm * 100) / 100) : "—");
}

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
  selectedStarConditionId: "global",
  starConditionOptions: [],
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

// Options that need another structure stay visible but disabled until it is chosen.
function renderStarConditionOptions() {
  const root = $("#starConditionSelect");
  root.replaceChildren();
  for (const item of state.starConditionOptions) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label + (item.status === "research" ? "（研究）" : "");
    option.disabled = item.available === false
      || (item.requires_structure_id && item.requires_structure_id !== state.selectedStructureId);
    root.append(option);
  }
  const selected = state.starConditionOptions.find(item => item.id === state.selectedStarConditionId);
  if (!selected || (selected.requires_structure_id && selected.requires_structure_id !== state.selectedStructureId)) {
    state.selectedStarConditionId = "global";
  }
  root.value = state.selectedStarConditionId;
}

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
  state.starConditionOptions = health.star_condition_options || [{id: "global", label: "全曲统一难度条件", available: true}];
  state.selectedStarConditionId = health.default_star_condition_id || "global";
  renderStarConditionOptions();
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
    row.style.setProperty("--group-color", group.color);
    row.setAttribute("role", "group");
    row.setAttribute("aria-label", group.name);
    const title = document.createElement("span");
    title.className = "style-group-name";
    title.setAttribute("aria-hidden", "true");
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
    section.style.setProperty("--group-color", group.color);
    section.setAttribute("role", "group");
    section.setAttribute("aria-label", group.name);
    const heading = document.createElement("span");
    heading.className = "group-swatch";
    heading.setAttribute("aria-hidden", "true");
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
  $("#wedgeStyle").textContent = state.selectedStyle;
  $("#panelStyleName").textContent = state.selectedStyle;
  $("#panelStyleName").style.color = color;
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
  $("#starConditionSelect").disabled = state.uploading || state.busy;
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
  setText("#previewSubtitle", "生成后显示逐小节规划。");
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
  const height = 110;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  const left = 14, right = width - 14, top = 12, bottom = height - 10;
  for (const row of [1, 2, 3]) {
    const y = top + (bottom - top) * row / 4;
    context.strokeStyle = "rgb(255 255 255 / 6%)";
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
      context.fillStyle = index % 4 === 0 ? "rgb(255 255 255 / 14%)" : "rgb(255 255 255 / 9%)";
      context.fillRect(x + gap / 2, bottom - 8, Math.max(1, stride - gap), 8);
      continue;
    }
    const color = STYLE_COLORS[bar.style] || STYLE_COLORS[state.selectedStyle] || "#ff66ab";
    const weight = Math.min(1, Math.max(.34, Number(bar.notes_per_beat || 0) / 3));
    const support = bar.role === "support";
    const barHeight = 20 + weight * 52;
    context.globalAlpha = support ? .50 + .30 * weight : .55 + .45 * weight;
    context.fillStyle = color;
    context.fillRect(x + gap / 2, bottom - barHeight, Math.max(1, stride - gap), barHeight);
    context.globalAlpha = 1;
    if (index > 0 && bar.segment_id >= 0 && bar.segment_id !== bars[index - 1].segment_id) {
      context.strokeStyle = "rgb(255 255 255 / 40%)";
      context.beginPath();
      context.moveTo(x, top);
      context.lineTo(x, bottom);
      context.stroke();
    }
    if (bar.section_id >= 0 && (index === 0 || bar.section_id !== bars[index - 1].section_id)) {
      context.strokeStyle = "rgb(255 102 171 / 95%)";
      context.lineWidth = 2;
      context.beginPath(); context.moveTo(x, top); context.lineTo(x, bottom); context.stroke();
      context.lineWidth = 1;
    }
    if (support && stride > 4) {
      context.fillStyle = "#ffcc22";
      context.fillRect(x + gap / 2, bottom - barHeight, Math.max(1, stride - gap), 2);
    } else if (barHeight > 40 && stride > 4) {
      context.fillStyle = "rgb(255 255 255 / 30%)";
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
  if (manifest.structure_id === "hierarchical" || manifest.structure_id === "peak-anchored") {
    setText("#previewSubtitle", `${manifest.bpm.toFixed(1)} BPM · ${styleSummary} · ${manifest.hierarchy_sections} 个段落 / ${manifest.hierarchy_phrases} 个乐句 · 粉线为段落边界，白线为乐句边界${unusedDetail}`);
  }
  if (manifest.structure_id === "peak-anchored" && manifest.hierarchy_density) {
    const shaping = manifest.hierarchy_density;
    const sections = shaping.sections || [];
    const peak = sections.filter(item => item.tier === "peak").length;
    const reduced = sections.filter(item => item.tier === "reduced").length;
    const change = (Number(shaping.note_budget_change) * 100).toFixed(1);
    setText("#previewSubtitle", `${$("#previewSubtitle").textContent} · 峰值段 ${peak} 个保持原密度，${reduced} 个段落降密 · 计划音符 ${change}%`);
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
    diagnostic.textContent = `密度诊断 · 音乐计划 ${execution.music_notes} 音 / 实际导出 ${execution.export_notes} 音 · 排键减键 ${execution.prepare_removed_notes + execution.layout_removed_notes} 音 · 导出减键 ${execution.export_removed_notes} 音 · 完整拍密度误差 ${mae} 音/拍 · 预算调整 ${budget.changed_beats || 0} 拍`;
    const placement = manifest.difficulty_placement;
    if (placement && placement.top_strain_peak_share != null) {
      const percent = value => `${Math.round(Number(value) * 100)}%`;
      const ratio = placement.lower_to_peak_exported_density_median;
      diagnostic.textContent += ` · 最难 ${placement.top_sections} 段 strain 有 ${percent(placement.top_strain_peak_share)} 落在峰值段（峰值段占 ${percent(placement.peak_bar_share)} 小节）`
        + (ratio == null ? "" : ` · 其余段落实际密度为峰值段的 ${percent(ratio)}`)
        + (placement.excluded_bars ? ` · 已排除 ${placement.excluded_styles.join("、")} ${placement.excluded_bars} 小节` : "");
    }
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
  setText("#stageMessage", job.message || "");
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
          syncTiming();
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
            syncTiming();
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
    star_condition_id: state.selectedStarConditionId,
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
      syncTiming();
      $("#tempoConfidence").textContent = "手动校准";
      setText("#tempoText", `${bpm.toFixed(1)} BPM · 首拍 ${$("#offsetInput").value} ms`);
      $("#tapHint").textContent = `已采样 ${intervals.length} 个间隔，可继续提高稳定性`;
      invalidatePreview("已更改拍点设置；重新生成后会更新此规划预览。");
    } else {
      $("#tapHint").textContent = "拍速超出 60–220 BPM，请重新点按";
    }
  }
}

const motion = {
  reduced: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  triangles: [],
  analyser: null,
  spectrum: null,
};

function seededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value = value + 0x6D2B79F5 | 0;
    let mixed = Math.imul(value ^ value >>> 15, 1 | value);
    mixed = mixed + Math.imul(mixed ^ mixed >>> 7, 61 | mixed) ^ mixed;
    return ((mixed ^ mixed >>> 14) >>> 0) / 4294967296;
  };
}

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

// beat phase from the player and the timing the chart will actually use, so the disc doubles as an offset check
function beatPulse(audio) {
  const bpm = Number($("#bpmInput").value);
  const offset = Number($("#offsetInput").value);
  if (!audio || audio.paused || !Number.isFinite(bpm) || bpm < 30 || !Number.isFinite(offset)) return 0;
  const beat = 60_000 / bpm;
  const phase = (((audio.currentTime * 1000 - offset) % beat) + beat) % beat;
  return Math.exp(-phase / beat * 7);
}

function connectAnalyser() {
  if (motion.analyser) return;
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextClass();
    const source = context.createMediaElementSource($("#audioPlayer"));
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = .72;
    source.connect(analyser);
    analyser.connect(context.destination);
    motion.analyser = analyser;
    motion.spectrum = new Uint8Array(analyser.frequencyBinCount);
    motion.context = context;
  } catch {
    motion.analyser = null;
  }
}

function drawTriangles(time, pulse) {
  const { context, width, height } = fitCanvas($("#triangles"));
  if (motion.triangles.length === 0) {
    const random = seededRandom(11);
    for (let index = 0; index < 70; index += 1) {
      motion.triangles.push({ x: random(), y: random(), size: 30 + Math.pow(random(), 1.6) * 240, speed: 14 + random() * 26, shade: random(), alpha: .02 + random() * .05 });
    }
  }
  context.clearRect(0, 0, width, height);
  for (const triangle of motion.triangles) {
    const span = height + triangle.size * 2;
    let y = (triangle.y * span - time * triangle.speed) % span;
    if (y < 0) y += span;
    y -= triangle.size;
    const x = triangle.x * (width + 200) - 100;
    context.fillStyle = `hsl(333 ${45 + triangle.shade * 30}% ${44 + triangle.shade * 22}% / ${triangle.alpha * (1 + pulse * .8)})`;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(x + triangle.size * .577, y + triangle.size);
    context.lineTo(x - triangle.size * .577, y + triangle.size);
    context.closePath();
    context.fill();
  }
}

function drawVisualiser(time, pulse, playing) {
  const { context, width, height } = fitCanvas($("#visualiser"));
  const radius = $(".disc-lg").getBoundingClientRect().width / 2 + 2;
  const bars = 120;
  context.clearRect(0, 0, width, height);
  if (playing && motion.analyser) motion.analyser.getByteFrequencyData(motion.spectrum);
  context.save();
  context.translate(width / 2, height / 2);
  context.rotate(time * .2);
  context.fillStyle = "rgb(255 255 255 / 24%)";
  const reach = Math.max(10, width / 2 - radius - 4);
  for (let index = 0; index < bars; index += 1) {
    let level;
    if (playing && motion.spectrum) {
      const bin = Math.floor(Math.pow(((index * 7) % bars) / bars, 1.6) * motion.spectrum.length * .7);
      level = motion.spectrum[bin] / 255;
    } else {
      level = .05 + .04 * Math.sin(time * 1.4 + index * .5);
    }
    const length = 3 + level * reach;
    context.save();
    context.rotate(index / bars * Math.PI * 2);
    context.fillRect(-radius * Math.PI / bars * .5, -radius - length, radius * Math.PI * 2 / bars * .5, length);
    context.restore();
  }
  context.restore();
  $(".disc-lg").style.transform = `scale(${1 + pulse * .045})`;
}

// the promo's mark: four warp threads (the keys) on a pink disc, a little 4K phrase falling down them,
// and one weft ribbon woven over threads 1 and 3, under 2 and 4
const LOGO_PHRASE = [0, 2, 1, 3, 0, 3, 1, 2, 0, 1, 2, 3, 1, 0, 3, 2];
const LOGO_TRIANGLES = (() => {
  const random = seededRandom(5);
  return Array.from({ length: 14 }, () => ({ size: .2 + random() * .5, x: random() * 2 - 1, speed: .3 + random() * .4, y: random() * 3, alpha: .04 + random() * .06 }));
})();

function drawLogo(canvas, time, beat) {
  const size = canvas.clientWidth;
  if (!size) return;
  const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
  const pixels = Math.round(size * dpr);
  if (canvas.width !== pixels) { canvas.width = pixels; canvas.height = pixels; }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, size, size);
  const radius = size / 2, inner = radius * .87;
  g.save();
  g.translate(radius, radius);
  g.fillStyle = "#fff";
  g.beginPath(); g.arc(0, 0, radius, 0, Math.PI * 2); g.fill();
  const fill = g.createLinearGradient(0, -inner, 0, inner);
  fill.addColorStop(0, "#ff7bb8"); fill.addColorStop(1, "#e6418c");
  g.fillStyle = fill;
  g.beginPath(); g.arc(0, 0, inner, 0, Math.PI * 2); g.fill();
  g.clip();
  if (size >= 60) {
    for (const triangle of LOGO_TRIANGLES) {
      const side = inner * triangle.size, x = triangle.x * inner;
      const y = ((triangle.y * inner - time * inner * triangle.speed * .5) % (3 * inner) + 3 * inner) % (3 * inner) - 1.5 * inner;
      g.fillStyle = `rgb(255 255 255 / ${triangle.alpha})`;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + side * .577, y + side); g.lineTo(x - side * .577, y + side); g.fill();
    }
  }
  const threads = [-.48, -.16, .16, .48].map((value) => value * inner);
  g.fillStyle = "rgb(255 255 255 / 28%)";
  for (const x of threads) g.fillRect(x - inner * .025, -inner, inner * .05, inner * 2);
  const note = inner * .11, travel = inner * 2.4, fall = beat * 4, step = beat / 2;
  for (let offset = -8; offset < 8; offset += 1) {
    const index = Math.floor(time / step) + offset;
    const column = LOGO_PHRASE[((index % 16) + 16) % 16];
    const y = -inner * 1.2 + ((time - index * step) / fall + .5) * travel;
    if (y < -inner - note || y > inner + note) continue;
    g.fillStyle = column === 1 || column === 2 ? "#bfe6ff" : "#fff";
    g.beginPath(); g.arc(threads[column], y, note, 0, Math.PI * 2); g.fill();
  }
  const weftY = inner * .28, thickness = inner * .13;
  const wave = (x) => weftY + Math.sin(x / inner * Math.PI * 1.55) * inner * .06;
  g.strokeStyle = "#fff"; g.lineWidth = thickness; g.lineCap = "round";
  g.beginPath();
  for (let x = -inner; x <= inner; x += Math.max(1, inner / 40)) x === -inner ? g.moveTo(x, wave(x)) : g.lineTo(x, wave(x));
  g.stroke();
  g.fillStyle = "#ef5a9f";
  for (const column of [1, 3]) g.fillRect(threads[column] - inner * .03, wave(threads[column]) - thickness * .7, inner * .06, thickness * 1.4);
  g.restore();
}

function drawLogos(time, beat) {
  for (const canvas of document.querySelectorAll("canvas.disc")) drawLogo(canvas, time, beat);
}

// while the song plays the phrase falls on the chart's own beat grid; otherwise at a calm idle tempo
function logoClock(audio, now, playing) {
  const bpm = Number($("#bpmInput").value);
  const offset = Number($("#offsetInput").value);
  if (playing && Number.isFinite(bpm) && bpm >= 30 && Number.isFinite(offset)) {
    return { time: audio.currentTime - offset / 1000, beat: 60 / bpm };
  }
  return { time: now, beat: .45 };
}

function startMotion() {
  const audio = $("#audioPlayer");
  audio.addEventListener("play", () => {
    connectAnalyser();
    motion.context?.resume?.();
  });
  if (motion.reduced) {
    const still = () => { drawTriangles(0, 0); drawVisualiser(0, 0, false); drawLogos(0, .45); };
    still();
    window.addEventListener("resize", still);
    return;
  }
  const frame = (now) => {
    const time = now / 1000;
    const playing = !audio.paused && !audio.classList.contains("hidden");
    const pulse = beatPulse(audio);
    if (!document.hidden) {
      const clock = logoClock(audio, time, playing);
      drawLogos(clock.time, clock.beat);
      drawTriangles(time + (playing ? audio.currentTime * .6 : 0), pulse);
      drawVisualiser(time, pulse, playing);
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

// CSP keeps font-src to 'self'; FontFace from bytes needs no fetch, so the embedded promo fonts still load
function loadFonts() {
  if (!("FontFace" in window)) return;
  const decode = (data) => Uint8Array.from(atob(data), (char) => char.charCodeAt(0)).buffer;
  const faces = [
    new FontFace("Outfit", decode(FONT_OUTFIT), { weight: "300 900" }),
    new FontFace("JetBrains Mono", decode(FONT_MONO), { weight: "500" }),
  ];
  for (const face of faces) {
    face.load().then((loaded) => {
      document.fonts.add(loaded);
      drawTimeline(state.renderedBars);
    }).catch(() => {});
  }
}

async function initialize() {
  loadFonts();
  startMotion();
  syncTargetStar();
  syncTiming();
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
    renderStarConditionOptions();
    invalidatePreview("已更改段落与乐句规划，重新生成后生效。");
    updateGenerateButton();
  });
  $("#starConditionSelect").addEventListener("change", (event) => {
    state.selectedStarConditionId = event.target.value;
    invalidatePreview("已更改生成器难度条件，重新生成后生效。");
    updateGenerateButton();
  });
  const star = $("#starInput");
  star.addEventListener("input", () => {
    syncTargetStar();
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
    syncTiming();
    invalidatePreview("已应用自动检测建议，重新生成后生效。");
  });
  for (const id of ["#bpmInput", "#offsetInput"]) {
    $(id).addEventListener("input", () => { $("#manualTiming").checked = true; syncTiming(); });
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
    setNotice("无法连接本机生成服务，请运行 studio.bat。");
  }
  updateGenerateButton();
  drawTimeline([]);
}

// Outfit (variable, Latin) and JetBrains Mono Medium, both SIL OFL 1.1; see THIRD_PARTY_NOTICES.md
const FONT_OUTFIT = "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";
const FONT_MONO = "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";

initialize();
