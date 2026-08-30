/* 视频工作室 — ChatGPT 式单页前端
 * 同源 API：/api/*，共享口令 Bearer 鉴权；文件直链也需鉴权，故一律 fetch(blob) → ObjectURL。
 */
"use strict";

/* ===== 常量与状态 ===== */
const TOKEN_KEY = "cvs_token";
const POLL_MS = 2000;
const GENERATION_ASPECT_RATIOS = Object.freeze(["16:9", "9:16"]);
const GENERATION_RESOLUTIONS = Object.freeze(["480p", "768p"]);
const GENERATION_CONFIG_KEYS = Object.freeze([
  "optimize_image", "remove_subtitle", "remove_watermark",
]);
const DIALOGUE_REVIEW_POLICIES = Object.freeze([
  "auto_continue", "review_required",
]);

const STATUS_TEXT = { queued: "排队中", processing: "处理中", done: "已完成", failed: "失败" };

const state = {
  token: null,
  conversations: [],
  currentId: null,
  detail: null,        // 当前会话详情
  file: null,          // composer 已选文件
  clientRequestId: null, // 幂等键：内容变更/成功才轮换，失败重试复用
  uploading: false,
  pollTimer: null,
  pollToken: 0,
  detailSeq: 0,        // 防止过期响应覆盖新渲染
  objectURLs: [],      // 当前 stream 渲染产生的 blob URL，重渲染前统一 revoke
  ppDetail: null,      // 后处理弹窗对应的会话详情
  ppAskDismissed: {},  // cid → true：后处理入口消息已点「否」（会话内记忆，重渲染不复活）
  ppResultExpanded: {}, // cid → 后处理结果是否由用户展开；轮询重渲染时保留，切会话重置
  segmentProductsExpanded: {}, // cid → 分段产物是否由用户展开；轮询重渲染时保留，切会话重置
  generationDrafts: {}, // cid → 最终视频表单草稿；轮询重渲染时保留用户输入
  generationSubmitting: {}, // cid → true：本页已有 /submit 请求在途，阻止重复提交
  historyDetails: {}, // cid → 近期会话的 GET 详情；只用于补齐历史识别信息
  historyThumbnailURLs: {}, // cid → 侧栏首帧 ObjectURL，与 stream 媒体生命周期隔离
  historyHydrating: false, // 受限并发补齐进行中，避免重复 GET 风暴
  generationConfigCapability: null, // 仅接受 /api/capabilities 的精确 generation_config 合同
  generationConfigCapabilityLoaded: false,
  dialogueReviewCapability: null,
  dialogueReviewCapabilityLoaded: false,
  dialogueReviewDrafts: {}, // cid → 本地校对稿；轮询保留，服务端 revision 变化时重置
  frameSelections: {}, // cid:scope → 选中的 segment/frame；轮询和展开切换时保留
  framePickerOpen: {}, // cid:scope → 图片下拉是否展开
  promptDraft: null,   // 当前图片优化提示词草稿；跨轮询保留，跨操作由统一守卫处理
  promptWorkspaceMode: {}, // cid:segment → generation/dialogue/image
  detailSig: null,     // 当前已渲染详情的状态签名：轮询比对，签名不变不碰 DOM（根治轮询闪烁）
};

class AuthError extends Error {}
class ApiError extends Error {
  constructor(message, code = "") {
    super(message);
    this.code = code;
  }
}

/* ===== 小工具 ===== */
const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function icon(name, cls) {
  const span = el("span", "ic" + (cls ? " " + cls : ""));
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", "#" + name);
  svg.appendChild(use);
  span.appendChild(svg);
  return span;
}

function fmtBytes(n) {
  if (!Number.isFinite(n) || n < 0) return "";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + " MB";
  return (n / 1024 / 1024 / 1024).toFixed(2) + " GB";
}

function apiErrorFromPayload(data, fallback) {
  const detail = data && (data.detail || data.error || data.message);
  if (detail && typeof detail === "object") {
    return new ApiError(
      String(detail.message || fallback),
      typeof detail.code === "string" ? detail.code : ""
    );
  }
  return new ApiError(detail ? String(detail) : fallback);
}

function showActionError(error, errorElement, controls = []) {
  errorElement.textContent = error && error.code === "client_refresh_required"
    ? "页面版本已更新，请刷新页面后重试。"
    : String(error && error.message ? error.message : error || "操作失败，请稍后重试");
  errorElement.hidden = false;
  for (const control of controls) control.disabled = false;
}

function createImagePromptDraft(conversationId, segmentIndex, prompt) {
  const value = prompt || {};
  const normalizedIndex = segmentIndex === null || segmentIndex === undefined
    ? 0 : (Number.isInteger(segmentIndex) && segmentIndex >= 0 ? segmentIndex : null);
  return {
    conversationId,
    segmentIndex: normalizedIndex,
    text: String(value.text || ""),
    savedText: String(value.text || ""),
    defaultText: String(value.default_text || ""),
    sha256: String(value.sha256 || ""),
    dirty: false,
    saving: false,
    save: null,
  };
}

function restoreImagePromptDefault(draft) {
  draft.text = draft.defaultText;
  draft.dirty = draft.text !== draft.savedText;
  return draft;
}

function mergeImagePromptDraft(draft, prompt) {
  if (!draft || !prompt) return draft;
  draft.defaultText = String(prompt.default_text || "");
  if (!draft.dirty) {
    draft.text = String(prompt.text || "");
    draft.savedText = draft.text;
    draft.sha256 = String(prompt.sha256 || "");
  }
  return draft;
}

function buildImagePromptPatch(draft) {
  if (!Number.isInteger(draft.segmentIndex) || draft.segmentIndex < 0) {
    throw new Error("图片优化提示词段号无效");
  }
  return {
    confirm: true,
    segment_index: draft.segmentIndex,
    expected_sha256: draft.sha256,
    prompt: draft.text,
  };
}

async function saveImageOptimizationPrompt(detail, segmentIndex, draft, request = apiJSON) {
  if (!Number.isInteger(segmentIndex) || segmentIndex < 0) {
    throw new Error("图片优化提示词段号无效");
  }
  if (!draft || draft.segmentIndex !== segmentIndex) {
    throw new Error("图片优化提示词段号已变化");
  }
  if (!imagePromptEditable(detail, segmentIndex)) {
    throw new Error("当前会话未开放图片优化编辑");
  }
  return request(
    "/api/conversations/" + encodeURIComponent(detail.id) + "/image-optimization-prompt",
    {method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify(buildImagePromptPatch(draft))},
  );
}

function promptScopeKey(conversationId, segmentIndex) {
  return conversationId + ":" + (Number.isInteger(segmentIndex) && segmentIndex >= 0 ? segmentIndex : "invalid");
}

function promptSegmentIndex(segment) {
  if (segment === null) return 0;
  return segment && Number.isInteger(segment.index) && segment.index > 0 ? segment.index : null;
}

function promptWorkspaceModes() {
  return [
    ["generation", "展开生成提示词"],
    ["dialogue", "展开段台词"],
    ["image", "展开图片优化"],
  ];
}

function postprocessAskDefault() {
  return "no";
}

function hasExactKeys(value, keys) {
  return !!value && typeof value === "object" && !Array.isArray(value)
    && Object.keys(value).sort().join("|") === keys.slice().sort().join("|");
}

function validGenerationConfig(value) {
  return hasExactKeys(value, GENERATION_CONFIG_KEYS)
    && GENERATION_CONFIG_KEYS.every((key) => typeof value[key] === "boolean");
}

function normalizeGenerationConfigCapability(payload) {
  const capability = payload && payload.generation_config;
  if (!capability || capability.supported !== true
      || capability.create_field !== "generation_config"
      || capability.encoding !== "multipart_json"
      || !hasExactKeys(capability.fields, GENERATION_CONFIG_KEYS)
      || !GENERATION_CONFIG_KEYS.every((key) => capability.fields[key] === "boolean")
      || !validGenerationConfig(capability.defaults)
      || capability.defaults.optimize_image !== true
      || capability.defaults.remove_subtitle !== false
      || capability.defaults.remove_watermark !== false) return null;
  return {
    supported: true,
    create_field: "generation_config",
    encoding: "multipart_json",
    fields: Object.assign({}, capability.fields),
    defaults: Object.assign({}, capability.defaults),
  };
}

function buildGenerationConfigCreateField(capability, value) {
  const normalized = normalizeGenerationConfigCapability({ generation_config: capability });
  if (!normalized || !validGenerationConfig(value)) return null;
  return {
    name: normalized.create_field,
    value: JSON.stringify({
      optimize_image: value.optimize_image,
      remove_subtitle: value.remove_subtitle,
      remove_watermark: value.remove_watermark,
    }),
  };
}

function normalizeDialogueReviewCapability(payload) {
  const capability = payload && payload.dialogue_review;
  if (!capability || capability.supported !== true
      || capability.create_field !== "dialogue_review_policy"
      || capability.default !== "auto_continue"
      || capability.commit_path !== "/api/conversations/{id}/dialogue-review/commit"
      || !Array.isArray(capability.policies)
      || capability.policies.length !== DIALOGUE_REVIEW_POLICIES.length
      || !DIALOGUE_REVIEW_POLICIES.every((value, index) => capability.policies[index] === value)) {
    return null;
  }
  return {
    supported: true,
    create_field: capability.create_field,
    default: capability.default,
    policies: capability.policies.slice(),
    commit_path: capability.commit_path,
  };
}

function buildDialogueReviewCreateField(capability, policy, mode) {
  const normalized = normalizeDialogueReviewCapability({ dialogue_review: capability });
  if (!normalized || mode !== "auto" || !normalized.policies.includes(policy)) return null;
  return { name: normalized.create_field, value: policy };
}

function buildDialogueReviewCommitPayload(review, lines, requestId) {
  if (!review || review.status !== "waiting" || review.editable !== true
      || !Number.isInteger(review.revision) || review.revision < 1
      || typeof review.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(review.sha256)
      || !Array.isArray(lines) || typeof requestId !== "string" || !requestId.trim()) {
    throw new Error("台词校对状态已变化，请刷新后重试");
  }
  return {
    confirm: true,
    client_request_id: requestId.trim(),
    expected_revision: review.revision,
    expected_sha256: review.sha256,
    lines: lines.map((line) => ({
      text: String(line.text || "").trim(),
      start_s: Number(line.start_s),
      end_s: Number(line.end_s),
    })),
  };
}

function generationConfigLabels(value) {
  if (!validGenerationConfig(value)) return [];
  return [
    value.optimize_image ? "图片优化" : "保留原图",
    value.remove_subtitle ? "去字幕" : "保留字幕",
    value.remove_watermark ? "去水印" : "保留水印",
  ];
}

function frozenGenerationConfig(detail) {
  const config = detail && detail.generation_config;
  const sha256 = String(detail && detail.generation_config_sha256 || "");
  if (!validGenerationConfig(config) || !/^[0-9a-f]{64}$/.test(sha256)) return null;
  return { config: Object.assign({}, config), sha256 };
}

async function saveActiveImagePrompt() {
  const draft = state.promptDraft;
  if (!draft || !draft.dirty || draft.saving || typeof draft.save !== "function") return !draft || !draft.dirty;
  draft.saving = true;
  try {
    return await draft.save();
  } finally {
    draft.saving = false;
  }
}

function dirtyPromptDecision() {
  const dialog = typeof document !== "undefined" ? $("draft-dialog") : null;
  if (!dialog || typeof dialog.showModal !== "function") {
    if (window.confirm("图片优化提示词尚未保存。确定保存后继续吗？")) return Promise.resolve("save");
    return Promise.resolve(window.confirm("丢弃未保存的修改吗？") ? "discard" : "cancel");
  }
  return new Promise((resolve) => {
    let settled = false;
    const finish = (decision) => {
      if (settled) return;
      settled = true;
      $("draft-save").onclick = null;
      $("draft-discard").onclick = null;
      $("draft-cancel").onclick = null;
      dialog.oncancel = null;
      dialog.close();
      resolve(decision);
    };
    $("draft-save").onclick = () => finish("save");
    $("draft-discard").onclick = () => finish("discard");
    $("draft-cancel").onclick = () => finish("cancel");
    dialog.oncancel = (event) => {
      event.preventDefault();
      finish("cancel");
    };
    dialog.showModal();
  });
}

async function guardDirtyPrompt() {
  const draft = state.promptDraft;
  if (!draft || !draft.dirty) return true;
  const decision = await dirtyPromptDecision();
  if (decision === "cancel") return false;
  if (decision === "discard") {
    draft.text = draft.savedText;
    draft.dirty = false;
    return true;
  }
  try {
    return await saveActiveImagePrompt();
  } catch (_) {
    return false;
  }
}

async function recoverLockedPostprocess(error, fetchLatest, inputs, lockHint, errorElement) {
  if (!error || error.code !== "postprocess_options_locked") return null;
  const latest = await fetchLatest();
  const options = latest && latest.postprocess && latest.postprocess.options;
  if (!options || typeof options !== "object") {
    throw new Error("服务端锁定选项校验失败，请刷新页面后重试");
  }
  for (const input of inputs) input.checked = options[input.value] === true;
  lockHint.hidden = false;
  errorElement.textContent = "选项已在其他页面锁定，已加载服务端选项，请直接确认";
  errorElement.hidden = false;
  return latest;
}

async function runSingleFlightPollCycle(isCurrent, load, schedule) {
  if (!isCurrent()) return;
  await load();
  if (isCurrent()) schedule();
}

// 幂等键；非安全上下文（如 LAN http 直连）无 crypto.randomUUID，退化为时间戳+随机串（满足后端 ^[0-9A-Za-z-]{8,64}$）
function newRequestId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return "rid-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 12);
}

/* 最终视频提交的纯契约能力：浏览器 UI 与无 DOM 测试共用，避免两套校验漂移。 */
function normalizeDialogueLines(dialogue) {
  let raw = dialogue;
  if (!Array.isArray(raw) && raw && typeof raw === "object") {
    raw = Array.isArray(raw.lines) ? raw.lines
      : (Array.isArray(raw.auto_lines) ? raw.auto_lines : []);
  }
  if (!Array.isArray(raw)) return [];
  return raw.map((line) => {
    if (!line || typeof line !== "object") return null;
    const start = Number(line.start_s !== undefined ? line.start_s : line.start);
    const end = Number(line.end_s !== undefined ? line.end_s : line.end);
    const text = String(line.text || "").trim();
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || start >= end || !text) return null;
    return { start_s: start, end_s: end, text };
  }).filter(Boolean);
}

function formatDialogueLines(dialogue) {
  return normalizeDialogueLines(dialogue)
    .map((line) => line.start_s + " - " + line.end_s + " | " + line.text)
    .join("\n");
}

function parseDialogueLines(text) {
  const rows = String(text || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  return rows.map((line, index) => {
    const match = line.match(/^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\|\s*(.+)$/);
    if (!match) throw new Error("第 " + (index + 1) + " 行格式应为：开始 - 结束 | 台词");
    const start = Number(match[1]);
    const end = Number(match[2]);
    const value = match[3].trim();
    if (start >= end) throw new Error("第 " + (index + 1) + " 行结束时间必须晚于开始时间");
    if (!value) throw new Error("第 " + (index + 1) + " 行台词不能为空");
    return { start_s: start, end_s: end, text: value };
  });
}

function buildCreateDialogueFields(mode, linesText) {
  if (!["auto", "none"].includes(mode)) {
    throw new Error("请选择台词模式");
  }
  return { dialogue_mode: mode };
}

function longVideoContract(detail) {
  const duration = Number(detail && detail.duration_s);
  const segmentCount = Number.isInteger(detail && detail.segment_count)
    && detail.segment_count > 0 ? detail.segment_count : null;
  const planReceipt = typeof (detail && detail.plan_receipt) === "string"
    && /^[0-9a-f]{64}$/.test(detail.plan_receipt) ? detail.plan_receipt : null;
  const hasHistoricalPlan = segmentCount !== null || planReceipt !== null;
  const isLong = Number.isFinite(duration) && duration > 0
    && (duration > 15 || (duration > 10 && hasHistoricalPlan));
  return {
    isLong,
    ready: !isLong || (segmentCount !== null && planReceipt !== null),
    segmentCount,
    planReceipt,
  };
}

function fitProfile(detail, aspectRatio) {
  const profile = detail && detail.fit_profiles && detail.fit_profiles[aspectRatio];
  if (!profile || typeof profile.fit_required !== "boolean"
      || profile.default_fit_mode !== (profile.fit_required ? "crop" : "none")) {
    throw new Error("服务端画幅适配建议无效，请刷新页面后重试");
  }
  return {
    fit_required: profile.fit_required,
    default_fit_mode: profile.default_fit_mode,
  };
}

function generationParameterDraft(detail) {
  const aspectRatio = detail && detail.aspect_ratio;
  const resolution = detail && detail.resolution;
  if (!GENERATION_ASPECT_RATIOS.includes(aspectRatio)) {
    throw new Error("服务端推荐画幅无效，请刷新页面后重试");
  }
  if (!GENERATION_RESOLUTIONS.includes(resolution)) {
    throw new Error("服务端推荐清晰度无效，请刷新页面后重试");
  }
  const profile = fitProfile(detail, aspectRatio);
  return {
    aspectRatio,
    resolution,
    fitMode: profile.default_fit_mode,
  };
}

function generationParameterSnapshot(detail) {
  const snapshot = {
    aspect_ratio: detail.aspect_ratio,
    resolution: detail.resolution,
    dialogue_mode: detail.dialogue && detail.dialogue.mode,
    fit_mode: detail.fit_mode,
    duration_s: detail.duration_s,
    segment_count: detail.segment_count,
  };
  if (longVideoContract(detail).isLong) {
    snapshot.fast_mode = generationFastMode(detail);
  }
  return snapshot;
}

function generationFastMode(detail) {
  return !!(detail && detail.generation && detail.generation.fast_mode === true);
}

function generationParameterSummary(detail) {
  const snapshot = generationParameterSnapshot(detail);
  const section = el("section", "res-section generation-parameter-summary");
  section.appendChild(el("h3", "res-h3", "生成参数"));
  const list = el("dl", "parameter-summary-grid");
  const dialogueLabels = {
    auto: "自动台词 / 保留源音轨", edit: "编辑识别台词",
    custom: "自定义台词", none: "无台词 / 静音",
  };
  const fitLabels = { none: "无需适配", crop: "裁切画面", pad: "留边完整展示" };
  const entries = [
    ["画幅", snapshot.aspect_ratio],
    ["清晰度", snapshot.resolution],
    ["台词模式", dialogueLabels[snapshot.dialogue_mode] || snapshot.dialogue_mode],
    ["适配方式", fitLabels[snapshot.fit_mode] || snapshot.fit_mode],
    ["实际总时长", Number.isFinite(Number(snapshot.duration_s))
      ? Number(snapshot.duration_s).toFixed(2) + " 秒" : "-"],
    ["长视频分段数", Number.isInteger(snapshot.segment_count)
      ? String(snapshot.segment_count) : "无（单段）"],
  ];
  entries.forEach(([label, value]) => {
    list.appendChild(el("dt", null, label));
    list.appendChild(el("dd", null, value || "-"));
  });
  section.appendChild(list);
  return section;
}

function buildSubmitPayload(input) {
  const dialogueMode = input.dialogueMode;
  if (!["auto", "edit", "custom", "none"].includes(dialogueMode)) {
    throw new Error("请选择台词模式");
  }
  const requestId = String(input.clientRequestId || "").trim();
  if (!requestId) throw new Error("缺少本次生成请求标识");
  if (input.isLong && dialogueMode !== "auto" && dialogueMode !== "none") {
    throw new Error("长视频仅支持保留完整源音轨或静音");
  }
  if (input.isLong && (typeof input.planReceipt !== "string"
      || !/^[0-9a-f]{64}$/.test(input.planReceipt))) {
    throw new Error("长视频生成计划尚未就绪，请刷新后重试");
  }
  if (!GENERATION_ASPECT_RATIOS.includes(input.aspectRatio)) throw new Error("请选择画幅");
  if (!GENERATION_RESOLUTIONS.includes(input.resolution)) throw new Error("请选择清晰度");

  let fitMode = "none";
  if (input.fitRequired) {
    fitMode = input.fitMode;
    if (!["crop", "pad"].includes(fitMode)) throw new Error("请选择裁切或留边以适配画幅");
  }

  const body = {
    confirm: true,
    client_request_id: requestId,
    dialogue_mode: dialogueMode,
    fit_mode: fitMode,
    aspect_ratio: input.aspectRatio,
    resolution: input.resolution,
  };
  if (input.isLong) {
    body.expected_plan_receipt = input.planReceipt;
    body.fast_mode = input.fastMode === true;
  }
  if (dialogueMode === "edit" || dialogueMode === "custom") {
    const lines = parseDialogueLines(input.linesText);
    if (lines.length === 0) throw new Error("请至少填写一行台词");
    body.lines = lines;
  }
  return body;
}

function generationAction(status, stage) {
  if (status === null || status === undefined) return "new";
  if (status === "failed") return stage === "stitch" ? "retry_stitch" : "retry";
  if (status === "resume_required") return "resume";
  return "none";
}

function generationRetryContract(detail) {
  const generation = detail && detail.generation;
  const action = generationAction(
    generation && generation.status,
    generation && generation.stage,
  );
  const longContract = longVideoContract(detail);
  if (!longContract.isLong) {
    return { action, paidTaskCount: action === "new" || action === "retry" ? 1 : 0 };
  }
  if (action === "new") return { action, paidTaskCount: longContract.segmentCount };
  if (action === "none" || action === "resume") {
    return { action, paidTaskCount: 0 };
  }
  const serverCount = generation && generation.retry_paid_segment_count;
  const validServerCount = Number.isInteger(serverCount)
    && serverCount >= 0 && serverCount <= longContract.segmentCount;
  return {
    action,
    paidTaskCount: validServerCount ? serverCount : null,
  };
}

function buildResumePayload(detail) {
  const generation = detail && detail.generation;
  const dialogue = detail && detail.dialogue;
  if (!generation || generation.status !== "resume_required") throw new Error("当前任务无需继续");
  if (typeof generation.client_request_id !== "string" || !generation.client_request_id.trim()) {
    throw new Error("缺少既有任务请求标识");
  }
  if (!dialogue || !["auto", "edit", "custom", "none"].includes(dialogue.mode)) {
    throw new Error("既有任务台词模式无效");
  }
  if (!["none", "crop", "pad"].includes(detail.fit_mode)) throw new Error("既有任务画幅模式无效");
  if (!GENERATION_ASPECT_RATIOS.includes(detail.aspect_ratio)
      || !GENERATION_RESOLUTIONS.includes(detail.resolution)) {
    throw new Error("既有任务生成参数无效");
  }

  const body = {
    confirm: true,
    client_request_id: generation.client_request_id,
    dialogue_mode: dialogue.mode,
    fit_mode: detail.fit_mode,
    aspect_ratio: detail.aspect_ratio,
    resolution: detail.resolution,
  };
  const longContract = longVideoContract(detail);
  if (longContract.isLong) {
    if (!longContract.ready) throw new Error("长视频生成计划尚未就绪，请刷新后重试");
    if (dialogue.mode !== "auto" && dialogue.mode !== "none") {
      throw new Error("长视频既有任务台词模式无效");
    }
    body.expected_plan_receipt = longContract.planReceipt;
    body.fast_mode = generationFastMode(detail);
  }
  if (dialogue.mode === "edit" || dialogue.mode === "custom") {
    if (!Array.isArray(dialogue.lines) || dialogue.lines.length === 0) throw new Error("既有任务台词缺失");
    body.lines = dialogue.lines;
  }
  return body;
}

function buildStitchRetryPayload(detail) {
  const generation = detail && detail.generation;
  if (!generation || generationAction(generation.status, generation.stage) !== "retry_stitch") {
    throw new Error("当前任务无需重试拼接");
  }
  const requestId = generation.client_request_id;
  const dialogue = detail && detail.dialogue;
  const longContract = longVideoContract(detail);
  if (!longContract.isLong || !longContract.ready) throw new Error("长视频生成计划尚未就绪，请刷新后重试");
  if (typeof requestId !== "string" || !requestId.trim()) throw new Error("缺少既有任务请求标识");
  if (!dialogue || !["auto", "none"].includes(dialogue.mode)) throw new Error("既有任务台词模式无效");
  if (!["none", "crop", "pad"].includes(detail.fit_mode)) throw new Error("既有任务画幅模式无效");
  if (!GENERATION_ASPECT_RATIOS.includes(detail.aspect_ratio)
      || !GENERATION_RESOLUTIONS.includes(detail.resolution)) {
    throw new Error("既有任务生成参数无效");
  }
  return {
    confirm: true,
    client_request_id: requestId,
    dialogue_mode: dialogue.mode,
    fit_mode: detail.fit_mode,
    aspect_ratio: detail.aspect_ratio,
    resolution: detail.resolution,
    expected_plan_receipt: longContract.planReceipt,
    fast_mode: generationFastMode(detail),
  };
}

function buildLongRetryPayload(detail, clientRequestId) {
  const generation = detail && detail.generation;
  const dialogue = detail && detail.dialogue;
  const longContract = longVideoContract(detail);
  if (!generation || generationAction(generation.status, generation.stage) !== "retry") {
    throw new Error("当前任务无需重试生成");
  }
  if (!longContract.isLong || !longContract.ready) throw new Error("长视频生成计划尚未就绪，请刷新后重试");
  if (!dialogue || !["auto", "none"].includes(dialogue.mode)) throw new Error("既有任务台词模式无效");
  return buildSubmitPayload({
    clientRequestId,
    dialogueMode: dialogue.mode,
    fitRequired: detail.fit_required === true,
    fitMode: detail.fit_mode,
    aspectRatio: detail.aspect_ratio,
    resolution: detail.resolution,
    isLong: true,
    fastMode: generationFastMode(detail),
    planReceipt: longContract.planReceipt,
  });
}

function fmtTime(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const now = new Date();
  const pad = (x) => String(x).padStart(2, "0");
  const hm = pad(d.getHours()) + ":" + pad(d.getMinutes());
  if (d.toDateString() === now.toDateString()) return hm;
  if (d.getFullYear() === now.getFullYear()) return pad(d.getMonth() + 1) + "-" + pad(d.getDate()) + " " + hm;
  return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate());
}

function fmtSec(x) {
  return Number.isFinite(x) ? Number(x).toFixed(1) : "?";
}

function shortId(value) {
  return typeof value === "string" && value ? value.slice(0, 8) : "--------";
}

function formatDuration(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds <= 0) return "时长待载入";
  if (seconds < 60) return seconds.toFixed(seconds < 10 ? 1 : 0) + " 秒";
  const minutes = Math.floor(seconds / 60);
  const tail = Math.round(seconds % 60);
  return minutes + " 分 " + String(tail).padStart(2, "0") + " 秒";
}

function formatElapsed(createdAt, endAt, nowMs = Date.now()) {
  const start = new Date(createdAt).getTime();
  const terminal = endAt ? new Date(endAt).getTime() : Number.NaN;
  if (!Number.isFinite(start)) return "已耗时未知";
  const end = Number.isFinite(terminal) ? Math.min(nowMs, Math.max(start, terminal)) : nowMs;
  const total = Math.max(0, Math.floor((end - start) / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) return `已耗时 ${hours}时 ${String(minutes).padStart(2, "0")}分`;
  if (minutes > 0) return `已耗时 ${minutes}分 ${String(seconds).padStart(2, "0")}秒`;
  return `已耗时 ${seconds}秒`;
}

function totalKeyframes(detail) {
  const segments = Array.isArray(detail && detail.segments) ? detail.segments : [];
  if (segments.length > 0) {
    return segments.reduce((sum, segment) =>
      sum + (Array.isArray(segment && segment.keyframes) ? segment.keyframes.length : 0), 0);
  }
  return Array.isArray(detail && detail.keyframes) ? detail.keyframes.length : 0;
}

function safeErrorSummary(raw, fallback = "阶段执行失败，请查看诊断后处理") {
  const code = raw && typeof raw === "object"
    ? String(raw.code || raw.error || "") : String(raw || "");
  const labels = new Map([
    ["submission_unknown", "提交结果未知，已禁止重复提交"],
    ["context_ir_query_unknown", "Context IR 查询结果未知，请继续原任务"],
    ["context_ir_resume_required", "Context IR 可从原任务继续"],
    ["prompt_fusion_failed", "最终提示词融合失败"],
    ["prompt_fusion_output_invalid", "最终提示词融合结果无效"],
    ["provider_rejected", "素材服务商未接受本次处理"],
    ["output_missing", "生成已结束，但成片尚未生成"],
    ["generation_path_removed", "该历史任务只能查看，不能重新生成"],
  ]);
  if (labels.has(code)) return labels.get(code);
  if (/(?:asr|voice_lines|codex voice|vocal classification|audio extract|transcri)/i.test(code)) {
    return "台词识别失败，未进入校对或后续生成";
  }
  if (/^[a-z][a-z0-9_]{2,80}$/.test(code)) {
    return fallback + `（${code}）`;
  }
  if (code && code.length <= 120 && !/[\r\n{}\[\]+]/.test(code)) return code;
  return fallback;
}

function diagnosticText(raw) {
  const text = raw && typeof raw === "object"
    ? JSON.stringify(raw, null, 2) : String(raw || "");
  return text.length > 1200 ? text.slice(0, 1200) + "\n…诊断内容已截断" : text;
}

function stageState(status, detail = "", count = "") {
  return { status, detail, count };
}

function operationTimeline(detail, nowMs = Date.now()) {
  const sourceReady = detail && detail.has_source === true;
  const analysis = detail && detail.status;
  const post = detail && detail.postprocess;
  const fusion = detail && detail.prompt_fusion;
  const generation = detail && detail.generation;
  const generationStatus = generation && generation.status;
  const generationStage = generation && generation.stage;
  const segments = Array.isArray(detail && detail.segments) ? detail.segments : [];
  const generationSegments = generation && Array.isArray(generation.segments)
    ? generation.segments : [];
  const frameCount = totalKeyframes(detail);
  const completedImages = ppCompletedFrames(detail);
  const totalImages = ppTotalFrames(detail);
  const fusionSegments = fusion && Array.isArray(fusion.segments) ? fusion.segments : [];
  const fusionDone = fusionSegments.filter((item) => item && item.status === "done").length;
  const h3Done = generationSegments.filter((item) => item && item.status === "succeeded").length;
  const h3Total = generationSegments.length || segments.length;
  const afterContext = ["h3", "stitch", "stitching"].includes(generationStage)
    || generationStatus === "succeeded" || detail.has_video === true;
  const review = detail && detail.dialogue_review;
  const reviewWaiting = !!review && review.status === "waiting";
  const reviewFrozen = !!review && review.status === "frozen";

  const stages = [
    {
      key: "source", label: "源视频 A",
      ...stageState(sourceReady ? "done" : analysis === "queued" ? "running" : "unknown",
        sourceReady ? "源视频已由服务器接收" : "等待服务器确认源视频"),
    },
    {
      key: "analysis", label: "分析",
      ...stageState(
        analysis === "done" || reviewWaiting || reviewFrozen ? "done"
          : analysis === "processing" ? "running"
            : analysis === "queued" ? "waiting"
              : analysis === "failed" ? "failed" : "unknown",
        analysis === "done" ? "分析产物已发布"
          : reviewWaiting || reviewFrozen ? "上传分析与台词识别已完成"
          : analysis === "processing" ? "正在抽帧并生成项目描述"
            : analysis === "queued" ? "等待分析" : "分析状态异常",
      ),
    },
    {
      key: "dialogue-review", label: "台词校对",
      ...stageState(
        reviewWaiting ? "attention"
          : reviewFrozen ? "done"
            : analysis === "failed" ? "blocked"
              : analysis === "done" ? "skipped" : "waiting",
        reviewWaiting ? "识别稿正在等待你校对并采用"
          : reviewFrozen ? (review.frozen_by === "user"
            ? "校对稿已冻结，正在继续同一任务"
            : "自动识别稿已冻结，无需中途确认")
            : analysis === "failed" ? "分析失败，未产生可校对台词"
              : "等待台词识别",
        review && Number.isInteger(review.revision) ? `v${review.revision}` : "",
      ),
    },
    {
      key: "index", label: "素材索引",
      ...stageState(
        analysis === "done" && frameCount > 0 ? "done"
          : analysis === "processing" && !reviewWaiting ? "running"
            : analysis === "failed" ? "blocked"
              : analysis === "done" ? "unknown" : "waiting",
        analysis === "done" && frameCount > 0
          ? "已发布关键帧与分段索引"
          : analysis === "done" ? "服务器未公开可用索引" : "随分析阶段建立",
        frameCount > 0 ? `${frameCount} 帧` : "",
      ),
    },
    {
      key: "image", label: "图片优化",
      ...stageState(
        post && post.status === "done" ? "done"
          : post && ["queued", "running"].includes(post.status) ? "running"
            : post && post.status === "failed" ? "failed"
              : (fusion || generation || detail.has_video) ? "skipped" : "waiting",
        post && post.status === "done" ? "优化素材已发布"
          : post && post.status === "failed" ? safeErrorSummary(post.error, "图片优化失败")
            : post ? "正在处理服务器已确认的素材" : "当前项目未公开图片优化任务",
        totalImages > 0 ? `${completedImages}/${totalImages} 帧` : "",
      ),
    },
    {
      key: "fusion", label: "Prompt Fusion",
      ...stageState(
        fusion && fusion.status === "done" ? "done"
          : fusion && fusion.status === "running" ? "running"
            : fusion && fusion.status === "failed" ? "failed"
              : fusion && fusion.status === "pending" ? "waiting"
                : generation ? "skipped" : "waiting",
        fusion && fusion.status === "done" ? "最终提示词已冻结"
          : fusion && fusion.status === "running" ? "正在融合逐段提示词"
            : fusion && fusion.status === "failed" ? safeErrorSummary(fusion.error, "Prompt Fusion 失败")
              : fusion ? "等待融合" : "当前项目未公开 Fusion 状态",
        fusionSegments.length > 0 ? `${fusionDone}/${fusionSegments.length} 段` : "",
      ),
    },
    {
      key: "context", label: "Context IR",
      ...stageState(
        ["context_ir", "context_ir_native"].includes(generationStage)
          ? ["failed"].includes(generationStatus) ? "failed"
            : ["resume_required", "submission_unknown"].includes(generationStatus) ? "attention"
              : "running"
          : afterContext && fusion ? "done"
            : afterContext ? "skipped" : "waiting",
        ["context_ir", "context_ir_native"].includes(generationStage)
          ? generationStatus === "submission_unknown" ? "查询结果未知，禁止重复提交"
            : generationStatus === "resume_required" ? "可继续原任务"
              : generationStatus === "failed" ? safeErrorSummary(generation.error, "Context IR 失败")
                : "正在优化最终提示词；服务器未公开逐段计数"
          : afterContext && fusion ? "Context IR 已完成"
            : afterContext ? "该历史任务未公开 Context IR" : "等待前序阶段",
      ),
    },
    {
      key: "h3", label: "H3 生成",
      ...stageState(
        generation && ["stitch", "stitching"].includes(generationStage) ? "done"
          : generationStatus === "succeeded" || detail.has_video === true ? "done"
            : generation && !["context_ir", "context_ir_native"].includes(generationStage)
              ? generationStatus === "failed" ? "failed"
                : ["resume_required", "submission_unknown"].includes(generationStatus) ? "attention"
                  : ["queued", "running", "submitting"].includes(generationStatus) ? "running" : "unknown"
              : "waiting",
        generationStatus === "submission_unknown" ? "供应商是否接单未知，禁止重复提交"
          : generationStatus === "resume_required" ? "继续原任务不会创建新请求"
            : generationStatus === "failed" && !["stitch", "stitching"].includes(generationStage)
              ? safeErrorSummary(generation.error, "H3 生成失败")
              : generation ? "逐段生成视频" : "等待生成任务",
        h3Total > 0 ? `${h3Done}/${h3Total} 段` : "",
      ),
    },
    {
      key: "stitch", label: "拼接",
      ...stageState(
        detail.has_video === true || generationStatus === "succeeded" ? "done"
          : generation && ["stitch", "stitching"].includes(generationStage)
            ? generationStatus === "failed" ? "failed"
              : ["resume_required", "submission_unknown"].includes(generationStatus) ? "attention" : "running"
            : "waiting",
        generation && ["stitch", "stitching"].includes(generationStage)
          ? generationStatus === "failed" ? safeErrorSummary(generation.error, "视频拼接失败")
            : "正在复用已完成分段合成成片"
          : detail.has_video === true ? "拼接产物已校验" : "等待全部分段",
        h3Total > 0 && (generationStatus === "succeeded" || ["stitch", "stitching"].includes(generationStage))
          ? `${h3Total} 段` : "",
      ),
    },
    {
      key: "output", label: "成片",
      ...stageState(
        detail.has_video === true ? "done"
          : generationStatus === "succeeded" ? "failed" : "waiting",
        detail.has_video === true ? "成片已由服务器校验并提交"
          : generationStatus === "succeeded" ? "生成成功，但成片未生成" : "等待最终成片",
      detail.has_video === true ? "可播放" : "等待成片",
      ),
    },
  ];

  const current = stages.find((stage) => ["failed", "attention"].includes(stage.status))
    || stages.find((stage) => stage.status === "running")
    || stages.find((stage) => stage.status === "unknown")
    || stages.find((stage) => stage.status === "waiting")
    || stages[stages.length - 1];
  const terminal = detail.has_video === true
    || ["failed"].includes(analysis)
    || ["failed", "submission_unknown", "resume_required"].includes(generationStatus);
  return {
    operationId: detail && detail.id,
    stages,
    current,
    elapsed: formatElapsed(detail && detail.created_at, terminal ? detail && detail.updated_at : null, nowMs),
    updated: fmtTime(detail && detail.updated_at),
  };
}

function stageStatusText(status) {
  return ({
    done: "已完成", running: "进行中", waiting: "等待中", blocked: "已阻塞",
    failed: "失败", attention: "需要处理", unknown: "状态未知", skipped: "未启用",
  })[status] || "状态未知";
}

function segmentJoinText(value) {
  return ({hard_cut: "硬切衔接", continuous: "连续衔接", crossfade: "淡化衔接"})[value]
    || "衔接方式未公开";
}

function skillMilestoneView(detail) {
  const milestone = detail && detail.skill_milestone;
  const skills = milestone && Array.isArray(milestone.skills) ? milestone.skills : [];
  if (!milestone || typeof milestone.id !== "string" || !/^skill-[0-9a-f]{64}$/.test(milestone.id)
      || !Number.isInteger(milestone.version) || skills.length === 0
      || skills.some((skill) => !skill || typeof skill.name !== "string"
        || typeof skill.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(skill.sha256))) return null;
  return {
    label: `Skill v${milestone.version} · ${milestone.id.slice(6, 14)}`,
    id: milestone.id,
    skills: skills.map((skill) => ({name: skill.name, short: skill.sha256.slice(0, 8), sha256: skill.sha256})),
  };
}

function materialIndexView(detail) {
  const candidate = detail && (detail.element_index || detail.material_index || detail.project_index);
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) return null;
  const groups = ["people", "entities", "scenes"].map((key) => {
    const source = candidate[key];
    const entries = source && typeof source === "object" && !Array.isArray(source)
      ? Object.entries(source).map(([id, value]) => ({
        id,
        description: value && typeof value === "object"
          ? String(value.source_visual_description || value.description || "") : "",
        occurrences: value && Array.isArray(value.occurrences) ? value.occurrences.length : 0,
      })) : [];
    return {key, entries};
  });
  const relations = candidate.relations && typeof candidate.relations === "object"
    && !Array.isArray(candidate.relations)
    ? Object.entries(candidate.relations).map(([id, value]) => ({
      id,
      subject: value && String(value.subject || value.subject_id || ""),
      predicate: value && String(value.predicate || value.relationship || ""),
      object: value && String(value.object || value.object_id || ""),
    })) : [];
  return {groups, relations};
}

function conversationThumbnailPath(detail) {
  const segments = Array.isArray(detail && detail.segments) ? detail.segments : [];
  if (segments.length > 0) {
    const paths = authoritativeSegmentKeyframePaths(detail, segments[0]);
    return paths.length > 0 ? paths[0] : null;
  }
  const names = Array.isArray(detail && detail.keyframes) ? detail.keyframes : [];
  return names.length > 0 && safeMediaBasename(names[0]) ? "keyframes/" + names[0] : null;
}

function errorDiagnostic(raw, label = "高级诊断") {
  if (raw === null || raw === undefined || raw === "") return null;
  const details = el("details", "error-diagnostic");
  details.appendChild(el("summary", null, label));
  details.appendChild(el("pre", null, diagnosticText(raw)));
  return details;
}

function hideOperationHeader() {
  const host = typeof document !== "undefined" ? $("operation-status") : null;
  if (!host) return;
  host.textContent = "";
  host.hidden = true;
}

function renderOperationHeader(detail) {
  const host = $("operation-status");
  const model = operationTimeline(detail);
  host.textContent = "";
  host.hidden = false;
  host.dataset.status = model.current.status;

  const summary = el("div", "operation-summary");
  const identity = el("div", "operation-identity");
  identity.appendChild(el("span", "operation-kicker", `项目 #${shortId(model.operationId)}`));
  const title = el("strong", "operation-current", model.current.label + " · " + stageStatusText(model.current.status));
  title.setAttribute("aria-label", `当前阶段：${model.current.label}，${stageStatusText(model.current.status)}`);
  identity.appendChild(title);
  if (model.current.count) identity.appendChild(el("span", "operation-count", model.current.count));
  summary.appendChild(identity);

  const meta = el("div", "operation-meta");
  meta.appendChild(el("span", null, model.elapsed));
  meta.appendChild(el("span", null, model.updated ? "服务器更新 " + model.updated : "更新时间未知"));
  summary.appendChild(meta);
  host.appendChild(summary);

  const currentDetail = el("p", "operation-current-detail", model.current.detail);
  host.appendChild(currentDetail);

  const timeline = el("ol", "operation-timeline");
  for (const stage of model.stages) {
    const item = el("li", `operation-stage status-${stage.status}`);
    if (stage.key === model.current.key) item.setAttribute("aria-current", "step");
    const marker = el("span", "operation-stage-marker", stage.status === "done" ? "✓" : "");
    marker.setAttribute("aria-hidden", "true");
    item.appendChild(marker);
    item.appendChild(el("span", "operation-stage-label", stage.label));
    item.appendChild(el("span", "operation-stage-state", stage.count || stageStatusText(stage.status)));
    item.title = stage.detail;
    timeline.appendChild(item);
  }
  host.appendChild(timeline);
}

function trackURL(url) {
  state.objectURLs.push(url);
  return url;
}

function revokeURLs() {
  for (const u of state.objectURLs) URL.revokeObjectURL(u);
  state.objectURLs = [];
}

function releaseTrackedURL(
  url,
  urls = state.objectURLs,
  revoke = (value) => URL.revokeObjectURL(value),
) {
  const index = urls.indexOf(url);
  if (index >= 0) urls.splice(index, 1);
  revoke(url);
}

function releaseTrackedURLs(urls, release = releaseTrackedURL) {
  for (const url of urls.splice(0)) release(url);
}

/* ===== API ===== */
async function api(path, options = {}) {
  const headers = Object.assign({}, options.headers);
  if (state.token) headers["Authorization"] = "Bearer " + state.token;
  let res;
  try {
    res = await fetch(path, Object.assign({}, options, { headers }));
  } catch (e) {
    throw new Error("无法连接服务器，请检查网络后重试");
  }
  if (res.status === 401) throw new AuthError("口令已失效");
  return res;
}

async function apiJSON(path, options = {}) {
  const res = await api(path, options);
  if (!res.ok) {
    const fallback = "请求失败（" + res.status + "），请稍后重试";
    try {
      const data = await res.json();
      throw apiErrorFromPayload(data, fallback);
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError(fallback);
    }
  }
  return res.json();
}

async function apiBlobURL(path) {
  const res = await api(path);
  if (!res.ok) throw new Error("文件加载失败（" + res.status + "）");
  const blob = await res.blob();
  return trackURL(URL.createObjectURL(blob));
}

function handleAuthError(err) {
  if (err instanceof AuthError) {
    sessionExpired();
    return true;
  }
  return false;
}

/* ===== 登录 / 会话鉴权 ===== */
function showLogin(message) {
  stopPolling();
  hideOperationHeader();
  releaseHistoryThumbnails();
  state.currentId = null;
  state.detail = null;
  state.generationConfigCapability = null;
  state.generationConfigCapabilityLoaded = false;
  state.dialogueReviewCapability = null;
  state.dialogueReviewCapabilityLoaded = false;
  $("app-view").hidden = true;
  $("login-view").hidden = false;
  const errEl = $("login-error");
  if (message) {
    errEl.textContent = message;
    errEl.hidden = false;
  } else {
    errEl.hidden = true;
  }
  $("login-token").value = "";
  setTimeout(() => $("login-token").focus(), 0);
}

function sessionExpired() {
  state.token = null;
  localStorage.removeItem(TOKEN_KEY);
  showLogin("登录状态已失效，请重新输入口令");
}

async function doLogin(token) {
  const btn = $("login-btn");
  const errEl = $("login-error");
  btn.disabled = true;
  btn.textContent = "验证中…";
  errEl.hidden = true;
  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    if (res.status === 401) {
      errEl.textContent = "口令不正确，请重新输入";
      errEl.hidden = false;
      return;
    }
    if (!res.ok) {
      errEl.textContent = "登录失败（" + res.status + "），请稍后重试";
      errEl.hidden = false;
      return;
    }
    state.token = token;
    localStorage.setItem(TOKEN_KEY, token);
    enterApp();
  } catch (e) {
    errEl.textContent = "无法连接服务器，请检查网络后重试";
    errEl.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "进入";
  }
}

function enterApp() {
  $("login-view").hidden = true;
  $("app-view").hidden = false;
  state.currentId = null;
  state.detail = null;
  resetGenerationConfigDisclosure();
  renderEmptyHero();
  void loadGenerationConfigCapability();
  refreshList(true);
}

/* ===== 侧栏会话列表 ===== */
async function refreshList(autoSelect) {
  try {
    const list = await apiJSON("/api/conversations");
    state.conversations = mergeConversationList(
      Array.isArray(list) ? list : [],
      state.conversations,
    );
    releaseHistoryThumbnails(new Set(state.conversations.map((item) => item.id)));
    renderList();
    if (autoSelect && state.conversations.length > 0 && !state.currentId) {
      selectConversation(state.conversations[0].id);
    }
    void hydrateConversationSummaries();
  } catch (err) {
    if (handleAuthError(err)) return;
    renderListError(err.message);
  }
}

function releaseHistoryThumbnails(keepIds = null) {
  for (const [id, url] of Object.entries(state.historyThumbnailURLs)) {
    if (keepIds && keepIds.has(id)) continue;
    URL.revokeObjectURL(url);
    delete state.historyThumbnailURLs[id];
    delete state.historyDetails[id];
  }
}

function mergeConversationList(incoming, previous) {
  const previousById = new Map((previous || []).map((item) => [item.id, item]));
  return incoming.map((item) => {
    const known = previousById.get(item.id);
    if (!known) return item;
    const merged = Object.assign({}, item);
    for (const field of [
      "generation", "navigation_status", "duration_s", "updated_at", "segment_count",
      "skill_milestone", "thumbnail_path", "prompt_fusion", "postprocess", "_hydrated",
    ]) {
      if (!Object.prototype.hasOwnProperty.call(item, field)
          && Object.prototype.hasOwnProperty.call(known, field)) {
        merged[field] = known[field];
      }
    }
    return merged;
  });
}

function syncConversationDetail(conversations, detail) {
  const summary = conversations.find((item) => item.id === detail.id);
  if (!summary) return false;
  summary.status = detail.status;
  summary.has_video = detail.has_video === true;
  summary.generation = detail.generation || null;
  summary.duration_s = detail.duration_s;
  summary.updated_at = detail.updated_at;
  summary.segment_count = detail.segment_count;
  summary.skill_milestone = detail.skill_milestone || null;
  summary.prompt_fusion = detail.prompt_fusion || null;
  summary.postprocess = detail.postprocess || null;
  summary.thumbnail_path = conversationThumbnailPath(detail);
  summary._hydrated = true;
  state.historyDetails[detail.id] = detail;
  if (Object.prototype.hasOwnProperty.call(detail, "navigation_status")) {
    summary.navigation_status = detail.navigation_status;
  }
  return true;
}

async function loadHistoryThumbnail(summary) {
  if (!summary || !summary.id || !summary.thumbnail_path
      || state.historyThumbnailURLs[summary.id]) return;
  try {
    const response = await api(
      "/api/conversations/" + encodeURIComponent(summary.id)
        + "/files/" + encodedMediaPath(summary.thumbnail_path),
    );
    if (!response.ok) return;
    const blob = await response.blob();
    if (!state.conversations.some((item) => item.id === summary.id)) return;
    state.historyThumbnailURLs[summary.id] = URL.createObjectURL(blob);
  } catch (error) {
    if (handleAuthError(error)) throw error;
  }
}

async function hydrateConversationSummaries() {
  if (state.historyHydrating || !state.token) return;
  const pending = state.conversations.filter((item) => item && item.id && item._hydrated !== true).slice(0, 24);
  if (pending.length === 0) return;
  state.historyHydrating = true;
  let cursor = 0;
  const worker = async () => {
    while (cursor < pending.length) {
      const summary = pending[cursor++];
      try {
        const detail = await apiJSON("/api/conversations/" + encodeURIComponent(summary.id));
        if (syncConversationDetail(state.conversations, detail)) {
          await loadHistoryThumbnail(summary);
        }
      } catch (error) {
        if (handleAuthError(error)) return;
        summary._hydrated = true;
      }
    }
  };
  try {
    await Promise.all([worker(), worker(), worker()]);
  } finally {
    state.historyHydrating = false;
    if (state.token) renderList();
  }
}

function renderList() {
  const nav = $("conv-list");
  nav.textContent = "";
  if (state.conversations.length === 0) {
    nav.appendChild(el("div", "conv-empty", "还没有会话\n在下方选择一段视频，开始第一次生成"));
    return;
  }
  for (const c of state.conversations) {
    const item = el("button", "conv-item" + (c.id === state.currentId ? " selected" : ""));
    item.type = "button";
    const thumb = el("span", "conv-thumb");
    const thumbnailURL = state.historyThumbnailURLs[c.id];
    if (thumbnailURL) {
      const image = el("img");
      image.src = thumbnailURL;
      image.alt = "";
      thumb.appendChild(image);
    } else {
      thumb.appendChild(icon("i-film"));
    }
    item.appendChild(thumb);
    const content = el("span", "conv-content");
    const heading = el("span", "conv-heading");
    heading.appendChild(el("span", "conv-title", c.title || "未命名项目"));
    const badgeState = conversationBadge(c);
    const badge = el("span", "badge " + badgeState.className, badgeState.text);
    heading.appendChild(badge);
    content.appendChild(heading);
    const identity = el("span", "conv-identity");
    identity.appendChild(el("span", "conv-id", "#" + shortId(c.id)));
    identity.appendChild(el("span", null, formatDuration(c.duration_s)));
    if (Number.isInteger(c.segment_count) && c.segment_count > 0) {
      identity.appendChild(el("span", null, c.segment_count + " 段"));
    }
    content.appendChild(identity);
    const footer = el("span", "conv-footer");
    const detail = state.historyDetails[c.id];
    const timeline = detail ? operationTimeline(detail) : null;
    footer.appendChild(el("span", null, timeline ? timeline.current.label : badgeState.text));
    footer.appendChild(el("span", "conv-time", fmtTime(c.updated_at || c.created_at)));
    footer.appendChild(el("span", "conv-output " + (c.has_video === true ? "is-ready" : "is-waiting"),
      c.has_video === true ? "成片已提交" : "等待成片"));
    content.appendChild(footer);
    item.appendChild(content);
    item.addEventListener("click", () => {
      selectConversation(c.id);
      closeDrawer();
    });
    nav.appendChild(item);
  }
}

function conversationBadge(conversation) {
  const item = conversation || {};
  const navigationBadges = {
    analysis_queued: { className: "queued", text: "分析排队中" },
    analysis_processing: { className: "processing", text: "分析中" },
    waiting_for_dialogue_review: { className: "processing", text: "等待台词校对" },
    analysis_failed: { className: "failed", text: "分析失败" },
    analysis_unknown: { className: "failed", text: "分析状态未知" },
    analysis_complete: { className: "analyzed", text: "分析完成" },
    generation_queued: { className: "processing", text: "生成排队中" },
    generation_running: { className: "processing", text: "生成中" },
    generation_failed: { className: "failed", text: "生成失败" },
    generation_submission_unknown: { className: "failed", text: "提交结果未知" },
    generation_resume_required: { className: "failed", text: "等待继续" },
    generation_unknown: { className: "failed", text: "生成状态未知" },
    output_missing: { className: "failed", text: "最终视频缺失" },
    completed: { className: "done", text: "已完成" },
    postprocessing: { className: "processing", text: "素材优化中" },
    postprocess_failed: { className: "failed", text: "素材优化失败" },
    // 兼容滚动升级期间旧后端的枚举；有最终视频时完成态始终统一为“已完成”。
    postprocess_done: { className: "done", text: "已完成" },
  };
  if (Object.prototype.hasOwnProperty.call(item, "navigation_status")) {
    return navigationBadges[item.navigation_status]
      || { className: "failed", text: "状态异常" };
  }

  const generationStatus = item.generation && item.generation.status;
  const generationBadges = {
    queued: { className: "processing", text: "生成排队中" },
    running: { className: "processing", text: "生成中" },
    failed: { className: "failed", text: "生成失败" },
    submission_unknown: { className: "failed", text: "提交结果未知" },
    resume_required: { className: "failed", text: "等待继续" },
  };
  if (generationBadges[generationStatus]) return generationBadges[generationStatus];
  if (generationStatus === "succeeded") {
    return item.has_video === true
      ? { className: "done", text: "已完成" }
      : { className: "failed", text: "最终视频缺失" };
  }
  if (generationStatus) return { className: "failed", text: "生成状态未知" };

  if (item.status === "done") {
    // 兼容尚未发布 navigation_status 的旧后端；新版响应仍只信上面的权威枚举。
    return item.has_video === true
      ? { className: "done", text: "已完成" }
      : { className: "analyzed", text: "分析完成" };
  }
  if (item.status === "failed") return { className: "failed", text: "失败" };
  if (item.status === "processing") return { className: "processing", text: "处理中" };
  return { className: "queued", text: STATUS_TEXT[item.status] || item.status || "排队中" };
}

function renderListError(msg) {
  const nav = $("conv-list");
  nav.textContent = "";
  const box = el("div", "conv-list-error");
  box.appendChild(el("div", null, "会话列表加载失败：" + msg));
  const retry = el("button", "btn btn-ghost", "重试");
  retry.type = "button";
  retry.addEventListener("click", () => refreshList(false));
  box.appendChild(retry);
  nav.appendChild(box);
}

/* ===== 抽屉（移动端） ===== */
function openDrawer() {
  $("sidebar").classList.add("open");
  $("drawer-backdrop").hidden = false;
  $("menu-btn").setAttribute("aria-expanded", "true");
}

function closeDrawer() {
  $("sidebar").classList.remove("open");
  $("drawer-backdrop").hidden = true;
  $("menu-btn").setAttribute("aria-expanded", "false");
}

/* ===== 底部创建抽屉 ===== */
function setComposerExpanded(expanded) {
  const dock = document.querySelector(".composer-dock");
  const toggle = $("composer-toggle");
  const panel = $("composer-panel");
  if (!dock || !toggle || !panel) return false;
  const next = expanded === true;
  dock.classList.toggle("is-collapsed", !next);
  panel.hidden = !next;
  toggle.setAttribute("aria-expanded", String(next));
  const action = next ? "收起创建抽屉" : "展开创建抽屉";
  toggle.setAttribute("aria-label", action);
  toggle.title = action;
  $("composer-toggle-label").textContent = next ? "收起" : "展开";
  return next;
}

/* ===== Stream 渲染 ===== */
function clearStream() {
  closeLightbox({ restoreFocus: false });
  revokeURLs();
  $("stream").textContent = "";
}

function renderEmptyHero() {
  stopPolling();
  hideOperationHeader();
  document.querySelector(".composer-dock").classList.remove("is-dialogue-review-waiting");
  setComposerExpanded(true);
  clearStream();
  $("main-title").textContent = "视频工作室";

  const inner = el("div", "stream-inner");
  const hero = el("div", "empty-hero");
  const iconBox = el("div", "empty-icon");
  iconBox.appendChild(icon("i-film"));
  hero.appendChild(iconBox);
  hero.appendChild(el("h2", null, "上传参考视频，生成复刻配方"));
  hero.appendChild(el("p", "empty-sub", "AI 会抽取关键帧并准备视频生成"));
  const steps = el("ol", "empty-steps");
  const items = [
    "点击回形针或把视频拖进输入框",
    "可选：填写备注，说明想复刻的镜头重点",
    "发送后等待处理，结果会出现在这里",
  ];
  items.forEach((t, i) => {
    const li = el("li");
    li.appendChild(el("span", "step-n", String(i + 1)));
    li.appendChild(el("span", null, t));
    steps.appendChild(li);
  });
  hero.appendChild(steps);
  inner.appendChild(hero);
  $("stream").appendChild(inner);
}

function renderSkeleton() {
  hideOperationHeader();
  clearStream();
  const inner = el("div", "stream-inner");
  inner.appendChild(el("div", "sk-block shimmer sk-user"));
  inner.appendChild(el("div", "sk-block shimmer sk-assistant"));
  $("stream").appendChild(inner);
}

function renderStreamError(msg) {
  clearStream();
  const inner = el("div", "stream-inner");
  const box = el("div", "stream-error");
  box.appendChild(el("p", null, msg));
  const retry = el("button", "btn btn-ghost", "重新加载");
  retry.type = "button";
  retry.addEventListener("click", () => {
    if (state.currentId) selectConversation(state.currentId);
  });
  box.appendChild(retry);
  inner.appendChild(box);
  $("stream").appendChild(inner);
}

function assistantHead(timeISO) {
  const head = el("div", "assistant-head");
  head.appendChild(icon("i-sparkle", "ic-accent"));
  head.appendChild(el("span", "assistant-name", "助手"));
  head.appendChild(el("span", "assistant-time", fmtTime(timeISO)));
  return head;
}

function renderUserBubble(detail) {
  const row = el("div", "msg-row msg-user");
  const bubble = el("div", "bubble-user");
  if (detail.note) bubble.appendChild(el("p", "bubble-note", detail.note));
  const fileRow = el("div", "bubble-file");
  fileRow.appendChild(icon("i-film"));
  fileRow.appendChild(el("span", "bubble-file-name", detail.title || "已上传视频"));
  bubble.appendChild(fileRow);
  row.appendChild(bubble);
  return row;
}

function renderFrozenGenerationConfig(detail) {
  const frozen = frozenGenerationConfig(detail);
  if (!frozen) return null;
  const section = el("section", "generation-config-receipt");
  section.setAttribute("aria-label", "已冻结的生成配置");
  const heading = el("div", "generation-config-receipt-heading");
  heading.appendChild(el("strong", null, "生成配置"));
  heading.appendChild(el("span", "stage-pill", "已冻结"));
  section.appendChild(heading);
  const chips = el("div", "generation-config-chips");
  for (const label of generationConfigLabels(frozen.config)) {
    chips.appendChild(el("span", "generation-config-chip", label));
  }
  section.appendChild(chips);
  section.appendChild(el(
    "p", "generation-config-freeze",
    "创建时已固定 · 配置指纹 #" + frozen.sha256.slice(0, 8),
  ));
  return section;
}

function dialogueReviewView(detail) {
  const review = detail && detail.dialogue_review;
  if (!review || !["waiting", "frozen"].includes(review.status)
      || !["recognized", "no_audio", "no_vocal", "vocal_unrecognized"].includes(review.outcome)
      || !Number.isInteger(review.revision) || review.revision < 1
      || typeof review.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(review.sha256)
      || !Array.isArray(review.lines)) return null;
  const lines = normalizeDialogueLines(review.lines);
  if (lines.length !== review.lines.length) return null;
  return Object.assign({}, review, { lines });
}

function dialogueReviewOutcomeText(review) {
  const count = review.lines.length;
  return ({
    recognized: `已识别 ${count} 行台词，请核对文字与时间码。`,
    no_audio: "未检测到音轨。你可以补充台词，或采用空稿按无台词继续。",
    no_vocal: "未检测到可信口播。你可以补充台词，或采用空稿继续。",
    vocal_unrecognized: "检测到人声，但未识别出可靠台词。请补充台词，或采用空稿继续。",
  })[review.outcome];
}

function dialogueReviewCommitErrorMessage(error) {
  const value = String(error && error.message ? error.message : error || "");
  const messages = {
    dialogue_review_conflict: "服务端台词稿已更新，请刷新后重新校对。",
    dialogue_review_read_only: "台词稿已冻结，当前任务已不能修改。",
    dialogue_review_not_waiting: "当前任务已不再等待台词校对，请刷新查看最新状态。",
    dialogue_review_unavailable: "当前任务没有可提交的台词校对稿。",
    invalid_dialogue_review_lines: "台词或时间码不符合要求，请逐行检查。",
  };
  return messages[value] || value || "采用台词稿失败，请重试";
}

function dialogueReviewDraft(detail, review) {
  const existing = state.dialogueReviewDrafts[detail.id];
  if (existing && existing.revision === review.revision
      && existing.sha256 === review.sha256) return existing;
  const draft = {
    revision: review.revision,
    sha256: review.sha256,
    lines: review.lines.map((line) => Object.assign({}, line)),
    requestId: newRequestId(),
    dirty: false,
    submitting: false,
  };
  state.dialogueReviewDrafts[detail.id] = draft;
  return draft;
}

function validateDialogueReviewDraft(draft, duration) {
  const limit = Number(duration);
  let previousStart = -1;
  return draft.lines.map((line, index) => {
    const text = String(line.text || "").trim();
    const start = Number(line.start_s);
    const end = Number(line.end_s);
    if (!text) throw new Error(`第 ${index + 1} 行台词不能为空`);
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || start >= end) {
      throw new Error(`第 ${index + 1} 行时间码无效`);
    }
    if (Number.isFinite(limit) && end > limit + 0.01) {
      throw new Error(`第 ${index + 1} 行超过视频时长 ${limit.toFixed(2)} 秒`);
    }
    if (start < previousStart) throw new Error(`第 ${index + 1} 行开始时间早于上一行`);
    previousStart = start;
    return { text, start_s: start, end_s: end };
  });
}

function renderDialogueReview(detail) {
  const review = dialogueReviewView(detail);
  if (!review) return null;
  const section = el("section", "dialogue-review-card");
  section.dataset.testid = "dialogue-review";
  const heading = el("div", "dialogue-review-heading");
  heading.appendChild(el("h3", "res-h3", review.status === "waiting"
    ? "校对识别台词" : "台词稿已冻结"));
  heading.appendChild(el("span", "stage-pill " + (review.status === "waiting"
    ? "status-attention" : "status-done"), review.status === "waiting" ? "等待你确认" : "只读"));
  section.appendChild(heading);
  section.appendChild(el("p", "dialogue-review-outcome", dialogueReviewOutcomeText(review)));

  if (review.status === "frozen") {
    section.appendChild(el("p", "dialogue-review-lock", review.frozen_by === "user"
      ? "已采用此稿并继续同一任务。下游已开始后，本稿不可修改；如需改台词，请创建新任务。"
      : "已按创建时的“自动识别并继续”冻结机器稿，无需中途确认。"));
    const frozen = el("details", "dialogue-review-readonly");
    frozen.appendChild(el("summary", null, `查看已冻结台词（${review.lines.length} 行） · v${review.revision} · #${review.sha256.slice(0, 8)}`));
    if (review.lines.length === 0) {
      frozen.appendChild(el("p", "final-help", "该冻结稿无台词。"));
    } else {
      const list = el("ol", "dialogue-review-readonly-lines");
      review.lines.forEach((line) => {
        list.appendChild(el("li", null, `${line.start_s.toFixed(2)}–${line.end_s.toFixed(2)} 秒 · ${line.text}`));
      });
      frozen.appendChild(list);
    }
    section.appendChild(frozen);
    return section;
  }

  const draft = dialogueReviewDraft(detail, review);
  const form = el("form", "dialogue-review-form");
  form.noValidate = true;
  const rows = el("div", "dialogue-review-lines");
  const error = el("p", "form-error");
  error.setAttribute("role", "alert");
  error.hidden = true;
  const updateDraft = () => {
    draft.dirty = true;
    draft.requestId = newRequestId();
    error.hidden = true;
  };
  const renderRows = () => {
    rows.textContent = "";
    if (draft.lines.length === 0) {
      rows.appendChild(el("p", "dialogue-review-empty", "当前识别稿为空。可新增台词；直接采用将按无台词继续。"));
      return;
    }
    draft.lines.forEach((line, index) => {
      const row = el("div", "dialogue-review-line");
      row.appendChild(el("span", "dialogue-review-line-index", String(index + 1)));
      for (const [field, label] of [["start_s", "开始秒"], ["end_s", "结束秒"]]) {
        const wrapper = el("label", "dialogue-review-time");
        wrapper.appendChild(el("span", "sr-only", `第 ${index + 1} 行${label}`));
        const input = el("input", "text-input");
        input.type = "number";
        input.inputMode = "decimal";
        input.min = "0";
        input.step = "0.01";
        input.value = String(line[field]);
        input.addEventListener("input", () => {
          line[field] = input.value;
          updateDraft();
        });
        wrapper.appendChild(input);
        row.appendChild(wrapper);
      }
      const textLabel = el("label", "dialogue-review-text");
      textLabel.appendChild(el("span", "sr-only", `第 ${index + 1} 行台词`));
      const textInput = el("input", "text-input");
      textInput.type = "text";
      textInput.value = line.text;
      textInput.placeholder = "台词内容";
      textInput.addEventListener("input", () => {
        line.text = textInput.value;
        updateDraft();
      });
      textLabel.appendChild(textInput);
      row.appendChild(textLabel);
      const remove = el("button", "btn btn-ghost dialogue-review-remove", "移除");
      remove.type = "button";
      remove.setAttribute("aria-label", `移除第 ${index + 1} 行台词`);
      remove.addEventListener("click", () => {
        draft.lines.splice(index, 1);
        updateDraft();
        renderRows();
      });
      row.appendChild(remove);
      rows.appendChild(row);
    });
  };
  renderRows();
  form.appendChild(rows);
  const actions = el("div", "dialogue-review-actions");
  const add = el("button", "btn btn-ghost", "新增一行");
  add.type = "button";
  add.addEventListener("click", () => {
    const duration = Number(detail.duration_s);
    const previous = draft.lines[draft.lines.length - 1];
    const start = previous ? Number(previous.end_s) : 0;
    if (Number.isFinite(duration) && start >= duration) {
      error.textContent = "已到视频末尾，请先调整上一行时间码";
      error.hidden = false;
      return;
    }
    draft.lines.push({
      start_s: Number.isFinite(start) ? start : 0,
      end_s: Number.isFinite(duration) ? Math.min(duration, (Number.isFinite(start) ? start : 0) + 1) : 1,
      text: "",
    });
    updateDraft();
    renderRows();
    const inputs = rows.querySelectorAll('input[type="text"]');
    if (inputs.length) inputs[inputs.length - 1].focus();
  });
  actions.appendChild(add);
  const submit = el("button", "btn btn-primary", "采用此稿并继续");
  submit.type = "submit";
  actions.appendChild(submit);
  form.appendChild(actions);
  form.appendChild(error);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (draft.submitting) return;
    try {
      const lines = validateDialogueReviewDraft(draft, detail.duration_s);
      const payload = buildDialogueReviewCommitPayload(review, lines, draft.requestId);
      draft.submitting = true;
      form.querySelectorAll("input,button").forEach((control) => { control.disabled = true; });
      submit.textContent = "正在采用…";
      await apiJSON(
        "/api/conversations/" + encodeURIComponent(detail.id) + "/dialogue-review/commit",
        { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) },
      );
      delete state.dialogueReviewDrafts[detail.id];
      await loadDetail(detail.id, true);
    } catch (err) {
      if (handleAuthError(err)) return;
      draft.submitting = false;
      error.textContent = dialogueReviewCommitErrorMessage(err);
      error.hidden = false;
      form.querySelectorAll("input,button").forEach((control) => { control.disabled = false; });
      submit.textContent = "采用此稿并继续";
    }
  });
  section.appendChild(form);
  section.appendChild(el("p", "dialogue-review-fingerprint", `机器稿 v${review.revision} · #${review.sha256.slice(0, 8)} · 提交后冻结并继续同一任务`));
  return section;
}

/* 状态时间线（queued / processing） */
function renderActivity(status) {
  const row = el("div", "msg-row");
  row.appendChild(assistantHead(null));
  const card = el("div", "activity-card");
  card.appendChild(el("p", "ac-title", status === "queued" ? "排队等待中" : "正在处理"));
  card.appendChild(el("p", "ac-sub", "通常需要几十秒到几分钟，可稍后再来查看"));
  const track = el("div", "progress-track");
  track.appendChild(el("div", "progress-fill"));
  card.appendChild(track);

  const stages = el("ol", "stages");
  // [标签, 状态]：done=已完成, active=进行中, pending=未开始
  const defs = status === "queued"
    ? [["上传完成", "done"], ["排队等待", "active"], ["抽取关键帧并生成提示词", "pending"]]
    : [["上传完成", "done"], ["排队等待", "done"], ["抽取关键帧并生成提示词", "active"]];
  for (const [label, st] of defs) {
    const li = el("li", "stage " + st);
    const ic = el("span", "stage-icon");
    if (st === "done") ic.appendChild(icon("i-check"));
    else if (st === "active") ic.appendChild(el("span", "pulse-dot"));
    li.appendChild(ic);
    li.appendChild(el("span", "stage-label", label));
    stages.appendChild(li);
  }
  card.appendChild(stages);
  row.appendChild(card);
  return row;
}

function renderFail(detail) {
  const row = el("div", "msg-row");
  row.appendChild(assistantHead(detail.updated_at));
  const card = el("div", "fail-card");
  card.appendChild(icon("i-alert", "ic-danger"));
  const body = el("div");
  const summary = safeErrorSummary(detail.error, "分析阶段执行失败");
  body.appendChild(el("p", "fail-title", summary.startsWith("台词识别失败")
    ? "台词识别失败" : "分析未完成"));
  body.appendChild(el("p", "fail-msg", summary));
  body.appendChild(el("p", "fail-tip", "成片流程尚未开始；刷新不会重复提交任务。"));
  const diagnostic = errorDiagnostic(detail.error);
  if (diagnostic) body.appendChild(diagnostic);
  card.appendChild(body);
  row.appendChild(card);
  return row;
}

/* 结果区（done） */
/* 视频区块：原始 / 最终成片共用同一加载骨架，靠标题与副标题区分 */
function videoSection(detail, file, title, sub) {
  const sec = el("section", "res-section");
  const h = el("h3", "res-h3", title);
  if (sub) h.appendChild(el("span", "res-count", sub));
  sec.appendChild(h);
  const wrap = el("div", "video-wrap");
  wrap.appendChild(el("div", "video-shimmer shimmer", "正在加载视频…"));
  const video = el("video");
  video.controls = true;
  video.playsInline = true;
  video.preload = "metadata";
  wrap.appendChild(video);
  sec.appendChild(wrap);
  apiBlobURL("/api/conversations/" + detail.id + "/files/" + file)
    .then((url) => {
      video.src = url;
      video.addEventListener("loadeddata", () => wrap.classList.add("is-ready"), { once: true });
      video.addEventListener("error", () => wrap.classList.add("is-error"), { once: true });
    })
    .catch(() => wrap.classList.add("is-error"));
  return sec;
}

/* 研究链路只展示 CID 已冻结的只读 Skill 摘要；不渲染任何本机路径。 */
function skillMilestoneSection(detail) {
  const milestone = skillMilestoneView(detail);
  if (!milestone) return null;
  const section = el("section", "res-section");
  section.dataset.testid = "skill-milestone";
  const card = el("div", "skill-card");
  const heading = el("div", "skill-heading");
  heading.appendChild(el("h3", "res-h3", "运行配方"));
  heading.appendChild(el("span", "skill-version", milestone.label));
  card.appendChild(heading);
  card.appendChild(el("p", "skill-freeze", "已按该项目冻结；后续 Skill 更新不会改变本项目的运行依据。"));
  const list = el("ul", "milestone-skills");
  const roles = {
    "video-maker": "视频分析与镜头规划",
    "image-postprocess": "图片优化与一致性",
    "video-prompt-fusion": "最终提示词融合",
  };
  for (const skill of milestone.skills) {
    const item = el("li");
    item.appendChild(el("strong", null, skill.name));
    item.appendChild(el("span", null, roles[skill.name] || "项目 Skill"));
    item.appendChild(el("code", null, skill.short));
    list.appendChild(item);
  }
  card.appendChild(list);
  const audit = el("details", "skill-audit");
  audit.appendChild(el("summary", null, "查看完整校验信息"));
  audit.appendChild(el("code", null, milestone.id));
  for (const skill of milestone.skills) {
    audit.appendChild(el("code", null, skill.name + " · " + skill.sha256));
  }
  card.appendChild(audit);
  section.appendChild(card);
  return section;
}

function renderMaterialIndexSection(detail) {
  const section = el("section", "res-section material-index-section");
  section.dataset.testid = "material-index";
  section.appendChild(el("h3", "res-h3", "素材与关系"));
  const facts = el("div", "material-facts");
  const segments = Array.isArray(detail && detail.segments) ? detail.segments : [];
  for (const [label, value] of [
    ["关键帧", totalKeyframes(detail) + " 帧"],
    ["视频分段", segments.length > 0 ? segments.length + " 段" : "单段"],
    ["源片时长", formatDuration(detail && detail.duration_s)],
  ]) {
    const fact = el("div", "material-fact");
    fact.appendChild(el("span", null, label));
    fact.appendChild(el("strong", null, value));
    facts.appendChild(fact);
  }
  section.appendChild(facts);

  const index = materialIndexView(detail);
  const cards = el("div", "material-card-grid");
  if (index) {
    const groupLabels = {people: "人物", entities: "物体", scenes: "场景"};
    for (const group of index.groups) {
      for (const entry of group.entries) {
        const card = el("article", "material-card");
        card.appendChild(el("span", "material-type", groupLabels[group.key] || group.key));
        card.appendChild(el("strong", null, entry.id));
        if (entry.description) card.appendChild(el("p", null, entry.description));
        card.appendChild(el("span", "material-occurrences", entry.occurrences + " 个分段记录"));
        cards.appendChild(card);
      }
    }
    for (const relation of index.relations) {
      const card = el("article", "material-card relation-card");
      card.appendChild(el("span", "material-type", "关系"));
      card.appendChild(el("strong", null, relation.id));
      card.appendChild(el("p", null,
        [relation.subject, relation.predicate, relation.object].filter(Boolean).join(" → ") || "关系说明未公开"));
      cards.appendChild(card);
    }
  } else {
    const unavailable = el("article", "material-card material-unavailable");
    unavailable.appendChild(el("span", "material-type", "元素索引"));
    unavailable.appendChild(el("strong", null, "人物与物体明细未公开"));
    unavailable.appendChild(el("p", null,
      "当前只读详情 API 未返回元素索引，因此页面不会从提示词猜测人物、物体或关系。"));
    cards.appendChild(unavailable);
  }

  for (const segment of segments) {
    const card = el("article", "material-card relation-card");
    const indexLabel = Number.isInteger(segment && segment.index) ? segment.index : "?";
    card.appendChild(el("span", "material-type", "分段关系"));
    card.appendChild(el("strong", null, `第 ${indexLabel} 段 · ${segmentJoinText(segment && segment.join_mode)}`));
    const start = Number(segment && segment.start_s);
    const end = Number(segment && segment.end_s);
    const frameCount = Array.isArray(segment && segment.keyframes) ? segment.keyframes.length : 0;
    card.appendChild(el("p", null,
      Number.isFinite(start) && Number.isFinite(end)
        ? `${start.toFixed(1)}–${end.toFixed(1)} 秒 · ${frameCount} 帧`
        : `${frameCount} 帧 · 时间边界未公开`));
    cards.appendChild(card);
  }
  section.appendChild(cards);
  return section;
}

function renderResults(detail) {
  const frag = document.createDocumentFragment();

  const headRow = el("div", "msg-row");
  headRow.appendChild(assistantHead(detail.updated_at));
  const doneCard = el("div", "activity-card");
  doneCard.appendChild(el("p", "ac-title", "处理完成"));
  doneCard.appendChild(el("p", "ac-sub", "关键帧与提示词已生成，可直接复制使用"));
  headRow.appendChild(doneCard);
  frag.appendChild(headRow);

  const milestone = skillMilestoneSection(detail);
  if (milestone) frag.appendChild(milestone);

  frag.appendChild(renderMaterialIndexSection(detail));

  // 原始视频（上传即存在，与生成物明确分区）
  if (detail.has_source) {
    frag.appendChild(videoSection(detail, "source.mp4", "原始视频", "上传的源素材"));
  }

  // 多段模式：逐段渲染「第 N 段」卡片；单段模式保持现有逻辑
  const segments = Array.isArray(detail.segments) ? detail.segments : [];
  if (segments.length > 0) {
    frag.appendChild(segmentProductsDisclosure(detail));
  } else {
    const names = Array.isArray(detail.keyframes) ? detail.keyframes : [];
    if (names.length > 0) {
      frag.appendChild(keyframesSection(detail));
    }
    const sourcePrompt = detail.source_prompt || detail.prompt;
    if (sourcePrompt) {
      const sec = el("section", "res-section");
      sec.appendChild(el("h3", "res-h3", "提示词与台词"));
      sec.appendChild(promptWorkspace(detail));
      frag.appendChild(sec);
    }
  }

  return frag;
}

function canOperate(detail) {
  // 缺少新契约字段的旧会话按只读处理，避免旧 UI 状态绕过服务端门控。
  return detail.read_only === false && detail.submit_enabled === true;
}

function generationDraft(detail) {
  let draft = state.generationDrafts[detail.id];
  const parameters = generationParameterDraft(detail);
  const parameterVersion = detail.aspect_ratio + "|" + detail.resolution
    + "|" + JSON.stringify(detail.fit_profiles);
  const hasFrozenGeneration = !!(detail.generation
    && typeof detail.generation === "object");
  if (!draft) {
    draft = {
      dialogueMode: "auto",
      editLinesText: formatDialogueLines(detail.dialogue),
      customLinesText: "",
      fastMode: longVideoContract(detail).isLong,
      ...parameters,
      parameterVersion,
      receiptVersion: detail.receipt_version,
    };
    state.generationDrafts[detail.id] = draft;
  }
  if (hasFrozenGeneration) {
    const dialogueMode = detail.dialogue && detail.dialogue.mode;
    if (!["auto", "edit", "custom", "none"].includes(dialogueMode)) {
      throw new Error("服务端冻结台词模式无效，请刷新页面后重试");
    }
    const fitMode = detail.fit_mode;
    const profile = fitProfile(detail, detail.aspect_ratio);
    if ((profile.fit_required && !["crop", "pad"].includes(fitMode))
        || (!profile.fit_required && fitMode !== "none")) {
      throw new Error("服务端冻结适配方式无效，请刷新页面后重试");
    }
    const frozenLines = formatDialogueLines(detail.dialogue);
    Object.assign(draft, parameters, {
      dialogueMode,
      fitMode,
      editLinesText: dialogueMode === "edit" ? frozenLines : draft.editLinesText,
      customLinesText: dialogueMode === "custom" ? frozenLines : draft.customLinesText,
      parameterVersion,
      receiptVersion: detail.receipt_version,
      parameterTouched: false,
      editTouched: false,
      frozen: true,
      fastMode: generationFastMode(detail),
    });
    return draft;
  }
  draft.frozen = false;
  if (draft.receiptVersion !== detail.receipt_version && !draft.editTouched) {
    draft.editLinesText = formatDialogueLines(detail.dialogue);
    draft.receiptVersion = detail.receipt_version;
  }
  if (draft.parameterVersion !== parameterVersion && !draft.parameterTouched) {
    Object.assign(draft, parameters, { parameterVersion });
  }
  return draft;
}

function choice(name, value, label, checked) {
  const item = el("label", "final-choice");
  const input = el("input");
  input.type = "radio";
  input.name = name;
  input.value = value;
  input.checked = checked;
  item.appendChild(input);
  item.appendChild(el("span", null, label));
  return item;
}

function setGenerationCardBusy(card, busy) {
  card.querySelectorAll("input, textarea, button").forEach((control) => {
    control.disabled = busy;
  });
  const button = card.querySelector(".generation-submit");
  if (button && busy) button.textContent = "正在提交…";
}

async function submitGeneration(detail, card) {
  if (!await guardDirtyPrompt()) return;
  const generation = detail.generation || {};
  const action = generationAction(generation.status, generation.stage);
  if (!canOperate(detail) || state.generationSubmitting[detail.id]
      || !["new", "retry", "retry_stitch"].includes(action)) return;
  const errorBox = card.querySelector(".generation-form-error");
  if (!postprocessReadyForGeneration(detail)) {
    errorBox.textContent = "素材优化尚未全部完成，不能生成最终视频";
    errorBox.hidden = false;
    return;
  }
  const draft = generationDraft(detail);
  const longContract = longVideoContract(detail);
  let body;
  try {
    if (action === "retry_stitch") {
      body = buildStitchRetryPayload(detail);
    } else if (longContract.isLong && action === "retry") {
      body = buildLongRetryPayload(detail, newRequestId());
    } else {
      const profile = fitProfile(detail, draft.aspectRatio);
      body = buildSubmitPayload({
        clientRequestId: newRequestId(),
        dialogueMode: draft.dialogueMode,
        linesText: draft.dialogueMode === "edit" ? draft.editLinesText : draft.customLinesText,
        fitRequired: profile.fit_required,
        fitMode: draft.fitMode,
        aspectRatio: draft.aspectRatio,
        resolution: draft.resolution,
        isLong: longContract.isLong,
        fastMode: draft.fastMode,
        planReceipt: longContract.planReceipt,
      });
    }
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
    return;
  }

  await postGeneration(detail, card, body);
}

function generationStageText(stage) {
  if (stage === "context_ir" || stage === "context_ir_native") return "Context IR 提示词优化";
  if (stage === "prompt_fusion") return "Prompt Fusion 最终提示词融合";
  if (stage === "h3") return "H3 分段视频生成";
  if (stage === "stitch") return "视频拼接";
  if (stage === "stitching") return "视频拼接";
  return stage ? "服务器阶段：" + String(stage) : "等待开始";
}

function generationSegmentStatusText(status) {
  const labels = {
    pending: "等待中", queued: "排队中", running: "生成中", succeeded: "已完成",
    failed: "失败", resume_required: "等待继续", submission_unknown: "状态未知",
  };
  return labels[status] || status || "等待中";
}

function generationSegmentLabel(segment, position) {
  const index = segment && Number.isInteger(segment.index) && segment.index > 0
    ? segment.index : position + 1;
  return "第 " + index + " 段 · "
    + generationSegmentStatusText(segment && segment.status);
}

function appendGenerationProgress(card, generation) {
  const segments = Array.isArray(generation.segments) ? generation.segments : [];
  if (segments.length === 0) return;
  const completed = segments.filter((segment) => segment && segment.status === "succeeded").length;
  const progress = el("div", "generation-progress");
  progress.appendChild(el("strong", null, "完成 " + completed + "/" + segments.length));
  progress.appendChild(el("span", null, "当前阶段：" + generationStageText(generation.stage)));
  const list = el("ol", "generation-segments");
  segments.forEach((segment, position) => {
    const item = el("li", "generation-segment status-" + String(segment.status || "pending"));
    item.appendChild(el("strong", null, generationSegmentLabel(segment, position)));
    const meta = el("span", null,
      Number.isInteger(segment && segment.attempt) && segment.attempt > 0
        ? "第 " + segment.attempt + " 次执行" : "等待服务端更新");
    item.appendChild(meta);
    if (segment.error) {
      item.appendChild(el("span", "generation-segment-error", safeErrorSummary(segment.error, "本段生成失败")));
      const diagnostic = errorDiagnostic(segment.error, "本段高级诊断");
      if (diagnostic) item.appendChild(diagnostic);
    }
    list.appendChild(item);
  });
  progress.appendChild(list);
  card.appendChild(progress);
}

function renderIntermediateStages(detail) {
  const fusion = detail && detail.prompt_fusion;
  const timeline = operationTimeline(detail);
  const contextStage = timeline.stages.find((stage) => stage.key === "context");
  if (!fusion && (!contextStage || ["waiting", "skipped"].includes(contextStage.status))) return null;
  const section = el("section", "res-section intermediate-stages");
  section.appendChild(el("h3", "res-h3", "中间阶段"));

  if (fusion) {
    const card = el("article", "intermediate-card");
    const fusionStage = timeline.stages.find((stage) => stage.key === "fusion");
    const heading = el("div", "intermediate-heading");
    heading.appendChild(el("strong", null, "Prompt Fusion"));
    heading.appendChild(el("span", `stage-pill status-${fusionStage.status}`,
      fusionStage.count || stageStatusText(fusionStage.status)));
    card.appendChild(heading);
    card.appendChild(el("p", "ac-sub", fusionStage.detail));
    if (fusion.error) {
      const diagnostic = errorDiagnostic(fusion.error);
      if (diagnostic) card.appendChild(diagnostic);
    }
    const segments = Array.isArray(fusion.segments) ? fusion.segments : [];
    if (segments.length > 0) {
      const list = el("ol", "intermediate-list");
      for (const segment of segments) {
        const item = el("li");
        item.appendChild(el("strong", null,
          `第 ${Number.isInteger(segment && segment.index) ? segment.index : "?"} 段`));
        item.appendChild(el("span", null,
          ({pending: "等待融合", running: "融合中", done: "已冻结", failed: "融合失败"})[segment && segment.status]
            || "状态未知"));
        if (segment && segment.status === "done" && typeof segment.final_prompt === "string"
            && segment.final_prompt.trim()) {
          const prompt = el("details", "fusion-prompt");
          prompt.appendChild(el("summary", null, "查看最终提示词"));
          prompt.appendChild(el("pre", null, segment.final_prompt));
          item.appendChild(prompt);
        }
        if (segment && segment.error) {
          item.appendChild(el("span", "generation-segment-error",
            safeErrorSummary(segment.error, "本段融合失败")));
          const diagnostic = errorDiagnostic(segment.error, "本段高级诊断");
          if (diagnostic) item.appendChild(diagnostic);
        }
        list.appendChild(item);
      }
      card.appendChild(list);
    }
    section.appendChild(card);
  }

  if (contextStage && !["waiting", "skipped"].includes(contextStage.status)) {
    const card = el("article", "intermediate-card");
    const heading = el("div", "intermediate-heading");
    heading.appendChild(el("strong", null, "Context IR"));
    heading.appendChild(el("span", `stage-pill status-${contextStage.status}`,
      contextStage.count || stageStatusText(contextStage.status)));
    card.appendChild(heading);
    card.appendChild(el("p", "ac-sub", contextStage.detail));
    const generation = detail && detail.generation;
    if (generation && generation.error && ["failed", "attention"].includes(contextStage.status)) {
      const diagnostic = errorDiagnostic(generation.error);
      if (diagnostic) card.appendChild(diagnostic);
    }
    section.appendChild(card);
  }
  return section;
}

async function resumeGeneration(detail, card) {
  if (!canOperate(detail) || state.generationSubmitting[detail.id]
      || generationAction(detail.generation && detail.generation.status) !== "resume") return;
  const errorBox = card.querySelector(".generation-form-error");
  let body;
  try {
    body = buildResumePayload(detail);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
    return;
  }
  await postGeneration(detail, card, body);
}

async function postGeneration(
  detail,
  card,
  body,
) {
  const generation = detail.generation || {};
  const errorBox = card.querySelector(".generation-form-error");
  errorBox.hidden = true;
  state.generationSubmitting[detail.id] = true;
  setGenerationCardBusy(card, true);
  let accepted = false;
  try {
    await requestGenerationSubmit(detail, body);
    accepted = true;
    await loadDetail(detail.id, true);
    // 极短暂的详情落盘延迟也不能开放第二次 POST；保持禁用并继续 GET 轮询。
    if (state.generationSubmitting[detail.id]) startPolling(detail.id);
  } catch (error) {
    if (handleAuthError(error)) return;
    showActionError(error, errorBox);
    // POST 断线时结果可能未知：先 GET 当前状态再开放重试，但绝不自动再次 POST。
    state.generationSubmitting[detail.id] = false;
    await loadDetail(detail.id, true);
  } finally {
    if (!accepted) state.generationSubmitting[detail.id] = false;
    if (!state.generationSubmitting[detail.id] && card.isConnected) {
      setGenerationCardBusy(card, false);
      const button = card.querySelector(".generation-submit");
      if (button) {
        const action = generationAction(generation.status, generation.stage);
        button.textContent = action === "retry_stitch" ? "仅重试拼接（0 新增付费）"
          : action === "retry" ? "重新生成失败内容" : "开始生成成片";
      }
    }
  }
}

/* 最终视频区：生成参数、异步状态、失败后的显式重试和历史成片都在同一卡片。 */
function renderFinalSection(detail) {
  const generation = detail.generation || { status: null, error: null, attempt: null };
  const showPublishedVideo = detail.has_video === true;
  const showStitchRecovery = generation.status === "failed" && generation.stage === "stitch";
  const published = document.createDocumentFragment();
  if (showPublishedVideo) {
    published.appendChild(videoSection(detail, "generated.mp4", "最终视频", "生成成片"));
    published.appendChild(generationParameterSummary(detail));
  }
  if (showPublishedVideo && !showStitchRecovery) return published;

  const sec = el("section", "res-section");
  published.appendChild(sec);
  const card = el("div", "final-card");
  card.appendChild(el("h3", "res-h3", "最终视频"));
  appendGenerationProgress(card, generation);

  if (generation.status === "queued" || generation.status === "running") {
    const status = el("div", "generation-status is-running");
    status.appendChild(el("strong", null,
      generation.status === "queued" ? "生成任务已排队" : "视频正在生成"));
    status.appendChild(el("span", null, generation.attempt ? "第 " + generation.attempt + " 次尝试" : "请稍候，页面会自动更新"));
    card.appendChild(status);
  } else if (generation.status === "failed" || generation.status === "submission_unknown") {
    const status = el("div", "generation-status is-error");
    const failedTitle = generation.stage === "stitch" ? "视频拼接失败" : "视频生成失败";
    status.appendChild(el("strong", null,
      generation.status === "submission_unknown" ? "提交结果未知" : failedTitle));
    status.appendChild(el("span", null, safeErrorSummary(generation.error, "视频任务执行失败")));
    card.appendChild(status);
    const diagnostic = errorDiagnostic(generation.error);
    if (diagnostic) card.appendChild(diagnostic);
  } else if (generation.status === "resume_required") {
    const status = el("div", "generation-status is-resume");
    status.appendChild(el("strong", null, "既有生成任务等待继续"));
    status.appendChild(el("span", null,
      generation.error ? safeErrorSummary(generation.error, "任务已保存，可从原进度继续")
        : "任务已保存，可从原进度继续"));
    card.appendChild(status);
  }

  if (generation.status === "submission_unknown") {
    card.appendChild(el("p", "final-caption", "提交结果未知，禁止重复提交；请先在 provider 侧核对任务。"));
    sec.appendChild(card);
    return published;
  }

  if (!canOperate(detail)) {
    card.appendChild(el("p", "final-caption", "此会话为只读状态，不能修改台词、画幅、后处理或再次生成。"));
    sec.appendChild(card);
    return published;
  }

  if (frozenGenerationConfig(detail) && generation.status === null) {
    const automatic = el("div", "generation-status is-running");
    automatic.appendChild(el("strong", null, "正在按生成配置继续处理"));
    automatic.appendChild(el("span", null, "服务端会自动运行至成片，无需再次确认或提交。"));
    card.appendChild(automatic);
    sec.appendChild(card);
    return published;
  }

  if (!postprocessReadyForGeneration(detail)) {
    card.appendChild(el("p", "final-warning", "素材优化尚未全部完成，不能生成最终视频"));
    sec.appendChild(card);
    return published;
  }

  const longContract = longVideoContract(detail);
  if (longContract.isLong && !longContract.ready) {
    card.appendChild(el("p", "final-warning", "长视频生成计划尚未就绪，请刷新后重试"));
    sec.appendChild(card);
    return published;
  }

  const retryContract = generationRetryContract(detail);
  if (longContract.isLong && ["retry", "retry_stitch"].includes(retryContract.action)
      && retryContract.paidTaskCount === null) {
    card.appendChild(el("p", "final-warning", "分段冻结状态不完整，无法安全计算本次付费子任务数，请刷新后重试"));
    sec.appendChild(card);
    return published;
  }

  if (longContract.isLong && ["new", "retry", "retry_stitch"].includes(retryContract.action)) {
    const notice = el("div", "long-video-notice");
    notice.appendChild(el("strong", null,
      "本次新增 " + retryContract.paidTaskCount + " 个付费生成子任务"));
    const noticeText = retryContract.action === "retry_stitch"
      ? "全部分段成片已复用，本次只在本地重试拼接。"
      : retryContract.action === "retry"
        ? "跨段连续性为 best effort；服务端按冻结模式复用成功段，本次只提交上方所示新增付费分段。"
        : "跨段连续性为 best effort；各段提示词将提交生成。";
    notice.appendChild(el("p", null, noticeText));
    card.appendChild(notice);
  }

  if (generationAction(generation.status, generation.stage) === "resume") {
    const locked = el("div", "resume-lock");
    locked.appendChild(el("strong", null, "继续既有生成任务"));
    locked.appendChild(el("p", null,
      "将使用已保存的请求标识和冻结输入继续查询既有生成任务。"));
    card.appendChild(locked);
    const errorBox = el("p", "form-error generation-form-error");
    errorBox.hidden = true;
    card.appendChild(errorBox);
    const row = el("div", "final-row");
    const button = el("button", "btn btn-primary generation-submit", "继续原任务（0 新增付费）");
    button.type = "button";
    button.addEventListener("click", () => {
      resumeGeneration(detail, card);
    });
    row.appendChild(button);
    row.appendChild(el("p", "final-caption", "继续原任务，不创建新的生成请求。"));
    card.appendChild(row);
    if (state.generationSubmitting[detail.id]) setGenerationCardBusy(card, true);
    sec.appendChild(card);
    return published;
  }


  if (longContract.isLong && (retryContract.action === "retry"
      || retryContract.action === "retry_stitch")) {
    const stitchOnly = retryContract.action === "retry_stitch";
    const locked = el("div", "resume-lock");
    locked.appendChild(el("strong", null, stitchOnly ? "重试本地拼接" : "重试失败的生成分段"));
    locked.appendChild(el("p", null, stitchOnly
      ? "复用原请求标识和全部成功分段，不创建新的付费生成子任务。"
      : "使用新的请求标识和逐段冻结输入；本次只提交服务端计算出的新增付费分段。"));
    card.appendChild(locked);
    const errorBox = el("p", "form-error generation-form-error");
    errorBox.hidden = true;
    card.appendChild(errorBox);
    const row = el("div", "final-row");
    const label = stitchOnly ? "仅重试拼接（0 新增付费）"
      : "重新生成失败分段（新增 " + retryContract.paidTaskCount + " 个付费任务）";
    const button = el("button", "btn btn-primary generation-submit", label);
    button.type = "button";
    button.addEventListener("click", () => submitGeneration(detail, card));
    row.appendChild(button);
    row.appendChild(el("p", "final-caption", stitchOnly
      ? "本次新增 0 个付费生成子任务。"
      : "本次新增 " + retryContract.paidTaskCount + " 个付费生成子任务。"));
    card.appendChild(row);
    if (state.generationSubmitting[detail.id]) setGenerationCardBusy(card, true);
    sec.appendChild(card);
    return published;
  }

  const draft = generationDraft(detail);
  const hasFrozenGeneration = draft.frozen === true;
  const busy = generation.status === "queued" || generation.status === "running"
    || !!state.generationSubmitting[detail.id];
  const aspectField = el("fieldset", "final-field");
  aspectField.appendChild(el("legend", null, "画幅"));
  const aspectChoices = el("div", "final-choices");
  for (const [value, label] of [["16:9", "横屏 16:9"], ["9:16", "竖屏 9:16"]]) {
    const item = choice("aspect-" + detail.id, value, label, draft.aspectRatio === value);
    item.querySelector("input").addEventListener("change", () => {
      draft.aspectRatio = value;
      draft.fitMode = fitProfile(detail, value).default_fit_mode;
      draft.parameterTouched = true;
      renderGenerationDynamic(detail);
    });
    aspectChoices.appendChild(item);
  }
  aspectField.appendChild(aspectChoices);
  aspectField.appendChild(el("p", "final-help", "系统已按实际输入帧的总几何损失预选。"));
  card.appendChild(aspectField);

  const resolutionField = el("fieldset", "final-field");
  resolutionField.appendChild(el("legend", null, "清晰度"));
  const resolutionChoices = el("div", "final-choices");
  for (const value of GENERATION_RESOLUTIONS) {
    const item = choice("resolution-" + detail.id, value, value, draft.resolution === value);
    item.querySelector("input").addEventListener("change", () => {
      draft.resolution = value;
      draft.parameterTouched = true;
    });
    resolutionChoices.appendChild(item);
  }
  resolutionField.appendChild(resolutionChoices);
  resolutionField.appendChild(el("p", "final-help", "系统已按源视频短边最接近档位预选。"));
  card.appendChild(resolutionField);

  const dialogueField = el("fieldset", "final-field");
  dialogueField.appendChild(el("legend", null, "台词模式"));
  const dialogueChoices = el("div", "final-choices");
  const modes = longContract.isLong ? [
    ["auto", "保留完整源音轨"],
    ["none", "静音"],
  ] : [
    ["auto", "自动台词"],
    ["edit", "编辑识别台词"],
    ["custom", "自定义台词"],
    ["none", "无台词"],
  ];
  if (longContract.isLong && draft.dialogueMode !== "auto" && draft.dialogueMode !== "none") {
    draft.dialogueMode = "auto";
  }
  for (const [value, label] of modes) {
    const item = choice("dialogue-" + detail.id, value, label, draft.dialogueMode === value);
    const input = item.querySelector("input");
    input.addEventListener("change", () => {
      draft.dialogueMode = value;
      const editor = card.querySelector(".dialogue-editor");
      const textarea = editor.querySelector("textarea");
      const editable = value === "edit" || value === "custom";
      editor.hidden = !editable;
      if (editable) textarea.value = value === "edit" ? draft.editLinesText : draft.customLinesText;
    });
    dialogueChoices.appendChild(item);
  }
  dialogueField.appendChild(dialogueChoices);
  const autoCount = normalizeDialogueLines(detail.dialogue).length;
  dialogueField.appendChild(el("p", "final-help", longContract.isLong
    ? "保留完整源音轨不会按段改写台词；选择静音将移除源音轨。"
    : (autoCount
      ? "自动识别到 " + autoCount + " 行台词；编辑模式会以这些台词预填。"
      : "未识别到自动台词；可改用自定义或无台词。")));
  card.appendChild(dialogueField);

  const editor = el("div", "dialogue-editor");
  editor.hidden = longContract.isLong
    || (draft.dialogueMode !== "edit" && draft.dialogueMode !== "custom");
  const textarea = el("textarea", "dialogue-textarea");
  textarea.rows = 5;
  textarea.placeholder = "0 - 1.5 | 第一行台词\n1.5 - 3 | 第二行台词";
  textarea.setAttribute("aria-label", "台词，每行格式为开始 - 结束 | 文本");
  textarea.value = draft.dialogueMode === "custom" ? draft.customLinesText : draft.editLinesText;
  textarea.addEventListener("input", () => {
    if (draft.dialogueMode === "edit") {
      draft.editLinesText = textarea.value;
      draft.editTouched = true;
    } else {
      draft.customLinesText = textarea.value;
    }
  });
  editor.appendChild(textarea);
  editor.appendChild(el("p", "final-help", "每行格式：开始秒 - 结束秒 | 台词。自定义模式不受识别结果限制。"));
  card.appendChild(editor);

  const selectedFitProfile = fitProfile(detail, draft.aspectRatio);
  if (selectedFitProfile.fit_required) {
    const fitField = el("fieldset", "final-field");
    fitField.appendChild(el("legend", null, "源画幅需要适配"));
    const fitChoices = el("div", "final-choices");
    for (const [value, label] of [["crop", "裁切画面"], ["pad", "留边完整展示"]]) {
      const item = choice("fit-" + detail.id, value, label, draft.fitMode === value);
      item.querySelector("input").addEventListener("change", () => {
        draft.fitMode = value;
        draft.parameterTouched = true;
      });
      fitChoices.appendChild(item);
    }
    fitField.appendChild(fitChoices);
    fitField.appendChild(el("p", "final-help", "必须选择一种方式后才能生成。"));
    card.appendChild(fitField);
  }

  const errorBox = el("p", "form-error generation-form-error");
  errorBox.hidden = true;
  card.appendChild(errorBox);
  const row = el("div", "final-row");
  const buttonLabel = generation.status === "failed"
    ? "重新生成（新增 " + retryContract.paidTaskCount + " 个付费任务）"
    : "开始生成成片（新增 " + retryContract.paidTaskCount + " 个付费任务）";
  const button = el("button", "btn btn-primary generation-submit", buttonLabel);
  button.type = "button";
  button.addEventListener("click", () => submitGeneration(detail, card));
  row.appendChild(button);
  row.appendChild(el("p", "final-caption generation-mode-caption", hasFrozenGeneration
    ? (busy ? "参数已按服务端冻结，正在等待生成结果"
      : "重试将使用上次服务端冻结的生成参数")
    : longContract.isLong
      ? "各段提示词将提交生成"
      : "源提示词将直接提交生成"));
  card.appendChild(row);
  if (hasFrozenGeneration) {
    card.querySelectorAll("input, textarea").forEach((control) => {
      control.disabled = true;
    });
  }
  if (busy) setGenerationCardBusy(card, true);
  sec.appendChild(card);
  return published;
}

function setDisclosureState(trigger, panel, expanded, labels) {
  trigger.setAttribute("aria-expanded", String(expanded));
  trigger.setAttribute("aria-label", expanded ? labels.collapse : labels.expand);
  if (panel) panel.hidden = !expanded;
  if (labels.expandText && labels.collapseText) {
    trigger.textContent = expanded ? labels.collapseText : labels.expandText;
  }
}

function keyframeDisclosureLabels(alt) {
  return { expand: "展开查看" + alt, collapse: "关闭" + alt + "大图" };
}

function safeMediaBasename(value) {
  return typeof value === "string"
    && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)
    && value !== "." && value !== "..";
}

function encodedMediaPath(path) {
  return path.split("/").map((part) => encodeURIComponent(part)).join("/");
}

/* 分段图片只接受详情接口发布的权威路径；布局不完整时不猜目录。 */
function authoritativeSegmentKeyframePaths(detail, segment) {
  const segments = Array.isArray(detail && detail.segments) ? detail.segments : [];
  const count = detail && detail.segment_count;
  const index = segment && segment.index;
  const names = segment && segment.keyframes;
  const paths = segment && segment.keyframe_paths;
  if (
    !Number.isInteger(count) || count < 1 || count !== segments.length
    || !Number.isInteger(index) || index < 1 || segments[index - 1] !== segment
    || !Array.isArray(names) || !names.length
    || !Array.isArray(paths) || paths.length !== names.length
    || names.some((name) => !safeMediaBasename(name))
  ) return [];
  const rootLayout = count === 1 && index === 1
    && paths.every((path, offset) => path === "keyframes/" + names[offset]);
  const segmentLayout = paths.every(
    (path, offset) => path === "segments/" + index + "/work/keyframes/" + names[offset],
  );
  return rootLayout || segmentLayout ? paths.slice() : [];
}

function authoritativePostprocessFrameGroups(detail, frames) {
  if (!Array.isArray(frames) || frames.some((frame) => typeof frame !== "string")) return [];
  const segments = Array.isArray(detail && detail.segments) ? detail.segments : [];
  if (!segments.length) {
    if (frames.some((frame) => !safeMediaBasename(frame))) return [];
    return frames.length ? [{ index: null, names: frames.slice(), paths: frames.map(
      (name) => "postprocessed/" + name,
    ) }] : [];
  }
  const count = detail && detail.segment_count;
  if (!Number.isInteger(count) || count !== segments.length || count < 1) return [];
  const groups = [];
  const consumed = new Set();
  for (const segment of segments) {
    const index = segment && segment.index;
    const names = segment && segment.keyframes;
    if (
      !Number.isInteger(index) || index < 1 || segments[index - 1] !== segment
      || !Array.isArray(names) || names.some((name) => !safeMediaBasename(name))
    ) return [];
    const keyframePaths = authoritativeSegmentKeyframePaths(detail, segment);
    if (!keyframePaths.length) return [];
    const rootLayout = count === 1
      && keyframePaths.every((path, offset) => path === "keyframes/" + names[offset]);
    const selected = [];
    for (const name of names) {
      const reference = rootLayout
        ? name
        : "segments/" + index + "/work/postprocessed/" + name;
      const position = frames.indexOf(reference);
      if (position < 0 || consumed.has(position)) continue;
      consumed.add(position);
      selected.push({ name, path: rootLayout ? "postprocessed/" + name : reference });
    }
    if (selected.length) groups.push({
      index,
      names: selected.map((item) => item.name),
      paths: selected.map((item) => item.path),
    });
  }
  return consumed.size === frames.length ? groups : [];
}

function validatedFramePromptMap(value, names) {
  if (!Array.isArray(value) || !Array.isArray(names) || value.length !== names.length) return null;
  const result = new Map();
  for (let offset = 0; offset < names.length; offset += 1) {
    const item = value[offset];
    if (!item || item.frame_name !== names[offset]
        || typeof item.text !== "string" || !item.text.trim()
        || typeof item.default_text !== "string" || !item.default_text.trim()
        || typeof item.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(item.sha256)) return null;
    result.set(item.frame_name, {
      text: item.text,
      defaultText: item.default_text,
      sha256: item.sha256,
    });
  }
  return result;
}

function frameViewerEntries(detail, requestedSegment = null) {
  if (!detail || typeof detail !== "object") return [];
  const segments = Array.isArray(detail.segments) ? detail.segments : [];
  let sources;
  if (requestedSegment !== null) {
    if (!segments.includes(requestedSegment)) return [];
    sources = [requestedSegment];
  } else if (segments.length > 0) {
    sources = segments;
  } else {
    sources = [{
      index: null,
      start_s: 0,
      end_s: detail.duration_s,
      join_mode: null,
      keyframes: Array.isArray(detail.keyframes) ? detail.keyframes : [],
      prompt: detail.source_prompt || detail.prompt || "",
      image_optimization_prompt: detail.image_optimization_prompt,
      image_optimization_prompts: null,
    }];
  }
  const post = detail.postprocess && typeof detail.postprocess === "object" ? detail.postprocess : null;
  const postGroups = authoritativePostprocessFrameGroups(
    detail,
    post && Array.isArray(post.frames) ? post.frames : [],
  );
  const postSegments = post && Array.isArray(post.segments) ? post.segments : [];
  const entries = [];
  for (const segment of sources) {
    const names = Array.isArray(segment && segment.keyframes) ? segment.keyframes : [];
    if (names.some((name) => !safeMediaBasename(name))) continue;
    const segmentIndex = Number.isInteger(segment && segment.index) ? segment.index : null;
    const paths = segmentIndex === null
      ? names.map((name) => "keyframes/" + name)
      : authoritativeSegmentKeyframePaths(detail, segment);
    if (paths.length !== names.length) continue;
    const promptMap = validatedFramePromptMap(segment && segment.image_optimization_prompts, names);
    const group = postGroups.find((item) => item.index === segmentIndex)
      || (segmentIndex === null ? postGroups.find((item) => item.index === null) : null);
    const optimizedByName = new Map();
    if (group) group.names.forEach((name, offset) => optimizedByName.set(name, group.paths[offset]));
    const postSegment = postSegments.find((item) => item && item.index === (segmentIndex || 1));
    for (let offset = 0; offset < names.length; offset += 1) {
      const name = names[offset];
      const optimizedPath = optimizedByName.get(name) || null;
      const framePrompt = promptMap && promptMap.get(name);
      const optimizationRequested = !!(post && post.options && post.options.optimize_image === true);
      const status = optimizedPath ? "processed"
        : optimizationRequested && post.status === "running" ? "running"
          : optimizationRequested && post.status === "failed" ? "failed"
            : optimizationRequested && post.status === "done" ? "unavailable" : "source";
      const start = Number(segment && segment.start_s);
      const end = Number(segment && segment.end_s);
      entries.push({
        id: `s${segmentIndex || 0}-f${offset + 1}-${name}`,
        segmentIndex,
        frameIndex: offset + 1,
        name,
        originalPath: paths[offset],
        optimizedPath,
        status,
        statusLabel: ({
          processed: "已优化", running: "处理中", failed: "处理失败",
          unavailable: "优化图未发布", source: "原图",
        })[status],
        generationPrompt: String(segment && segment.prompt || detail.source_prompt || detail.prompt || ""),
        imagePrompt: framePrompt ? framePrompt.text
          : String(segment && segment.image_optimization_prompt && segment.image_optimization_prompt.text
            || detail.image_optimization_prompt && detail.image_optimization_prompt.text || ""),
        relation: segmentIndex === null
          ? `单段项目 · 第 ${offset + 1} 帧`
          : `第 ${segmentIndex} 段 · ${segmentJoinText(segment && segment.join_mode)} · 第 ${offset + 1} 帧`,
        timeRange: Number.isFinite(start) && Number.isFinite(end)
          ? `${start.toFixed(1)}–${end.toFixed(1)} 秒` : "时间边界未公开",
        completedFrames: postSegment && Number.isInteger(postSegment.completed_frames)
          ? postSegment.completed_frames : null,
        totalFrames: postSegment && Number.isInteger(postSegment.total_frames)
          ? postSegment.total_frames : null,
      });
    }
  }
  return entries;
}

function selectedFrameEntry(entries, selectedId) {
  if (!Array.isArray(entries) || entries.length === 0) return null;
  return entries.find((entry) => entry.id === selectedId)
    || entries.find((entry) => entry.status === "running")
    || entries[0];
}

function frameViewerScope(detail, segment, context) {
  const index = Number.isInteger(segment && segment.index) ? segment.index : 0;
  return `${detail.id}:${context}:s${index}`;
}

function frameMediaCard(detail, path, title, alt, urls) {
  const card = el("figure", "frame-media-card shimmer");
  card.appendChild(el("figcaption", "frame-media-title", title));
  const button = el("button", "frame-media-button");
  button.type = "button";
  button.disabled = true;
  button.setAttribute("aria-label", "放大查看" + title);
  const img = el("img");
  img.alt = alt;
  button.appendChild(img);
  card.appendChild(button);
  apiBlobURL("/api/conversations/" + encodeURIComponent(detail.id)
    + "/files/" + encodedMediaPath(path))
    .then((url) => {
      if (card.isConnected === false) {
        releaseTrackedURL(url);
        return;
      }
      urls.push(url);
      img.src = url;
      img.addEventListener("load", () => {
        card.classList.remove("shimmer");
        card.classList.add("is-loaded");
      }, { once: true });
      button.disabled = false;
      button.addEventListener("click", () => openLightbox(img.src, img.alt, button));
    })
    .catch(() => {
      card.classList.remove("shimmer");
      card.appendChild(el("div", "kf-err", "加载失败"));
    });
  return card;
}

function framePickerOption(detail, entry, selected, urls, onSelect) {
  const option = el("button", "frame-picker-option");
  option.type = "button";
  option.setAttribute("role", "option");
  option.setAttribute("aria-selected", String(selected));
  option.dataset.frameId = entry.id;
  const thumb = el("span", "frame-picker-thumb shimmer");
  const img = el("img");
  img.alt = "";
  thumb.appendChild(img);
  option.appendChild(thumb);
  const text = el("span", "frame-picker-option-text");
  text.appendChild(el("strong", null,
    (entry.segmentIndex === null ? "单段" : "第 " + entry.segmentIndex + " 段")
      + " · 第 " + entry.frameIndex + " 帧"));
  text.appendChild(el("small", null, entry.name));
  option.appendChild(text);
  option.appendChild(el("span", "frame-status status-" + entry.status, entry.statusLabel));
  option.addEventListener("click", () => onSelect(entry.id));
  apiBlobURL("/api/conversations/" + encodeURIComponent(detail.id)
    + "/files/" + encodedMediaPath(entry.optimizedPath || entry.originalPath))
    .then((url) => {
      if (option.isConnected === false) {
        releaseTrackedURL(url);
        return;
      }
      urls.push(url);
      img.src = url;
      img.addEventListener("load", () => thumb.classList.remove("shimmer"), { once: true });
    })
    .catch(() => thumb.classList.remove("shimmer"));
  return option;
}

function frameInspector(detail, segment = null, options = {}) {
  const context = options.context || "frames";
  const mode = options.mode === "image" ? "image" : "generation";
  const scope = frameViewerScope(detail, segment, context);
  const entries = frameViewerEntries(detail, segment);
  const wrap = el("div", "frame-inspector");
  const pickerURLs = [];
  let detailURLs = [];
  let selected = selectedFrameEntry(entries, state.frameSelections[scope]);
  if (!selected) {
    wrap.appendChild(el("p", "prompt-unavailable", "当前没有可展示的图片"));
    return { node: wrap, dispose: () => {} };
  }
  state.frameSelections[scope] = selected.id;

  const picker = el("details", "frame-picker");
  picker.open = state.framePickerOpen[scope] === true;
  const summary = el("summary", "frame-picker-summary");
  const selectedLabel = el("span", "frame-picker-selected");
  const selectedStatus = el("span", "frame-status");
  summary.appendChild(selectedLabel);
  summary.appendChild(selectedStatus);
  picker.appendChild(summary);
  const list = el("div", "frame-picker-list");
  list.setAttribute("role", "listbox");
  list.setAttribute("aria-label", "按分段和帧选择图片");
  picker.appendChild(list);
  wrap.appendChild(picker);
  const selectedHost = el("div", "frame-inspector-detail");
  wrap.appendChild(selectedHost);
  let optionsBuilt = false;

  const renderSelected = () => {
    releaseTrackedURLs(detailURLs);
    detailURLs = [];
    selectedHost.textContent = "";
    selectedLabel.textContent = (selected.segmentIndex === null ? "单段" : "第 " + selected.segmentIndex + " 段")
      + " · 第 " + selected.frameIndex + " 帧 · " + selected.name;
    selectedStatus.className = "frame-status status-" + selected.status;
    selectedStatus.textContent = selected.statusLabel;
    for (const option of list.querySelectorAll("[role=option]")) {
      option.setAttribute("aria-selected", String(option.dataset.frameId === selected.id));
    }

    const compare = el("div", "frame-compare");
    compare.appendChild(frameMediaCard(
      detail, selected.originalPath, "原图",
      selected.relation + "原图", detailURLs,
    ));
    if (selected.optimizedPath) {
      compare.appendChild(frameMediaCard(
        detail, selected.optimizedPath, "优化图",
        selected.relation + "优化图", detailURLs,
      ));
    } else {
      const missing = el("div", "frame-media-card frame-media-placeholder");
      missing.appendChild(el("span", "frame-media-title", "优化图"));
      missing.appendChild(el("p", null, selected.status === "source"
        ? "本次未生成优化图" : selected.statusLabel));
      compare.appendChild(missing);
    }
    selectedHost.appendChild(compare);
    const relation = el("div", "frame-relation");
    relation.appendChild(el("strong", null, "素材关系"));
    relation.appendChild(el("span", null, selected.relation));
    relation.appendChild(el("span", null, selected.timeRange));
    if (selected.completedFrames !== null && selected.totalFrames !== null) {
      relation.appendChild(el("span", null,
        "本段优化 " + selected.completedFrames + "/" + selected.totalFrames + " 帧"));
    }
    selectedHost.appendChild(relation);
    if (options.showPrompt !== false) {
      const prompt = mode === "image" ? selected.imagePrompt : selected.generationPrompt;
      selectedHost.appendChild(prompt
        ? promptCard(prompt, [], mode === "image" ? "该帧图片优化提示词" : "该帧生成提示词")
        : el("p", "prompt-unavailable", mode === "image"
          ? "该帧没有已发布的图片优化提示词" : "该帧没有已发布的生成提示词"));
    }
  };

  const buildOptions = () => {
    if (optionsBuilt) return;
    optionsBuilt = true;
    for (const entry of entries) {
      list.appendChild(framePickerOption(
        detail, entry, entry.id === selected.id, pickerURLs,
        (id) => {
          selected = selectedFrameEntry(entries, id);
          state.frameSelections[scope] = selected.id;
          picker.open = false;
          state.framePickerOpen[scope] = false;
          renderSelected();
          summary.focus();
        },
      ));
    }
  };
  if (picker.open) buildOptions();
  picker.addEventListener("toggle", () => {
    state.framePickerOpen[scope] = picker.open;
    if (picker.open) buildOptions();
  });
  renderSelected();
  return {
    node: wrap,
    dispose: () => {
      releaseTrackedURLs(detailURLs);
      releaseTrackedURLs(pickerURLs);
    },
  };
}

/* 关键帧网格：长视频分段用小缩略图按钮，其他结果保持现有网格。 */
function kfGrid(detail, names, pathPrefix, altPrefix, options = {}) {
  const grid = el("div", "kf-grid");
  if (options.compact) grid.classList.add("kf-grid-compact");
  const hasAuthoritativePaths = Object.prototype.hasOwnProperty.call(options, "paths");
  if (hasAuthoritativePaths && (
    !Array.isArray(options.paths) || options.paths.length !== names.length
  )) return grid;
  for (let offset = 0; offset < names.length; offset += 1) {
    const name = names[offset];
    const path = hasAuthoritativePaths
      ? options.paths[offset]
      : pathPrefix + "/" + name;
    const fig = el("figure", "kf-card shimmer");
    const img = el("img");
    img.alt = altPrefix + name;
    let button = null;
    if (options.expandable) {
      button = el("button", "kf-expand-button");
      button.type = "button";
      button.disabled = true;
      setDisclosureState(button, null, false, keyframeDisclosureLabels(img.alt));
      button.appendChild(img);
      button.addEventListener("click", () => openLightbox(img.src, img.alt, button));
      fig.appendChild(button);
    } else {
      fig.appendChild(img);
    }
    grid.appendChild(fig);
    apiBlobURL("/api/conversations/" + detail.id + "/files/" + encodedMediaPath(path))
      .then((url) => {
        if (fig.isConnected === false) {
          releaseTrackedURL(url);
          return;
        }
        if (options.onURL) options.onURL(url);
        img.src = url;
        img.addEventListener("load", () => {
          fig.classList.remove("shimmer");
          fig.classList.add("is-loaded");
        }, { once: true });
        if (button) button.disabled = false;
        else img.addEventListener("click", () => openLightbox(img.src, img.alt));
      })
      .catch(() => {
        fig.classList.remove("shimmer");
        fig.appendChild(el("div", "kf-err", "加载失败"));
      });
  }
  return grid;
}

let disclosureSeq = 0;

function createDisclosure(labels, buildContent, options = {}) {
  const wrap = el("div", options.wrapClass || "disclosure");
  const button = el("button", options.buttonClass || "disclosure-toggle");
  button.type = "button";
  const panel = el("div", options.panelClass || "disclosure-panel");
  panel.id = (options.idPrefix || "disclosure") + "-" + (++disclosureSeq);
  button.setAttribute("aria-controls", panel.id);
  let rendered = false;
  const ensureContent = () => {
    if (rendered) return;
    panel.appendChild(buildContent());
    rendered = true;
  };
  const initialExpanded = options.expanded === true;
  if (initialExpanded) ensureContent();
  setDisclosureState(button, panel, initialExpanded, labels);
  const toggle = () => {
    const expanded = panel.hidden;
    if (expanded) ensureContent();
    setDisclosureState(button, panel, expanded, labels);
    if (!expanded && rendered && options.onDispose) {
      options.onDispose(panel);
      panel.textContent = "";
      rendered = false;
    }
    if (options.onChange) options.onChange(expanded);
  };
  button.addEventListener("click", () => {
    if (options.onBeforeChange) {
      const allowed = options.onBeforeChange();
      if (allowed && typeof allowed.then === "function") {
        allowed.then((resolved) => { if (resolved !== false) toggle(); });
      } else if (allowed !== false) {
        toggle();
      }
      return;
    }
    toggle();
  });
  wrap.appendChild(button);
  wrap.appendChild(panel);
  return wrap;
}

function dialogueText(lines) {
  if (Array.isArray(lines)) {
    return lines.map((line) => typeof line === "string" ? line : String(line && line.text || ""))
      .filter(Boolean).join("\n");
  }
  return normalizeDialogueLines(lines).map((line) => line.text).join("\n");
}

function imagePromptEditable(detail, segmentIndex) {
  const generationStarted = !!(detail.generation && detail.generation.status);
  const postprocessStarted = !!(detail.postprocess && detail.postprocess.status);
  const capabilities = detail.postprocess_capabilities;
  return Number.isInteger(segmentIndex) && segmentIndex >= 0
    && !!capabilities && capabilities.optimize_image === true
    && canOperate(detail) && !generationStarted && !postprocessStarted;
}

function readOnlyImageFramePromptText(value) {
  if (!Array.isArray(value) || value.length !== 9) return null;
  const seen = new Set();
  const sections = [];
  for (const item of value) {
    const name = item && item.frame_name;
    const text = item && item.text;
    const defaultText = item && item.default_text;
    const digest = item && item.sha256;
    if (
      !safeMediaBasename(name) || seen.has(name)
      || typeof text !== "string" || !text.trim()
      || typeof defaultText !== "string" || !defaultText.trim()
      || typeof digest !== "string" || !/^[0-9a-f]{64}$/.test(digest)
    ) return null;
    seen.add(name);
    sections.push(name + "\n" + text);
  }
  return sections.join("\n\n");
}

function promptWorkspace(detail, segment = null, disposeHooks = null) {
  const segmentIndex = promptSegmentIndex(segment);
  const scope = promptScopeKey(detail.id, segmentIndex);
  const isLong = Array.isArray(detail.segments) && detail.segments.length > 0;
  const generationText = String(segment ? segment.prompt || "" : detail.source_prompt || detail.prompt || "");
  const lines = segment ? segment.lines : detail.dialogue;
  const imagePrompt = segment ? segment.image_optimization_prompt : detail.image_optimization_prompt;
  const wrap = el("div", "prompt-workspace");
  const tabs = el("div", "prompt-workspace-tabs");
  tabs.setAttribute("role", "tablist");
  const panel = el("div", "prompt-workspace-panel");
  panel.id = "prompt-workspace-" + (++disclosureSeq);
  panel.setAttribute("role", "tabpanel");
  const modes = promptWorkspaceModes();
  const buttons = {};
  let currentMode = state.promptWorkspaceMode[scope] || null;
  let disposeMode = null;
  if (Array.isArray(disposeHooks)) {
    disposeHooks.push(() => {
      if (disposeMode) disposeMode();
      disposeMode = null;
    });
  }

  const renderMode = () => {
    if (disposeMode) disposeMode();
    disposeMode = null;
    panel.textContent = "";
    panel.hidden = currentMode === null;
    for (const [mode, label] of modes) {
      const selected = mode === currentMode;
      buttons[mode].classList.toggle("is-active", selected);
      buttons[mode].setAttribute("aria-selected", String(selected));
      buttons[mode].textContent = selected ? label.replace("展开", "收起") : label;
    }
    if (currentMode === null) return;
    if (currentMode === "dialogue") {
      panel.appendChild(promptCard(dialogueText(lines) || "暂无段台词"));
      return;
    }
    if (currentMode === "generation") {
      const editable = !isLong && sourcePromptEditable(detail);
      const inspector = frameInspector(detail, segment, {
        context: "generation-prompt",
        mode: "generation",
        showPrompt: !editable,
      });
      if (frameViewerEntries(detail, segment).length) {
        panel.appendChild(inspector.node);
        disposeMode = inspector.dispose;
      } else if (!editable) {
        panel.appendChild(promptCard(generationText));
      }
      if (!editable) return;
      panel.appendChild(editablePromptCard({
        text: generationText,
        defaultText: generationText,
        ariaLabel: "修改源提示词",
        saveLabel: "保存提示词",
        showRestore: false,
        onSave: async (text) => {
          let payload;
          try {
            payload = await apiJSON("/api/conversations/" + encodeURIComponent(detail.id) + "/prompt", {
              method: "PATCH",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({confirm: true, expected_sha256: detail.source_prompt_sha256, prompt: text}),
            });
          } catch (error) {
            if (!error || error.code !== "prompt_changed") throw error;
            const latest = await apiJSON("/api/conversations/" + encodeURIComponent(detail.id));
            if (!latest || typeof latest.source_prompt !== "string"
                || !/^[0-9a-f]{64}$/.test(String(latest.source_prompt_sha256 || ""))) {
              throw new Error("最新提示词详情校验失败，请刷新页面后重试");
            }
            Object.assign(detail, latest);
            const conflict = new Error("提示词已在其他页面更新，已加载最新版本，请重新编辑");
            conflict.latestText = latest.source_prompt;
            throw conflict;
          }
          if (!payload || typeof payload.prompt !== "string"
              || typeof payload.final_prompt !== "string"
              || !/^[0-9a-f]{64}$/.test(String(payload.sha256 || ""))) {
            throw new Error("源提示词保存响应校验失败");
          }
          detail.source_prompt = payload.prompt;
          detail.source_prompt_sha256 = payload.sha256;
          detail.prompt = payload.final_prompt;
          return payload.prompt;
        },
      }));
      return;
    }
    const inspector = frameInspector(detail, segment, {
      context: "image-optimization",
      mode: "image",
      showPrompt: true,
    });
    const entries = frameViewerEntries(detail, segment);
    const hasFramePrompt = entries.some((entry) => !!entry.imagePrompt);
    if (entries.length) {
      panel.appendChild(inspector.node);
      disposeMode = inspector.dispose;
      if (hasFramePrompt) return;
    }
    if (!imagePrompt || typeof imagePrompt.text !== "string") {
      panel.appendChild(el("p", "prompt-unavailable", "当前会话没有可编辑的图片优化提示词"));
      return;
    }
    if (!imagePromptEditable(detail, segmentIndex)) {
      panel.appendChild(promptCard(imagePrompt.text));
      return;
    }
    let draft = state.promptDraft;
    if (!draft || draft.conversationId !== detail.id || draft.segmentIndex !== segmentIndex) {
      draft = createImagePromptDraft(detail.id, segmentIndex, imagePrompt);
      state.promptDraft = draft;
    } else {
      mergeImagePromptDraft(draft, imagePrompt);
    }
    const card = editablePromptCard({
      text: draft.text,
      defaultText: draft.defaultText,
      ariaLabel: "修改图片优化提示词",
      saveLabel: "保存图片优化",
      showRestore: true,
      onInput: (text) => {
        draft.text = text;
        draft.dirty = text !== draft.savedText;
      },
      onRestore: () => restoreImagePromptDefault(draft).text,
      onSave: async (text) => {
        draft.text = text;
        const payload = await saveImageOptimizationPrompt(detail, segmentIndex, draft);
        const saved = payload.image_optimization_prompt || payload;
        if (!saved || typeof saved.text !== "string" || !/^[0-9a-f]{64}$/.test(String(saved.sha256 || ""))) {
          throw new Error("图片优化提示词保存响应校验失败");
        }
        draft.text = saved.text;
        draft.savedText = saved.text;
        draft.defaultText = String(saved.default_text || draft.defaultText);
        draft.sha256 = saved.sha256;
        draft.dirty = false;
        if (segment) segment.image_optimization_prompt = saved;
        else detail.image_optimization_prompt = saved;
        return saved.text;
      },
    });
    draft.save = card.save;
    panel.appendChild(card.node);
  };

  for (const [mode, label] of modes) {
    const button = el("button", "segment-prompt-toggle", label);
    button.type = "button";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", panel.id);
    button.addEventListener("click", async () => {
      if (!await guardDirtyPrompt()) return;
      currentMode = mode === currentMode ? null : mode;
      state.promptWorkspaceMode[scope] = currentMode;
      renderMode();
    });
    buttons[mode] = button;
    tabs.appendChild(button);
  }
  wrap.appendChild(tabs);
  wrap.appendChild(panel);
  renderMode();
  return wrap;
}

function editablePromptCard(options) {
  const node = el("div", "prompt-card");
  const head = el("div", "prompt-head");
  head.appendChild(el("span", "prompt-hint", options.ariaLabel));
  const copy = el("button", "copy-btn", "复制");
  copy.type = "button";
  head.appendChild(copy);
  node.appendChild(head);
  const textarea = el("textarea", "dialogue-textarea prompt-editor");
  textarea.rows = 14;
  textarea.value = options.text;
  textarea.setAttribute("aria-label", options.ariaLabel);
  node.appendChild(textarea);
  const error = el("p", "form-error");
  error.hidden = true;
  node.appendChild(error);
  const actions = el("div", "final-row prompt-edit-actions");
  const saveButton = el("button", "btn btn-primary", options.saveLabel);
  saveButton.type = "button";
  actions.appendChild(saveButton);
  if (options.showRestore) {
    const restore = el("button", "btn", "恢复默认");
    restore.type = "button";
    restore.addEventListener("click", () => {
      textarea.value = options.onRestore();
      if (options.onInput) options.onInput(textarea.value);
    });
    actions.appendChild(restore);
  }
  node.appendChild(actions);
  textarea.addEventListener("input", () => options.onInput && options.onInput(textarea.value));
  copy.addEventListener("click", () => copyText(textarea.value));
  const save = async () => {
    saveButton.disabled = true;
    error.hidden = true;
    try {
      textarea.value = await options.onSave(textarea.value);
      return true;
    } catch (err) {
      if (typeof err.latestText === "string") textarea.value = err.latestText;
      showActionError(err, error, [saveButton]);
      return false;
    } finally {
      saveButton.disabled = false;
    }
  };
  saveButton.addEventListener("click", save);
  return options.showRestore ? {node, save} : node;
}

function sourcePromptEditable(detail) {
  return canOperate(detail)
    && (!detail.generation || detail.generation.status === null)
    && typeof detail.source_prompt_sha256 === "string"
    && /^[0-9a-f]{64}$/.test(detail.source_prompt_sha256);
}

/* prompt 卡片（复制按钮 + 全文；源提示词、IR、单段/多段共用） */
function promptCard(text, actions = [], hint = "用于视频生成") {
  const card = el("div", "prompt-card");
  const head = el("div", "prompt-head");
  head.appendChild(el("span", "prompt-hint", hint));
  for (const action of actions) head.appendChild(action);
  const output = el("pre", "prompt-text", text);
  const copyBtn = el("button", "copy-btn");
  copyBtn.type = "button";
  copyBtn.appendChild(icon("i-copy"));
  const copyLabel = el("span", null, "复制");
  copyBtn.appendChild(copyLabel);
  copyBtn.addEventListener("click", async () => {
    const ok = await copyText(output.textContent);
    copyBtn.classList.add("copied");
    copyLabel.textContent = ok ? "已复制" : "复制失败";
    setTimeout(() => {
      copyBtn.classList.remove("copied");
      copyLabel.textContent = "复制";
    }, 1600);
  });
  head.appendChild(copyBtn);
  card.appendChild(head);
  card.appendChild(output);
  return card;
}

/* 单段模式：关键帧区；「优化后」与失败提示统一走聊天消息（renderPpChat） */
function keyframesSection(detail) {
  const sec = el("section", "res-section");
  const names = Array.isArray(detail.keyframes) ? detail.keyframes : [];
  const h = el("h3", "res-h3", "关键帧");
  h.appendChild(el("span", "res-count", names.length + " 张"));
  sec.appendChild(h);
  sec.appendChild(frameInspector(detail, null, {
    context: "source-frames",
    mode: "generation",
    showPrompt: false,
  }).node);
  return sec;
}

/* 多段模式：逐段「第 N 段」卡片；帧媒体由提示词工作区的单帧查看器按需展示。 */
function renderSegments(detail) {
  const disposeHooks = arguments[1] || [];
  const frag = document.createDocumentFragment();
  const headSec = el("section", "res-section");
  const h = el("h3", "res-h3", "分段产物");
  h.appendChild(el("span", "res-count", detail.segments.length + " 段"));
  headSec.appendChild(h);
  frag.appendChild(headSec);

  for (const seg of detail.segments) {
    const n = seg.index;
    const card = el("section", "res-section seg-card");
    const sh = el("h3", "res-h3", "第 " + n + " 段");
    sh.appendChild(el("span", "res-count", fmtSec(seg.start_s) + "s – " + fmtSec(seg.end_s) + "s"));
    card.appendChild(sh);
    card.appendChild(promptWorkspace(detail, seg, disposeHooks));
    frag.appendChild(card);
  }
  return frag;
}

function resetSegmentProductsDisclosure(id) {
  delete state.segmentProductsExpanded[id];
}

function segmentProductsDisclosure(detail, buildContent = null) {
  const disposeHooks = [];
  const labels = {
    expand: "展开分段产物",
    collapse: "收起分段产物",
    expandText: "展开分段产物",
    collapseText: "收起分段产物",
  };
  return createDisclosure(labels, buildContent || (() => renderSegments(detail, disposeHooks)), {
    idPrefix: "segment-products",
    wrapClass: "segment-products-disclosure",
    buttonClass: "btn btn-primary segment-products-toggle",
    panelClass: "segment-products-panel",
    expanded: state.segmentProductsExpanded[detail.id] === true,
    onBeforeChange: () => state.promptDraft && state.promptDraft.dirty ? guardDirtyPrompt() : true,
    onChange: (expanded) => { state.segmentProductsExpanded[detail.id] = expanded; },
    onDispose: (panel) => {
      if (activeLightboxDisclosure && panel.contains(activeLightboxDisclosure)) {
        closeLightbox({ restoreFocus: false });
      }
      for (const dispose of disposeHooks.splice(0)) dispose();
    },
  });
}

/* 关键帧放大查看：点击开、点任意处或 Esc 关 */
let lightboxEl = null;
let activeLightboxDisclosure = null;
let lightboxReturnFocus = null;

function openLightbox(src, alt, trigger = null) {
  if (!lightboxEl) {
    lightboxEl = el("div", "lightbox");
    lightboxEl.setAttribute("role", "dialog");
    lightboxEl.setAttribute("aria-label", "查看大图");
    lightboxEl.setAttribute("aria-modal", "true");
    lightboxEl.hidden = true;
    const close = el("button", "lightbox-close", "×");
    close.type = "button";
    close.setAttribute("aria-label", "关闭大图");
    close.addEventListener("click", (event) => {
      event.stopPropagation();
      closeLightbox();
    });
    lightboxEl.appendChild(close);
    lightboxEl.appendChild(el("img"));
    lightboxEl.addEventListener("click", closeLightbox);
    document.body.appendChild(lightboxEl);
  }
  if (activeLightboxDisclosure) {
    setDisclosureState(
      activeLightboxDisclosure.trigger,
      null,
      false,
      activeLightboxDisclosure.labels,
    );
  }
  activeLightboxDisclosure = trigger
    ? { trigger, labels: keyframeDisclosureLabels(alt || "关键帧") }
    : null;
  if (activeLightboxDisclosure) {
    setDisclosureState(trigger, null, true, activeLightboxDisclosure.labels);
  }
  const focusOrigin = trigger || document.activeElement;
  lightboxReturnFocus = focusOrigin && typeof focusOrigin.focus === "function"
    ? focusOrigin : null;
  const img = lightboxEl.querySelector("img");
  img.setAttribute("src", src);
  img.setAttribute("alt", alt || "");
  lightboxEl.hidden = false;
  lightboxEl.classList.add("is-open");
  document.addEventListener("keydown", onLightboxKey);
  lightboxEl.querySelector(".lightbox-close").focus();
}

function closeLightbox({ restoreFocus = true } = {}) {
  if (!lightboxEl || lightboxEl.hidden) return;
  lightboxEl.classList.remove("is-open");
  lightboxEl.hidden = true;
  document.removeEventListener("keydown", onLightboxKey);
  const img = lightboxEl.querySelector("img");
  img.removeAttribute("src");
  img.removeAttribute("alt");
  if (activeLightboxDisclosure) {
    const disclosure = activeLightboxDisclosure;
    activeLightboxDisclosure = null;
    setDisclosureState(disclosure.trigger, null, false, disclosure.labels);
  }
  const focusTarget = lightboxReturnFocus;
  lightboxReturnFocus = null;
  if (restoreFocus && focusTarget && focusTarget.isConnected !== false) {
    focusTarget.focus();
  } else {
    const active = document.activeElement;
    if (active && lightboxEl.contains(active) && typeof active.blur === "function") active.blur();
  }
}

function onLightboxKey(e) {
  if (e.key === "Escape") {
    closeLightbox();
  } else if (e.key === "Tab" && lightboxEl && !lightboxEl.hidden) {
    e.preventDefault();
    lightboxEl.querySelector(".lightbox-close").focus();
  }
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_) {
    // 降级：隐藏 textarea + execCommand
    try {
      const ta = el("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch (e) {
      return false;
    }
  }
}

/* ===== 后处理弹窗 ===== */
function ppCheckedOptions() {
  const options = { remove_subtitle: false, remove_brand: false, optimize_image: false };
  document.querySelectorAll('#pp-form input[name="opt"]:checked').forEach((c) => {
    options[c.value] = true;
  });
  return options;
}

function updatePpConfirm() {
  $("pp-confirm").disabled = !Object.values(ppCheckedOptions()).some(Boolean);
}

function openPostprocessModal(detail) {
  if (!canOperate(detail)) return;
  state.ppDetail = detail;
  const last = detail.postprocess && detail.postprocess.options;
  const capabilities = detail.postprocess_capabilities || {
    remove_subtitle: detail.postprocess_enabled === true,
    remove_brand: detail.postprocess_enabled === true,
    optimize_image: false,
  };
  document.querySelectorAll('#pp-form input[name="opt"]').forEach((c) => {
    c.disabled = !capabilities[c.value];
    c.closest("label").hidden = c.disabled;
    c.checked = !c.disabled && (last ? last[c.value] === true : c.value !== "optimize_image");
  });
  $("pp-lock-hint").hidden = !last;
  $("pp-error").hidden = true;
  updatePpConfirm();
  $("pp-dialog").showModal();
}

async function requestOpenPostprocessModal(detail) {
  if (!await guardDirtyPrompt()) return;
  openPostprocessModal(detail);
}

function closePostprocessModal() {
  $("pp-dialog").close();
  state.ppDetail = null;
}

async function submitPostprocess(event) {
  event.preventDefault();
  if (!await guardDirtyPrompt()) return;
  const detail = state.ppDetail;
  if (!detail || !canOperate(detail)) {
    closePostprocessModal();
    return;
  }
  const btn = $("pp-confirm");
  const errEl = $("pp-error");
  btn.disabled = true;
  errEl.hidden = true;
  try {
    await apiJSON("/api/conversations/" + detail.id + "/postprocess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ options: ppCheckedOptions(), confirm: true }),
    });
    closePostprocessModal();
    // 刷新进入「处理中…」；loadDetail 见到 postprocess.status==running 会沿用 2s 轮询到终态
    loadDetail(detail.id, true);
  } catch (err) {
    if (handleAuthError(err)) return;
    try {
      const inputs = Array.from(document.querySelectorAll('#pp-form input[name="opt"]'));
      const latest = await recoverLockedPostprocess(
        err,
        () => apiJSON("/api/conversations/" + encodeURIComponent(detail.id)),
        inputs,
        $("pp-lock-hint"),
        errEl,
      );
      if (latest) {
        state.ppDetail = latest;
        if (state.currentId === detail.id) state.detail = latest;
      } else {
        showActionError(err, errEl, [btn]);
      }
    } catch (recoveryError) {
      showActionError(recoveryError, errEl, [btn]);
    }
    updatePpConfirm();
  }
}

/* ===== 后处理聊天消息 ===== */
/* 选项标签单一来源：弹窗勾选项 DOM（index.html），避免与 HTML 重复硬编码 */
let ppOptionLabelsCache = null;

function ppOptionLabels() {
  if (!ppOptionLabelsCache) {
    ppOptionLabelsCache = {};
    document.querySelectorAll('#pp-form input[name="opt"]').forEach((c) => {
      ppOptionLabelsCache[c.value] = c.nextElementSibling ? c.nextElementSibling.textContent : c.value;
    });
  }
  return ppOptionLabelsCache;
}

function ppCheckedLabels(options) {
  const opts = options || {};
  const labels = ppOptionLabels();
  return Object.keys(labels)
    .filter((k) => opts[k] === true)
    .map((k) => labels[k]);
}

/* 用户消息：后处理请求摘要（勾选项 chips） */
function renderPpUserBubble(pp) {
  const row = el("div", "msg-row msg-user");
  const bubble = el("div", "bubble-user pp-bubble");
  bubble.appendChild(el("span", "pp-bubble-label", "后处理"));
  for (const label of ppCheckedLabels(pp.options)) {
    bubble.appendChild(el("span", "pp-chip", label));
  }
  row.appendChild(bubble);
  return row;
}

/* 「优化后」结果区（done）：多段逐段分组；单段按一个虚拟段统一处理 */
function ppFramesSection(detail, frames, disposeHooks = null) {
  const wrap = el("div");
  const published = authoritativePostprocessFrameGroups(detail, frames);
  if (!published.length) return wrap;
  const inspector = frameInspector(detail, null, {
    context: "optimized-frames",
    mode: "image",
    showPrompt: true,
  });
  wrap.appendChild(inspector.node);
  if (Array.isArray(disposeHooks)) disposeHooks.push(inspector.dispose);
  return wrap;
}

function ppResultDisclosure(detail, frames) {
  const disposeHooks = [];
  const labels = {
    expand: "展开优化后素材",
    collapse: "收起优化后素材",
    expandText: "展开优化后素材",
    collapseText: "收起优化后素材",
  };
  return createDisclosure(labels, () => {
    const card = el("div", "activity-card");
    card.appendChild(ppFramesSection(detail, frames, disposeHooks));
    return card;
  }, {
    idPrefix: "pp-result",
    wrapClass: "pp-result-disclosure",
    buttonClass: "btn btn-primary pp-result-toggle",
    panelClass: "pp-result-panel",
    expanded: state.ppResultExpanded[detail.id] === true,
    onChange: (expanded) => { state.ppResultExpanded[detail.id] = expanded; },
    onDispose: () => {
      for (const dispose of disposeHooks.splice(0)) dispose();
    },
  });
}

/* 后处理目标帧总数：多段 = 各段 keyframes 之和；单段 = detail.keyframes 长度 */
function ppTotalFrames(detail) {
  const segments = Array.isArray(detail.segments) ? detail.segments : [];
  if (segments.length > 0) {
    return segments.reduce((sum, seg) => sum + (Array.isArray(seg.keyframes) ? seg.keyframes.length : 0), 0);
  }
  return Array.isArray(detail.keyframes) ? detail.keyframes.length : 0;
}

/* 运行中 canonical frames 仍按 manifest-last 发布；真实进度来自后端逐 receipt
   投影的 segment.completed_frames。旧记录没有 segments 时才回退 frames。 */
function ppCompletedFrames(detail) {
  const pp = detail && detail.postprocess;
  const segments = pp && Array.isArray(pp.segments) ? pp.segments : [];
  if (segments.length > 0 && segments.every((segment) =>
    Number.isInteger(segment.completed_frames) && segment.completed_frames >= 0
    && Number.isInteger(segment.total_frames) && segment.total_frames >= 0)) {
    return segments.reduce((sum, segment) =>
      sum + Math.min(segment.completed_frames, segment.total_frames), 0);
  }
  return pp && Array.isArray(pp.frames) ? pp.frames.length : 0;
}

function postprocessReadyForGeneration(detail) {
  const pp = detail && detail.postprocess;
  if (pp === null || pp === undefined) return true;
  if (!pp || typeof pp !== "object" || pp.status !== "done"
      || !Array.isArray(pp.segments) || pp.segments.length === 0) return false;
  const duration = Number(detail && detail.duration_s);
  if (!Number.isFinite(duration) || duration <= 0) return false;
  const longContract = longVideoContract(detail);
  const isLong = longContract.isLong;
  const expectedIndexes = isLong
    ? (Array.isArray(detail.segments) ? detail.segments.map((segment) => segment && segment.index) : [])
    : [0];
  if ((isLong && (!longContract.ready
        || !Number.isInteger(longContract.segmentCount)
        || expectedIndexes.length !== longContract.segmentCount))
      || expectedIndexes.length === 0
      || expectedIndexes.some((index) => !Number.isInteger(index) || (isLong ? index <= 0 : index !== 0))
      || (isLong && expectedIndexes.some((index, position) => index !== position + 1))
      || new Set(expectedIndexes).size !== expectedIndexes.length
      || pp.segments.length !== expectedIndexes.length) return false;
  const actualIndexes = new Set();
  for (const segment of pp.segments) {
    if (!segment || typeof segment !== "object" || segment.status !== "done"
        || !Number.isInteger(segment.index) || !expectedIndexes.includes(segment.index)
        || actualIndexes.has(segment.index)
        || !Number.isInteger(segment.revision) || segment.revision < 1
        || !Number.isInteger(segment.completed_frames) || segment.completed_frames < 0
        || !Number.isInteger(segment.total_frames) || segment.total_frames <= 0
        || segment.completed_frames !== segment.total_frames
        || segment.stage !== "done" || segment.error !== null) return false;
    actualIndexes.add(segment.index);
  }
  return actualIndexes.size === expectedIndexes.length;
}

async function requestGenerationSubmit(detail, body, request = apiJSON) {
  if (!postprocessReadyForGeneration(detail)) {
    throw new Error("素材优化尚未全部完成，不能生成最终视频");
  }
  return request("/api/conversations/" + encodeURIComponent(detail.id) + "/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function postprocessSegmentStatus(status) {
  const labels = new Map([
    ["queued", "等待处理"],
    ["running", "处理中"],
    ["done", "已完成"],
    ["failed", "失败"],
  ]);
  return labels.get(status) || "状态未知";
}

function safePostprocessStage(stage) {
  const labels = new Map([
    ["queued", "等待处理"],
    ["text", "移除文字/字幕"],
    ["brand", "移除常见 Logo/图标"],
    ["seedream", "优化图片质量"],
    ["publishing", "正在发布结果"],
    ["done", "已完成"],
  ]);
  return labels.get(stage) || "处理中";
}

function isPostprocessSubmissionUnknown(segment) {
  return !!segment && segment.error === "submission_unknown";
}

function safePostprocessError(error) {
  const code = error && typeof error === "object" ? error.code : error;
  const labels = new Map([
    ["revision_conflict", "分段状态已更新，请刷新后重试"],
    ["submission_unknown", "提交结果未知，请谨慎确认后再重试"],
    ["timeout", "处理超时，请稍后重试"],
    ["postprocess_failed", "图片处理失败，请重试本段"],
    ["provider_failed", "图片处理失败，请重试本段"],
  ]);
  return labels.get(code) || "本段处理失败，请重试或联系管理员";
}

async function retryPostprocessSegment(detail, segment, request = apiJSON, confirmUnknown, onAccepted) {
  const retryable = segment && segment.status === "failed";
  const duration = Number(detail && detail.duration_s);
  const knownDuration = Number.isFinite(duration) && duration > 0;
  const isLong = knownDuration && longVideoContract(detail).isLong;
  const validIndex = knownDuration && (isLong ? segment && segment.index > 0 : segment && segment.index === 0);
  if (!retryable || !validIndex || !Number.isInteger(segment.index)
      || !Number.isInteger(segment.revision)) return false;
  if (isPostprocessSubmissionUnknown(segment)
      && (!confirmUnknown || confirmUnknown() !== true)) return false;
  if (onAccepted) onAccepted();
  try {
    await request(
      "/api/conversations/" + encodeURIComponent(detail.id) + "/postprocess/segments/" + segment.index + "/retry",
      {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({confirm: true, expected_revision: segment.revision})},
    );
  } catch (error) {
    throw new Error(safePostprocessError(error));
  }
  return true;
}

function renderPostprocessSegments(detail, pp) {
  const wrap = el("div", "pp-segment-statuses");
  for (const segment of Array.isArray(pp.segments) ? pp.segments : []) {
    const row = el("div", "pp-segment-status");
    const segmentLabel = segment.index === 0 ? "当前视频" : "第 " + segment.index + " 段";
    const title = el("div", "pp-segment-title", segmentLabel + " · " + postprocessSegmentStatus(segment.status));
    row.appendChild(title);
    const completed = Number.isInteger(segment.completed_frames) ? segment.completed_frames : 0;
    const total = Number.isInteger(segment.total_frames) ? segment.total_frames : 0;
    row.appendChild(el("p", "ac-sub", `已完成 ${completed}/${total} 帧`));
    if (segment.stage) row.appendChild(el("p", "ac-sub", "阶段：" + safePostprocessStage(segment.stage)));
    if (segment.error) row.appendChild(el("p", "fail-msg", safePostprocessError(segment.error)));
    if (isPostprocessSubmissionUnknown(segment)) {
      row.appendChild(el("p", "pp-billing-warning", "提交结果未知；人工重试可能重复计费，请谨慎确认。"));
    }
    if (segment.status === "failed" && Number.isInteger(segment.revision)) {
      const retry = el("button", "btn btn-ghost pp-segment-retry", "重试本段");
      retry.type = "button";
      retry.addEventListener("click", async () => {
        if (!await guardDirtyPrompt()) return;
        try {
          const retried = await retryPostprocessSegment(
            detail,
            segment,
            apiJSON,
            () => window.confirm("该段上次提交结果未知，重试可能重复计费。确认仍要重试本段吗？"),
            () => { retry.disabled = true; }, // revision CAS + 立即禁用共同阻止双击
          );
          if (!retried) return;
          await loadDetail(detail.id, true);
        } catch (err) {
          retry.disabled = false;
          const message = el("p", "form-error", safePostprocessError(err));
          row.appendChild(message);
        }
      });
      row.appendChild(retry);
    }
    wrap.appendChild(row);
  }
  return wrap;
}

/* 助手消息：running 进行中卡 / done 优化后结果 / failed 错误卡（动态区，轮询期间单独重渲染） */
function renderPpAssistant(detail, pp) {
  const row = el("div", "msg-row");
  row.appendChild(assistantHead(detail.updated_at));
  if (pp.status === "running") {
    const card = el("div", "activity-card");
    card.appendChild(el("p", "ac-title", "正在优化素材…"));
    // 实时进度：n = receipt 投影的逐帧完成数；m = 目标帧总数
    const total = ppTotalFrames(detail);
    if (total > 0) {
      const done = ppCompletedFrames(detail);
      card.appendChild(el("p", "ac-sub", `已完成 ${done}/${total} 帧（每帧约需 1 分钟）`));
    }
    const track = el("div", "progress-track");
    track.appendChild(el("div", "progress-fill"));
    card.appendChild(track);
    card.appendChild(renderPostprocessSegments(detail, pp));
    row.appendChild(card);
  } else if (pp.status === "failed") {
    const card = el("div", "fail-card");
    card.appendChild(icon("i-alert", "ic-danger"));
    const body = el("div");
    body.appendChild(el("p", "fail-title", "后处理失败"));
    body.appendChild(el("p", "fail-msg", safePostprocessError(pp.error)));
    if (Array.isArray(pp.frames) && pp.frames.length) {
      body.appendChild(el("p", "fail-tip", "已成功优化的帧保留"));
    }
    card.appendChild(body);
    if (Array.isArray(pp.segments) && pp.segments.length > 0) {
      card.appendChild(renderPostprocessSegments(detail, pp));
    } else {
      card.appendChild(el("p", "fail-tip", "分段状态不完整，请刷新页面后重试"));
    }
    row.appendChild(card);
  } else if (pp.status === "done") {
    const frames = Array.isArray(pp.frames) ? pp.frames : [];
    if (frames.length) {
      row.appendChild(ppResultDisclosure(detail, frames));
    } else {
      const card = el("div", "activity-card");
      card.appendChild(el("p", "ac-title", "后处理完成"));
      card.appendChild(el("p", "ac-sub", "所有目标帧均已处理"));
      row.appendChild(card);
    }
  }
  return row;
}

/* 后处理入口消息：仅 postprocess 尚未开始且接口开放时，结果区末尾提问。
   “设置素材处理”只打开配置；真正提交发生在弹窗的“开始素材处理”。
   running/done 时不显示——renderPpChat 的进行中卡/结果卡接管。 */
function renderPpAsk(detail) {
  if (!shouldRenderPostprocessAsk(detail)) return null;
  const row = el("div", "msg-row");
  row.appendChild(assistantHead(detail.updated_at));
  const card = el("div", "activity-card pp-ask-card");
  card.appendChild(el("p", "pp-ask-text", "需要处理关键帧素材吗？"));
  card.appendChild(el("p", "ac-sub", "设置选项不会提交；在确认弹窗中点击“开始素材处理”后才会执行。"));
  if (state.ppAskDismissed[detail.id]) {
    card.appendChild(el("p", "pp-ask-ended", "已保持原素材；未提交素材处理任务。"));
  } else {
    const actions = el("div", "pp-ask-actions");
    const defaultChoice = postprocessAskDefault();
    const yes = el("button", defaultChoice === "yes" ? "btn btn-primary pp-ask-btn is-selected" : "btn btn-ghost pp-ask-btn", "设置素材处理");
    yes.type = "button";
    yes.addEventListener("click", () => requestOpenPostprocessModal(detail));
    const no = el("button", defaultChoice === "no" ? "btn btn-primary pp-ask-btn is-selected" : "btn btn-ghost pp-ask-btn", "保持原素材");
    no.type = "button";
    no.addEventListener("click", () => {
      state.ppAskDismissed[detail.id] = true;
      actions.replaceWith(el("p", "pp-ask-ended", "已保持原素材；未提交素材处理任务。"));
    });
    actions.appendChild(yes);
    actions.appendChild(no);
    card.appendChild(actions);
  }
  row.appendChild(card);
  return row;
}

function shouldRenderPostprocessAsk(detail) {
  if (frozenGenerationConfig(detail)) return false;
  const capabilities = detail.postprocess_capabilities || {};
  const enabled = Object.values(capabilities).some((value) => value === true) || detail.postprocess_enabled === true;
  const pp = detail.postprocess || {};
  return enabled && canOperate(detail) && !pp.status;
}

/* postprocess 存在即渲染：用户摘要 + 助手消息（动态区，renderPpDynamic 随轮询更新） */
function renderPpChat(detail) {
  const pp = detail.postprocess || {};
  if (!pp.status || !pp.options) return null;
  const frag = document.createDocumentFragment();
  frag.appendChild(renderPpUserBubble(pp));
  frag.appendChild(renderPpAssistant(detail, pp));
  return frag;
}

function renderDetail(detail) {
  document.querySelector(".composer-dock").classList.toggle(
    "is-dialogue-review-waiting",
    !!(detail.dialogue_review && detail.dialogue_review.status === "waiting"),
  );
  renderOperationHeader(detail);
  renderStable(detail);
  renderPpDynamic(detail);
  renderGenerationDynamic(detail);
}

/* 稳定区：用户气泡 + 结果区 + 后处理入口 + 最终视频；中间留 .pp-dynamic 插槽给后处理聊天 */
function renderStable(detail) {
  clearStream();
  const inner = el("div", "stream-inner");
  inner.appendChild(renderUserBubble(detail));
  const frozenConfig = renderFrozenGenerationConfig(detail);
  if (frozenConfig) inner.appendChild(frozenConfig);
  const dialogueReview = renderDialogueReview(detail);
  if (dialogueReview) inner.appendChild(dialogueReview);
  if (detail.status === "queued" || detail.status === "processing") {
    if (!(detail.dialogue_review && detail.dialogue_review.status === "waiting")) {
      inner.appendChild(renderActivity(detail.status));
    }
  } else if (detail.status === "failed") {
    inner.appendChild(renderFail(detail));
  } else if (detail.status === "done") {
    inner.appendChild(renderResults(detail));
    const ppAsk = renderPpAsk(detail);
    if (ppAsk) inner.appendChild(ppAsk);
    inner.appendChild(el("div", "pp-dynamic"));
    inner.appendChild(el("div", "generation-dynamic"));
  }
  $("stream").appendChild(inner);
}

/* 动态区：后处理聊天（用户摘要 + 状态卡/结果）。running 轮询期间只重渲染本区，
   稳定区的 <video>/<img> 引用不重建，避免每 2s 全量重建导致媒体反复重载闪烁 */
function renderPpDynamic(detail) {
  const slot = document.querySelector(".pp-dynamic");
  if (!slot) return;
  slot.textContent = "";
  const ppChat = renderPpChat(detail);
  if (ppChat) slot.appendChild(ppChat);
}

/* 生成任务进度独立刷新，避免每个分段状态变化都重建原视频和关键帧。 */
function renderGenerationDynamic(detail) {
  const slot = document.querySelector(".generation-dynamic");
  if (!slot) return;
  slot.textContent = "";
  const intermediate = renderIntermediateStages(detail);
  if (intermediate) slot.appendChild(intermediate);
  slot.appendChild(renderFinalSection(detail));
}

/* ===== 会话详情 + 轮询 ===== */
/* 详情状态签名：
   stable 变（状态机/产物内容变化）→ 全量重渲染一次；
   仅 dyn 变（后处理 running 时 frames 逐帧增长）→ 只刷新后处理动态区；
   generation 变 → 只刷新最终视频区，保留原视频 DOM；
   完全不变 → 什么都不做（连 DOM 都不碰，杜绝每 2s 清空重建媒体引发的闪烁）。
   dyn 取逐帧进度和分段阶段：它们在 stable 不变时随轮询增长。 */
function detailSignature(detail) {
  // stable 覆盖稳定区渲染消费的全部字段（未覆盖字段如 title/note 由创建后不变兜底，
  // pp.options 与 status 原子落盘——见审查记录）；dyn 只跟后处理进度（frames 单调增长）。
  const pp = detail.postprocess || null;
  const segments = Array.isArray(detail.segments) ? detail.segments : [];
  const stable = JSON.stringify([
    detail.id,
    detail.title,
    detail.note,
    detail.status,
    detail.read_only === true,
    detail.submit_enabled === true,
    detail.fit_required === true,
    detail.fit_mode,
    detail.aspect_ratio,
    detail.resolution,
    detail.fit_profiles || null,
    detail.duration_s,
    detail.receipt_version,
    detail.source_prompt || null,
    detail.source_prompt_sha256 || null,
    detail.image_optimization_prompt || null,
    detail.postprocess_capabilities || null,
    detail.dialogue || null,
    detail.dialogue_review || null,
    detail.generation_config || null,
    detail.generation_config_sha256 || null,
    detail.skill_milestone || null,
    detail.element_index || detail.material_index || detail.project_index || null,
    pp ? pp.status : "",
    pp && pp.error ? pp.error : "",
    Array.isArray(detail.keyframes) ? detail.keyframes.join(",") : "",
    detail.prompt || "",
    segments.map((seg) => [
      seg.index,
      Array.isArray(seg.keyframes) ? seg.keyframes.join(",") : "",
      seg.prompt || "",
      Array.isArray(seg.lines) ? seg.lines.join("\n") : "",
      seg.image_optimization_prompt || null,
    ]),
    detail.has_video ? 1 : 0,
  ]);
  const dyn = JSON.stringify([
    ppCompletedFrames(detail),
    pp && Array.isArray(pp.segments) ? pp.segments : null,
  ]);
  const generation = JSON.stringify([
    detail.plan_receipt || null,
    Number.isInteger(detail.segment_count) ? detail.segment_count : null,
    detail.generation || null,
    detail.prompt_fusion || null,
    detail.has_video ? 1 : 0,
  ]);
  return { stable, dyn, generation };
}

async function loadDetail(id, silent) {
  const seq = ++state.detailSeq;
  if (!silent) renderSkeleton();
  try {
    const detail = await apiJSON("/api/conversations/" + encodeURIComponent(id));
    if (seq !== state.detailSeq || state.currentId !== id) return; // 已切换会话
    if (detail.generation && detail.generation.status !== null) {
      state.generationSubmitting[id] = false;
    }
    state.detail = detail;
    if (syncConversationDetail(state.conversations, detail)) {
      renderList();
      const summary = state.conversations.find((item) => item.id === detail.id);
      if (summary) void loadHistoryThumbnail(summary).then(() => {
        if (state.token) renderList();
      });
    }
    renderOperationHeader(detail);
    const sig = detailSignature(detail);
    if (!silent) {
      // 手动切换 / 首次进入：全量渲染照旧
      renderDetail(detail);
    } else if (!state.detailSig || sig.stable !== state.detailSig.stable) {
      // 轮询：stable 变（queued→processing→done、后处理进入/离开 running、产物变化）→ 全量一次
      renderDetail(detail);
    } else {
      if (sig.dyn !== state.detailSig.dyn) {
        // 轮询：仅后处理进度增长 → 只刷动态区，稳定区 <video>/<img> 引用不重建
        renderPpDynamic(detail);
      }
      if (sig.generation !== state.detailSig.generation) {
        // 分段生成 / 拼接进度变化只刷新最终视频卡片，不重载源视频。
        renderGenerationDynamic(detail);
      }
    }
    // 签名完全不变 → 不碰 DOM（根治轮询闪烁的关键）
    state.detailSig = sig;
    if (shouldPollDetail(detail) || state.generationSubmitting[id]) {
      startPolling(id);
    } else {
      stopPolling();
      refreshList(false); // 终态：同步侧栏徽章（轻量更新，不动 stream）
    }
  } catch (err) {
    if (handleAuthError(err)) return;
    if (seq !== state.detailSeq) return;
    state.detailSig = null;
    renderStreamError("会话加载失败：" + err.message);
    if (silent && state.currentId === id) startPolling(id);
    else stopPolling();
  }
}

function shouldPollDetail(detail) {
  if (!detail || typeof detail !== "object") return false;
  const ppStatus = detail.postprocess && detail.postprocess.status;
  const fusionStatus = detail.prompt_fusion && detail.prompt_fusion.status;
  const generationStatus = detail.generation && detail.generation.status;
  const navigationStatus = detail.navigation_status;
  const automaticPending = !!frozenGenerationConfig(detail)
    && detail.status === "done" && detail.has_video !== true && !generationStatus;
  return detail.status === "queued"
    || detail.status === "processing"
    || ppStatus === "queued"
    || ppStatus === "running"
    || fusionStatus === "pending"
    || fusionStatus === "running"
    || generationStatus === "queued"
    || generationStatus === "running"
    || automaticPending
    || ["analysis_queued", "analysis_processing", "generation_queued",
      "generation_running", "postprocessing"].includes(navigationStatus);
}

async function selectConversation(id) {
  if (state.uploading) return; // 上传中不切换，避免打断
  if (state.currentId !== id && !await guardDirtyPrompt()) return;
  delete state.ppResultExpanded[id];
  resetSegmentProductsDisclosure(id);
  state.currentId = id;
  const conv = state.conversations.find((c) => c.id === id);
  $("main-title").textContent = (conv && conv.title) || "会话";
  renderList();
  loadDetail(id, false);
}

function startPolling(id) {
  stopPolling();
  const token = state.pollToken;
  const isCurrent = () => state.pollToken === token && state.currentId === id;
  const schedule = () => {
    if (!isCurrent()) return;
    state.pollTimer = setTimeout(async () => {
      state.pollTimer = null;
      await runSingleFlightPollCycle(
        isCurrent,
        () => loadDetail(id, true),
        schedule,
      );
    }, POLL_MS);
  };
  schedule();
}

function stopPolling() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
  state.pollToken += 1;
}

/* ===== Composer ===== */
function setComposerError(msg) {
  const box = $("composer-error");
  if (msg) {
    box.textContent = msg;
    box.hidden = false;
  } else {
    box.hidden = true;
  }
}

function resetGenerationConfigDisclosure() {
  $("generation-config").open = true;
}

function selectedGenerationConfig() {
  return {
    optimize_image: $("generation-optimize-image").checked,
    remove_subtitle: $("generation-remove-subtitle").checked,
    remove_watermark: $("generation-remove-watermark").checked,
  };
}

function applyGenerationConfigDefaults(defaults) {
  if (!validGenerationConfig(defaults)) return;
  $("generation-optimize-image").checked = defaults.optimize_image;
  $("generation-remove-subtitle").checked = defaults.remove_subtitle;
  $("generation-remove-watermark").checked = defaults.remove_watermark;
}

function renderGenerationConfigControl() {
  const capability = state.generationConfigCapability;
  const loaded = state.generationConfigCapabilityLoaded;
  $("generation-config-fields").disabled = state.uploading || !capability;
  if (!loaded) {
    $("generation-config-summary-text").textContent = "正在读取服务器配置能力…";
    $("generation-config-status").textContent = "配置能力确认前不会发送额外字段。";
    return;
  }
  if (!capability) {
    $("generation-config-summary-text").textContent = "使用服务器默认配置";
    $("generation-config-status").textContent = "当前服务器未声明可选配置；页面不会发送未知字段。";
    return;
  }
  $("generation-config-summary-text").textContent = generationConfigLabels(
    selectedGenerationConfig(),
  ).join(" · ");
  $("generation-config-status").textContent = "提交一次后将按此配置自动运行至成片，中途无需确认。";
}

function dialogueReviewPolicy() {
  const checked = document.querySelector('input[name="dialogue-review-policy"]:checked');
  return checked ? checked.value : "auto_continue";
}

function selectDialogueReviewPolicy(policy) {
  const value = DIALOGUE_REVIEW_POLICIES.includes(policy) ? policy : "auto_continue";
  const radio = document.querySelector(`input[name="dialogue-review-policy"][value="${value}"]`);
  if (radio) radio.checked = true;
}

function renderDialogueReviewPolicyControl() {
  const fieldset = $("dialogue-review-fields");
  const loaded = state.dialogueReviewCapabilityLoaded;
  const capability = state.dialogueReviewCapability;
  const automatic = dialogueMode() === "auto";
  if (!automatic) selectDialogueReviewPolicy("auto_continue");
  fieldset.disabled = state.uploading || !capability || !automatic;
  if (!automatic) {
    $("dialogue-review-policy-status").textContent = "无台词模式不会运行语音识别，将直接继续后续流程。";
  } else if (!loaded) {
    $("dialogue-review-policy-status").textContent = "能力确认前使用自动继续，且不发送额外字段。";
  } else if (!capability) {
    selectDialogueReviewPolicy("auto_continue");
    $("dialogue-review-policy-status").textContent = "当前服务器未声明校对能力；使用服务器默认配置，不发送未知字段。";
  } else if (dialogueReviewPolicy() === "review_required") {
    $("dialogue-review-policy-status").textContent = "识别完成后会持久等待；采用台词稿后恢复同一任务。";
  } else {
    $("dialogue-review-policy-status").textContent = "默认无人值守：机器稿自动冻结并继续至成片。";
  }
}

async function loadGenerationConfigCapability() {
  const token = state.token;
  state.generationConfigCapability = null;
  state.generationConfigCapabilityLoaded = false;
  state.dialogueReviewCapability = null;
  state.dialogueReviewCapabilityLoaded = false;
  renderGenerationConfigControl();
  renderDialogueReviewPolicyControl();
  try {
    const payload = await apiJSON("/api/capabilities");
    if (token !== state.token) return;
    state.generationConfigCapability = normalizeGenerationConfigCapability(payload);
    state.dialogueReviewCapability = normalizeDialogueReviewCapability(payload);
    if (state.generationConfigCapability) {
      applyGenerationConfigDefaults(state.generationConfigCapability.defaults);
    }
    if (state.dialogueReviewCapability) {
      selectDialogueReviewPolicy(state.dialogueReviewCapability.default);
    }
  } catch (error) {
    if (handleAuthError(error)) return;
    if (token !== state.token) return;
    state.generationConfigCapability = null;
    state.dialogueReviewCapability = null;
  } finally {
    if (token === state.token) {
      state.generationConfigCapabilityLoaded = true;
      state.dialogueReviewCapabilityLoaded = true;
      renderGenerationConfigControl();
      renderDialogueReviewPolicyControl();
    }
  }
}

function sourceMode() {
  const checked = document.querySelector('input[name="source-mode"]:checked');
  return checked ? checked.value : "link"; // 与 HTML 默认 checked 一致：默认视频链接
}

function voiceMode() {
  const checked = document.querySelector('input[name="voice-mode"]:checked');
  return checked ? checked.value : "keep";
}

function dialogueMode() {
  const checked = document.querySelector('input[name="dialogue-mode"]:checked');
  return checked ? checked.value : "auto";
}

// 口播转换切换：翻译模式才显示语言填空（必填）
function setVoiceMode() {
  const translate = voiceMode() === "translate";
  $("lang-input").hidden = !translate;
  $("lang-input").required = translate;
  state.clientRequestId = newRequestId(); // 内容变 = 新意图 = 新键
  setComposerError(null);
  updateSendBtn();
}

function setDialogueMode() {
  renderDialogueReviewPolicyControl();
  state.clientRequestId = newRequestId();
  setComposerError(null);
  updateSendBtn();
}

// 来源二选一：同一时刻只存在一种输入，切换即清空另一边
function setSourceMode(mode) {
  const isUpload = mode === "upload";
  if (isUpload) {
    $("url-input").value = "";
  } else {
    clearFile();
  }
  state.clientRequestId = newRequestId(); // 内容变 = 新意图 = 新键
  $("attach-btn").hidden = !isUpload;
  $("file-hint").hidden = !isUpload;
  $("url-row").hidden = isUpload;
  $("url-input").required = !isUpload;
  setComposerError(null);
  updateSendBtn();
}

function updateSendBtn() {
  const ready = sourceMode() === "upload" ? !!state.file : !!$("url-input").value.trim();
  // 翻译模式必须填目标语言
  const langReady = voiceMode() !== "translate" || !!$("lang-input").value.trim();
  $("send-btn").disabled = state.uploading || !ready || !langReady;
}

function isVideoFile(file) {
  if (!file) return false;
  if (file.type && file.type.startsWith("video/")) return true;
  return /\.(mp4|mov|webm)$/i.test(file.name || "");
}

function pickFile(file) {
  setComposerError(null);
  if (!file) return;
  if (!isVideoFile(file)) {
    setComposerError("仅支持视频文件（如 MP4 / MOV），请重新选择");
    return;
  }
  state.file = file;
  state.clientRequestId = newRequestId(); // 内容变 = 新意图 = 新键
  $("file-chip-name").textContent = file.name;
  $("file-chip-size").textContent = fmtBytes(file.size);
  $("file-chip").hidden = false;
  updateSendBtn();
}

function clearFile() {
  state.file = null;
  $("file-input").value = "";
  $("file-chip").hidden = true;
  updateSendBtn();
}

function setUploading(on) {
  state.uploading = on;
  $("composer-toggle").disabled = on;
  $("attach-btn").disabled = on;
  $("note-input").disabled = on;
  $("url-input").disabled = on;
  $("file-remove").disabled = on;
  $("lang-input").disabled = on;
  document.querySelectorAll('input[name="source-mode"]').forEach((r) => {
    r.disabled = on;
  });
  document.querySelectorAll('input[name="voice-mode"]').forEach((r) => {
    r.disabled = on;
  });
  document.querySelectorAll('input[name="dialogue-mode"]').forEach((r) => {
    r.disabled = on;
  });
  renderGenerationConfigControl();
  renderDialogueReviewPolicyControl();
  updateSendBtn();
  if (!on) $("upload-progress").hidden = true;
}

function uploadConversation({
  file, url, note, requestId, voiceMode: mode, targetLanguage, dialogue,
  generationConfig, generationConfigCapability,
  dialogueReviewPolicy: reviewPolicy, dialogueReviewCapability,
}, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/conversations");
    xhr.setRequestHeader("Authorization", "Bearer " + state.token);
    xhr.responseType = "json";
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && e.total > 0) onProgress(e.loaded / e.total);
    });
    xhr.addEventListener("load", () => {
      // 200 = 幂等命中返回既有会话；201 = 新建成功
      if (xhr.status === 200 || xhr.status === 201) {
        resolve(xhr.response);
      } else if (xhr.status === 401) {
        reject(new AuthError("口令已失效"));
      } else if (xhr.status === 429) {
        // 429 有两种来源：排队满 / IP 限流，按 detail 文案区分
        const data = xhr.response;
        const detail = data && data.detail ? String(data.detail) : "";
        reject(new Error(detail.indexOf("queued") >= 0
          ? "当前排队任务较多，请稍后再试"
          : "操作过于频繁，请稍后再试"));
      } else {
        const data = xhr.response;
        reject(apiErrorFromPayload(
          data,
          "上传失败（" + xhr.status + "），请稍后重试"
        ));
      }
    });
    xhr.addEventListener("error", () => reject(new Error("网络异常，上传未完成，请重试")));
    xhr.addEventListener("abort", () => reject(new Error("上传已中断，请重试")));
    const fd = new FormData();
    if (file) fd.append("file", file, file.name);
    else if (url) fd.append("reference_url", url);
    if (note) fd.append("note", note);
    if (requestId) fd.append("client_request_id", requestId);
    fd.append("voice_mode", mode || "keep");
    if (mode === "translate" && targetLanguage) fd.append("target_language", targetLanguage);
    fd.append("dialogue_mode", dialogue.dialogue_mode);
    const reviewField = buildDialogueReviewCreateField(
      dialogueReviewCapability,
      reviewPolicy,
      dialogue.dialogue_mode,
    );
    if (reviewField) fd.append(reviewField.name, reviewField.value);
    const configField = buildGenerationConfigCreateField(
      generationConfigCapability,
      generationConfig,
    );
    if (configField) fd.append(configField.name, configField.value);
    xhr.send(fd);
  });
}

async function handleSend(event) {
  event.preventDefault();
  if (state.uploading) return;
  if (!await guardDirtyPrompt()) return;
  const mode = sourceMode();
  const file = mode === "upload" ? state.file : null;
  const url = mode === "link" ? $("url-input").value.trim() : "";
  const note = $("note-input").value.trim();
  const vMode = voiceMode();
  const targetLanguage = $("lang-input").value.trim();
  const generationConfig = state.generationConfigCapability
    ? selectedGenerationConfig() : null;
  const reviewPolicy = dialogueReviewPolicy();
  let dialogue;
  if (!file && !url) {
    setComposerError(mode === "upload" ? "请先选择视频文件" : "请先粘贴视频链接");
    return;
  }
  if (vMode === "translate" && !targetLanguage) {
    setComposerError("请填写翻译目标语言");
    return;
  }
  try {
    dialogue = buildCreateDialogueFields(dialogueMode(), "");
  } catch (err) {
    setComposerError(err.message);
    return;
  }

  setComposerError(null);
  setUploading(true);
  const progress = $("upload-progress");
  const fill = $("up-fill");
  const label = $("up-label");
  progress.hidden = false;
  fill.style.width = "0%";
  // 链接分支没有本地上传进度：下载发生在服务端
  label.textContent = url ? "服务器正在下载视频…" : "正在上传 0%";

  try {
    const created = await uploadConversation(
      {
        file, url, note, requestId: state.clientRequestId,
        voiceMode: vMode, targetLanguage, dialogue, generationConfig,
        generationConfigCapability: state.generationConfigCapability,
        dialogueReviewPolicy: reviewPolicy,
        dialogueReviewCapability: state.dialogueReviewCapability,
      },
      (ratio) => {
        if (url) return;
        const pct = Math.round(ratio * 100);
        fill.style.width = pct + "%";
        label.textContent = pct >= 100 ? "上传完成，等待处理…" : "正在上传 " + pct + "%";
      }
    );
    // 成功：清空 composer、换新幂等键，刷新列表并选中新会话
    state.clientRequestId = newRequestId();
    clearFile();
    $("note-input").value = "";
    $("url-input").value = "";
    const keepRadio = document.querySelector('input[name="voice-mode"][value="keep"]');
    keepRadio.checked = true;
    $("lang-input").value = "";
    $("lang-input").hidden = true;
    $("lang-input").required = false;
    const autoDialogue = document.querySelector('input[name="dialogue-mode"][value="auto"]');
    autoDialogue.checked = true;
    selectDialogueReviewPolicy("auto_continue");
    renderDialogueReviewPolicyControl();
    if (state.generationConfigCapability) {
      applyGenerationConfigDefaults(state.generationConfigCapability.defaults);
      renderGenerationConfigControl();
    }
    setUploading(false);
    await refreshList(false);
    if (created && created.id) {
      selectConversation(created.id);
    }
  } catch (err) {
    setUploading(false);
    if (handleAuthError(err)) return;
    if (err.code === "video_duration_exceeds_h3_limit") {
      window.alert(err.message);
      setComposerError(err.message);
      return;
    }
    // 422 no audio track in video → 换成可读引导（本产品仅处理带口播视频）
    setComposerError(err.message.includes("no audio track in video")
      ? "该视频没有音轨，本产品仅支持带口播的视频"
      : err.message);
  }
}

/* ===== 事件绑定与启动 ===== */
function bindEvents() {
  $("login-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const token = $("login-token").value.trim();
    if (!token) {
      $("login-error").textContent = "请输入访问口令";
      $("login-error").hidden = false;
      return;
    }
    doLogin(token);
  });

  $("logout-btn").addEventListener("click", async () => {
    if (!await guardDirtyPrompt()) return;
    state.token = null;
    localStorage.removeItem(TOKEN_KEY);
    showLogin(null);
  });

  $("menu-btn").addEventListener("click", openDrawer);
  $("drawer-backdrop").addEventListener("click", closeDrawer);
  $("composer-toggle").addEventListener("click", () => {
    setComposerExpanded($("composer-toggle").getAttribute("aria-expanded") !== "true");
  });

  $("new-chat-btn").addEventListener("click", async () => {
    if (state.uploading) return;
    if (!await guardDirtyPrompt()) return;
    state.currentId = null;
    state.detail = null;
    resetGenerationConfigDisclosure();
    renderList();
    renderEmptyHero();
    closeDrawer();
    $("note-input").focus();
  });

  $("attach-btn").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => {
    pickFile(e.target.files && e.target.files[0]);
  });
  $("file-remove").addEventListener("click", clearFile);

  // 来源 radio 互斥切换
  document.querySelectorAll('input[name="source-mode"]').forEach((radio) => {
    radio.addEventListener("change", () => setSourceMode(radio.value));
  });

  // 口播转换 radio 切换
  document.querySelectorAll('input[name="voice-mode"]').forEach((radio) => {
    radio.addEventListener("change", setVoiceMode);
  });
  document.querySelectorAll('input[name="dialogue-mode"]').forEach((radio) => {
    radio.addEventListener("change", setDialogueMode);
  });
  $("dialogue-review-fields").addEventListener("change", () => {
    if (!state.dialogueReviewCapability || state.uploading || dialogueMode() !== "auto") return;
    state.clientRequestId = newRequestId();
    setComposerError(null);
    renderDialogueReviewPolicyControl();
  });
  $("lang-input").addEventListener("input", () => {
    state.clientRequestId = newRequestId(); // 内容变 = 新意图 = 新键
    setComposerError(null);
    updateSendBtn();
  });

  $("url-input").addEventListener("input", () => {
    state.clientRequestId = newRequestId(); // 内容变 = 新意图 = 新键
    setComposerError(null);
    updateSendBtn();
  });
  $("generation-config-fields").addEventListener("change", () => {
    if (!state.generationConfigCapability || state.uploading) return;
    state.clientRequestId = newRequestId();
    setComposerError(null);
    renderGenerationConfigControl();
  });

  $("pp-form").addEventListener("submit", submitPostprocess);
  $("pp-cancel").addEventListener("click", closePostprocessModal);
  $("pp-form").addEventListener("change", updatePpConfirm);

  const composer = $("composer");
  composer.addEventListener("submit", handleSend);
  composer.addEventListener("dragenter", (e) => {
    e.preventDefault();
    if (!state.uploading && sourceMode() === "upload") composer.classList.add("drag-over");
  });
  composer.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (!state.uploading && sourceMode() === "upload") composer.classList.add("drag-over");
  });
  composer.addEventListener("dragleave", (e) => {
    if (!composer.contains(e.relatedTarget)) composer.classList.remove("drag-over");
  });
  composer.addEventListener("drop", (e) => {
    e.preventDefault();
    composer.classList.remove("drag-over");
    if (state.uploading || sourceMode() !== "upload") return;
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    pickFile(file);
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 768) closeDrawer();
  });
  window.addEventListener("beforeunload", (event) => {
    if (!state.promptDraft || !state.promptDraft.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
}

function boot() {
  state.clientRequestId = newRequestId();
  bindEvents();
  setSourceMode(sourceMode());
  setDialogueMode();
  const saved = localStorage.getItem(TOKEN_KEY);
  if (saved) {
    state.token = saved;
    enterApp(); // refreshList 遇 401 会自动回登录页
  } else {
    showLogin(null);
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    buildLongRetryPayload,
    buildCreateDialogueFields,
    buildDialogueReviewCommitPayload,
    buildDialogueReviewCreateField,
    buildGenerationConfigCreateField,
    buildImagePromptPatch,
    buildStitchRetryPayload,
    buildResumePayload,
    buildSubmitPayload,
    apiErrorFromPayload,
    authoritativePostprocessFrameGroups,
    authoritativeSegmentKeyframePaths,
    canOperate,
    createImagePromptDraft,
    clearStream,
    closeLightbox,
    conversationBadge,
    conversationThumbnailPath,
    createDisclosure,
    detailSignature,
    diagnosticText,
    dialogueReviewOutcomeText,
    dialogueReviewCommitErrorMessage,
    dialogueReviewView,
    renderDialogueReview,
    dirtyPromptDecision,
    frameInspector,
    frameViewerEntries,
    selectedFrameEntry,
    fitProfile,
    formatDuration,
    formatElapsed,
    formatDialogueLines,
    frozenGenerationConfig,
    generationDraft,
    generationConfigLabels,
    generationAction,
    generationParameterDraft,
    generationParameterSnapshot,
    generationRetryContract,
    generationSegmentLabel,
    imagePromptEditable,
    isPostprocessSubmissionUnknown,
    longVideoContract,
    materialIndexView,
    mergeConversationList,
    mergeImagePromptDraft,
    normalizeGenerationConfigCapability,
    normalizeDialogueReviewCapability,
    normalizeDialogueLines,
    operationTimeline,
    parseDialogueLines,
    postprocessReadyForGeneration,
    ppCompletedFrames,
    postprocessSegmentStatus,
    postprocessAskDefault,
    promptSegmentIndex,
    promptWorkspaceModes,
    openLightbox,
    releaseTrackedURLs,
    releaseTrackedURL,
    releaseHistoryThumbnails,
    readOnlyImageFramePromptText,
    resetGenerationConfigDisclosure,
    resetSegmentProductsDisclosure,
    requestGenerationSubmit,
    retryPostprocessSegment,
    restoreImagePromptDefault,
    recoverLockedPostprocess,
    runSingleFlightPollCycle,
    safePostprocessError,
    safeErrorSummary,
    safePostprocessStage,
    saveImageOptimizationPrompt,
    segmentProductsDisclosure,
    segmentJoinText,
    setComposerExpanded,
    setDisclosureState,
    showActionError,
    shouldRenderPostprocessAsk,
    shouldPollDetail,
    shortId,
    skillMilestoneView,
    syncConversationDetail,
    totalKeyframes,
    validateDialogueReviewDraft,
  };
}

if (typeof document !== "undefined") boot();
