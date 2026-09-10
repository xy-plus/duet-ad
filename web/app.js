/* 视频工作室 — ChatGPT 式单页前端
 * 同源 API：/api/*，共享口令 Bearer 鉴权；文件直链也需鉴权，故一律 fetch(blob) → ObjectURL。
 */
"use strict";

/* ===== 常量与状态 ===== */
const TOKEN_KEY = "cvs_token";
const COMPOSER_SNAPSHOTS_KEY = "cvs_minimal_creation_snapshots_v1";
const POLL_MS = 2000;
const GENERATION_ASPECT_RATIOS = Object.freeze(["16:9", "9:16"]);
const GENERATION_RESOLUTIONS = Object.freeze(["480p", "768p"]);
const MINIMAL_CREATION_VERSION = 1;
const TARGET_LANGUAGE_MAX_CHARS = 80;
const SAME_AS_SOURCE_LANGUAGE = "与原视频相同";
const SOURCE_URL_PLACEHOLDER = "粘贴 http(s) 视频直链或视频页链接";
const MINIMAL_PROCESSING_DEFAULTS = Object.freeze({
  optimize_image: true,
  remove_subtitle: true,
  remove_logo: true,
});
const REPLACEMENT_IMAGE_TYPES = Object.freeze([
  "image/jpeg", "image/png", "image/webp",
]);
const DIALOGUE_MODES = Object.freeze(["auto_rewrite", "generate"]);

const STATUS_TEXT = { queued: "排队中", processing: "处理中", done: "已完成", failed: "失败" };

const state = {
  token: null,
  authEpoch: 0,       // 同一口令重新登录也必须使旧异步请求失效
  conversations: [],
  currentId: null,
  detail: null,        // 当前会话详情
  file: null,          // composer 已选文件
  replacementImages: [],
  replacementPreviewURLs: [],
  replacementPreviewOwnerId: null,
  clientRequestId: null, // 幂等键：内容变更/成功才轮换，失败重试复用
  uploading: false,
  pollTimer: null,
  pollToken: 0,
  detailSeq: 0,        // 防止过期响应覆盖新渲染
  objectURLs: [],      // 当前 stream 渲染产生的 blob URL，重渲染前统一 revoke
  historyDetails: {}, // cid → 用户显式选中过的会话详情；不为侧栏批量加载
  historyThumbnailURLs: {}, // cid → 侧栏首帧 ObjectURL，与 stream 媒体生命周期隔离
  historyHydrating: false, // 受限并发补齐进行中，避免重复 GET 风暴
  minimalCreationCapability: null,
  minimalCreationCapabilityLoaded: false,
  composerDrafts: {}, // cid → 本页提交的可序列化创建输入
  viewingSubmittedConfig: false, // 当前配置属于既有项目，仅展示；新建项目后才允许再次提交
  submittedReplacements: [], // 当前展示的是已提交参考图元数据，而非可再次提交的 File
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
  const detail = data && data.detail;
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

function hasExactKeys(value, keys) {
  return !!value && typeof value === "object" && !Array.isArray(value)
    && Object.keys(value).sort().join("|") === keys.slice().sort().join("|");
}

function normalizeMinimalCreationCapability(payload) {
  const capability = payload && payload.minimal_creation;
  const dialogue = capability && capability.dialogue;
  const replacement = capability && capability.replacement;
  const defaults = capability && capability.defaults;
  if (!capability || capability.supported !== true
      || capability.version !== MINIMAL_CREATION_VERSION
      || capability.endpoint !== "/api/conversations"
      || capability.encoding !== "multipart/form-data"
      || capability.request_field !== "generation_request"
      || capability.replacement_image_field !== "replacement_image"
      || !Array.isArray(capability.aspect_ratios)
      || capability.aspect_ratios.join("|") !== GENERATION_ASPECT_RATIOS.join("|")
      || !Array.isArray(capability.resolutions)
      || capability.resolutions.join("|") !== GENERATION_RESOLUTIONS.join("|")
      || !hasExactKeys(defaults, ["fit_mode", "optimize_image", "remove_subtitle", "remove_logo", "add_subtitles"])
      || defaults.add_subtitles !== false
      || defaults.fit_mode !== "auto"
      || defaults.optimize_image !== true
      || defaults.remove_subtitle !== true
      || defaults.remove_logo !== true
      || !hasExactKeys(dialogue, ["mode", "modes", "translation", "max_brief_chars", "optional", "default"])
      || !Array.isArray(dialogue.modes)
      || dialogue.modes.join("|") !== DIALOGUE_MODES.join("|")
      || dialogue.mode !== "auto_rewrite"
      || dialogue.translation !== true
      || !Number.isInteger(dialogue.max_brief_chars)
      || dialogue.max_brief_chars < 1
      || dialogue.optional !== true
      || !hasExactKeys(dialogue.default, ["mode", "target_language"])
      || dialogue.default.mode !== "auto_rewrite"
      || dialogue.default.target_language !== SAME_AS_SOURCE_LANGUAGE
      || !replacement || !hasExactKeys(replacement, [
        "supported", "accept", "max_bytes", "max_images", "max_instruction_chars",
      ]) || replacement.supported !== true
      || !Array.isArray(replacement.accept)
      || replacement.accept.join("|") !== REPLACEMENT_IMAGE_TYPES.join("|")
      || !Number.isInteger(replacement.max_bytes) || replacement.max_bytes < 1
      || replacement.max_images !== 4
      || !Number.isInteger(replacement.max_instruction_chars)
      || replacement.max_instruction_chars < 1) return null;
  return {
    endpoint: capability.endpoint,
    request_field: capability.request_field,
    replacement_image_field: capability.replacement_image_field,
    aspect_ratios: capability.aspect_ratios.slice(),
    resolutions: capability.resolutions.slice(),
    dialogue: {
      mode: dialogue.mode,
      modes: dialogue.modes.slice(),
      translation: dialogue.translation,
      max_brief_chars: dialogue.max_brief_chars,
      optional: dialogue.optional,
      default: {
        mode: dialogue.default.mode,
        target_language: dialogue.default.target_language,
      },
    },
    replacement: {
      accept: replacement.accept.slice(),
      max_bytes: replacement.max_bytes,
      max_images: replacement.max_images,
      max_instruction_chars: replacement.max_instruction_chars,
    },
  };
}

function buildMinimalGenerationRequest(input, capability) {
  const normalized = normalizeMinimalCreationCapability({ minimal_creation: capability });
  if (!normalized) throw new Error("生成服务尚未支持当前创建方式");
  const aspectRatio = String(input && input.aspectRatio || "");
  const resolution = String(input && input.resolution || "");
  const targetLanguage = String(input && input.targetLanguage || "").trim();
  const dialogueMode = input && input.dialogueMode === "generate" ? "generate" : "auto_rewrite";
  const dialogueBrief = String(input && input.dialogueBrief || "").trim();
  const replacementImageCount = Number.isInteger(input && input.replacementImageCount)
    ? input.replacementImageCount : (input && input.hasReplacementImage ? 1 : 0);
  const hasReplacementImage = replacementImageCount > 0;
  const replacementInstruction = String(input && input.replacementInstruction || "").trim();
  if (!normalized.aspect_ratios.includes(aspectRatio)) throw new Error("请选择画幅");
  if (!normalized.resolutions.includes(resolution)) throw new Error("请选择清晰度");
  if (!targetLanguage) throw new Error("请填写目标语言");
  if (targetLanguage.length > TARGET_LANGUAGE_MAX_CHARS) throw new Error("目标语言超过长度限制");
  if (dialogueMode === "generate" && !dialogueBrief) throw new Error("请填写口播描述");
  if (dialogueBrief.length > normalized.dialogue.max_brief_chars) {
    throw new Error("口播描述超过长度限制");
  }
  if (replacementImageCount < 0 || replacementImageCount > normalized.replacement.max_images) {
    throw new Error("参考图数量超出限制");
  }
  if (hasReplacementImage !== !!replacementInstruction) {
    throw new Error("参考图与替换说明需要一起提供");
  }
  if (replacementInstruction.length > normalized.replacement.max_instruction_chars) {
    throw new Error("替换说明内容过长");
  }
  return {
    version: MINIMAL_CREATION_VERSION,
    output: {
      aspect_ratio: aspectRatio,
      resolution,
      fit_mode: "auto",
    },
    processing: Object.assign({}, MINIMAL_PROCESSING_DEFAULTS),
    dialogue: dialogueMode === "generate"
      ? {
        mode: "generate",
        target_language: targetLanguage,
        brief: dialogueBrief,
      }
      : {
        mode: "auto_rewrite",
        target_language: targetLanguage,
      },
    replacement_guidance: hasReplacementImage ? {
      instruction: replacementInstruction,
      image_field: normalized.replacement_image_field,
    } : null,
    add_subtitles: !!(input && input.addSubtitles === true),
  };
}

function replacementImageSnapshot(image) {
  return {
    filename: String(image.name || "参考图"),
    bytes: Number.isFinite(image.size) ? image.size : null,
    media_type: String(image.type || ""),
  };
}

function creationInputSnapshot({
  file = null, url = "", replacementImage = null, replacementImages = null,
} = {}) {
  const source = file ? {
    mode: "upload",
    filename: String(file.name || "原视频"),
    bytes: Number.isFinite(file.size) ? file.size : null,
  } : {
    mode: "link",
    reference_url: String(url || ""),
  };
  const images = Array.isArray(replacementImages)
    ? replacementImages.slice() : replacementImage ? [replacementImage] : [];
  const imageSnapshots = images.map(replacementImageSnapshot);
  return {
    version: 1,
    source,
    // 保留首图投影，以便旧任务、旧客户端和单图 API 调用保持兼容。
    replacement_image: imageSnapshots[0] || null,
    replacement_images: imageSnapshots,
  };
}

function normalizeCreationInputSnapshot(value, { allowLocalPreview = false } = {}) {
  if (!value || typeof value !== "object" || Array.isArray(value)
      || value.version !== MINIMAL_CREATION_VERSION
      || !Object.prototype.hasOwnProperty.call(value, "source")
      || !Object.prototype.hasOwnProperty.call(value, "replacement_image")) return null;
  const rawSource = value.source;
  let source = null;
  if (rawSource && typeof rawSource === "object" && !Array.isArray(rawSource)
      && rawSource.mode === "link") {
    if (rawSource.reference_url === null) {
      source = { mode: "link", reference_url: null };
    } else if (typeof rawSource.reference_url === "string" && rawSource.reference_url.trim()) {
      source = { mode: "link", reference_url: rawSource.reference_url.trim() };
    } else {
      return null;
    }
  } else if (rawSource && rawSource.mode === "upload"
      && typeof rawSource.filename === "string" && rawSource.filename.trim()
      && Number.isInteger(rawSource.bytes) && rawSource.bytes >= 0) {
    source = {
      mode: "upload",
      filename: rawSource.filename.trim(),
      bytes: rawSource.bytes,
    };
  } else {
    return null;
  }
  const normalizeReplacement = (rawReplacement) => {
    if (rawReplacement === null) return null;
    const previewURL = rawReplacement && rawReplacement.preview_url;
    if (!rawReplacement || typeof rawReplacement !== "object" || Array.isArray(rawReplacement)
        || typeof rawReplacement.filename !== "string" || !rawReplacement.filename.trim()
        || !Number.isInteger(rawReplacement.bytes) || rawReplacement.bytes < 0
        || (allowLocalPreview
          ? typeof rawReplacement.media_type !== "string"
            || (!!rawReplacement.media_type
              && !REPLACEMENT_IMAGE_TYPES.includes(rawReplacement.media_type))
          : !REPLACEMENT_IMAGE_TYPES.includes(rawReplacement.media_type))
        || (allowLocalPreview
          ? previewURL !== undefined && typeof previewURL !== "string"
          : typeof previewURL !== "string" || !previewURL.trim())) return undefined;
    return {
      filename: rawReplacement.filename.trim(),
      bytes: rawReplacement.bytes,
      media_type: rawReplacement.media_type,
      preview_url: typeof previewURL === "string" ? previewURL.trim() : "",
    };
  };
  const replacementImage = normalizeReplacement(value.replacement_image);
  if (replacementImage === undefined) return null;
  const hasReplacementImages = Object.prototype.hasOwnProperty.call(value, "replacement_images");
  let replacementImages = replacementImage ? [replacementImage] : [];
  if (hasReplacementImages) {
    if (!Array.isArray(value.replacement_images)
        || value.replacement_images.length > 4
        || (!!replacementImage) !== (value.replacement_images.length > 0)) return null;
    replacementImages = value.replacement_images.map(normalizeReplacement);
    if (replacementImages.some((image) => !image)) return null;
    const first = replacementImages[0] || null;
    if (!!first !== !!replacementImage
        || (first && (first.filename !== replacementImage.filename
          || first.bytes !== replacementImage.bytes
          || first.media_type !== replacementImage.media_type))) return null;
  }
  return {
    version: MINIMAL_CREATION_VERSION,
    source,
    replacement_image: replacementImage,
    replacement_images: replacementImages,
  };
}

function resolveCreationInputSnapshot(response, localInput = null) {
  if (response && typeof response === "object"
      && Object.prototype.hasOwnProperty.call(response, "creation_input")) {
    return normalizeCreationInputSnapshot(response.creation_input);
  }
  return normalizeCreationInputSnapshot(localInput, { allowLocalPreview: true });
}

function storedComposerDraft(conversationId) {
  if (!conversationId || typeof sessionStorage === "undefined") return null;
  try {
    const all = JSON.parse(sessionStorage.getItem(COMPOSER_SNAPSHOTS_KEY) || "{}");
    const record = all && typeof all === "object" ? all[conversationId] : null;
    return record && typeof record === "object" ? record : null;
  } catch (_error) {
    return null;
  }
}

function rememberSubmittedComposer(conversationId, request, inputs, response = null) {
  if (!conversationId) return;
  const input = resolveCreationInputSnapshot(response, creationInputSnapshot(inputs));
  const record = { request, input };
  state.composerDrafts[conversationId] = record;
  if (input && input.replacement_images.length && state.replacementPreviewURLs.length) {
    state.replacementPreviewOwnerId = conversationId;
  }
  if (typeof sessionStorage === "undefined") return;
  try {
    const all = JSON.parse(sessionStorage.getItem(COMPOSER_SNAPSHOTS_KEY) || "{}");
    const snapshots = all && typeof all === "object" ? all : {};
    snapshots[conversationId] = { request, input };
    sessionStorage.setItem(COMPOSER_SNAPSHOTS_KEY, JSON.stringify(snapshots));
  } catch (_error) {
    // 浏览器禁用会话存储时，服务端 effective_request 仍可恢复标量配置。
  }
}

function clearComposerSnapshots() {
  state.composerDrafts = {};
  state.viewingSubmittedConfig = false;
  state.file = null;
  state.replacementImages = [];
  state.submittedReplacements = [];
  revokeReplacementPreviewURLs();
  state.replacementPreviewOwnerId = null;
  if (typeof sessionStorage !== "undefined") {
    try {
      sessionStorage.removeItem(COMPOSER_SNAPSHOTS_KEY);
    } catch (_error) {
      // 显式退出时尽力清理；浏览器拒绝存储访问不影响退出。
    }
  }
}

function projectComposerModel(detail, localDraft = null) {
  const value = detail && typeof detail === "object" ? detail : {};
  const serverRequest = value.effective_request;
  const fallbackRequest = localDraft && localDraft.request;
  const request = serverRequest && serverRequest.version === MINIMAL_CREATION_VERSION
    ? serverRequest
    : fallbackRequest && fallbackRequest.version === MINIMAL_CREATION_VERSION
      ? fallbackRequest : null;
  const output = request && request.output;
  const dialogue = request && request.dialogue;
  if (!request || !output || !dialogue
      || !GENERATION_ASPECT_RATIOS.includes(output.aspect_ratio)
      || !GENERATION_RESOLUTIONS.includes(output.resolution)
      || typeof dialogue.target_language !== "string"
      || !dialogue.target_language.trim()) return null;
  const input = resolveCreationInputSnapshot(value, localDraft && localDraft.input);
  const guidance = request.replacement_guidance;
  const receiptReplacement = value.input_receipt
    && value.input_receipt.replacement_image;
  const replacementUsed = !!guidance || !!receiptReplacement;
  return {
    aspectRatio: output.aspect_ratio,
    resolution: output.resolution,
    dialogueMode: dialogue.mode === "generate" ? "generate" : "auto_rewrite",
    addSubtitles: request.add_subtitles === true,
    dialogueBrief: dialogue.mode === "generate" && typeof dialogue.brief === "string"
      ? dialogue.brief : "",
    languageMode: dialogue.target_language.trim() === SAME_AS_SOURCE_LANGUAGE
      ? "same" : "other",
    targetLanguage: dialogue.target_language.trim() === SAME_AS_SOURCE_LANGUAGE
      ? "" : dialogue.target_language.trim(),
    replacementInstruction: guidance && typeof guidance.instruction === "string"
      ? guidance.instruction : "",
    source: input && input.source,
    replacementImage: input && input.replacement_image
      ? input.replacement_image
      : replacementUsed ? { filename: "已提交参考图", bytes: null, media_type: "", preview_url: "" }
        : null,
    replacementImages: input && input.replacement_images.length
      ? input.replacement_images
      : replacementUsed ? [{ filename: "已提交参考图", bytes: null, media_type: "", preview_url: "" }]
        : [],
  };
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

function formatProjectTimestamp(iso) {
  const value = new Date(iso);
  if (Number.isNaN(value.getTime())) return "未知";
  const pad = (part) => String(part).padStart(2, "0");
  return `${value.getFullYear()}-${pad(value.getMonth() + 1)}-${pad(value.getDate())} `
    + `${pad(value.getHours())}:${pad(value.getMinutes())}:${pad(value.getSeconds())}`;
}

function safeErrorSummary(raw, fallback = "阶段执行失败，请查看诊断后处理") {
  const code = raw && typeof raw === "object"
    ? String(raw.code || raw.error || "") : String(raw || "");
  const labels = new Map([
    ["submission_unknown", "提交结果未知，已禁止重复提交"],
    ["context_ir_query_unknown", "Context IR 查询结果未知，请继续原任务"],
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

function projectPhaseLabel(detail) {
  const value = detail || {};
  const progress = value.project_progress || {};
  const percent = Number(progress.percent);
  if (Number.isFinite(percent)) {
    if (percent >= 57) return "AI 正在生成视频";
    if (percent >= 24) return "AI 正在思考创意";
    return "AI 正在理解视频";
  }

  const generation = value.generation;
  const stage = generation && generation.stage;
  if (["h3", "h3-max-r2v", "picsart-resume", "stitch", "stitching", "stitch-resume"].includes(stage)) {
    return "AI 正在生成视频";
  }
  if (["video-maker", "video-maker-resume", "translation", "video-maker-project-index", "object-replace",
    "seedream-reference-sheet",
    "context_ir", "context_ir_native"].includes(stage)) {
    return "AI 正在思考创意";
  }
  const fusionStatus = value.prompt_fusion && value.prompt_fusion.status;
  const postprocessStatus = value.postprocess && value.postprocess.status;
  if (["running", "done", "failed"].includes(fusionStatus)
      || ["queued", "running", "done", "failed"].includes(postprocessStatus)) {
    return "AI 正在思考创意";
  }
  return "AI 正在理解视频";
}

function projectProgressModel(detail, nowMs = Date.now()) {
  const value = detail || {};
  const authoritative = value.project_progress;
  const allowedStates = ["queued", "running", "succeeded", "failed"];
  const coherentProgress = authoritative && (
    (authoritative.status === "queued" && authoritative.percent === 0)
    || (authoritative.status === "running" && authoritative.percent >= 1 && authoritative.percent <= 99)
    || (authoritative.status === "succeeded" && authoritative.percent === 100)
    || (authoritative.status === "failed" && authoritative.percent >= 0 && authoritative.percent <= 99)
  );
  const progress = authoritative && Number.isInteger(authoritative.percent)
      && authoritative.percent >= 0 && authoritative.percent <= 100
      && allowedStates.includes(authoritative.status) && coherentProgress
    ? {
      title: String(value.title || "未命名项目"),
      percent: authoritative.percent,
      state: authoritative.status,
    }
    : {
      title: String(value.title || "未命名项目"),
      percent: 0,
      state: "failed",
    };

  const terminal = ["succeeded", "failed"].includes(progress.state);
  const elapsed = terminal && !value.updated_at
    ? "未知"
    : formatElapsed(value.created_at, terminal ? value.updated_at : null, nowMs)
      .replace(/^已耗时\s*/, "");
  return {
    ...progress,
    startedAt: formatProjectTimestamp(value.created_at),
    elapsedLabel: terminal ? "总耗时" : "当前耗时",
    elapsed,
    phase: progress.state === "succeeded" ? "视频生成完成"
      : progress.state === "failed" ? "视频生成未完成"
        : projectPhaseLabel(value),
    loading: !terminal,
  };
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

function hideOperationHeader() {
  const host = typeof document !== "undefined" ? $("operation-status") : null;
  if (!host) return;
  host.textContent = "";
  host.hidden = true;
}

function renderOperationHeader(detail) {
  const host = $("operation-status");
  const model = projectProgressModel(detail);
  host.textContent = "";
  host.hidden = false;
  host.dataset.status = model.state;

  const header = el("div", "project-progress-header");
  const fields = el("div", "project-progress-fields");
  const project = el("span", "project-progress-field project-progress-project");
  project.appendChild(el("span", "project-progress-label", "项目"));
  project.appendChild(el("strong", "project-progress-title", model.title));
  fields.appendChild(project);
  const started = el("span", "project-progress-field");
  started.appendChild(el("span", "project-progress-label", "发起时间"));
  started.appendChild(el("span", "project-progress-time", model.startedAt));
  fields.appendChild(started);
  const elapsed = el("span", "project-progress-field");
  elapsed.appendChild(el("span", "project-progress-label", model.elapsedLabel));
  elapsed.appendChild(el("span", "project-progress-time", model.elapsed));
  fields.appendChild(elapsed);
  header.appendChild(fields);
  host.appendChild(header);

  const phase = el("div", "project-phase");
  phase.setAttribute("role", "status");
  if (model.loading) {
    const spinner = el("span", "project-phase-spinner");
    spinner.setAttribute("aria-hidden", "true");
    phase.appendChild(spinner);
  }
  phase.appendChild(el("span", "project-phase-label", model.phase));
  host.appendChild(phase);
}

function trackURL(url) {
  state.objectURLs.push(url);
  return url;
}

function revokeURLs() {
  for (const u of state.objectURLs) URL.revokeObjectURL(u);
  state.objectURLs = [];
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
  state.minimalCreationCapability = null;
  state.minimalCreationCapabilityLoaded = false;
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
  state.authEpoch += 1;
  state.token = null;
  localStorage.removeItem(TOKEN_KEY);
  clearComposerSnapshots();
  resetComposerForNewProject();
  clearStream();
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
    state.authEpoch += 1;
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
  resetComposerForNewProject();
  renderEmptyHero();
  void loadMinimalCreationCapability();
  refreshList(false);
}

/* ===== 侧栏会话列表 ===== */
async function refreshList() {
  try {
    const list = await apiJSON("/api/conversations");
    state.conversations = mergeConversationList(
      Array.isArray(list) ? list : [],
      state.conversations,
    );
    releaseHistoryThumbnails(new Set(state.conversations.map((item) => item.id)));
    renderList();
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
      "skill_milestone", "thumbnail_path", "prompt_fusion", "postprocess",
      "project_progress", "error",
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
  summary.project_progress = detail.project_progress || null;
  summary.error = detail.error || null;
  summary.generation = detail.generation || null;
  summary.duration_s = detail.duration_s;
  summary.updated_at = detail.updated_at;
  summary.segment_count = detail.segment_count;
  summary.skill_milestone = detail.skill_milestone || null;
  summary.prompt_fusion = detail.prompt_fusion || null;
  summary.postprocess = detail.postprocess || null;
  summary.thumbnail_path = conversationThumbnailPath(detail);
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
  const pending = state.conversations.filter((item) => item && item.id
    && item.thumbnail_path && !state.historyThumbnailURLs[item.id]);
  if (pending.length === 0) return;
  state.historyHydrating = true;
  let cursor = 0;
  const worker = async () => {
    while (cursor < pending.length) {
      const summary = pending[cursor++];
      try {
        await loadHistoryThumbnail(summary);
      } catch (error) {
        if (handleAuthError(error)) return;
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
    identity.appendChild(el("span", null, formatDuration(c.duration_s)));
    content.appendChild(identity);
    const footer = el("span", "conv-footer");
    const progress = projectProgressModel(c);
    footer.appendChild(el("span", null, progress.elapsed));
    const cardTime = progress.loading ? c.created_at : (c.updated_at || c.created_at);
    footer.appendChild(el("span", "conv-time", fmtTime(cardTime)));
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
  if (Object.prototype.hasOwnProperty.call(item, "project_progress")) {
    const progress = projectProgressModel(item);
    if (progress.state === "succeeded") {
      return item.has_video === true
        ? { className: "done", text: "已完成" }
        : { className: "failed", text: "最终视频缺失" };
    }
    if (progress.state === "failed") return { className: "failed", text: "生成失败" };
    if (progress.state === "running") return { className: "processing", text: "生成中" };
    return { className: "queued", text: "排队中" };
  }
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
  revokeURLs();
  $("stream").textContent = "";
}

function renderEmptyHero() {
  stopPolling();
  hideOperationHeader();
  setComposerExpanded(true);
  clearStream();
  $("main-title").textContent = "视频工作室";

  const inner = el("div", "stream-inner");
  const hero = el("div", "empty-hero");
  const iconBox = el("div", "empty-icon");
  iconBox.appendChild(icon("i-film"));
  hero.appendChild(iconBox);
  hero.appendChild(el("h2", null, "提供原视频，生成新视频"));
  hero.appendChild(el("p", "empty-sub", "Agent 会自动改编台词，你只需选择最终视频语种与生成配置"));
  const steps = el("ol", "empty-steps");
  const items = [
    "粘贴原视频链接，也可切换为上传",
    "选择画幅、清晰度与最终视频语种",
    "点击开始生成，等待新视频完成",
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
  if (detail.has_source === true) {
    const preview = videoPlayer(detail, "source.mp4");
    preview.classList.add("bubble-source-video");
    preview.setAttribute("aria-label", "原视频");
    bubble.appendChild(preview);
  }
  row.appendChild(bubble);
  return row;
}

/* 失败项目续跑：优先复用 Picsart 前置产物续跑；409 picsart_resume_not_ready 时
   回退到更靠前的 pre-Picsart 续跑；两个入口都只接受 failed 项目，其余错误直接抛出。 */
async function resumeFailedProject(conversationId, request = apiJSON) {
  const options = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm: true }),
  };
  try {
    return await request(
      "/api/conversations/" + encodeURIComponent(conversationId) + "/resume-picsart",
      options,
    );
  } catch (error) {
    if (!error || error.code !== "picsart_resume_not_ready") throw error;
    return request(
      "/api/conversations/" + encodeURIComponent(conversationId) + "/resume-pre-picsart",
      options,
    );
  }
}

function renderFail(detail) {
  const row = el("div", "msg-row");
  row.appendChild(assistantHead(detail.updated_at));
  const card = el("div", "fail-card");
  card.appendChild(icon("i-alert", "ic-danger"));
  const body = el("div");
  body.appendChild(el("p", "fail-title", "生成未完成"));
  body.appendChild(el("p", "fail-msg", "项目未能完成，请稍后重试或联系管理员。"));
  if (detail.error) {
    body.appendChild(el("p", "fail-tip", safeErrorSummary(detail.error, "生成失败")));
  }
  if (detail.status === "failed") {
    const errorBox = el("p", "form-error");
    errorBox.setAttribute("role", "alert");
    errorBox.hidden = true;
    const actions = el("div", "fail-actions");
    const resume = el("button", "btn btn-primary", "续跑生成");
    resume.type = "button";
    resume.addEventListener("click", async () => {
      if (!window.confirm("将从已保存的进度继续生成，不会重复创建项目。确认续跑吗？")) return;
      resume.disabled = true;
      errorBox.hidden = true;
      try {
        await resumeFailedProject(detail.id);
        // 202：项目已重新排队，详情轮询接管后续状态刷新
        await loadDetail(detail.id, true);
      } catch (err) {
        if (handleAuthError(err)) return;
        showActionError(err, errorBox, [resume]);
      }
    });
    actions.appendChild(resume);
    body.appendChild(actions);
    body.appendChild(errorBox);
  } else {
    body.appendChild(el("p", "fail-tip", "刷新页面不会重复提交任务。"));
  }
  card.appendChild(body);
  row.appendChild(card);
  return row;
}

/* 结果区（done） */
/* 视频区块：原始 / 最终成片共用同一加载骨架，靠标题与副标题区分 */
function videoPlayer(detail, file) {
  const wrap = el("div", "video-wrap");
  wrap.appendChild(el("div", "video-shimmer shimmer", "正在加载视频…"));
  const video = el("video");
  video.controls = true;
  video.playsInline = true;
  video.preload = "metadata";
  wrap.appendChild(video);
  apiBlobURL("/api/conversations/" + detail.id + "/files/" + file)
    .then((url) => {
      video.src = url;
      video.addEventListener("loadeddata", () => wrap.classList.add("is-ready"), { once: true });
      video.addEventListener("error", () => wrap.classList.add("is-error"), { once: true });
    })
    .catch(() => wrap.classList.add("is-error"));
  return wrap;
}

function videoSection(detail, file, title, sub) {
  const sec = el("section", "res-section");
  const h = el("h3", "res-h3", title);
  if (sub) h.appendChild(el("span", "res-count", sub));
  sec.appendChild(h);
  sec.appendChild(videoPlayer(detail, file));
  return sec;
}

function renderResults(detail) {
  const frag = document.createDocumentFragment();
  if (detail.has_video === true) {
    frag.appendChild(videoSection(detail, "generated.mp4", "新视频", "已完成"));
  }
  return frag;
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

function renderDetail(detail) {
  renderOperationHeader(detail);
  renderStable(detail);
}

/* 稳定区：用户气泡 + 失败卡（含续跑入口）/ 结果区 */
function renderStable(detail) {
  clearStream();
  const inner = el("div", "stream-inner");
  inner.appendChild(renderUserBubble(detail));
  const progress = projectProgressModel(detail);
  if (progress.state === "failed") {
    inner.appendChild(renderFail(detail));
  } else if (progress.state === "succeeded" && detail.has_video === true) {
    inner.appendChild(renderResults(detail));
  }
  $("stream").appendChild(inner);
}

/* ===== 会话详情 + 轮询 ===== */
/* 极简详情只在可见内容变化时重建。百分比和阶段标题由
   renderOperationHeader 单独刷新，隐藏的技术字段不能触发视频重载。 */
function detailSignature(detail) {
  return {
    stable: JSON.stringify([
      detail.id,
      detail.title,
      detail.note,
      detail.project_progress && detail.project_progress.status || null,
      detail.error || null,
      detail.has_source === true,
      detail.has_video ? 1 : 0,
    ]),
  };
}

async function loadDetail(id, silent) {
  if (!id || state.currentId !== id) return;
  const seq = ++state.detailSeq;
  if (!silent) renderSkeleton();
  try {
    const detail = await apiJSON("/api/conversations/" + encodeURIComponent(id));
    if (seq !== state.detailSeq || state.currentId !== id) return; // 已切换会话
    state.detail = detail;
    if (!silent) applyProjectComposer(detail);
    if (syncConversationDetail(state.conversations, detail)) {
      renderList();
      const summary = state.conversations.find((item) => item.id === detail.id);
      if (summary) void loadHistoryThumbnail(summary).then(() => {
        if (state.token) renderList();
      });
    }
    renderOperationHeader(detail);
    const sig = detailSignature(detail);
    // 手动切换 / 首次进入全量渲染；轮询仅在 stable 签名变化时全量一次，
    // 签名完全不变 → 不碰 DOM（根治轮询闪烁的关键）
    if (!silent || !state.detailSig || sig.stable !== state.detailSig.stable) {
      renderDetail(detail);
    }
    state.detailSig = sig;
    if (shouldPollDetail(detail)) {
      startPolling(id);
    } else {
      stopPolling();
    }
  } catch (err) {
    if (handleAuthError(err)) return;
    if (seq !== state.detailSeq || state.currentId !== id) return;
    state.detailSig = null;
    if (silent) {
      renderStreamError("会话加载失败：" + err.message);
      startPolling(id);
      return;
    }
    state.currentId = null;
    state.detail = null;
    resetComposerForNewProject();
    renderList();
    renderEmptyHero();
    setComposerError("项目加载失败，已回到新建项目：" + err.message);
  }
}

function shouldPollDetail(detail) {
  if (!detail || typeof detail !== "object") return false;
  const progressState = projectProgressModel(detail).state;
  return progressState === "queued" || progressState === "running";
}

async function selectConversation(id) {
  if (state.uploading) return; // 上传中不切换，避免打断
  const switchingProject = state.currentId !== id;
  state.currentId = id;
  if (switchingProject) {
    state.detail = null;
    prepareComposerForProjectLoad();
  }
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

function sourceMode() {
  const checked = document.querySelector('input[name="source-mode"]:checked');
  return checked ? checked.value : "link";
}

function dialogueWriteMode() {
  const checked = document.querySelector('input[name="dialogue-write-mode"]:checked');
  return checked ? checked.value : "auto_rewrite";
}

function syncDialogueModeControls() {
  const isGenerate = dialogueWriteMode() === "generate";
  $("dialogue-brief-fields").hidden = !isGenerate;
  $("dialogue-brief").required = isGenerate;
  $("dialogue-brief").disabled = state.uploading || state.viewingSubmittedConfig || !isGenerate;
}

function targetLanguageMode() {
  const checked = document.querySelector('input[name="target-language-mode"]:checked');
  return checked ? checked.value : "same";
}

function resolveTargetLanguage(mode, otherLanguage) {
  if (mode === "same") return SAME_AS_SOURCE_LANGUAGE;
  if (mode === "other") return String(otherLanguage || "").trim();
  return "";
}

function selectedTargetLanguage() {
  return resolveTargetLanguage(targetLanguageMode(), $("lang-input").value);
}

function syncTargetLanguageControls() {
  const isOther = targetLanguageMode() === "other";
  $("translation-fields").hidden = !isOther;
  $("lang-input").required = isOther;
  $("lang-input").disabled = state.uploading || state.viewingSubmittedConfig || !isOther;
}

function selectedAspectRatio() {
  const checked = document.querySelector('input[name="aspect-ratio"]:checked');
  return checked ? checked.value : "";
}

function selectedResolution() {
  const checked = document.querySelector('input[name="resolution"]:checked');
  return checked ? checked.value : "";
}

function renderCreationServiceStatus() {
  const status = $("creation-service-status");
  if (!state.minimalCreationCapabilityLoaded) {
    status.textContent = "正在连接生成服务…";
    status.classList.remove("is-error");
  } else if (!state.minimalCreationCapability) {
    status.textContent = "生成服务尚未升级，暂时无法创建新项目";
    status.classList.add("is-error");
  } else {
    status.textContent = "配置完成后即可一次提交并自动生成";
    status.classList.remove("is-error");
  }
  updateReplacementHint();
  updateSendBtn();
}

async function loadMinimalCreationCapability() {
  const token = state.token;
  state.minimalCreationCapability = null;
  state.minimalCreationCapabilityLoaded = false;
  renderCreationServiceStatus();
  try {
    const payload = await apiJSON("/api/capabilities");
    if (token !== state.token) return;
    const normalized = normalizeMinimalCreationCapability(payload);
    state.minimalCreationCapability = normalized ? payload.minimal_creation : null;
    if (normalized) {
      $("dialogue-brief").maxLength = normalized.dialogue.max_brief_chars;
      $("replacement-instruction").maxLength = normalized.replacement.max_instruction_chars;
      $("replacement-image-input").accept = normalized.replacement.accept.join(",");
      renderReplacementImages();
    }
  } catch (error) {
    if (handleAuthError(error)) return;
    if (token !== state.token) return;
    state.minimalCreationCapability = null;
  } finally {
    if (token === state.token) {
      state.minimalCreationCapabilityLoaded = true;
      renderCreationServiceStatus();
    }
  }
}

function syncSourceModeView(mode) {
  const isUpload = mode === "upload";
  const radio = document.querySelector(`input[name="source-mode"][value="${isUpload ? "upload" : "link"}"]`);
  if (radio) radio.checked = true;
  $("attach-btn").hidden = !isUpload;
  $("file-hint").hidden = !isUpload;
  $("url-row").hidden = isUpload;
  $("url-input").required = !isUpload;
}

function showSourceFileSummary(filename, bytes) {
  $("file-chip-name").textContent = filename || "已提交原视频";
  $("file-chip-size").textContent = Number.isFinite(bytes) ? fmtBytes(bytes) : "已提交";
  $("file-chip").hidden = false;
}

// 来源二选一：同一时刻只存在一种输入，切换即清空另一边
function setSourceMode(mode) {
  const isUpload = mode === "upload";
  $("url-input").placeholder = SOURCE_URL_PLACEHOLDER;
  if (isUpload) {
    $("url-input").value = "";
  } else {
    clearFile();
  }
  state.clientRequestId = newRequestId(); // 内容变 = 新意图 = 新键
  syncSourceModeView(mode);
  setComposerError(null);
  updateSendBtn();
}

function updateSendBtn() {
  const ready = sourceMode() === "upload" ? !!state.file : !!$("url-input").value.trim();
  const targetLanguage = selectedTargetLanguage();
  const langReady = !!targetLanguage && targetLanguage.length <= TARGET_LANGUAGE_MAX_CHARS;
  const briefReady = dialogueWriteMode() !== "generate" || !!$("dialogue-brief").value.trim();
  const hasInstruction = !!$("replacement-instruction").value.trim();
  const replacementReady = (state.replacementImages.length > 0) === hasInstruction;
  const replacementSizeReady = !state.minimalCreationCapability || state.replacementImages.every(
    (image) => image.size <= state.minimalCreationCapability.replacement.max_bytes,
  );
  const configReady = GENERATION_ASPECT_RATIOS.includes(selectedAspectRatio())
    && GENERATION_RESOLUTIONS.includes(selectedResolution())
    && !!state.minimalCreationCapability;
  $("send-btn").disabled = state.uploading || state.viewingSubmittedConfig
    || !state.minimalCreationCapability
    || !ready || !langReady || !briefReady || !replacementReady
    || !replacementSizeReady || !configReady;
}

function isVideoFile(file) {
  if (!file) return false;
  if (file.type && file.type.startsWith("video/")) return true;
  return /\.(mp4|mov|webm)$/i.test(file.name || "");
}

function pickFile(file) {
  if (state.viewingSubmittedConfig) return;
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

function isReferenceImageFile(file) {
  if (!file) return false;
  if (file.type) return REPLACEMENT_IMAGE_TYPES.includes(file.type);
  return /\.(jpe?g|png|webp)$/i.test(file.name || "");
}

function updateReplacementHint() {
  const imageCount = state.replacementImages.length;
  const submittedCount = state.submittedReplacements.length;
  const hasAnyImage = imageCount + submittedCount > 0;
  const hasInstruction = !!$("replacement-instruction").value.trim();
  const capability = state.minimalCreationCapability;
  const tooLarge = !!(capability && state.replacementImages.some(
    (image) => image.size > capability.replacement.max_bytes,
  ));
  const hint = $("replacement-hint");
  hint.classList.toggle("is-error", hasAnyImage !== hasInstruction || tooLarge);
  hint.textContent = tooLarge
    ? "参考图过大，请选择不超过 " + fmtBytes(capability.replacement.max_bytes) + " 的图片"
    : hasAnyImage !== hasInstruction ? "请同时提供参考图和替换说明"
    : submittedCount ? "该项目创建时已提交 " + submittedCount + " 张参考图；如需再次创建，请重新选择图片"
    : imageCount ? imageCount + " 张参考图和替换说明将在创建时一起提交"
      : "参考图与替换说明需要一起提供";
}

function revokeReplacementPreviewURLs() {
  state.replacementPreviewURLs.forEach((url) => {
    if (url) URL.revokeObjectURL(url);
  });
  state.replacementPreviewURLs = [];
}

function renderReplacementImages() {
  const submitted = state.submittedReplacements.length > 0;
  const images = submitted ? state.submittedReplacements : state.replacementImages;
  const list = $("replacement-image-list");
  list.textContent = "";
  list.hidden = images.length === 0;
  images.forEach((image, index) => {
    const chip = el("div", "reference-image-chip");
    const previewURL = state.replacementPreviewURLs[index];
    if (previewURL) {
      const preview = el("img");
      preview.src = previewURL;
      preview.alt = "参考图 " + (index + 1) + " 预览";
      chip.appendChild(preview);
    }
    chip.appendChild(el("span", null, "参考图 " + (index + 1) + " · " + image.filename));
    if (!submitted) {
      const remove = el("button", "icon-btn icon-btn-sm");
      remove.type = "button";
      remove.setAttribute("aria-label", "移除参考图 " + (index + 1));
      remove.appendChild(icon("i-close"));
      remove.addEventListener("click", () => clearReplacementImage(index));
      chip.appendChild(remove);
    }
    list.appendChild(chip);
  });
  const maxImages = state.minimalCreationCapability
    ? state.minimalCreationCapability.replacement.max_images : 4;
  $("replacement-select").hidden = submitted;
  $("replacement-select").textContent = images.length
    ? "添加参考图（" + images.length + "/" + maxImages + "）" : "上传参考图（最多 " + maxImages + " 张）";
  $("replacement-select").disabled = state.uploading || state.viewingSubmittedConfig
    || images.length >= maxImages;
}

function resetReplacementImageDisplay() {
  revokeReplacementPreviewURLs();
  state.replacementImages = [];
  state.replacementPreviewOwnerId = null;
  state.submittedReplacements = [];
  $("replacement-image-input").value = "";
  renderReplacementImages();
}

function clearReplacementImage(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.replacementImages.length) return;
  const [previewURL] = state.replacementPreviewURLs.splice(index, 1);
  if (previewURL) URL.revokeObjectURL(previewURL);
  state.replacementImages.splice(index, 1);
  $("replacement-image-input").value = "";
  state.clientRequestId = newRequestId();
  renderReplacementImages();
  updateReplacementHint();
  updateSendBtn();
}

function pickReplacementImages(files) {
  if (state.viewingSubmittedConfig) return;
  setComposerError(null);
  const images = Array.from(files || []);
  if (images.length === 0) return;
  const capability = state.minimalCreationCapability;
  const maxImages = capability ? capability.replacement.max_images : 4;
  if (state.replacementImages.length + images.length > maxImages) {
    setComposerError("最多可选择 " + maxImages + " 张参考图");
    $("replacement-image-input").value = "";
    return;
  }
  if (images.some((image) => !isReferenceImageFile(image))) {
    setComposerError("参考图仅支持 JPG、PNG 或 WebP");
    $("replacement-image-input").value = "";
    return;
  }
  if (capability && images.some((image) => image.size > capability.replacement.max_bytes)) {
    setComposerError("参考图过大，请选择不超过 " + fmtBytes(capability.replacement.max_bytes) + " 的图片");
    $("replacement-image-input").value = "";
    return;
  }
  state.submittedReplacements = [];
  state.replacementImages.push(...images);
  state.replacementPreviewURLs.push(...images.map((image) => URL.createObjectURL(image)));
  state.replacementPreviewOwnerId = null;
  state.clientRequestId = newRequestId();
  $("replacement-image-input").value = "";
  renderReplacementImages();
  updateReplacementHint();
  updateSendBtn();
}

function showUnavailableSource(message) {
  document.querySelectorAll('input[name="source-mode"]').forEach((control) => {
    control.checked = false;
  });
  state.file = null;
  $("file-input").value = "";
  $("file-chip").hidden = true;
  $("attach-btn").hidden = true;
  $("file-hint").hidden = true;
  $("url-row").hidden = false;
  $("url-input").required = false;
  $("url-input").value = "";
  $("url-input").placeholder = message;
}

function prepareComposerForProjectLoad() {
  $("add-subtitles").checked = false;
  state.viewingSubmittedConfig = true;
  showUnavailableSource("正在加载项目配置…");
  resetReplacementImageDisplay();
  $("replacement-instruction").value = "";
  $("dialogue-brief").value = "";
  document.querySelectorAll(
    'input[name="aspect-ratio"], input[name="resolution"], input[name="target-language-mode"], input[name="dialogue-write-mode"]'
  ).forEach((control) => { control.checked = false; });
  $("lang-input").value = "";
  setComposerError(null);
  syncDialogueModeControls();
  syncTargetLanguageControls();
  updateReplacementHint();
  setUploading(false);
}

async function loadSubmittedReplacementPreview(conversationId, previewUrl, index = null) {
  const safePath = creationInputReplacementPreviewPath(conversationId, previewUrl, null, index);
  if (!safePath) return;
  try {
    const response = await api(safePath);
    if (!response.ok) return;
    const objectUrl = URL.createObjectURL(await response.blob());
    if (state.currentId !== conversationId
        || !state.submittedReplacements[index === null ? 0 : index - 1]
        || state.submittedReplacements[index === null ? 0 : index - 1].preview_url !== previewUrl) {
      URL.revokeObjectURL(objectUrl);
      return;
    }
    const previewIndex = index === null ? 0 : index - 1;
    if (state.replacementPreviewURLs[previewIndex]) {
      URL.revokeObjectURL(state.replacementPreviewURLs[previewIndex]);
    }
    state.replacementPreviewURLs[previewIndex] = objectUrl;
    state.replacementPreviewOwnerId = conversationId;
    renderReplacementImages();
  } catch (error) {
    if (handleAuthError(error)) return;
    // 配置文字仍可展示；预览失败不改变已冻结配置。
  }
}

function creationInputReplacementPreviewPath(conversationId, previewUrl, origin = null, index = null) {
  if (!conversationId || typeof previewUrl !== "string" || !previewUrl.trim()) return null;
  const basePath = "/api/conversations/" + encodeURIComponent(conversationId) + "/creation-input/";
  const expectedPath = index === null
    ? basePath + "replacement-image"
    : Number.isInteger(index) && index >= 1 && index <= 4
      ? basePath + "replacement-images/" + index : null;
  if (!expectedPath) return null;
  const expectedOrigin = origin || (typeof window !== "undefined" && window.location
    ? window.location.origin : "");
  if (!expectedOrigin) return null;
  try {
    const base = new URL(expectedOrigin);
    const candidate = new URL(previewUrl, base);
    if (candidate.origin !== base.origin || candidate.pathname !== expectedPath
        || candidate.search || candidate.hash) return null;
    return expectedPath;
  } catch (_error) {
    return null;
  }
}

function applyProjectComposer(detail) {
  $("add-subtitles").checked = false;
  const memoryDraft = state.composerDrafts[detail.id] || null;
  const localDraft = memoryDraft || storedComposerDraft(detail.id);
  const model = projectComposerModel(detail, localDraft);
  const hasServerInput = Object.prototype.hasOwnProperty.call(detail, "creation_input");
  const serverInput = hasServerInput
    ? normalizeCreationInputSnapshot(detail.creation_input) : null;
  const localPreviewAllowed = !hasServerInput
    || !!(serverInput && serverInput.replacement_images.length);
  const preservedPreviews = localPreviewAllowed
    && state.replacementPreviewOwnerId === detail.id
    ? state.replacementPreviewURLs.slice() : [];
  if (preservedPreviews.length) state.replacementPreviewURLs = [];

  state.viewingSubmittedConfig = true;
  state.file = null;
  $("file-input").value = "";
  $("file-chip").hidden = true;
  $("url-input").value = "";
  $("url-input").placeholder = SOURCE_URL_PLACEHOLDER;
  resetReplacementImageDisplay();
  $("replacement-instruction").value = "";
  $("dialogue-brief").value = "";

  if (!model) {
    preservedPreviews.forEach((url) => URL.revokeObjectURL(url));
    showUnavailableSource("历史项目未保存来源信息");
    document.querySelectorAll(
      'input[name="aspect-ratio"], input[name="resolution"], input[name="target-language-mode"], input[name="dialogue-write-mode"]'
    ).forEach((control) => { control.checked = false; });
    $("lang-input").value = "";
    setComposerError("该历史项目未保存创建配置，无法恢复原选项");
    syncDialogueModeControls();
    syncTargetLanguageControls();
    updateReplacementHint();
    setUploading(false);
    return false;
  }

  const aspect = document.querySelector(`input[name="aspect-ratio"][value="${model.aspectRatio}"]`);
  const resolution = document.querySelector(`input[name="resolution"][value="${model.resolution}"]`);
  const language = document.querySelector(`input[name="target-language-mode"][value="${model.languageMode}"]`);
  const writeMode = document.querySelector(`input[name="dialogue-write-mode"][value="${model.dialogueMode}"]`);
  if (aspect) aspect.checked = true;
  if (resolution) resolution.checked = true;
  if (language) language.checked = true;
  if (writeMode) writeMode.checked = true;
  $("lang-input").value = model.targetLanguage;
  $("dialogue-brief").value = model.dialogueBrief;
  $("replacement-instruction").value = model.replacementInstruction;
  $("add-subtitles").checked = model.addSubtitles;
  setComposerError(null);

  if (model.source && model.source.mode === "upload") {
    syncSourceModeView("upload");
    showSourceFileSummary(model.source.filename, model.source.bytes);
  } else if (model.source && model.source.mode === "link") {
    syncSourceModeView("link");
    $("url-input").value = model.source.reference_url || "";
    if (model.source.reference_url === null) {
      $("url-input").placeholder = "链接来源已提交";
    }
  } else {
    showUnavailableSource("历史项目未保存来源信息");
  }

  if (model.replacementImages.length) {
    state.submittedReplacements = model.replacementImages;
    if (preservedPreviews.length) {
      state.replacementPreviewURLs = preservedPreviews;
      state.replacementPreviewOwnerId = detail.id;
    }
    renderReplacementImages();
    const indexedPreviews = hasServerInput && detail.creation_input
      && Object.prototype.hasOwnProperty.call(detail.creation_input, "replacement_images");
    if (!preservedPreviews.length) {
      model.replacementImages.forEach((image, index) => {
        if (image.preview_url) {
          void loadSubmittedReplacementPreview(
            detail.id, image.preview_url, indexedPreviews ? index + 1 : null,
          );
        }
      });
    }
  } else {
    preservedPreviews.forEach((url) => URL.revokeObjectURL(url));
  }

  state.clientRequestId = newRequestId();
  syncDialogueModeControls();
  syncTargetLanguageControls();
  updateReplacementHint();
  setUploading(false);
  return true;
}

function resetComposerForNewProject() {
  $("add-subtitles").checked = false;
  state.viewingSubmittedConfig = false;
  state.file = null;
  $("file-input").value = "";
  $("file-chip").hidden = true;
  $("url-input").value = "";
  $("url-input").placeholder = SOURCE_URL_PLACEHOLDER;
  resetReplacementImageDisplay();
  $("replacement-instruction").value = "";
  $("dialogue-brief").value = "";
  document.querySelector('input[name="source-mode"][value="link"]').checked = true;
  document.querySelector('input[name="aspect-ratio"][value="9:16"]').checked = true;
  document.querySelector('input[name="resolution"][value="768p"]').checked = true;
  document.querySelector('input[name="target-language-mode"][value="same"]').checked = true;
  document.querySelector('input[name="dialogue-write-mode"][value="auto_rewrite"]').checked = true;
  $("lang-input").value = "";
  state.clientRequestId = newRequestId();
  syncSourceModeView("link");
  syncDialogueModeControls();
  syncTargetLanguageControls();
  setComposerError(null);
  updateReplacementHint();
  setUploading(false);
}

function setUploading(on) {
  state.uploading = on;
  const controlsDisabled = on || state.viewingSubmittedConfig;
  $("add-subtitles").disabled = controlsDisabled;
  $("composer-toggle").disabled = on;
  $("attach-btn").disabled = controlsDisabled;
  $("file-input").disabled = controlsDisabled;
  $("url-input").disabled = controlsDisabled;
  $("file-remove").disabled = controlsDisabled;
  $("replacement-select").disabled = controlsDisabled;
  $("replacement-image-input").disabled = controlsDisabled;
  document.querySelectorAll("#replacement-image-list .icon-btn").forEach((button) => {
    button.disabled = controlsDisabled;
  });
  $("replacement-instruction").disabled = controlsDisabled;
  document.querySelectorAll('input[name="source-mode"]').forEach((r) => {
    r.disabled = controlsDisabled;
  });
  document.querySelectorAll('input[name="target-language-mode"]').forEach((r) => {
    r.disabled = controlsDisabled;
  });
  document.querySelectorAll('input[name="dialogue-write-mode"]').forEach((r) => {
    r.disabled = controlsDisabled;
  });
  document.querySelectorAll('input[name="aspect-ratio"], input[name="resolution"]').forEach((r) => {
    r.disabled = controlsDisabled;
  });
  syncDialogueModeControls();
  syncTargetLanguageControls();
  updateSendBtn();
  if (!on) $("upload-progress").hidden = true;
}

function uploadConversation({
  file, url, requestId, generationRequest, replacementImages = null, replacementImage = null, capability,
}, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", capability.endpoint);
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
    fd.append("client_request_id", requestId);
    fd.append(capability.request_field, JSON.stringify(generationRequest));
    const images = Array.isArray(replacementImages)
      ? replacementImages : replacementImage ? [replacementImage] : [];
    images.forEach((image) => {
      fd.append(capability.replacement_image_field, image, image.name);
    });
    xhr.send(fd);
  });
}

async function handleSend(event) {
  event.preventDefault();
  if (state.uploading || state.viewingSubmittedConfig) return;
  const mode = sourceMode();
  const file = mode === "upload" ? state.file : null;
  const url = mode === "link" ? $("url-input").value.trim() : "";
  const capability = state.minimalCreationCapability;
  let generationRequest;
  if (!file && !url) {
    setComposerError(mode === "upload" ? "请先选择视频文件" : "请先粘贴视频链接");
    return;
  }
  if (!capability) {
    setComposerError("生成服务尚未升级，暂时无法创建新项目");
    return;
  }
  try {
    generationRequest = buildMinimalGenerationRequest({
      aspectRatio: selectedAspectRatio(),
      resolution: selectedResolution(),
      dialogueMode: dialogueWriteMode(),
      dialogueBrief: $("dialogue-brief").value,
      targetLanguage: selectedTargetLanguage(),
      replacementImageCount: state.replacementImages.length,
      replacementInstruction: $("replacement-instruction").value,
      addSubtitles: $("add-subtitles").checked,
    }, capability);
  } catch (err) {
    setComposerError(err.message);
    return;
  }

  setComposerError(null);
  setUploading(true);
  const submitEpoch = state.authEpoch;
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
        file, url, requestId: state.clientRequestId,
        generationRequest,
        replacementImages: state.replacementImages,
        capability,
      },
      (ratio) => {
        if (url) return;
        const pct = Math.round(ratio * 100);
        fill.style.width = pct + "%";
        label.textContent = pct >= 100 ? "上传完成，等待处理…" : "正在上传 " + pct + "%";
      }
    );
    if (submitEpoch !== state.authEpoch) return;
    // 成功：保留本次配置供用户在可折叠窗口中查看，只轮换下一次提交的幂等键。
    if (created && created.id) {
      rememberSubmittedComposer(
        created.id,
        created.effective_request || generationRequest,
        { file, url, replacementImages: state.replacementImages },
        created,
      );
      // 响应成功即冻结，不等待列表或详情请求，避免同一付费任务被再次提交。
      state.viewingSubmittedConfig = true;
      state.file = null;
      state.replacementImages = [];
      $("file-input").value = "";
      $("replacement-image-input").value = "";
    }
    state.clientRequestId = newRequestId();
    setUploading(false);
    await refreshList(false);
    if (created && created.id) {
      selectConversation(created.id);
    }
  } catch (err) {
    if (submitEpoch !== state.authEpoch) return;
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
    state.authEpoch += 1;
    state.token = null;
    localStorage.removeItem(TOKEN_KEY);
    clearComposerSnapshots();
    resetComposerForNewProject();
    clearStream();
    showLogin(null);
  });

  $("menu-btn").addEventListener("click", openDrawer);
  $("drawer-backdrop").addEventListener("click", closeDrawer);
  $("composer-toggle").addEventListener("click", () => {
    setComposerExpanded($("composer-toggle").getAttribute("aria-expanded") !== "true");
  });

  $("new-chat-btn").addEventListener("click", async () => {
    if (state.uploading) return;
    state.currentId = null;
    state.detail = null;
    resetComposerForNewProject();
    renderList();
    renderEmptyHero();
    closeDrawer();
    $(sourceMode() === "upload" ? "attach-btn" : "url-input").focus();
  });

  $("attach-btn").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => {
    pickFile(e.target.files && e.target.files[0]);
  });
  $("file-remove").addEventListener("click", clearFile);
  $("replacement-select").addEventListener("click", () => $("replacement-image-input").click());
  $("replacement-image-input").addEventListener("change", (event) => {
    pickReplacementImages(event.target.files);
  });

  // 来源 radio 互斥切换
  document.querySelectorAll('input[name="source-mode"]').forEach((radio) => {
    radio.addEventListener("change", () => setSourceMode(radio.value));
  });

  document.querySelectorAll('input[name="aspect-ratio"], input[name="resolution"]').forEach((control) => {
    control.addEventListener("change", () => {
      state.clientRequestId = newRequestId();
      setComposerError(null);
      updateSendBtn();
    });
  });
  document.querySelectorAll('input[name="target-language-mode"]').forEach((control) => {
    control.addEventListener("change", () => {
      state.clientRequestId = newRequestId();
      setComposerError(null);
      syncTargetLanguageControls();
      updateSendBtn();
      if (control.value === "other" && control.checked) $("lang-input").focus();
    });
  });
  document.querySelectorAll('input[name="dialogue-write-mode"]').forEach((control) => {
    control.addEventListener("change", () => {
      state.clientRequestId = newRequestId();
      setComposerError(null);
      syncDialogueModeControls();
      updateSendBtn();
      if (control.value === "generate" && control.checked) $("dialogue-brief").focus();
    });
  });
  $("add-subtitles").addEventListener("change", () => {
    state.clientRequestId = newRequestId();
    setComposerError(null);
  });
  $("dialogue-brief").addEventListener("input", () => {
    state.clientRequestId = newRequestId();
    setComposerError(null);
    updateSendBtn();
  });
  $("lang-input").addEventListener("input", () => {
    state.clientRequestId = newRequestId(); // 内容变 = 新意图 = 新键
    setComposerError(null);
    updateSendBtn();
  });
  $("url-input").addEventListener("input", () => {
    state.clientRequestId = newRequestId();
    setComposerError(null);
    updateSendBtn();
  });
  $("replacement-instruction").addEventListener("input", () => {
    state.clientRequestId = newRequestId();
    setComposerError(null);
    updateReplacementHint();
    updateSendBtn();
  });

  const composer = $("composer");
  composer.addEventListener("submit", handleSend);
  composer.addEventListener("dragenter", (e) => {
    e.preventDefault();
    if (!state.uploading && !state.viewingSubmittedConfig && sourceMode() === "upload") {
      composer.classList.add("drag-over");
    }
  });
  composer.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (!state.uploading && !state.viewingSubmittedConfig && sourceMode() === "upload") {
      composer.classList.add("drag-over");
    }
  });
  composer.addEventListener("dragleave", (e) => {
    if (!composer.contains(e.relatedTarget)) composer.classList.remove("drag-over");
  });
  composer.addEventListener("drop", (e) => {
    e.preventDefault();
    composer.classList.remove("drag-over");
    if (state.uploading || state.viewingSubmittedConfig || sourceMode() !== "upload") return;
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    pickFile(file);
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 768) closeDrawer();
  });
}

function boot() {
  state.clientRequestId = newRequestId();
  bindEvents();
  setSourceMode(sourceMode());
  syncDialogueModeControls();
  syncTargetLanguageControls();
  updateReplacementHint();
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
    apiErrorFromPayload,
    authoritativeSegmentKeyframePaths,
    buildMinimalGenerationRequest,
    clearStream,
    conversationBadge,
    conversationThumbnailPath,
    creationInputSnapshot,
    creationInputReplacementPreviewPath,
    detailSignature,
    formatDuration,
    formatElapsed,
    mergeConversationList,
    normalizeCreationInputSnapshot,
    normalizeMinimalCreationCapability,
    projectComposerModel,
    projectPhaseLabel,
    projectProgressModel,
    releaseHistoryThumbnails,
    resolveCreationInputSnapshot,
    resolveTargetLanguage,
    resumeFailedProject,
    runSingleFlightPollCycle,
    safeErrorSummary,
    setComposerExpanded,
    shortId,
    shouldPollDetail,
    showActionError,
    syncConversationDetail,
  };
}

if (typeof document !== "undefined") boot();
