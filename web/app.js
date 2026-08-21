/* 视频工作室 — ChatGPT 式单页前端
 * 同源 API：/api/*，共享口令 Bearer 鉴权；文件直链也需鉴权，故一律 fetch(blob) → ObjectURL。
 */
"use strict";

/* ===== 常量与状态 ===== */
const TOKEN_KEY = "cvs_token";
const POLL_MS = 2000;

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
  detailSeq: 0,        // 防止过期响应覆盖新渲染
  objectURLs: [],      // 当前 stream 渲染产生的 blob URL，重渲染前统一 revoke
  ppDetail: null,      // 后处理弹窗对应的会话详情
  ppAskDismissed: {},  // cid → true：后处理入口消息已点「否」（会话内记忆，重渲染不复活）
  generationDrafts: {}, // cid → 最终视频表单草稿；轮询重渲染时保留用户输入
  generationSubmitting: {}, // cid → true：本页已有 /submit 请求在途，阻止重复提交
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

function longVideoContract(detail) {
  const isLong = Number(detail && detail.duration_s) > 10;
  const segmentCount = Number.isInteger(detail && detail.segment_count)
    && detail.segment_count > 0 ? detail.segment_count : null;
  const planReceipt = typeof (detail && detail.plan_receipt) === "string"
    && /^[0-9a-f]{64}$/.test(detail.plan_receipt) ? detail.plan_receipt : null;
  return {
    isLong,
    ready: !isLong || (segmentCount !== null && planReceipt !== null),
    segmentCount,
    planReceipt,
  };
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
  };
  if (input.isLong) body.expected_plan_receipt = input.planReceipt;
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

  const body = {
    confirm: true,
    client_request_id: generation.client_request_id,
    dialogue_mode: dialogue.mode,
    fit_mode: detail.fit_mode,
  };
  const longContract = longVideoContract(detail);
  if (longContract.isLong) {
    if (!longContract.ready) throw new Error("长视频生成计划尚未就绪，请刷新后重试");
    if (dialogue.mode !== "auto" && dialogue.mode !== "none") {
      throw new Error("长视频既有任务台词模式无效");
    }
    body.expected_plan_receipt = longContract.planReceipt;
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
  return {
    confirm: true,
    client_request_id: requestId,
    dialogue_mode: dialogue.mode,
    fit_mode: detail.fit_mode,
    expected_plan_receipt: longContract.planReceipt,
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
    isLong: true,
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
  state.currentId = null;
  state.detail = null;
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
  renderEmptyHero();
  refreshList(true);
}

/* ===== 侧栏会话列表 ===== */
async function refreshList(autoSelect) {
  try {
    const list = await apiJSON("/api/conversations");
    state.conversations = Array.isArray(list) ? list : [];
    renderList();
    if (autoSelect && state.conversations.length > 0 && !state.currentId) {
      selectConversation(state.conversations[0].id);
    }
  } catch (err) {
    if (handleAuthError(err)) return;
    renderListError(err.message);
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
    item.appendChild(el("span", "conv-title", c.title || "未命名会话"));
    const meta = el("span", "conv-meta");
    const badge = el("span", "badge " + (c.status || "queued"), STATUS_TEXT[c.status] || c.status || "");
    meta.appendChild(badge);
    meta.appendChild(el("span", "conv-time", fmtTime(c.created_at)));
    item.appendChild(meta);
    item.addEventListener("click", () => {
      selectConversation(c.id);
      closeDrawer();
    });
    nav.appendChild(item);
  }
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

/* ===== Stream 渲染 ===== */
function clearStream() {
  revokeURLs();
  $("stream").textContent = "";
}

function renderEmptyHero() {
  stopPolling();
  clearStream();
  $("main-title").textContent = "视频工作室";

  const inner = el("div", "stream-inner");
  const hero = el("div", "empty-hero");
  const iconBox = el("div", "empty-icon");
  iconBox.appendChild(icon("i-film"));
  hero.appendChild(iconBox);
  hero.appendChild(el("h2", null, "上传参考视频，生成复刻配方"));
  hero.appendChild(el("p", "empty-sub", "AI 会抽取关键帧并准备 H3 视频生成"));
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
  body.appendChild(el("p", "fail-title", "处理失败"));
  body.appendChild(el("p", "fail-msg", detail.error || "后端未返回具体原因"));
  body.appendChild(el("p", "fail-tip", "输入准备未完成；请保留上方错误信息后重试，若重复出现请联系管理员"));
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

function renderResults(detail) {
  const frag = document.createDocumentFragment();

  const headRow = el("div", "msg-row");
  headRow.appendChild(assistantHead(detail.updated_at));
  const doneCard = el("div", "activity-card");
  doneCard.appendChild(el("p", "ac-title", "处理完成"));
  doneCard.appendChild(el("p", "ac-sub", "关键帧与提示词已生成，可直接复制使用"));
  headRow.appendChild(doneCard);
  frag.appendChild(headRow);

  // 原始视频（上传即存在，与生成物明确分区）
  if (detail.has_source) {
    frag.appendChild(videoSection(detail, "source.mp4", "原始视频", "上传的源素材"));
  }

  // 多段模式：逐段渲染「第 N 段」卡片；单段模式保持现有逻辑
  const segments = Array.isArray(detail.segments) ? detail.segments : [];
  if (segments.length > 0) {
    frag.appendChild(renderSegments(detail));
  } else {
    const names = Array.isArray(detail.keyframes) ? detail.keyframes : [];
    if (names.length > 0) {
      frag.appendChild(keyframesSection(detail));
    }
    const sourcePrompt = detail.source_prompt || detail.prompt;
    if (sourcePrompt) {
      const sec = el("section", "res-section");
      sec.appendChild(el("h3", "res-h3", "H3 源提示词"));
      sec.appendChild(renderSourcePromptCard(detail, sourcePrompt));
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
  if (!draft) {
    draft = {
      dialogueMode: "auto",
      editLinesText: formatDialogueLines(detail.dialogue),
      customLinesText: "",
      fitMode: "",
      receiptVersion: detail.receipt_version,
    };
    state.generationDrafts[detail.id] = draft;
  } else if (draft.receiptVersion !== detail.receipt_version && !draft.editTouched) {
    draft.editLinesText = formatDialogueLines(detail.dialogue);
    draft.receiptVersion = detail.receipt_version;
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
  const generation = detail.generation || {};
  const action = generationAction(generation.status, generation.stage);
  if (!canOperate(detail) || state.generationSubmitting[detail.id]
      || !["new", "retry", "retry_stitch"].includes(action)) return;
  const draft = generationDraft(detail);
  const longContract = longVideoContract(detail);
  const errorBox = card.querySelector(".generation-form-error");
  let body;
  try {
    if (action === "retry_stitch") {
      body = buildStitchRetryPayload(detail);
    } else if (longContract.isLong && action === "retry") {
      body = buildLongRetryPayload(detail, newRequestId());
    } else {
      body = buildSubmitPayload({
        clientRequestId: newRequestId(),
        dialogueMode: draft.dialogueMode,
        linesText: draft.dialogueMode === "edit" ? draft.editLinesText : draft.customLinesText,
        fitRequired: detail.fit_required === true,
        fitMode: draft.fitMode,
        isLong: longContract.isLong,
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
  if (stage === "h3") return "H3 子任务生成";
  if (stage === "stitch") return "视频拼接";
  if (stage === "stitching") return "视频拼接";
  return stage ? String(stage) : "等待开始";
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
    const meta = el("span", null, "chain：" + (segment.chain_id || "-")
      + " · join：" + (segment.join_mode || "-")
      + " · 尝试 " + (segment.attempt || 0));
    item.appendChild(meta);
    if (segment.error) item.appendChild(el("span", "generation-segment-error", segment.error));
    list.appendChild(item);
  });
  progress.appendChild(list);
  card.appendChild(progress);
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
    await apiJSON("/api/conversations/" + encodeURIComponent(detail.id) + "/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    accepted = true;
    await loadDetail(detail.id, true);
    // 极短暂的详情落盘延迟也不能开放第二次 POST；保持禁用并继续 GET 轮询。
    if (state.generationSubmitting[detail.id]) startPolling(detail.id);
  } catch (error) {
    if (handleAuthError(error)) return;
    errorBox.textContent = error.message;
    errorBox.hidden = false;
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
        button.textContent = action === "retry_stitch" ? "重试拼接"
          : action === "retry" ? "重试生成" : "生成最终视频";
      }
    }
  }
}

/* 最终视频区：H3 生成参数、异步状态、失败后的显式重试和历史成片都在同一卡片。 */
function renderFinalSection(detail) {
  const generation = detail.generation || { status: null, error: null, attempt: null };
  const showPublishedVideo = detail.has_video === true;
  const showStitchRecovery = generation.status === "failed" && generation.stage === "stitch";
  const published = document.createDocumentFragment();
  if (showPublishedVideo) {
    published.appendChild(videoSection(detail, "generated.mp4", "最终视频", "H3 生成成片"));
  }
  if (showPublishedVideo && !showStitchRecovery) return published;

  const sec = el("section", "res-section");
  published.appendChild(sec);
  const card = el("div", "final-card");
  card.appendChild(el("h3", "res-h3", "最终视频 · H3"));
  appendGenerationProgress(card, generation);

  if (generation.status === "queued" || generation.status === "running") {
    const status = el("div", "generation-status is-running");
    status.appendChild(el("strong", null,
      generation.status === "queued" ? "H3 已排队" : "H3 正在生成"));
    status.appendChild(el("span", null, generation.attempt ? "第 " + generation.attempt + " 次尝试" : "请稍候，页面会自动更新"));
    card.appendChild(status);
  } else if (generation.status === "failed" || generation.status === "submission_unknown") {
    const status = el("div", "generation-status is-error");
    const failedTitle = generation.stage === "stitch" ? "视频拼接失败" : "H3 生成失败";
    status.appendChild(el("strong", null,
      generation.status === "submission_unknown" ? "提交结果未知" : failedTitle));
    const errorText = generation.error || "后端未返回具体原因";
    status.appendChild(el("span", null, errorText));
    card.appendChild(status);
  } else if (generation.status === "resume_required") {
    const status = el("div", "generation-status is-resume");
    status.appendChild(el("strong", null, "既有 H3 任务等待继续"));
    status.appendChild(el("span", null, generation.error || "任务已保存，可从原进度继续"));
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
      "本次新增 " + retryContract.paidTaskCount + " 个付费 H3 子任务"));
    const noticeText = retryContract.action === "retry_stitch"
      ? "全部分段成片已复用，本次只在本地重试拼接。"
      : retryContract.action === "retry"
        ? "跨段连续性为 best effort；成功段复用，失败时只重做失败段及同链下游。"
        : "跨段连续性为 best effort；首次生成覆盖全部逐段冻结输入。";
    notice.appendChild(el("p", null, noticeText));
    card.appendChild(notice);
  }

  if (generationAction(generation.status, generation.stage) === "resume") {
    const locked = el("div", "resume-lock");
    locked.appendChild(el("strong", null, "继续既有 H3 任务"));
    locked.appendChild(el("p", null,
      "将使用已保存的请求标识和冻结输入继续查询既有 H3 任务。"));
    card.appendChild(locked);
    const errorBox = el("p", "form-error generation-form-error");
    errorBox.hidden = true;
    card.appendChild(errorBox);
    const row = el("div", "final-row");
    const button = el("button", "btn btn-primary generation-submit", "继续 H3");
    button.type = "button";
    button.addEventListener("click", () => {
      resumeGeneration(detail, card);
    });
    row.appendChild(button);
    row.appendChild(el("p", "final-caption", "继续原任务，不创建新的 H3 attempt。"));
    card.appendChild(row);
    if (state.generationSubmitting[detail.id]) setGenerationCardBusy(card, true);
    sec.appendChild(card);
    return published;
  }


  if (longContract.isLong && (retryContract.action === "retry"
      || retryContract.action === "retry_stitch")) {
    const stitchOnly = retryContract.action === "retry_stitch";
    const locked = el("div", "resume-lock");
    locked.appendChild(el("strong", null, stitchOnly ? "重试本地拼接" : "重试失败的 H3 分段"));
    locked.appendChild(el("p", null, stitchOnly
      ? "复用原请求标识和全部成功分段，不创建新的付费 H3 子任务。"
      : "使用新的请求标识和逐段冻结输入；成功段复用，只重做失败段及同链下游。"));
    card.appendChild(locked);
    const errorBox = el("p", "form-error generation-form-error");
    errorBox.hidden = true;
    card.appendChild(errorBox);
    const row = el("div", "final-row");
    const label = stitchOnly ? "重试拼接" : "重试生成";
    const button = el("button", "btn btn-primary generation-submit", label);
    button.type = "button";
    button.addEventListener("click", () => submitGeneration(detail, card));
    row.appendChild(button);
    row.appendChild(el("p", "final-caption", stitchOnly
      ? "本次新增 0 个付费 H3 子任务。"
      : "本次新增 " + retryContract.paidTaskCount + " 个付费 H3 子任务。"));
    card.appendChild(row);
    if (state.generationSubmitting[detail.id]) setGenerationCardBusy(card, true);
    sec.appendChild(card);
    return published;
  }

  const draft = generationDraft(detail);
  const busy = generation.status === "queued" || generation.status === "running"
    || !!state.generationSubmitting[detail.id];
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

  if (detail.fit_required === true) {
    const fitField = el("fieldset", "final-field");
    fitField.appendChild(el("legend", null, "源画幅需要适配"));
    const fitChoices = el("div", "final-choices");
    for (const [value, label] of [["crop", "裁切画面"], ["pad", "留边完整展示"]]) {
      const item = choice("fit-" + detail.id, value, label, draft.fitMode === value);
      item.querySelector("input").addEventListener("change", () => { draft.fitMode = value; });
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
  const buttonLabel = generation.status === "failed" ? "重试生成" : "生成最终视频";
  const button = el("button", "btn btn-primary generation-submit", buttonLabel);
  button.type = "button";
  button.addEventListener("click", () => submitGeneration(detail, card));
  row.appendChild(button);
  row.appendChild(el("p", "final-caption generation-mode-caption", busy
    ? "正在等待 H3 生成结果"
    : "H3 源提示词将直接提交生成"));
  card.appendChild(row);
  if (busy) setGenerationCardBusy(card, true);
  sec.appendChild(card);
  return published;
}

/* 关键帧网格：blob 化加载 + 点击放大灯箱（单段/多段/优化后共用；pathPrefix 即 files 白名单路径） */
function kfGrid(detail, names, pathPrefix, altPrefix) {
  const grid = el("div", "kf-grid");
  for (const name of names) {
    const fig = el("figure", "kf-card shimmer");
    const img = el("img");
    img.alt = altPrefix + name;
    fig.appendChild(img);
    grid.appendChild(fig);
    apiBlobURL("/api/conversations/" + detail.id + "/files/" + pathPrefix + "/" + encodeURIComponent(name))
      .then((url) => {
        img.src = url;
        img.addEventListener("load", () => {
          fig.classList.remove("shimmer");
          fig.classList.add("is-loaded");
        }, { once: true });
        img.addEventListener("click", () => openLightbox(img.src, img.alt));
      })
      .catch(() => {
        fig.classList.remove("shimmer");
        fig.appendChild(el("div", "kf-err", "加载失败"));
      });
  }
  return grid;
}

function sourcePromptEditable(detail) {
  return canOperate(detail)
    && (!detail.generation || detail.generation.status === null)
    && typeof detail.source_prompt_sha256 === "string"
    && /^[0-9a-f]{64}$/.test(detail.source_prompt_sha256);
}

function renderSourcePromptCard(detail, text) {
  const editable = sourcePromptEditable(detail);
  const editButton = el("button", "copy-btn edit-prompt-btn", "修改提示词");
  editButton.type = "button";
  const card = promptCard(text, editable ? [editButton] : []);
  if (!editable) return card;

  const output = card.querySelector(".prompt-text");
  const textarea = el("textarea", "dialogue-textarea prompt-editor");
  textarea.rows = 14;
  textarea.value = text;
  textarea.hidden = true;
  textarea.setAttribute("aria-label", "修改 H3 源提示词");
  card.appendChild(textarea);
  const error = el("p", "form-error");
  error.hidden = true;
  card.appendChild(error);
  const controls = el("div", "final-row prompt-edit-actions");
  controls.hidden = true;
  const save = el("button", "btn btn-primary", "保存提示词");
  save.type = "button";
  const cancel = el("button", "btn", "取消");
  cancel.type = "button";
  controls.appendChild(save);
  controls.appendChild(cancel);
  card.appendChild(controls);

  const showEditor = (shown) => {
    output.hidden = shown;
    textarea.hidden = !shown;
    controls.hidden = !shown;
    editButton.hidden = shown;
    error.hidden = true;
  };
  editButton.addEventListener("click", () => showEditor(true));
  cancel.addEventListener("click", () => {
    textarea.value = output.textContent;
    showEditor(false);
  });
  save.addEventListener("click", async () => {
    save.disabled = true;
    cancel.disabled = true;
    error.hidden = true;
    try {
      const payload = await apiJSON(
        "/api/conversations/" + encodeURIComponent(detail.id) + "/prompt",
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            confirm: true,
            expected_sha256: detail.source_prompt_sha256,
            prompt: textarea.value,
          }),
        },
      );
      if (!payload || typeof payload.prompt !== "string"
          || typeof payload.final_prompt !== "string"
          || typeof payload.sha256 !== "string"
          || !/^[0-9a-f]{64}$/.test(payload.sha256)) {
        throw new Error("源提示词保存响应校验失败");
      }
      output.textContent = payload.prompt;
      detail.source_prompt = payload.prompt;
      detail.source_prompt_sha256 = payload.sha256;
      detail.prompt = payload.final_prompt;
      showEditor(false);
      await loadDetail(detail.id, true);
    } catch (saveError) {
      if (handleAuthError(saveError)) return;
      error.textContent = "保存失败：" + saveError.message;
      error.hidden = false;
    } finally {
      save.disabled = false;
      cancel.disabled = false;
    }
  });
  return card;
}

/* prompt 卡片（复制按钮 + 全文；源提示词、IR、单段/多段共用） */
function promptCard(text, actions = []) {
  const card = el("div", "prompt-card");
  const head = el("div", "prompt-head");
  head.appendChild(el("span", "prompt-hint", "用于 H3 视频生成"));
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
  sec.appendChild(kfGrid(detail, names, "keyframes", "关键帧 "));
  return sec;
}

/* 多段模式：逐段「第 N 段」卡片（段关键帧 grid + 段提示词 + 段台词） */
function renderSegments(detail) {
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
    const names = Array.isArray(seg.keyframes) ? seg.keyframes : [];
    if (names.length) {
      card.appendChild(kfGrid(detail, names, "segments/" + n + "/work/keyframes", "第 " + n + " 段关键帧 "));
    }
    if (seg.prompt) {
      card.appendChild(el("h4", "res-sub", "逐段冻结的 H3 提示词"));
      card.appendChild(promptCard(seg.prompt));
    }
    if (Array.isArray(seg.lines) && seg.lines.length) {
      const lines = el("div", "lines-card");
      lines.appendChild(el("h4", "res-sub", "段台词"));
      const ul = el("ul", "lines-list");
      for (const line of seg.lines) {
        ul.appendChild(el("li", null, line));
      }
      lines.appendChild(ul);
      card.appendChild(lines);
    }
    frag.appendChild(card);
  }
  return frag;
}

/* 关键帧放大查看：点击开、点任意处或 Esc 关 */
let lightboxEl = null;

function openLightbox(src, alt) {
  if (!lightboxEl) {
    lightboxEl = el("div", "lightbox");
    lightboxEl.setAttribute("role", "dialog");
    lightboxEl.setAttribute("aria-label", "查看大图");
    lightboxEl.appendChild(el("img"));
    lightboxEl.addEventListener("click", closeLightbox);
    document.body.appendChild(lightboxEl);
  }
  const img = lightboxEl.querySelector("img");
  img.src = src;
  img.alt = alt || "";
  lightboxEl.classList.add("is-open");
  document.addEventListener("keydown", onLightboxKey);
}

function closeLightbox() {
  if (!lightboxEl) return;
  lightboxEl.classList.remove("is-open");
  document.removeEventListener("keydown", onLightboxKey);
}

function onLightboxKey(e) {
  if (e.key === "Escape") closeLightbox();
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
  const options = { remove_subtitle: false, remove_brand: false };
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
  document.querySelectorAll('#pp-form input[name="opt"]').forEach((c) => {
    c.checked = last ? last[c.value] === true : true; // 已运行过 → 预填上次选项（后端锁定）；首次 → 默认全选
  });
  $("pp-lock-hint").hidden = !last;
  $("pp-error").hidden = true;
  updatePpConfirm();
  $("pp-dialog").showModal();
}

function closePostprocessModal() {
  $("pp-dialog").close();
  state.ppDetail = null;
}

async function submitPostprocess(event) {
  event.preventDefault();
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
    if (err.message.includes("options changed since last run")) {
      // 后端锁定上次选项：回填并提示直接确认，避免反复踩 409
      const last = detail.postprocess && detail.postprocess.options;
      document.querySelectorAll('#pp-form input[name="opt"]').forEach((c) => {
        c.checked = last ? last[c.value] === true : true;
      });
      $("pp-lock-hint").hidden = false;
      errEl.textContent = "选项与上次不一致，已按上次选项预填，请直接确认";
    } else {
      errEl.textContent = err.message;
    }
    errEl.hidden = false;
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
function ppFramesSection(detail, frames) {
  const wrap = el("div");
  const segs = (Array.isArray(detail.segments) && detail.segments.length > 0)
    ? detail.segments
    : [{ index: null }]; // 单段：帧名为裸文件名，路径前缀为空
  for (const seg of segs) {
    const n = seg.index;
    const prefix = n == null ? "" : "segments/" + n + "/work/postprocessed/";
    const own = frames.filter((f) => f.startsWith(prefix)).map((f) => f.slice(prefix.length));
    if (!own.length) continue;
    const title = n == null ? "优化后" : "第 " + n + " 段优化后";
    const pathPrefix = n == null ? "postprocessed" : "segments/" + n + "/work/postprocessed";
    const block = el("div", "pp-seg");
    block.appendChild(el("h4", "res-sub", title));
    block.appendChild(kfGrid(detail, own, pathPrefix, title + " "));
    wrap.appendChild(block);
  }
  return wrap;
}

/* 后处理目标帧总数：多段 = 各段 keyframes 之和；单段 = detail.keyframes 长度 */
function ppTotalFrames(detail) {
  const segments = Array.isArray(detail.segments) ? detail.segments : [];
  if (segments.length > 0) {
    return segments.reduce((sum, seg) => sum + (Array.isArray(seg.keyframes) ? seg.keyframes.length : 0), 0);
  }
  return Array.isArray(detail.keyframes) ? detail.keyframes.length : 0;
}

/* 助手消息：running 进行中卡 / done 优化后结果 / failed 错误卡（动态区，轮询期间单独重渲染） */
function renderPpAssistant(detail, pp) {
  const row = el("div", "msg-row");
  row.appendChild(assistantHead(detail.updated_at));
  if (pp.status === "running") {
    const card = el("div", "activity-card");
    card.appendChild(el("p", "ac-title", "正在优化素材…"));
    // 实时进度：n = postprocess.frames 已完成数（后端逐帧写回）；m = 目标帧总数
    const total = ppTotalFrames(detail);
    if (total > 0) {
      const done = Array.isArray(pp.frames) ? pp.frames.length : 0;
      card.appendChild(el("p", "ac-sub", `已完成 ${done}/${total} 帧（每帧约需 1 分钟）`));
    }
    const track = el("div", "progress-track");
    track.appendChild(el("div", "progress-fill"));
    card.appendChild(track);
    row.appendChild(card);
  } else if (pp.status === "failed") {
    const card = el("div", "fail-card");
    card.appendChild(icon("i-alert", "ic-danger"));
    const body = el("div");
    body.appendChild(el("p", "fail-title", "后处理失败"));
    body.appendChild(el("p", "fail-msg", pp.error || "后端未返回具体原因"));
    if (Array.isArray(pp.frames) && pp.frames.length) {
      body.appendChild(el("p", "fail-tip", "已成功优化的帧保留"));
    }
    card.appendChild(body);
    row.appendChild(card);
  } else if (pp.status === "done") {
    const frames = Array.isArray(pp.frames) ? pp.frames : [];
    const card = el("div", "activity-card");
    if (frames.length) {
      card.appendChild(ppFramesSection(detail, frames));
    } else {
      card.appendChild(el("p", "ac-title", "后处理完成"));
      card.appendChild(el("p", "ac-sub", "所有目标帧均已处理"));
    }
    row.appendChild(card);
  }
  return row;
}

/* 后处理入口消息：postprocess 未做（或 failed 可重试）且接口开放时，结果区末尾提问。
   点「是」打开弹窗（消息流照旧）；点「否」原位标记已结束，不再弹窗（会话内记忆）。
   running/done 时不显示——renderPpChat 的进行中卡/结果卡接管。 */
function renderPpAsk(detail) {
  const pp = detail.postprocess || {};
  if (!detail.postprocess_enabled || !canOperate(detail)) return null;
  if (pp.status === "running" || pp.status === "done") return null;
  const row = el("div", "msg-row");
  row.appendChild(assistantHead(detail.updated_at));
  const card = el("div", "activity-card pp-ask-card");
  card.appendChild(el("p", "pp-ask-text", "是否优化素材？"));
  if (state.ppAskDismissed[detail.id]) {
    card.appendChild(el("p", "pp-ask-ended", "已跳过优化，素材保持原样"));
  } else {
    const actions = el("div", "pp-ask-actions");
    const yes = el("button", "btn btn-primary pp-ask-btn", "是");
    yes.type = "button";
    yes.addEventListener("click", () => openPostprocessModal(detail));
    const no = el("button", "btn btn-ghost pp-ask-btn", "否");
    no.type = "button";
    no.addEventListener("click", () => {
      state.ppAskDismissed[detail.id] = true;
      actions.replaceWith(el("p", "pp-ask-ended", "已跳过优化，素材保持原样"));
    });
    actions.appendChild(yes);
    actions.appendChild(no);
    card.appendChild(actions);
  }
  row.appendChild(card);
  return row;
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
  renderStable(detail);
  renderPpDynamic(detail);
  renderGenerationDynamic(detail);
}

/* 稳定区：用户气泡 + 结果区 + 后处理入口 + 最终视频；中间留 .pp-dynamic 插槽给后处理聊天 */
function renderStable(detail) {
  clearStream();
  const inner = el("div", "stream-inner");
  inner.appendChild(renderUserBubble(detail));
  if (detail.status === "queued" || detail.status === "processing") {
    inner.appendChild(renderActivity(detail.status));
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

/* H3 任务进度独立刷新，避免每个分段状态变化都重建原视频和关键帧。 */
function renderGenerationDynamic(detail) {
  const slot = document.querySelector(".generation-dynamic");
  if (!slot) return;
  slot.textContent = "";
  slot.appendChild(renderFinalSection(detail));
}

/* ===== 会话详情 + 轮询 ===== */
/* 详情状态签名：
   stable 变（状态机/产物内容变化）→ 全量重渲染一次；
   仅 dyn 变（后处理 running 时 frames 逐帧增长）→ 只刷新后处理动态区；
   generation 变 → 只刷新最终视频区，保留原视频 DOM；
   完全不变 → 什么都不做（连 DOM 都不碰，杜绝每 2s 清空重建媒体引发的闪烁）。
   dyn 取 postprocess.frames 长度：只有它会在 stable 不变时随轮询增长。 */
function detailSignature(detail) {
  // stable 覆盖稳定区渲染消费的全部字段（未覆盖字段如 title/note 由创建后不变兜底，
  // pp.options 与 status 原子落盘——见审查记录）；dyn 只跟后处理进度（frames 单调增长）。
  const pp = detail.postprocess || null;
  const segments = Array.isArray(detail.segments) ? detail.segments : [];
  const stable = JSON.stringify([
    detail.status,
    detail.read_only === true,
    detail.submit_enabled === true,
    detail.fit_required === true,
    detail.fit_mode,
    detail.duration_s,
    detail.receipt_version,
    detail.source_prompt || null,
    detail.source_prompt_sha256 || null,
    detail.dialogue || null,
    pp ? pp.status : "",
    pp && pp.error ? pp.error : "",
    Array.isArray(detail.keyframes) ? detail.keyframes.join(",") : "",
    detail.prompt || "",
    segments.map((seg) => [
      seg.index,
      Array.isArray(seg.keyframes) ? seg.keyframes.join(",") : "",
      seg.prompt || "",
      Array.isArray(seg.lines) ? seg.lines.join("\n") : "",
    ]),
    detail.has_video ? 1 : 0,
  ]);
  const dyn = pp && Array.isArray(pp.frames) ? pp.frames.length : 0;
  const generation = JSON.stringify([
    detail.plan_receipt || null,
    Number.isInteger(detail.segment_count) ? detail.segment_count : null,
    detail.generation || null,
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
        // 分段 H3 / 拼接进度变化只刷新最终视频卡片，不重载源视频。
        renderGenerationDynamic(detail);
      }
    }
    // 签名完全不变 → 不碰 DOM（根治轮询闪烁的关键）
    state.detailSig = sig;
    const ppRunning = !!(detail.postprocess && detail.postprocess.status === "running");
    const generationRunning = !!(detail.generation
      && (detail.generation.status === "queued" || detail.generation.status === "running"));
    if (detail.status === "queued" || detail.status === "processing" || ppRunning
        || generationRunning || state.generationSubmitting[id]) {
      startPolling(id);
    } else {
      stopPolling();
      refreshList(false); // 终态：同步侧栏徽章（轻量更新，不动 stream）
    }
  } catch (err) {
    if (handleAuthError(err)) return;
    if (seq !== state.detailSeq) return;
    stopPolling();
    renderStreamError("会话加载失败：" + err.message);
  }
}

function selectConversation(id) {
  if (state.uploading) return; // 上传中不切换，避免打断
  state.currentId = id;
  const conv = state.conversations.find((c) => c.id === id);
  $("main-title").textContent = (conv && conv.title) || "会话";
  renderList();
  loadDetail(id, false);
}

function startPolling(id) {
  stopPolling();
  state.pollTimer = setInterval(() => {
    if (state.currentId !== id) {
      stopPolling();
      return;
    }
    loadDetail(id, true);
  }, POLL_MS);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
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
  return checked ? checked.value : "link"; // 与 HTML 默认 checked 一致：默认视频链接
}

function voiceMode() {
  const checked = document.querySelector('input[name="voice-mode"]:checked');
  return checked ? checked.value : "keep";
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
  updateSendBtn();
  if (!on) $("upload-progress").hidden = true;
}

function uploadConversation({ file, url, note, requestId, voiceMode: mode, targetLanguage }, onProgress) {
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
    xhr.send(fd);
  });
}

async function handleSend(event) {
  event.preventDefault();
  if (state.uploading) return;
  const mode = sourceMode();
  const file = mode === "upload" ? state.file : null;
  const url = mode === "link" ? $("url-input").value.trim() : "";
  const note = $("note-input").value.trim();
  const vMode = voiceMode();
  const targetLanguage = $("lang-input").value.trim();
  if (!file && !url) {
    setComposerError(mode === "upload" ? "请先选择视频文件" : "请先粘贴视频链接");
    return;
  }
  if (vMode === "translate" && !targetLanguage) {
    setComposerError("请填写翻译目标语言");
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
      { file, url, note, requestId: state.clientRequestId, voiceMode: vMode, targetLanguage },
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

  $("logout-btn").addEventListener("click", () => {
    state.token = null;
    localStorage.removeItem(TOKEN_KEY);
    showLogin(null);
  });

  $("menu-btn").addEventListener("click", openDrawer);
  $("drawer-backdrop").addEventListener("click", closeDrawer);

  $("new-chat-btn").addEventListener("click", () => {
    if (state.uploading) return;
    state.currentId = null;
    state.detail = null;
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
}

function boot() {
  state.clientRequestId = newRequestId();
  bindEvents();
  setSourceMode(sourceMode());
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
    buildStitchRetryPayload,
    buildResumePayload,
    buildSubmitPayload,
    apiErrorFromPayload,
    canOperate,
    detailSignature,
    formatDialogueLines,
    generationDraft,
    generationAction,
    generationRetryContract,
    generationSegmentLabel,
    longVideoContract,
    normalizeDialogueLines,
    parseDialogueLines,
  };
}

if (typeof document !== "undefined") boot();
