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
};

class AuthError extends Error {}

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
    let msg = "请求失败（" + res.status + "），请稍后重试";
    try {
      const data = await res.json();
      if (data && (data.detail || data.error || data.message)) {
        msg = String(data.detail || data.error || data.message);
      }
    } catch (_) { /* 保留默认文案 */ }
    throw new Error(msg);
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
  hero.appendChild(el("p", "empty-sub", "AI 会抽取关键帧并生成 Seedance 提示词"));
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
  body.appendChild(el("p", "fail-tip", "请确认视频可正常播放、格式常见（如 MP4 / MOV），然后重新上传"));
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
    if (detail.prompt) {
      const sec = el("section", "res-section");
      sec.appendChild(el("h3", "res-h3", "Seedance 提示词"));
      sec.appendChild(promptCard(detail.prompt));
      frag.appendChild(sec);
    }
  }

  return frag;
}

/* 最终视频区（结果区最后一段）：已提交生成则播放成片，否则显示「待提交生成」（提交接口预留未开放） */
function renderFinalSection(detail) {
  if (detail.has_video) {
    return videoSection(detail, "generated.mp4", "最终视频", "Seedance 生成成片");
  }
  const sec = el("section", "res-section");
  const card = el("div", "final-card");
  card.appendChild(el("h3", "res-h3", "最终视频"));
  const row = el("div", "final-row");
  const btnWrap = el("span", "submit-wrap");
  const btn = el("button", "btn btn-primary", "生成最终视频");
  btn.type = "button";
  btn.disabled = true;
  btn.setAttribute("aria-describedby", "final-caption");
  btnWrap.appendChild(btn);
  row.appendChild(btnWrap);
  const cap = el("p", "final-caption", "待提交生成（接口预留，当前阶段未开放）");
  cap.id = "final-caption";
  row.appendChild(cap);
  card.appendChild(row);
  sec.appendChild(card);
  return sec;
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

/* prompt 卡片（复制按钮 + 全文；单段/多段共用） */
function promptCard(text) {
  const card = el("div", "prompt-card");
  const head = el("div", "prompt-head");
  head.appendChild(el("span", "prompt-hint", "复制后可直接粘贴到 Seedance"));
  const copyBtn = el("button", "copy-btn");
  copyBtn.type = "button";
  copyBtn.appendChild(icon("i-copy"));
  const copyLabel = el("span", null, "复制");
  copyBtn.appendChild(copyLabel);
  copyBtn.addEventListener("click", async () => {
    const ok = await copyText(text);
    copyBtn.classList.add("copied");
    copyLabel.textContent = ok ? "已复制" : "复制失败";
    setTimeout(() => {
      copyBtn.classList.remove("copied");
      copyLabel.textContent = "复制";
    }, 1600);
  });
  head.appendChild(copyBtn);
  card.appendChild(head);
  card.appendChild(el("pre", "prompt-text", text));
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
      card.appendChild(el("h4", "res-sub", "Seedance 提示词"));
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
  const options = { change_bg: false, face_hold: false, remove_subtitle: false, remove_brand: false };
  document.querySelectorAll('#pp-form input[name="opt"]:checked').forEach((c) => {
    options[c.value] = true;
  });
  return options;
}

function updatePpConfirm() {
  $("pp-confirm").disabled = !Object.values(ppCheckedOptions()).some(Boolean);
}

function openPostprocessModal(detail) {
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
  if (!detail) return;
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

/* 助手消息：running 进行中卡 / done 优化后结果 / failed 错误卡（detail 轮询重渲染自然保持） */
function renderPpAssistant(detail, pp) {
  const row = el("div", "msg-row");
  row.appendChild(assistantHead(detail.updated_at));
  if (pp.status === "running") {
    const card = el("div", "activity-card");
    card.appendChild(el("p", "ac-title", "后处理进行中…"));
    card.appendChild(el("p", "ac-sub", "正在逐帧优化关键帧，通常需要一两分钟"));
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
      card.appendChild(el("p", "ac-sub", "无适用帧（如勾选「含人脸遮挡」但未检出人脸）"));
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
  if (!detail.postprocess_enabled) return null;
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

/* postprocess 存在即渲染：用户摘要 + 助手消息（renderDetail 全量重渲染，随轮询自然更新） */
function renderPpChat(detail) {
  const pp = detail.postprocess || {};
  if (!pp.status || !pp.options) return null;
  const frag = document.createDocumentFragment();
  frag.appendChild(renderPpUserBubble(pp));
  frag.appendChild(renderPpAssistant(detail, pp));
  return frag;
}

function renderDetail(detail) {
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
    const ppChat = renderPpChat(detail);
    if (ppChat) inner.appendChild(ppChat);
    inner.appendChild(renderFinalSection(detail));
  }
  $("stream").appendChild(inner);
}

/* ===== 会话详情 + 轮询 ===== */
async function loadDetail(id, silent) {
  const seq = ++state.detailSeq;
  if (!silent) renderSkeleton();
  try {
    const detail = await apiJSON("/api/conversations/" + encodeURIComponent(id));
    if (seq !== state.detailSeq || state.currentId !== id) return; // 已切换会话
    state.detail = detail;
    renderDetail(detail);
    if (detail.status === "queued" || detail.status === "processing"
        || (detail.postprocess && detail.postprocess.status === "running")) {
      startPolling(id);
    } else {
      stopPolling();
      refreshList(false); // 终态：同步侧栏徽章
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
  return checked ? checked.value : "upload";
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
        const msg = data && (data.detail || data.error || data.message)
          ? String(data.detail || data.error || data.message)
          : "上传失败（" + xhr.status + "），请稍后重试";
        reject(new Error(msg));
      }
    });
    xhr.addEventListener("error", () => reject(new Error("网络异常，上传未完成，请重试")));
    xhr.addEventListener("abort", () => reject(new Error("上传已中断，请重试")));
    const fd = new FormData();
    if (file) fd.append("file", file, file.name);
    else if (url) fd.append("reference_url", url);
    if (note) fd.append("note", note);
    if (requestId) fd.append("client_request_id", requestId);
    fd.append("voice_mode", mode || "none");
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

boot();
