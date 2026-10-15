import { mappingSupportsRequest } from "@hypit/generation";
import type {
  GenerationRequest,
  GenerationWireMapping,
  GenerationWireRequest,
} from "@hypit/generation";
import type { CanonicalValue, CapabilityRef } from "@hypit/protocol";

/**
 * 火山引擎 ark（OpenAI 兼容）直连的 wire 映射。
 *
 * Provider 不 import 模型包：端口表由 @hypit/seedance、@hypit/seedream 拥有，
 * 本文件只以数据形式声明 capability 身份与 ark 字段名。
 *
 * 已核实的 ark 契约（火山方舟「创建视频生成任务」文档 + 首轮真实 build 的
 * wire-request/wire-receipt 留痕）：
 * - Seedance 任务：`POST {baseUrl}/contents/generations/tasks`，body 为
 *   `{model, content: [...], duration, resolution, ratio, generate_audio}`。
 *   content 项：text；image_url（URL 或 data:image/<小写格式>;base64,…，
 *   role: first_frame / last_frame / reference_image）；video_url
 *   （仅公网 URL，role: reference_video）；audio_url（仅公网 URL，
 *   role: reference_audio）。
 * - Seedream 图片：`POST {baseUrl}/images/generations`，`size` 为 `宽x高`
 *   像素，`image` 为参考图数组（URL 或 base64 data URI）。
 */

const SEEDANCE_MODULE = { name: "@hypit/seedance", version: "1" } as const;
const SEEDREAM_MODULE = { name: "@hypit/seedream", version: "1" } as const;

/** 本 Provider 以 endpoint config 的 capability 字段选择注册哪一族 capability。 */
export const arkSeedanceCapabilities = ["seedance-2", "seedance-2-fast", "seedance-2-mini", "seedance-2.5"] as const;
export type ArkSeedanceCapability = typeof arkSeedanceCapabilities[number];
export const arkSeedreamCapability = "seedream-5-lite" as const;

/** 一个已核实的 ark 视频模型的能力档位。 */
export type ArkSeedanceModelLimits = {
  readonly durations: { readonly min: number; readonly max: number };
  readonly resolutions: readonly string[];
  readonly maxReferenceImages: number;
  readonly maxReferenceVideos: number;
  readonly maxReferenceAudios: number;
};

const SEEDANCE_20_REFERENCES = { maxReferenceImages: 9, maxReferenceVideos: 3, maxReferenceAudios: 3 } as const;

/**
 * 按 model ID 前缀匹配；顺序即优先级（mini/fast 先于 2-0 裸前缀）。
 * 未知模型落在保守兜底：让不认识的新模型显式受限，而不是放行后被供应商拒单。
 */
const SEEDANCE_MODEL_LIMITS: readonly (readonly [string, ArkSeedanceModelLimits])[] = [
  ["doubao-seedance-2-0-mini", {
    durations: { min: 4, max: 15 }, resolutions: ["480p", "720p"], ...SEEDANCE_20_REFERENCES,
  }],
  ["doubao-seedance-2-0-fast", {
    durations: { min: 4, max: 15 }, resolutions: ["480p", "720p"], ...SEEDANCE_20_REFERENCES,
  }],
  ["doubao-seedance-2-0", {
    durations: { min: 4, max: 15 }, resolutions: ["480p", "720p", "1080p", "4k"], ...SEEDANCE_20_REFERENCES,
  }],
  ["doubao-seedance-2-5", {
    // 2.5 公开资料：4-30 秒、30 图 / 10 视频 / 10 音频参考；1080p/4K 处于灰度，未核实前不放行
    durations: { min: 4, max: 30 }, resolutions: ["480p", "720p"],
    maxReferenceImages: 30, maxReferenceVideos: 10, maxReferenceAudios: 10,
  }],
];

const SEEDANCE_LIMITS_FALLBACK: ArkSeedanceModelLimits = {
  durations: { min: 4, max: 15 }, resolutions: ["480p", "720p"], ...SEEDANCE_20_REFERENCES,
};

export function arkSeedanceModelLimits(model: string): ArkSeedanceModelLimits {
  for (const [prefix, limits] of SEEDANCE_MODEL_LIMITS) {
    if (model.startsWith(prefix)) return limits;
  }
  return SEEDANCE_LIMITS_FALLBACK;
}

export function arkSeedanceCapabilityRef(name: ArkSeedanceCapability): CapabilityRef {
  return { module: SEEDANCE_MODULE, name };
}

export function arkSeedreamCapabilityRef(): CapabilityRef {
  return { module: SEEDREAM_MODULE, name: arkSeedreamCapability };
}

function scalar(request: GenerationRequest, port: string): string | number | boolean | undefined {
  const value = request.ports[port]?.[0];
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean" ? value : undefined;
}

function count(request: GenerationRequest, port: string): number {
  return request.ports[port]?.length ?? 0;
}

/**
 * ark 的参考视频/音频只接受公网可访问 URL（文档明确，不接受 base64）。
 * endpoint config 同时配置 assetPublishDir 与 assetPublicBaseUrl 时，
 * Provider 会把本地字节发布成公网 URL 上送；未配置则显式拒绝这两个端口。
 */
export const ARK_REFERENCE_URL_REQUIRED_REASON =
  "ark 的参考视频/音频只接受公网可访问的 URL；请在 endpoint config 同时配置 "
  + "assetPublishDir 与 assetPublicBaseUrl（把本地资源发布为公网 URL），或移除这些参考";

/**
 * 请求的 ark 支持性判定：返回 undefined 表示支持，否则返回拒绝原因。
 * 同时覆盖执行期（ports 完整）与 plan 期（媒体端口以 pendingInputs 命名出现）。
 * `publishedAssets` 表示 endpoint 是否配置了公网资产发布通道。
 */
export function arkSeedanceUnsupportedReason(
  request: GenerationRequest,
  pendingInputs: readonly string[],
  model: string,
  publishedAssets: boolean,
): string | undefined {
  const limits = arkSeedanceModelLimits(model);
  const present = new Set([...Object.keys(request.ports).filter((port) => count(request, port) > 0), ...pendingInputs]);
  if (!publishedAssets && (present.has("referenceVideo") || present.has("referenceAudio"))) {
    return ARK_REFERENCE_URL_REQUIRED_REASON;
  }
  if (scalar(request, "webSearch") === true) {
    return "ark Seedance 直连不支持 web_search=true";
  }
  const duration = scalar(request, "duration");
  if (duration === -1) {
    return "ark 需要显式秒数时长；-1（模型自选）未在 ark 文档中核实支持";
  }
  if (typeof duration === "number"
    && (!Number.isInteger(duration) || duration < limits.durations.min || duration > limits.durations.max)) {
    return `ark 模型 ${model} 只接受 ${limits.durations.min}-${limits.durations.max} 秒整数时长，收到 ${duration}`;
  }
  const resolution = scalar(request, "resolution");
  if (typeof resolution === "string" && !limits.resolutions.includes(resolution)) {
    return `ark 模型 ${model} 只接受 ${limits.resolutions.join("/")} 分辨率，收到 ${resolution}`;
  }
  if (count(request, "referenceImage") > limits.maxReferenceImages) {
    return `ark 模型 ${model} 参考图最多 ${limits.maxReferenceImages} 张，收到 ${count(request, "referenceImage")} 张`;
  }
  return undefined;
}

export function arkSeedreamUnsupportedReason(request: GenerationRequest): string | undefined {
  if (scalar(request, "nsfwCheck") === false) {
    return "ark 的内容审核不可关闭；nsfw-check 只能写 true";
  }
  return undefined;
}

/** Seedance 端口 → ark 任务字段；`content` 数组在 normalize 阶段组装。 */
export function arkSeedanceMapping(capability: ArkSeedanceCapability, model: string): GenerationWireMapping {
  return {
    capability: arkSeedanceCapabilityRef(capability),
    result: "video",
    routes: [{ model }],
    fields: {
      prompt: { as: "value", field: "prompt" },
      firstFrame: { as: "url", field: "first_frame" },
      lastFrame: { as: "url", field: "last_frame" },
      referenceImage: { as: "urlArray", field: "reference_images" },
      referenceVideo: { as: "urlArray", field: "reference_videos" },
      referenceAudio: { as: "urlArray", field: "reference_audios" },
      resolution: { as: "value", field: "resolution" },
      aspectRatio: { as: "value", field: "ratio" },
      duration: { as: "value", field: "duration" },
      generateAudio: { as: "value", field: "generate_audio" },
      webSearch: { as: "value", field: "web_search" },
    },
  };
}

/** Seedream 端口 → ark 图片生成字段；size 在 normalize 阶段由 aspectRatio+quality 折算。 */
export function arkSeedreamMapping(model: string): GenerationWireMapping {
  return {
    capability: arkSeedreamCapabilityRef(),
    result: "image",
    routes: [{ model }],
    fields: {
      prompt: { as: "value", field: "prompt" },
      aspectRatio: { as: "value", field: "aspect_ratio" },
      quality: { as: "value", field: "quality" },
      outputFormat: { as: "value", field: "output_format" },
      nsfwCheck: { as: "value", field: "nsfw_check" },
      images: { as: "urlArray", field: "image" },
    },
  };
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function asStringArray(value: CanonicalValue | undefined): readonly string[] {
  if (value === undefined) return [];
  assert(Array.isArray(value) && value.every((item) => typeof item === "string"), "ark 参考输入必须是 URL 数组");
  return value as readonly string[];
}

/** prompt 里作者已写的 ark 参考占位符（@图像1/@视频1/@音频1 等）。 */
const ARK_REFERENCE_PLACEHOLDER = /@(?:图像|图片|视频|音频)\d+/u;

/**
 * 把通用 wire 形状组装成 ark 任务创建 body。
 *
 * content 顺序：text → 参考图 → 首帧 → 尾帧 → 参考视频 → 参考音频。
 * ark 用 prompt 里的 @图像N/@视频N/@音频N 占位符按类型、按 content 顺序指代参考
 * 素材；作者 prompt 没带占位符且存在参考素材时，自动在 prompt 末尾注入一行
 * 占位符清单，保证顺序对应关系存在（作者已写占位符时原样保留，不改动措辞）。
 *
 * generate_audio 显式写入作者选择的布尔值；web_search 为 false 时省略
 *（true 已在支持性判定被拒绝，ark 视频生成无此参数）。
 */
export function normalizeArkSeedanceRequest(wire: GenerationWireRequest): Record<string, CanonicalValue> {
  const input = { ...(wire.input as Record<string, CanonicalValue>) };
  const prompt = input.prompt;
  assert(typeof prompt === "string" && prompt.length > 0, "ark Seedance 请求缺少 prompt");
  assert(input.web_search !== true, "web_search=true 已在支持性判定被拒绝");
  assert(typeof input.duration === "number", "ark Seedance 请求缺少 duration");
  assert(typeof input.resolution === "string", "ark Seedance 请求缺少 resolution");
  assert(typeof input.ratio === "string", "ark Seedance 请求缺少 ratio");
  const referenceImages = asStringArray(input.reference_images);
  const referenceVideos = asStringArray(input.reference_videos);
  const referenceAudios = asStringArray(input.reference_audios);

  let text = prompt;
  const referenceCount = referenceImages.length + referenceVideos.length + referenceAudios.length;
  if (referenceCount > 0 && !ARK_REFERENCE_PLACEHOLDER.test(text)) {
    const tokens = [
      ...referenceImages.map((_url, index) => `@图像${index + 1}`),
      ...referenceVideos.map((_url, index) => `@视频${index + 1}`),
      ...referenceAudios.map((_url, index) => `@音频${index + 1}`),
    ];
    text = `${text}\n参考素材：${tokens.join("、")}`;
  }

  const content: CanonicalValue[] = [{ type: "text", text }];
  for (const url of referenceImages) {
    content.push({ type: "image_url", image_url: { url }, role: "reference_image" });
  }
  if (typeof input.first_frame === "string") {
    content.push({ type: "image_url", image_url: { url: input.first_frame }, role: "first_frame" });
  }
  if (typeof input.last_frame === "string") {
    content.push({ type: "image_url", image_url: { url: input.last_frame }, role: "last_frame" });
  }
  for (const url of referenceVideos) {
    content.push({ type: "video_url", video_url: { url }, role: "reference_video" });
  }
  for (const url of referenceAudios) {
    content.push({ type: "audio_url", audio_url: { url }, role: "reference_audio" });
  }
  return {
    model: wire.model,
    content,
    duration: input.duration,
    resolution: input.resolution,
    ratio: input.ratio,
    generate_audio: input.generate_audio === true,
  };
}

/** quality 档位 → 长边像素（seedream-5-lite 端口表注释：basic 2K / high 3K / ultra 4K）。 */
const SEEDREAM_LONG_SIDE: Readonly<Record<string, number>> = {
  basic: 2048,
  high: 3072,
  ultra: 4096,
};

function even(value: number): number {
  return Math.max(2, Math.round(value / 2) * 2);
}

/** 按比例与质量档折算 `宽x高`；ark 的 size 接受精确像素。 */
export function arkSeedreamSize(aspectRatio: string, quality: string): string {
  const longSide = SEEDREAM_LONG_SIDE[quality];
  assert(longSide !== undefined, `ark Seedream 不支持 quality=${quality}`);
  const parts = aspectRatio.split(":").map(Number);
  assert(parts.length === 2 && parts.every((part) => Number.isFinite(part) && part > 0),
    `ark Seedream 无法解析 aspectRatio=${aspectRatio}`);
  const [w, h] = parts as [number, number];
  const width = w >= h ? longSide : even((longSide * w) / h);
  const height = w >= h ? even((longSide * h) / w) : longSide;
  return `${width}x${height}`;
}

/**
 * 把通用 wire 形状组装成 ark 图片生成 body。
 * nsfw_check 不落进 body：ark 始终执行内容审核（false 已在支持性判定拒绝）。
 * 有 `image` 参考图即图生图/多图融合；参考图接受 URL 或 base64 data URI。
 */
export function normalizeArkSeedreamRequest(wire: GenerationWireRequest): Record<string, CanonicalValue> {
  const input = { ...(wire.input as Record<string, CanonicalValue>) };
  const prompt = input.prompt;
  assert(typeof prompt === "string" && prompt.length > 0, "ark Seedream 请求缺少 prompt");
  assert(input.nsfw_check !== false, "nsfw_check=false 已在支持性判定被拒绝");
  const aspectRatio = input.aspect_ratio;
  const quality = input.quality;
  assert(typeof aspectRatio === "string" && typeof quality === "string", "ark Seedream 请求缺少 aspect_ratio/quality");
  const body: Record<string, CanonicalValue> = {
    model: wire.model,
    prompt,
    size: arkSeedreamSize(aspectRatio, quality),
    response_format: "url",
  };
  if (typeof input.output_format === "string") body.output_format = input.output_format;
  if (Array.isArray(input.image) && input.image.length > 0) body.image = input.image;
  return body;
}

/** 请求只使用了映射能覆盖的端口（plan 期 pendingInputs 另行判定）。 */
export function arkMappingSupportsRequest(mapping: GenerationWireMapping, value: unknown): boolean {
  return mappingSupportsRequest(mapping, value);
}
