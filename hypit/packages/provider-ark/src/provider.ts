import {
  defineEndpointPackage,
  wakeAfter,
} from "@hypit/endpoint-kit";
import type {
  AsyncEndpoint,
  EndpointCredential,
  EndpointInvocationContext,
  EndpointOutcome,
  EndpointRequest,
  EndpointStartContext,
  EndpointSupport,
} from "@hypit/endpoint-kit";
import {
  compileWireRequest,
  generationTypes,
  sealGeneratedImageSet,
  sealGeneratedVideoSet,
} from "@hypit/generation";
import type { GenerationArtifactUrlResolver, GenerationRequest, GenerationWireMapping } from "@hypit/generation";
import { nodeJournalContextKey } from "@hypit/node-journal";
import type { NodeJournalCarrier, NodeJournalSession } from "@hypit/node-journal";
import { canonicalize } from "@hypit/protocol";
import type { CanonicalValue } from "@hypit/protocol";
import type { CredentialRef, ResourceStore } from "@hypit/runtime";

import { createArkAssetPublisher } from "./assets.js";
import type { ArkAssetPublisher } from "./assets.js";
import {
  ARK_REFERENCE_URL_REQUIRED_REASON,
  arkSeedanceCapabilities,
  arkSeedanceMapping,
  arkSeedanceUnsupportedReason,
  arkSeedreamCapability,
  arkSeedreamMapping,
  arkSeedreamUnsupportedReason,
  normalizeArkSeedanceRequest,
  normalizeArkSeedreamRequest,
} from "./mapping.js";

export const arkProviderModuleRef = { name: "@hypit/provider-ark", version: "1" } as const;

/** Provider 内部常量（非 config 项；config 只有 capability/baseUrl/model/apiKey 四个显式必填键）。 */
const SUBMIT_TIMEOUT_MS = 30_000;
const POLL_TIMEOUT_MS = 30_000;
const IMAGE_TIMEOUT_MS = 300_000;
const DOWNLOAD_TIMEOUT_MS = 300_000;
const POLL_INTERVAL_MS = 10_000;
const OPERATION_TIMEOUT_MS = 30 * 60_000;

const TASK_CONTRACT = "hypit.ark-task@1";

export type ArkCapabilityKind = "seedance" | "seedream";

export type CreateArkProviderOptions = {
  readonly instance: string;
  readonly pool: string;
  readonly capability: ArkCapabilityKind;
  readonly baseUrl: string;
  readonly model: string;
  readonly apiKey: CredentialRef;
  /**
   * 公网资产发布通道（两个键必须同时提供，activation 已校验）：参考视频/音频
   * 和超大图片经它发布为公网 URL 上送；缺省不启用，这些参考显式拒绝。
   */
  readonly assetPublish?: {
    readonly dir: string;
    readonly publicBaseUrl: string;
  };
  /** 测试注入；生产缺省为全局 fetch（这是进程内固定依赖，不是用户配置缺省值）。 */
  readonly fetch?: typeof globalThis.fetch;
};

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function object(value: unknown, subject: string): Record<string, unknown> {
  assert(value !== null && typeof value === "object" && !Array.isArray(value), `${subject} 必须是对象`);
  return value as Record<string, unknown>;
}

function text(value: unknown, subject: string): string {
  assert(typeof value === "string" && value.length > 0, `${subject} 必须是非空字符串`);
  return value;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

class ArkHttpError extends Error {
  readonly status: number;
  constructor(status: number, method: string, path: string, body: string) {
    let detail = body.slice(0, 500);
    try {
      const parsed = object(JSON.parse(body), "ark 错误响应").error;
      if (parsed !== undefined) {
        const error = object(parsed, "ark 错误");
        detail = `${text(error.code, "ark 错误码")}: ${text(error.message, "ark 错误信息")}`;
      }
    } catch { /* 保留原始文本 */ }
    super(`ark ${method} ${path} 返回 HTTP ${status}: ${detail}`);
    this.status = status;
  }
}

function failure(error: unknown): EndpointOutcome {
  return {
    status: "failed",
    failure: { code: error instanceof ArkHttpError ? "ARK_HTTP_ERROR" : "ARK_ERROR", message: message(error) },
  };
}

function apiSecret(credentials: Readonly<Record<string, EndpointCredential>>): string {
  const secret = credentials.apiKey?.secret;
  assert(typeof secret === "string" && secret.length > 0, "ark apiKey 凭据不可用；请配置 endpoint 的 apiKey（如 env 存储的 ARK_API_KEY）");
  return secret;
}

/** 调用上下文上由 runtime 层挂的节点 journal 会话；不存在时（非 build 路径）跳过 wire 留痕。 */
function journalOf(context: unknown): NodeJournalSession | undefined {
  return (context as NodeJournalCarrier)[nodeJournalContextKey];
}

/** 图片 data URI 的体积上限（字节）；超过则走公网发布通道（若启用）。 */
const DATA_URI_IMAGE_MAX_BYTES = 10 * 1024 * 1024;

/**
 * 引用资产 → ark 可读的 URL。
 * 小图片用 data URI 内联（首轮真实 build 已验证 ark 接受）；参考视频/音频与
 * 超大图片按文档只接受公网 URL：启用发布通道时落盘为公网资产，未启用显式失败。
 */
function arkAssetResolver(
  resources: ResourceStore,
  publish?: ArkAssetPublisher,
): GenerationArtifactUrlResolver {
  return async (artifact) => {
    const isImage = artifact.mediaType.startsWith("image/");
    if (isImage && (publish === undefined || artifact.size <= DATA_URI_IMAGE_MAX_BYTES)) {
      const bytes = await resources.get(artifact.resource);
      assert(bytes !== undefined && bytes.byteLength === artifact.size, `ark 引用资源 ${artifact.resource} 不可用`);
      return `data:${artifact.mediaType};base64,${Buffer.from(bytes).toString("base64")}`;
    }
    assert(publish !== undefined, ARK_REFERENCE_URL_REQUIRED_REASON);
    return await publish(artifact, resources);
  };
}

class ArkClient {
  readonly #baseUrl: string;
  readonly #fetcher: typeof globalThis.fetch;

  constructor(baseUrl: string, fetcher: typeof globalThis.fetch) {
    const url = new URL(baseUrl);
    assert(url.protocol === "https:" || url.hostname === "localhost" || url.hostname === "127.0.0.1",
      "ark baseUrl 必须使用 HTTPS（或本机回环）");
    this.#baseUrl = baseUrl.replace(/\/+$/u, "");
    this.#fetcher = fetcher;
  }

  /** 一次 HTTP 往返；wire 请求与原始响应写入节点目录（有 journal 会话时）。 */
  async request(
    method: "GET" | "POST",
    path: string,
    secret: string,
    body: Record<string, CanonicalValue> | undefined,
    timeoutMs: number,
    journal?: NodeJournalSession,
  ): Promise<Record<string, unknown>> {
    const url = `${this.#baseUrl}${path}`;
    // wire-request.json 只记有 payload 的提交（submit/图片生成）；poll 的 GET 不覆盖提交体，
    // wire-receipt.json 始终是最近一次响应，任务结束时即终态响应。
    if (body !== undefined) await journal?.recordWireRequest({ method, url, body });
    const response = await this.#fetcher(url, {
      method,
      signal: AbortSignal.timeout(timeoutMs),
      headers: {
        authorization: `Bearer ${secret}`,
        ...(body === undefined ? {} : { "content-type": "application/json" }),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    const textBody = await response.text();
    let parsed: unknown;
    try {
      parsed = textBody.length === 0 ? {} : JSON.parse(textBody);
    } catch {
      parsed = undefined;
    }
    await journal?.recordWireReceipt({ status: response.status, body: parsed ?? textBody.slice(0, 2_000) });
    if (!response.ok) throw new ArkHttpError(response.status, method, path, textBody);
    return object(parsed, `ark ${path} 响应`);
  }

  async download(url: string): Promise<{ readonly bytes: Uint8Array; readonly mediaType: string }> {
    const response = await this.#fetcher(url, { signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS) });
    if (!response.ok) throw new Error(`ark 产物下载返回 HTTP ${response.status}`);
    return {
      bytes: new Uint8Array(await response.arrayBuffer()),
      mediaType: response.headers.get("content-type")?.split(";", 1)[0]?.trim() ?? "application/octet-stream",
    };
  }
}

type ArkTaskHandle = {
  readonly contract: typeof TASK_CONTRACT;
  readonly taskId: string;
  readonly startedAt: number;
  readonly videoUrl?: string;
};

function parseHandle(value: unknown): ArkTaskHandle {
  const handle = object(value, "ark 任务 handle");
  assert(handle.contract === TASK_CONTRACT, "ark 任务 handle 契约不匹配");
  const taskId = text(handle.taskId, "ark 任务 handle.taskId");
  const startedAt = handle.startedAt;
  assert(typeof startedAt === "number", "ark 任务 handle.startedAt 无效");
  return {
    contract: TASK_CONTRACT,
    taskId,
    startedAt,
    ...(typeof handle.videoUrl === "string" ? { videoUrl: handle.videoUrl } : {}),
  };
}

function taskFailure(task: Record<string, unknown>, taskId: string): EndpointOutcome | undefined {
  if (task.status !== "failed") return undefined;
  let detail = "供应商未给出错误信息";
  if (task.error !== undefined) {
    const error = object(task.error, "ark 任务错误");
    detail = `${typeof error.code === "string" ? error.code : "UNKNOWN"}: ${typeof error.message === "string" ? error.message : ""}`;
  }
  return {
    status: "failed",
    receipt: { id: taskId },
    failure: { code: "ARK_TASK_FAILED", message: `ark 任务 ${taskId} 失败: ${detail}` },
  };
}

function seedanceEndpoint(
  client: ArkClient,
  mappings: ReadonlyMap<string, GenerationWireMapping>,
  model: string,
  publish: ArkAssetPublisher | undefined,
): AsyncEndpoint {
  return {
    async start(context: EndpointStartContext): Promise<EndpointOutcome> {
      try {
        const mapping = mappings.get(context.need.capability.name);
        assert(mapping !== undefined, `ark 未注册 capability ${context.need.capability.name}`);
        const rejection = arkSeedanceUnsupportedReason(
          context.need.constraints as unknown as GenerationRequest, [], model, publish !== undefined);
        assert(rejection === undefined, rejection ?? "");
        const secret = apiSecret(context.credentials);
        await context.reportProgress?.({ phase: `编译 ark 请求: ${mapping.routes[0]!.model}` });
        const wire = await compileWireRequest(
          mapping,
          context.need.constraints as unknown as GenerationRequest,
          arkAssetResolver(context.resources, publish),
        );
        const body = normalizeArkSeedanceRequest(wire);
        await context.reportProgress?.({ phase: `提交 ark 任务: ${String(body.model)}` });
        const response = await client.request("POST", "/contents/generations/tasks", secret, body, SUBMIT_TIMEOUT_MS, journalOf(context));
        const taskId = text(response.id, "ark 任务响应缺少 id");
        const handle = canonicalize({ contract: TASK_CONTRACT, taskId, startedAt: Date.now() } satisfies ArkTaskHandle);
        const receipt = { id: taskId };
        await context.checkpoint?.({ handle, receipt });
        return { ...wakeAfter(handle, POLL_INTERVAL_MS, Date.now(), { phase: "submitted" }), receipt };
      } catch (error) {
        return failure(error);
      }
    },

    async poll(context): Promise<EndpointOutcome> {
      try {
        const handle = parseHandle(context.handle);
        if (Date.now() - handle.startedAt > OPERATION_TIMEOUT_MS) {
          return {
            status: "failed",
            receipt: { id: handle.taskId },
            failure: { code: "ARK_OPERATION_TIMEOUT", message: `ark 任务 ${handle.taskId} 超过 ${OPERATION_TIMEOUT_MS}ms 未结束；远端结果未知` },
          };
        }
        const task = await client.request("GET", `/contents/generations/tasks/${encodeURIComponent(handle.taskId)}`,
          apiSecret(context.credentials), undefined, POLL_TIMEOUT_MS, journalOf(context));
        const rejected = taskFailure(task, handle.taskId);
        if (rejected !== undefined) return rejected;
        if (task.status === "queued" || task.status === "running") {
          return wakeAfter(canonicalize(handle), POLL_INTERVAL_MS, Date.now(), { phase: String(task.status) });
        }
        assert(task.status === "succeeded", `ark 任务返回未知状态 ${String(task.status)}`);
        const content = object(task.content, "ark 任务结果");
        const videoUrl = text(content.video_url, "ark 任务结果缺少 video_url");
        return {
          status: "ready",
          handle: canonicalize({ ...handle, videoUrl } satisfies ArkTaskHandle),
          receipt: { id: handle.taskId },
        };
      } catch (error) {
        return failure(error);
      }
    },

    async collect(context): Promise<EndpointOutcome> {
      try {
        const handle = parseHandle(context.handle);
        assert(handle.videoUrl !== undefined, "ark 任务 handle 缺少 videoUrl");
        await context.reportProgress?.({ phase: "下载 ark 生成视频" });
        const downloaded = await client.download(handle.videoUrl);
        assert(downloaded.mediaType.startsWith("video/"), `ark 产物不是视频（${downloaded.mediaType}）`);
        const artifact = await context.resources.put(downloaded.bytes, downloaded.mediaType);
        return {
          status: "completed",
          result: { value: { kind: "inline", value: canonicalize(sealGeneratedVideoSet({ videos: [artifact] })) } },
          receipt: { id: handle.taskId },
        };
      } catch (error) {
        return failure(error);
      }
    },
  };
}

async function seedreamGenerate(
  client: ArkClient,
  model: string,
  context: EndpointInvocationContext,
  publish: ArkAssetPublisher | undefined,
): Promise<{ readonly value: { readonly kind: "inline"; readonly value: CanonicalValue } }> {
  const rejection = arkSeedreamUnsupportedReason(context.need.constraints as unknown as GenerationRequest);
  assert(rejection === undefined, rejection ?? "");
  const wire = await compileWireRequest(
    arkSeedreamMapping(model),
    context.need.constraints as unknown as GenerationRequest,
    arkAssetResolver(context.resources, publish),
  );
  const body = normalizeArkSeedreamRequest(wire);
  await context.reportProgress?.({ phase: `生成 ark 图片: ${model}` });
  const response = await client.request("POST", "/images/generations", apiSecret(context.credentials), body, IMAGE_TIMEOUT_MS, journalOf(context));
  const data = response.data;
  assert(Array.isArray(data) && data.length > 0, "ark 图片响应缺少 data");
  const artifacts = [];
  for (const [index, item] of data.entries()) {
    const entry = object(item, `ark 图片 ${index + 1}`);
    if (typeof entry.b64_json === "string" && entry.b64_json.length > 0) {
      const bytes = Uint8Array.from(Buffer.from(entry.b64_json, "base64"));
      artifacts.push(await context.resources.put(bytes, "image/png"));
      continue;
    }
    const url = text(entry.url, `ark 图片 ${index + 1} 缺少 url`);
    const downloaded = await client.download(url);
    assert(downloaded.mediaType.startsWith("image/"), `ark 产物不是图片（${downloaded.mediaType}）`);
    artifacts.push(await context.resources.put(downloaded.bytes, downloaded.mediaType));
  }
  return { value: { kind: "inline", value: canonicalize(sealGeneratedImageSet({ images: artifacts })) } };
}

/** 支持性判定绑定配置的 ark 模型与发布通道：时长/分辨率/参考数量按核实档位。 */
function seedanceSupport(model: string, publishedAssets: boolean): (request: EndpointRequest) => EndpointSupport {
  return (request) => {
    const reason = arkSeedanceUnsupportedReason(
      request.constraints as unknown as GenerationRequest,
      request.pendingInputs?.map((input) => input.input) ?? [],
      model,
      publishedAssets,
    );
    return reason === undefined ? { status: "supported" } : { status: "unsupported", reason };
  };
}

function seedreamSupport(request: EndpointRequest): EndpointSupport {
  const reason = arkSeedreamUnsupportedReason(request.constraints as unknown as GenerationRequest);
  return reason === undefined ? { status: "supported" } : { status: "unsupported", reason };
}

export function createArkProvider(options: CreateArkProviderOptions) {
  assert(options.instance.trim().length > 0, "ark endpoint instance 不能为空");
  assert(options.pool.trim().length > 0, "ark endpoint pool 不能为空");
  assert(options.model.trim().length > 0, "ark model 不能为空（必须与 ark 控制台开通的模型 ID 一致）");
  const client = new ArkClient(options.baseUrl, options.fetch ?? globalThis.fetch);
  const publish = options.assetPublish === undefined ? undefined : createArkAssetPublisher(options.assetPublish);

  if (options.capability === "seedream") {
    return defineEndpointPackage({
      module: arkProviderModuleRef,
      facet: "seedream",
      instance: options.instance,
      pool: options.pool,
      credentials: { apiKey: options.apiKey },
      credentialInputs: { apiKey: { label: "ark API Key（ARK_API_KEY）" } },
      pricing: { kind: "page", url: "https://www.volcengine.com/pricing" },
      capabilities: [{
        capability: { module: { name: "@hypit/seedream", version: "1" }, name: arkSeedreamCapability },
        returns: generationTypes.imageSet,
        lifecycle: "immediate",
        supports: seedreamSupport,
        handler: async (context) => await seedreamGenerate(client, options.model, context, publish),
      }],
    });
  }

  const mappings = new Map(arkSeedanceCapabilities.map((name) => [name, arkSeedanceMapping(name, options.model)] as const));
  const endpoint = seedanceEndpoint(client, mappings, options.model, publish);
  return defineEndpointPackage({
    module: arkProviderModuleRef,
    facet: "seedance",
    instance: options.instance,
    pool: options.pool,
    credentials: { apiKey: options.apiKey },
    credentialInputs: { apiKey: { label: "ark API Key（ARK_API_KEY）" } },
    pricing: { kind: "page", url: "https://www.volcengine.com/pricing" },
    capabilities: arkSeedanceCapabilities.map((name) => ({
      capability: { module: { name: "@hypit/seedance", version: "1" }, name },
      returns: generationTypes.videoSet,
      lifecycle: "asynchronous" as const,
      supports: seedanceSupport(options.model, publish !== undefined),
      endpoint,
    })),
  });
}
