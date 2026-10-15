import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { EndpointRegistry, MemoryResourceStore } from "@hypit/driver-node";
import { generationTypes } from "@hypit/generation";
import { nodeJournalContextKey, openNodeJournals } from "@hypit/node-journal";
import { canonicalize } from "@hypit/protocol";
import type { CapabilityRef, Need } from "@hypit/protocol";
import { credentialRef } from "@hypit/endpoint-kit";
import { sealSeedanceRequest } from "@hypit/seedance";
import { sealSeedreamRequest } from "@hypit/seedream";

import { createArkProvider } from "../src/provider.js";
import hypitPackage from "../src/activation.js";

const SEEDANCE_CAPABILITY: CapabilityRef = { module: { name: "@hypit/seedance", version: "1" }, name: "seedance-2-mini" };
const SEEDREAM_CAPABILITY: CapabilityRef = { module: { name: "@hypit/seedream", version: "1" }, name: "seedream-5-lite" };

type FetchCall = { readonly url: string; readonly method: string; readonly body?: unknown };

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function binaryResponse(bytes: readonly number[], mediaType: string): Response {
  return new Response(new Uint8Array(bytes), { status: 200, headers: { "content-type": mediaType } });
}

function seedanceNeed(constraints: unknown): Need {
  return {
    id: "take",
    result: "rec_take",
    capability: SEEDANCE_CAPABILITY,
    returns: generationTypes.videoSet,
    constraints: canonicalize(constraints),
  };
}

const miniConstraints = sealSeedanceRequest("seedance-2-mini", {
  prompt: ["一只猫走过镜头"],
  duration: [5],
  resolution: ["720p"],
  aspectRatio: ["9:16"],
  generateAudio: [false],
  webSearch: [false],
} as Parameters<typeof sealSeedanceRequest>[1]);

function makeContext(need: Need, resources: MemoryResourceStore, extras: Record<string | symbol, unknown> = {}) {
  return {
    command: { kind: "fulfill-need" as const, id: `command:${need.id}`, need },
    need,
    resources,
    credentials: { apiKey: { secret: "sk-test" } },
    operation: "op_1",
    checkpoint: async (_checkpoint: unknown) => undefined,
    ...extras,
  };
}

async function install(pkg: { install(r: EndpointRegistry): unknown }): Promise<EndpointRegistry> {
  const registry = new EndpointRegistry();
  await pkg.install(registry);
  return registry;
}

describe("provider-ark seedance（异步任务：submit → poll → 下载）", () => {
  it("完整流程并写 wire-request/wire-receipt 到节点目录", async () => {
    const calls: FetchCall[] = [];
    let polls = 0;
    const fetcher: typeof globalThis.fetch = async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      const body = typeof init?.body === "string" ? JSON.parse(init.body) as unknown : undefined;
      calls.push({ url, method, ...(body === undefined ? {} : { body }) });
      if (url === "https://ark.test/api/v3/contents/generations/tasks" && method === "POST") {
        return jsonResponse(200, { id: "cgt-1" });
      }
      if (url === "https://ark.test/api/v3/contents/generations/tasks/cgt-1") {
        polls += 1;
        return polls === 1
          ? jsonResponse(200, { id: "cgt-1", status: "running" })
          : jsonResponse(200, { id: "cgt-1", status: "succeeded", content: { video_url: "https://cdn.test/v.mp4" } });
      }
      if (url === "https://cdn.test/v.mp4") return binaryResponse([1, 2, 3], "video/mp4");
      throw new Error(`unexpected fetch ${url}`);
    };
    const pkg = createArkProvider({
      instance: "ark.seedance", pool: "ark", capability: "seedance",
      baseUrl: "https://ark.test/api/v3", model: "doubao-seedance-1-0-pro-250528",
      apiKey: credentialRef("env", "ARK_API_KEY"), fetch: fetcher,
    });
    const registry = await install(pkg);
    const need = seedanceNeed(miniConstraints);
    const resolution = registry.resolve(need);
    expect(resolution.status).toBe("resolved");
    if (resolution.status !== "resolved") throw new Error("unreachable");
    expect(resolution.registration.kind).toBe("asynchronous");
    if (resolution.registration.kind !== "asynchronous") throw new Error("unreachable");
    const endpoint = resolution.registration.endpoint;

    const resources = new MemoryResourceStore();
    const dataRoot = await mkdtemp(join(tmpdir(), "hypit-ark-test-"));
    const journal = await openNodeJournals(dataRoot).beginBuild("build_ark");
    const session = await journal.beginNode({ seq: 1, capability: "@hypit/seedance@1#seedance-2-mini", nodeId: "take", endpoint: "ark.seedance" });
    const checkpoints: unknown[] = [];
    const base = makeContext(need, resources, {
      checkpoint: async (checkpoint: unknown) => { checkpoints.push(checkpoint); },
      [nodeJournalContextKey]: session,
    });

    const started = await endpoint.start(base as never);
    expect(started.status).toBe("pending");
    expect(started.receipt).toEqual({ id: "cgt-1" });
    expect(checkpoints).toHaveLength(1);

    const submit = calls[0]!;
    expect(submit.method).toBe("POST");
    const submitBody = submit.body as Record<string, unknown>;
    expect(submitBody.model).toBe("doubao-seedance-1-0-pro-250528");
    expect(submitBody.content).toEqual([{ type: "text", text: "一只猫走过镜头" }]);
    expect(submitBody.duration).toBe(5);
    expect(submitBody.resolution).toBe("720p");
    expect(submitBody.ratio).toBe("9:16");
    expect(submitBody.generate_audio).toBe(false);

    const handle = (started as { handle: unknown }).handle;
    const first = await endpoint.poll({ ...base, handle } as never);
    expect(first.status).toBe("pending");
    const second = await endpoint.poll({ ...base, handle } as never);
    expect(second.status).toBe("ready");
    const readyHandle = (second as { handle: unknown }).handle;

    if (endpoint.collect === undefined) throw new Error("endpoint 缺少 collect");
    const collected = await endpoint.collect({ ...base, handle: readyHandle } as never);
    expect(collected.status).toBe("completed");
    if (collected.status !== "completed") throw new Error("unreachable");
    const value = collected.result.value;
    if (value.kind !== "inline") throw new Error("expected inline value");
    const videos = (value.value as { videos: { resource: `res_${string}`; size: number; mediaType: string }[] }).videos;
    expect(videos).toHaveLength(1);
    expect(videos[0]!.mediaType).toBe("video/mp4");
    expect([...(await resources.get(videos[0]!.resource))!]).toEqual([1, 2, 3]);

    // wire payload 留痕：提交体与最近一次终态响应
    const wireRequest = JSON.parse(await readFile(join(session.dir, "wire-request.json"), "utf8")) as { body: Record<string, unknown> };
    expect(wireRequest.body.model).toBe("doubao-seedance-1-0-pro-250528");
    const wireReceipt = JSON.parse(await readFile(join(session.dir, "wire-receipt.json"), "utf8")) as { status: number; body: { status: string } };
    expect(wireReceipt.status).toBe(200);
    expect(wireReceipt.body.status).toBe("succeeded");
  });

  it("submit 被 ark 拒绝时返回 failed outcome，不抛出", async () => {
    const fetcher: typeof globalThis.fetch = async () =>
      jsonResponse(400, { error: { code: "InvalidParameter", message: "model is not supported" } });
    const pkg = createArkProvider({
      instance: "ark.seedance", pool: "ark", capability: "seedance",
      baseUrl: "https://ark.test/api/v3", model: "doubao-seedance-1-0-pro-250528",
      apiKey: credentialRef("env", "ARK_API_KEY"), fetch: fetcher,
    });
    const registry = await install(pkg);
    const need = seedanceNeed(miniConstraints);
    const resolution = registry.resolve(need);
    if (resolution.status !== "resolved" || resolution.registration.kind !== "asynchronous") throw new Error("unreachable");
    const outcome = await resolution.registration.endpoint.start(makeContext(need, new MemoryResourceStore()) as never);
    expect(outcome.status).toBe("failed");
    if (outcome.status !== "failed") throw new Error("unreachable");
    expect(outcome.failure.message).toMatch(/InvalidParameter/);
    expect(outcome.failure.message).toMatch(/400/);
  });

  it("远端任务失败时 poll 返回 failed 并携带供应商错误", async () => {
    const fetcher: typeof globalThis.fetch = async () =>
      jsonResponse(200, { id: "cgt-9", status: "failed", error: { code: "ContentFiltered", message: "prompt blocked" } });
    const pkg = createArkProvider({
      instance: "ark.seedance", pool: "ark", capability: "seedance",
      baseUrl: "https://ark.test/api/v3", model: "doubao-seedance-1-0-pro-250528",
      apiKey: credentialRef("env", "ARK_API_KEY"), fetch: fetcher,
    });
    const registry = await install(pkg);
    const need = seedanceNeed(miniConstraints);
    const resolution = registry.resolve(need);
    if (resolution.status !== "resolved" || resolution.registration.kind !== "asynchronous") throw new Error("unreachable");
    const handle = canonicalize({ contract: "hypit.ark-task@1", taskId: "cgt-9", startedAt: Date.now() });
    const outcome = await resolution.registration.endpoint.poll({
      ...makeContext(need, new MemoryResourceStore()), handle,
    } as never);
    expect(outcome.status).toBe("failed");
    if (outcome.status !== "failed") throw new Error("unreachable");
    expect(outcome.failure.message).toMatch(/ContentFiltered/);
  });

  it("不支持的端口组合在 resolve 阶段即被拒绝（web_search / 参考视频）", async () => {
    const pkg = createArkProvider({
      instance: "ark.seedance", pool: "ark", capability: "seedance",
      baseUrl: "https://ark.test/api/v3", model: "doubao-seedance-2-0-mini-260615",
      apiKey: credentialRef("env", "ARK_API_KEY"),
      fetch: (async () => { throw new Error("不应发起任何 HTTP 调用"); }) as typeof globalThis.fetch,
    });
    const registry = await install(pkg);
    const withWebSearch = seedanceNeed(sealSeedanceRequest("seedance-2-mini", {
      prompt: ["x"], duration: [5], resolution: ["720p"], aspectRatio: ["9:16"],
      generateAudio: [false], webSearch: [true],
    } as Parameters<typeof sealSeedanceRequest>[1]));
    const rejectedSearch = registry.resolve(withWebSearch);
    expect(rejectedSearch.status).toBe("unsupported");
    if (rejectedSearch.status !== "unsupported") throw new Error("unreachable");
    expect(rejectedSearch.rejections[0]!.reason).toMatch(/web_search/);

    const withRefVideo = seedanceNeed(sealSeedanceRequest("seedance-2-mini", {
      prompt: ["x"], duration: [5], resolution: ["720p"], aspectRatio: ["9:16"],
      generateAudio: [false], webSearch: [false],
      referenceVideo: [{ role: "video", artifact: { kind: "blob", resource: "res_v", size: 9, mediaType: "video/mp4" } }],
    } as Parameters<typeof sealSeedanceRequest>[1]));
    const rejectedVideo = registry.resolve(withRefVideo);
    expect(rejectedVideo.status).toBe("unsupported");
    if (rejectedVideo.status !== "unsupported") throw new Error("unreachable");
    expect(rejectedVideo.rejections[0]!.reason).toMatch(/公网/);
  });

  it("启用发布通道后参考视频经公网 URL 上送（内容寻址幂等）", async () => {
    const assetDir = await mkdtemp(join(tmpdir(), "hypit-ark-publish-"));
    const calls: FetchCall[] = [];
    const fetcher: typeof globalThis.fetch = async (input, init) => {
      const body = typeof init?.body === "string" ? JSON.parse(init.body) as unknown : undefined;
      calls.push({ url: String(input), method: init?.method ?? "GET", ...(body === undefined ? {} : { body }) });
      return jsonResponse(200, { id: "cgt-3" });
    };
    const pkg = createArkProvider({
      instance: "ark.seedance", pool: "ark", capability: "seedance",
      baseUrl: "https://ark.test/api/v3", model: "doubao-seedance-2-0-mini-260615",
      apiKey: credentialRef("env", "ARK_API_KEY"), fetch: fetcher,
      assetPublish: { dir: assetDir, publicBaseUrl: "https://cdn.test/assets" },
    });
    const registry = await install(pkg);
    const resources = new MemoryResourceStore();
    const videoRef = await resources.put(new TextEncoder().encode("fake-video"), "video/mp4");
    const audioRef = await resources.put(new TextEncoder().encode("fake-audio"), "audio/mpeg");
    const need = seedanceNeed(sealSeedanceRequest("seedance-2-mini", {
      prompt: ["跟着参考动起来"], duration: [5], resolution: ["720p"], aspectRatio: ["9:16"],
      generateAudio: [false], webSearch: [false],
      referenceVideo: [{ role: "video", artifact: videoRef }],
      referenceAudio: [{ role: "audio", artifact: audioRef }],
    } as Parameters<typeof sealSeedanceRequest>[1]));
    // resolve 不再拒绝（发布通道启用）
    const resolution = registry.resolve(need);
    expect(resolution.status).toBe("resolved");
    if (resolution.status !== "resolved" || resolution.registration.kind !== "asynchronous") throw new Error("unreachable");
    const outcome = await resolution.registration.endpoint.start(makeContext(need, resources) as never);
    expect(outcome.status).toBe("pending");
    const body = calls[0]!.body as Record<string, unknown>;
    const content = body.content as Record<string, unknown>[];
    expect(content[0]).toEqual({ type: "text", text: "跟着参考动起来\n参考素材：@视频1、@音频1" });
    expect(content[1]).toMatchObject({ type: "video_url", role: "reference_video" });
    expect(content[2]).toMatchObject({ type: "audio_url", role: "reference_audio" });
    const videoUrl = (content[1]!.video_url as { url: string }).url;
    expect(videoUrl).toMatch(/^https:\/\/cdn\.test\/assets\/[0-9a-f]{64}\.mp4$/u);
    expect((content[2]!.audio_url as { url: string }).url).toMatch(/^https:\/\/cdn\.test\/assets\/[0-9a-f]{64}\.mp3$/u);
    // 幂等：同一字节再发布不产生新文件
    const { readdir, readFile: readAsset } = await import("node:fs/promises");
    const files = await readdir(assetDir);
    expect(files).toHaveLength(2);
    expect(await readAsset(join(assetDir, files.find((f) => f.endsWith(".mp4"))!), "utf8")).toBe("fake-video");
    expect(videoUrl.split("/").at(-1)).toBe(files.find((f) => f.endsWith(".mp4")));
  });

  it("generate_audio=true 与参考图现在是受支持的组合", async () => {
    const calls: FetchCall[] = [];
    const fetcher: typeof globalThis.fetch = async (input, init) => {
      const body = typeof init?.body === "string" ? JSON.parse(init.body) as unknown : undefined;
      calls.push({ url: String(input), method: init?.method ?? "GET", ...(body === undefined ? {} : { body }) });
      return jsonResponse(200, { id: "cgt-2" });
    };
    const pkg = createArkProvider({
      instance: "ark.seedance", pool: "ark", capability: "seedance",
      baseUrl: "https://ark.test/api/v3", model: "doubao-seedance-2-0-mini-260615",
      apiKey: credentialRef("env", "ARK_API_KEY"), fetch: fetcher,
    });
    const registry = await install(pkg);
    const resources = new MemoryResourceStore();
    const refBytes = new TextEncoder().encode("fake-png");
    const ref = await resources.put(refBytes, "image/png");
    const need = seedanceNeed(sealSeedanceRequest("seedance-2-mini", {
      prompt: ["让参考图里的猫走到窗台"], duration: [10], resolution: ["720p"], aspectRatio: ["9:16"],
      generateAudio: [true], webSearch: [false],
      referenceImage: [{ role: "image", artifact: ref }],
    } as Parameters<typeof sealSeedanceRequest>[1]));
    const resolution = registry.resolve(need);
    expect(resolution.status).toBe("resolved");
    if (resolution.status !== "resolved" || resolution.registration.kind !== "asynchronous") throw new Error("unreachable");
    const outcome = await resolution.registration.endpoint.start(makeContext(need, resources) as never);
    expect(outcome.status).toBe("pending");
    const body = calls[0]!.body as Record<string, unknown>;
    expect(body.generate_audio).toBe(true);
    expect(body.duration).toBe(10);
    const content = body.content as Record<string, unknown>[];
    expect(content[0]).toEqual({ type: "text", text: "让参考图里的猫走到窗台\n参考素材：@图像1" });
    expect(content[1]).toMatchObject({ type: "image_url", role: "reference_image" });
    expect(String((content[1]!.image_url as { url: string }).url)).toMatch(/^data:image\/png;base64,/u);
  });
});

describe("provider-ark seedream（同步图片生成）", () => {
  const liteConstraints = sealSeedreamRequest({
    prompt: ["一张海报"],
    aspectRatio: ["9:16"],
    quality: ["basic"],
    outputFormat: ["png"],
    nsfwCheck: [true],
  } as Parameters<typeof sealSeedreamRequest>[0]);

  it("POST /images/generations 并下载产物", async () => {
    const calls: FetchCall[] = [];
    const fetcher: typeof globalThis.fetch = async (input, init) => {
      const url = String(input);
      const body = typeof init?.body === "string" ? JSON.parse(init.body) as unknown : undefined;
      calls.push({ url, method: init?.method ?? "GET", ...(body === undefined ? {} : { body }) });
      if (url === "https://ark.test/api/v3/images/generations") {
        return jsonResponse(200, { data: [{ url: "https://cdn.test/i.png" }] });
      }
      if (url === "https://cdn.test/i.png") return binaryResponse([9, 8, 7], "image/png");
      throw new Error(`unexpected fetch ${url}`);
    };
    const pkg = createArkProvider({
      instance: "ark.seedream", pool: "ark", capability: "seedream",
      baseUrl: "https://ark.test/api/v3", model: "doubao-seedream-5-0-pro-260628",
      apiKey: credentialRef("env", "ARK_API_KEY"), fetch: fetcher,
    });
    const registry = await install(pkg);
    const need: Need = {
      id: "img", result: "rec_img", capability: SEEDREAM_CAPABILITY,
      returns: generationTypes.imageSet, constraints: canonicalize(liteConstraints),
    };
    const resolution = registry.resolve(need);
    if (resolution.status !== "resolved" || resolution.registration.kind !== "immediate") throw new Error("unreachable");
    const dataRoot = await mkdtemp(join(tmpdir(), "hypit-ark-test-"));
    const journal = await openNodeJournals(dataRoot).beginBuild("build_img");
    const session = await journal.beginNode({ seq: 1, capability: "@hypit/seedream@1#seedream-5-lite", nodeId: "img" });
    const resources = new MemoryResourceStore();
    const result = await resolution.registration.handler({
      ...makeContext(need, resources, { [nodeJournalContextKey]: session }),
    } as never);
    if (result.value.kind !== "inline") throw new Error("expected inline value");
    const images = (result.value.value as { images: { resource: `res_${string}`; mediaType: string }[] }).images;
    expect(images).toHaveLength(1);
    expect(images[0]!.mediaType).toBe("image/png");
    expect([...(await resources.get(images[0]!.resource))!]).toEqual([9, 8, 7]);

    const body = calls[0]!.body as Record<string, unknown>;
    expect(calls[0]!.url).toBe("https://ark.test/api/v3/images/generations");
    expect(body.model).toBe("doubao-seedream-5-0-pro-260628");
    expect(body.size).toBe("1152x2048");
    expect(body.response_format).toBe("url");

    const wireRequest = JSON.parse(await readFile(join(session.dir, "wire-request.json"), "utf8")) as { body: Record<string, unknown> };
    expect(wireRequest.body.prompt).toBe("一张海报");
    const wireReceipt = JSON.parse(await readFile(join(session.dir, "wire-receipt.json"), "utf8")) as { status: number };
    expect(wireReceipt.status).toBe(200);
  });

  it("nsfw-check=false 在 resolve 阶段被拒绝", async () => {
    const pkg = createArkProvider({
      instance: "ark.seedream", pool: "ark", capability: "seedream",
      baseUrl: "https://ark.test/api/v3", model: "doubao-seedream-5-0-pro-260628",
      apiKey: credentialRef("env", "ARK_API_KEY"),
      fetch: (async () => { throw new Error("不应发起任何 HTTP 调用"); }) as typeof globalThis.fetch,
    });
    const registry = await install(pkg);
    const need: Need = {
      id: "img", result: "rec_img", capability: SEEDREAM_CAPABILITY,
      returns: generationTypes.imageSet,
      constraints: canonicalize(sealSeedreamRequest({
        prompt: ["x"], aspectRatio: ["1:1"], quality: ["basic"], outputFormat: ["png"], nsfwCheck: [false],
      } as Parameters<typeof sealSeedreamRequest>[0])),
    };
    const resolution = registry.resolve(need);
    expect(resolution.status).toBe("unsupported");
    if (resolution.status !== "unsupported") throw new Error("unreachable");
    expect(resolution.rejections[0]!.reason).toMatch(/nsfw/);
  });
});

describe("provider-ark activation（config 显式必填）", () => {
  const adapter = hypitPackage.hostFacets[0]!.implementation as unknown as {
    activate(context: {
      hostStateRoot: string; dataRoot: string; instance: string; pool?: string; config: Record<string, unknown>;
    }): Promise<{ endpoint: unknown }>;
  };
  const activate = (config: Record<string, unknown>) =>
    // activate 是同步抛错的实现，包一层 Promise 统一用 rejects 断言
    Promise.resolve().then(() => adapter.activate({ ...baseContext, config }));
  const baseContext = {
    hostStateRoot: "/tmp/host", dataRoot: "/tmp/data", instance: "ark.seedance", pool: "ark",
  };
  const fullConfig = {
    capability: "seedance",
    baseUrl: "https://ark.cn-beijing.volces.com/api/v3",
    model: "doubao-seedance-1-0-pro-250528",
    apiKey: { store: "env", key: "ARK_API_KEY" },
  };

  it("缺 baseUrl/model/apiKey/capability 任何一个都报错", async () => {
    for (const key of Object.keys(fullConfig)) {
      const config = { ...fullConfig, [key]: undefined };
      delete (config as Record<string, unknown>)[key];
      await expect(activate(config)).rejects.toThrow();
    }
  });

  it("拒绝未知 config 键与非法 capability 值", async () => {
    await expect(activate({ ...fullConfig, timeoutMs: 1000 })).rejects.toThrow(/timeoutMs/);
    await expect(activate({ ...fullConfig, capability: "gpt" })).rejects.toThrow(/capability/);
  });

  it("完整 config 正常激活", async () => {
    const activation = await activate(fullConfig);
    expect((activation.endpoint as { instance: { id: string } }).instance.id).toBe("ark.seedance");
  });

  it("assetPublishDir/assetPublicBaseUrl 只给一个报错，成对给出可激活", async () => {
    await expect(activate({ ...fullConfig, assetPublishDir: "/tmp/assets" })).rejects.toThrow(/同时/);
    await expect(activate({ ...fullConfig, assetPublicBaseUrl: "https://cdn.example/a" })).rejects.toThrow(/同时/);
    const activation = await activate({
      ...fullConfig,
      assetPublishDir: "/tmp/assets",
      assetPublicBaseUrl: "https://cdn.example/a",
    });
    expect((activation.endpoint as { instance: { id: string } }).instance.id).toBe("ark.seedance");
    await expect(activate({
      ...fullConfig,
      assetPublishDir: "relative/dir",
      assetPublicBaseUrl: "https://cdn.example/a",
    })).rejects.toThrow(/绝对路径/);
    await expect(activate({
      ...fullConfig,
      assetPublishDir: "/tmp/assets",
      assetPublicBaseUrl: "ftp://cdn.example/a",
    })).rejects.toThrow(/assetPublicBaseUrl/);
  });
});
