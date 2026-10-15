import { describe, expect, it } from "vitest";

import {
  assertMappingCoversPorts,
  compileWireRequest,
  mappingSupportsRequest,
  selectWireModelForRequest,
} from "@hypit/generation";
import type { GenerationArtifactUrlResolver } from "@hypit/generation";
import type { BlobRef } from "@hypit/protocol";
import { sealSeedanceRequest, seedancePorts } from "@hypit/seedance";
import type { SeedanceModel } from "@hypit/seedance";
import { sealSeedreamRequest, seedream5LitePorts } from "@hypit/seedream";

import {
  ARK_REFERENCE_URL_REQUIRED_REASON,
  arkSeedanceCapabilities,
  arkSeedanceMapping,
  arkSeedanceModelLimits,
  arkSeedanceUnsupportedReason,
  arkSeedreamCapability,
  arkSeedreamMapping,
  arkSeedreamUnsupportedReason,
  normalizeArkSeedanceRequest,
  normalizeArkSeedreamRequest,
} from "../src/mapping.js";

const MINI = "doubao-seedance-2-0-mini-260615";
const STANDARD = "doubao-seedance-2-0-260128";
const ARK_SEEDREAM_MODEL = "doubao-seedream-5-0-pro-260628";

const resolve: GenerationArtifactUrlResolver = async (artifact: BlobRef) =>
  `https://assets.test/${artifact.resource}`;

const frame: BlobRef = { kind: "blob", resource: "res_frame", size: 10, mediaType: "image/png" };
const refImage = (id: string): { role: string; artifact: BlobRef } => ({
  role: "image",
  artifact: { kind: "blob", resource: `res_${id}`, size: 5, mediaType: "image/png" },
});

function miniRequest(overrides: Record<string, readonly unknown[]> = {}) {
  return sealSeedanceRequest("seedance-2-mini", {
    prompt: ["一只猫走过镜头"],
    duration: [5],
    resolution: ["720p"],
    aspectRatio: ["9:16"],
    generateAudio: [false],
    webSearch: [false],
    ...overrides,
  } as Parameters<typeof sealSeedanceRequest>[1]);
}

describe("ark seedance mapping", () => {
  it("端口映射到 ark 字段并组装 content 数组；generate_audio 显式写入作者选择", async () => {
    const mapping = arkSeedanceMapping("seedance-2-mini", MINI);
    const request = miniRequest({
      generateAudio: [true],
      firstFrame: [{ role: "image", artifact: frame }],
    });
    const wire = await compileWireRequest(mapping, request, resolve);
    expect(wire.model).toBe(MINI);
    expect(normalizeArkSeedanceRequest(wire)).toEqual({
      model: MINI,
      content: [
        { type: "text", text: "一只猫走过镜头" },
        { type: "image_url", image_url: { url: "https://assets.test/res_frame" }, role: "first_frame" },
      ],
      duration: 5,
      resolution: "720p",
      ratio: "9:16",
      generate_audio: true,
    });
  });

  it("last_frame 进入 content，web_search=false 不写入 body", async () => {
    const mapping = arkSeedanceMapping("seedance-2-mini", MINI);
    const request = miniRequest({
      firstFrame: [{ role: "image", artifact: frame }],
      lastFrame: [{ role: "image", artifact: { kind: "blob", resource: "res_last", size: 8, mediaType: "image/png" } }],
    });
    const body = normalizeArkSeedanceRequest(await compileWireRequest(mapping, request, resolve));
    expect(body.content).toEqual([
      { type: "text", text: "一只猫走过镜头" },
      { type: "image_url", image_url: { url: "https://assets.test/res_frame" }, role: "first_frame" },
      { type: "image_url", image_url: { url: "https://assets.test/res_last" }, role: "last_frame" },
    ]);
    expect(body.generate_audio).toBe(false);
    expect(body).not.toHaveProperty("web_search");
  });

  it("参考图映射为 role=reference_image 的 content 项，prompt 无占位符时自动注入", async () => {
    const mapping = arkSeedanceMapping("seedance-2-mini", MINI);
    const request = miniRequest({
      referenceImage: [refImage("a"), refImage("b")],
    });
    const wire = await compileWireRequest(mapping, request, resolve);
    expect(mappingSupportsRequest(mapping, request)).toBe(true);
    const body = normalizeArkSeedanceRequest(wire);
    expect(body.content).toEqual([
      { type: "text", text: "一只猫走过镜头\n参考素材：@图像1、@图像2" },
      { type: "image_url", image_url: { url: "https://assets.test/res_a" }, role: "reference_image" },
      { type: "image_url", image_url: { url: "https://assets.test/res_b" }, role: "reference_image" },
    ]);
  });

  it("作者已写占位符时 prompt 原样保留", async () => {
    const mapping = arkSeedanceMapping("seedance-2-mini", MINI);
    const request = miniRequest({
      prompt: ["让 @图像1 里的猫走向镜头"],
      referenceImage: [refImage("a")],
    });
    const body = normalizeArkSeedanceRequest(await compileWireRequest(mapping, request, resolve));
    expect((body.content as { text: string }[])[0]!.text).toBe("让 @图像1 里的猫走向镜头");
  });

  it("参考图与首尾帧共存：参考图在前，占位符编号对应参考图顺序", async () => {
    const mapping = arkSeedanceMapping("seedance-2-mini", MINI);
    // 端口表不允许 referenceImage 与 firstFrame 共存（atMostOneOf），这里直接构造 wire 输入测 normalize 顺序
    const wire = await compileWireRequest(mapping, miniRequest({ referenceImage: [refImage("a")] }), resolve);
    const withFrame = {
      ...wire,
      input: { ...(wire.input as Record<string, unknown>), first_frame: "https://assets.test/res_frame" },
    };
    const body = normalizeArkSeedanceRequest(withFrame as typeof wire);
    expect(body.content).toEqual([
      { type: "text", text: "一只猫走过镜头\n参考素材：@图像1" },
      { type: "image_url", image_url: { url: "https://assets.test/res_a" }, role: "reference_image" },
      { type: "image_url", image_url: { url: "https://assets.test/res_frame" }, role: "first_frame" },
    ]);
  });

  it("referenceVideo/referenceAudio：未配置发布通道显式拒绝，配置后放行", () => {
    const withVideo = miniRequest({
      referenceVideo: [{ role: "video", artifact: { kind: "blob", resource: "res_v", size: 9, mediaType: "video/mp4" } }],
    });
    expect(arkSeedanceUnsupportedReason(withVideo, [], MINI, false)).toBe(ARK_REFERENCE_URL_REQUIRED_REASON);
    // plan 阶段以 pendingInputs 出现同样拒绝
    expect(arkSeedanceUnsupportedReason(miniRequest(), ["referenceAudio"], MINI, false))
      .toBe(ARK_REFERENCE_URL_REQUIRED_REASON);
    // 配置 assetPublishDir/assetPublicBaseUrl 后（publishedAssets=true）放行
    expect(arkSeedanceUnsupportedReason(withVideo, [], MINI, true)).toBeUndefined();
    expect(arkSeedanceUnsupportedReason(miniRequest(), ["referenceAudio"], MINI, true)).toBeUndefined();
  });

  it("web_search=true 显式拒绝（ark 无此能力）", () => {
    expect(arkSeedanceUnsupportedReason(miniRequest({ webSearch: [true] }), [], MINI, false)).toMatch(/web_search/);
    expect(arkSeedanceUnsupportedReason(miniRequest(), [], MINI, false)).toBeUndefined();
  });

  it("时长/分辨率按配置的 ark 模型档位判定", () => {
    // mini 档位：4-15 秒、480p/720p（与 seedance-2-mini 端口表一致，常态请求全通过）
    expect(arkSeedanceUnsupportedReason(miniRequest({ duration: [4] }), [], MINI, false)).toBeUndefined();
    expect(arkSeedanceUnsupportedReason(miniRequest({ duration: [15] }), [], MINI, false)).toBeUndefined();
    // 2.5 端口表允许的 20 秒超出 2.0 家族档位 → 显式拒绝
    const v25Long = sealSeedanceRequest("seedance-2.5", {
      prompt: ["x"], duration: [20], resolution: ["720p"], aspectRatio: ["16:9"],
      generateAudio: [false], webSearch: [false],
    } as Parameters<typeof sealSeedanceRequest>[1]);
    expect(arkSeedanceUnsupportedReason(v25Long, [], MINI, false)).toMatch(/4-15/);
    expect(arkSeedanceUnsupportedReason(v25Long, [], "doubao-seedance-2-5-260628", false)).toBeUndefined();
    // 2.5 端口表允许的 -1（模型自选）未在 ark 文档核实 → 显式拒绝
    const v25Auto = sealSeedanceRequest("seedance-2.5", {
      prompt: ["x"], duration: [-1], resolution: ["720p"], aspectRatio: ["16:9"],
      generateAudio: [false], webSearch: [false],
    } as Parameters<typeof sealSeedanceRequest>[1]);
    expect(arkSeedanceUnsupportedReason(v25Auto, [], "doubao-seedance-2-5-260628", false)).toMatch(/-1/);
    // 分辨率：standard 模型允许 4k，mini 拒绝
    const standard4k = sealSeedanceRequest("seedance-2", {
      prompt: ["x"], duration: [5], resolution: ["4k"], aspectRatio: ["16:9"],
      generateAudio: [false], webSearch: [false],
    } as Parameters<typeof sealSeedanceRequest>[1]);
    expect(arkSeedanceUnsupportedReason(standard4k, [], STANDARD, false)).toBeUndefined();
    expect(arkSeedanceUnsupportedReason(standard4k, [], MINI, false)).toMatch(/480p\/720p/);
  });

  it("未知模型落保守兜底档位", () => {
    const limits = arkSeedanceModelLimits("doubao-seedance-9-9-future");
    expect(limits.resolutions).toEqual(["480p", "720p"]);
    expect(limits.durations).toEqual({ min: 4, max: 15 });
  });

  it("映射完整覆盖四个 seedance 端口表（assertMappingCoversPorts）", () => {
    for (const capability of arkSeedanceCapabilities) {
      assertMappingCoversPorts(seedancePorts[capability as SeedanceModel], arkSeedanceMapping(capability, MINI));
    }
  });

  it("参考视频/音频映射为 video_url/audio_url content 项", () => {
    const wire = {
      model: MINI,
      input: {
        prompt: "跟着 @视频1 动起来，配上 @音频1 的声音",
        duration: 5, resolution: "720p", ratio: "9:16", generate_audio: false,
        reference_videos: ["https://assets.test/v1.mp4"],
        reference_audios: ["https://assets.test/a1.mp3"],
      },
    };
    const body = normalizeArkSeedanceRequest(wire);
    expect(body.content).toEqual([
      { type: "text", text: "跟着 @视频1 动起来，配上 @音频1 的声音" },
      { type: "video_url", video_url: { url: "https://assets.test/v1.mp4" }, role: "reference_video" },
      { type: "audio_url", audio_url: { url: "https://assets.test/a1.mp3" }, role: "reference_audio" },
    ]);
  });

  it("selectWireModelForRequest 对全端口请求选出配置的 ark 模型", async () => {
    const mapping = arkSeedanceMapping("seedance-2-mini", MINI);
    const request = miniRequest({ referenceImage: [refImage("a")] });
    expect(selectWireModelForRequest(mapping, request)).toBe(MINI);
  });
});

describe("ark seedream mapping", () => {
  function liteRequest(overrides: Record<string, readonly unknown[]> = {}) {
    return sealSeedreamRequest({
      prompt: ["一张海报"],
      aspectRatio: ["9:16"],
      quality: ["basic"],
      outputFormat: ["png"],
      nsfwCheck: [true],
      ...overrides,
    } as Parameters<typeof sealSeedreamRequest>[0]);
  }

  it("端口映射到 ark 字段，aspectRatio+quality 折算为 size 像素", async () => {
    const mapping = arkSeedreamMapping(ARK_SEEDREAM_MODEL);
    const wire = await compileWireRequest(mapping, liteRequest(), resolve);
    expect(normalizeArkSeedreamRequest(wire)).toEqual({
      model: ARK_SEEDREAM_MODEL,
      prompt: "一张海报",
      size: "1152x2048",
      output_format: "png",
      response_format: "url",
    });
  });

  it("参考图写入 image 数组（图生图/多图融合）；无参考图时省略", async () => {
    const mapping = arkSeedreamMapping(ARK_SEEDREAM_MODEL);
    const withRef = normalizeArkSeedreamRequest(await compileWireRequest(mapping, liteRequest({
      images: [{ role: "image", artifact: frame }],
    }), resolve));
    expect(withRef.image).toEqual(["https://assets.test/res_frame"]);
    const withoutRef = normalizeArkSeedreamRequest(await compileWireRequest(mapping, liteRequest(), resolve));
    expect(withoutRef).not.toHaveProperty("image");
  });

  it("quality 档位映射到长边像素", async () => {
    const mapping = arkSeedreamMapping(ARK_SEEDREAM_MODEL);
    const size = async (quality: string, ratio: string) => {
      const wire = await compileWireRequest(mapping, liteRequest({ quality: [quality], aspectRatio: [ratio] }), resolve);
      return normalizeArkSeedreamRequest(wire).size;
    };
    expect(await size("basic", "1:1")).toBe("2048x2048");
    expect(await size("high", "16:9")).toBe("3072x1728");
    expect(await size("ultra", "3:4")).toBe("3072x4096");
  });

  it("nsfwCheck=false 显式拒绝（ark 内容审核不可关闭）", () => {
    expect(arkSeedreamUnsupportedReason(liteRequest({ nsfwCheck: [false] }))).toMatch(/nsfw/);
    expect(arkSeedreamUnsupportedReason(liteRequest())).toBeUndefined();
  });

  it("映射完整覆盖 seedream 端口表", () => {
    const mapping = arkSeedreamMapping(ARK_SEEDREAM_MODEL);
    expect(mapping.capability.name).toBe(arkSeedreamCapability);
    assertMappingCoversPorts(seedream5LitePorts, mapping);
  });
});
