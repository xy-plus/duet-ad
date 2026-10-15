import { createHash } from "node:crypto";
import { stat, writeFile } from "node:fs/promises";
import { isAbsolute, join } from "node:path";

import type { BlobRef } from "@hypit/protocol";
import type { ResourceStore } from "@hypit/runtime";

/**
 * 把本地 resource-store 的字节发布为公网可访问 URL，供 ark 拉取参考视频/音频
 *（以及超过 data URI 合理大小的图片）。ark 的 reference_video/reference_audio
 * 只接受公网 URL，这条通道是本 Provider 支持它们的唯一路径。
 *
 * 文件名 = 内容 SHA-256 + 扩展名：内容寻址使发布幂等，同一字节重复发布直接
 * 复用已有文件，也便于手工清理（同名即同内容）。
 */

export type ArkAssetPublishOptions = {
  /** 本地发布目录（绝对路径，须已存在且可写）。 */
  readonly dir: string;
  /** 公网 URL 前缀，发布结果为 `<publicBaseUrl>/<filename>`。 */
  readonly publicBaseUrl: string;
};

export type ArkAssetPublisher = (artifact: BlobRef, resources: ResourceStore) => Promise<string>;

const MEDIA_TYPE_EXTENSIONS: Readonly<Record<string, string>> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/webp": "webp",
  "image/gif": "gif",
  "video/mp4": "mp4",
  "video/webm": "webm",
  "video/quicktime": "mov",
  "audio/mpeg": "mp3",
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/mp4": "m4a",
  "audio/aac": "aac",
  "audio/ogg": "ogg",
  "audio/flac": "flac",
};

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

export function createArkAssetPublisher(options: ArkAssetPublishOptions): ArkAssetPublisher {
  assert(isAbsolute(options.dir), `assetPublishDir 必须是绝对路径，收到 ${options.dir}`);
  const url = new URL(options.publicBaseUrl);
  assert(url.protocol === "https:" || url.hostname === "localhost" || url.hostname === "127.0.0.1",
    "assetPublicBaseUrl 必须使用 HTTPS（或本机回环）");
  const base = options.publicBaseUrl.replace(/\/+$/u, "");
  return async (artifact, resources) => {
    const extension = MEDIA_TYPE_EXTENSIONS[artifact.mediaType];
    assert(extension !== undefined, `无法为 mediaType ${artifact.mediaType} 确定发布文件扩展名`);
    const info = await stat(options.dir).catch(() => undefined);
    assert(info?.isDirectory() === true, `assetPublishDir 不存在或不是目录: ${options.dir}`);
    const bytes = await resources.get(artifact.resource);
    assert(bytes !== undefined && bytes.byteLength === artifact.size,
      `ark 引用资源 ${artifact.resource} 不可用`);
    const name = `${createHash("sha256").update(bytes).digest("hex")}.${extension}`;
    const path = join(options.dir, name);
    // wx：已存在即内容相同的重发布，直接复用（文件名即内容 hash）
    await writeFile(path, bytes, { flag: "wx" }).catch((error: unknown) => {
      if (error instanceof Error && "code" in error && error.code === "EEXIST") return;
      throw error;
    });
    return `${base}/${name}`;
  };
}
