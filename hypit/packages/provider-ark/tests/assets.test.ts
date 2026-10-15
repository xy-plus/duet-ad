import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";

import { MemoryResourceStore } from "@hypit/driver-node";

import { createArkAssetPublisher } from "../src/assets.js";

const dirs: string[] = [];
async function freshDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "hypit-ark-assets-"));
  dirs.push(dir);
  return dir;
}

afterAll(async () => {
  await Promise.all(dirs.map((dir) => rm(dir, { recursive: true, force: true })));
});

const BASE = "https://8.166.140.227:3301/.tmp-ab";

async function storeWith(bytes: Uint8Array, mediaType: string) {
  const store = new MemoryResourceStore();
  const ref = await store.put(bytes, mediaType);
  return { store, ref };
}

describe("ark asset 发布器", () => {
  it("字节写入发布目录，文件名为内容 hash + 扩展名，返回公网 URL", async () => {
    const dir = await freshDir();
    const publish = createArkAssetPublisher({ dir, publicBaseUrl: BASE });
    const { store, ref } = await storeWith(new TextEncoder().encode("fake-mp4"), "video/mp4");
    const url = await publish(ref, store);
    expect(url).toMatch(new RegExp(`^${BASE.replace(/[./:]/gu, "\\$&")}/[0-9a-f]{64}\\.mp4$`));
    const [name] = await readdir(dir);
    expect(url).toBe(`${BASE}/${name}`);
    expect(await readFile(join(dir, name!), "utf8")).toBe("fake-mp4");
  });

  it("同样字节重复发布幂等：同一 URL，不重复写文件", async () => {
    const dir = await freshDir();
    const publish = createArkAssetPublisher({ dir, publicBaseUrl: `${BASE}/` });
    const bytes = new TextEncoder().encode("same-content");
    const first = await storeWith(bytes, "audio/mpeg");
    const second = await storeWith(bytes, "audio/mpeg");
    const url1 = await publish(first.ref, first.store);
    const url2 = await publish(second.ref, second.store);
    expect(url1).toBe(url2);
    expect(url1).toMatch(/\.mp3$/u);
    expect(await readdir(dir)).toHaveLength(1);
  });

  it("不同字节得到不同文件名", async () => {
    const dir = await freshDir();
    const publish = createArkAssetPublisher({ dir, publicBaseUrl: BASE });
    const a = await storeWith(new TextEncoder().encode("a"), "video/mp4");
    const b = await storeWith(new TextEncoder().encode("b"), "video/mp4");
    expect(await publish(a.ref, a.store)).not.toBe(await publish(b.ref, b.store));
  });

  it("发布目录不存在时显式报错", async () => {
    const publish = createArkAssetPublisher({
      dir: join(await freshDir(), "missing"),
      publicBaseUrl: BASE,
    });
    const { store, ref } = await storeWith(new TextEncoder().encode("x"), "video/mp4");
    await expect(publish(ref, store)).rejects.toThrow(/assetPublishDir.*不存在/);
  });

  it("无法确定扩展名的 mediaType 显式报错", async () => {
    const dir = await freshDir();
    const publish = createArkAssetPublisher({ dir, publicBaseUrl: BASE });
    const { store, ref } = await storeWith(new TextEncoder().encode("x"), "application/x-weird");
    await expect(publish(ref, store)).rejects.toThrow(/mediaType/);
  });

  it("publicBaseUrl 必须公网可访问（https 或本机回环）", () => {
    expect(() => createArkAssetPublisher({ dir: "/tmp/x", publicBaseUrl: "ftp://example.com/a" }))
      .toThrow(/publicBaseUrl/i);
  });
});
