import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { openNodeJournals } from "../src/index.js";

async function freshDataRoot(): Promise<string> {
  return await mkdtemp(join(tmpdir(), "hypit-node-journal-"));
}

async function readJson(path: string): Promise<Record<string, unknown>> {
  const value = JSON.parse(await readFile(path, "utf8")) as unknown;
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error(`${path} is not an object`);
  return value as Record<string, unknown>;
}

const REF_A = { resource: "res_a", size: 3, mediaType: "image/png" } as const;
const REF_B = { resource: "res_b", size: 4, mediaType: "video/mp4" } as const;

describe("node-journal 落盘结构", () => {
  it("beginBuild/beginNode/record*/endNode 生成 spec 规定的目录与文件", async () => {
    const dataRoot = await freshDataRoot();
    const journals = openNodeJournals(dataRoot);
    const build = await journals.beginBuild("build_01");
    expect(build.dir).toBe(join(dataRoot, "builds", "build_01", "node-journal"));

    const node = await build.beginNode({ seq: 1, capability: "@hypit/seedream@1#seedream-5-lite", nodeId: "opening-image", endpoint: "ark.seedream" });
    expect(node.dir).toBe(join(build.dir, "nodes", "0001-seedream-5-lite"));

    await node.recordRequest({ capability: "seedream-5-lite", constraints: { ports: { prompt: ["a cat"] } } });
    await node.recordWireRequest({ model: "doubao-seedream-5-0-pro-260628", prompt: "a cat" });
    await node.recordWireReceipt({ status: 200, body: { data: [{ url: "https://example/x.png" }] } });
    await node.recordReceipt({ status: "completed" });
    await node.recordOutputs([REF_A]);
    await node.endNode("completed");

    const dir = node.dir;
    const request = await readJson(join(dir, "request.json"));
    expect(request).toEqual({ capability: "seedream-5-lite", constraints: { ports: { prompt: ["a cat"] } } });
    const wireRequest = await readJson(join(dir, "wire-request.json"));
    expect(wireRequest).toEqual({ model: "doubao-seedream-5-0-pro-260628", prompt: "a cat" });
    const wireReceipt = await readJson(join(dir, "wire-receipt.json"));
    expect(wireReceipt).toEqual({ status: 200, body: { data: [{ url: "https://example/x.png" }] } });
    const receipt = await readJson(join(dir, "receipt.json"));
    expect(receipt).toEqual({ status: "completed" });

    const meta = await readJson(join(dir, "meta.json"));
    expect(meta.seq).toBe(1);
    expect(meta.nodeId).toBe("opening-image");
    expect(meta.capability).toBe("@hypit/seedream@1#seedream-5-lite");
    expect(meta.endpoint).toBe("ark.seedream");
    expect(meta.status).toBe("completed");
    expect(typeof meta.startedAt).toBe("string");
    expect(typeof meta.endedAt).toBe("string");
    expect(meta.outputs).toEqual([REF_A]);

    const manifest = (await readFile(join(build.dir, "manifest.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line) as Record<string, unknown>);
    expect(manifest).toHaveLength(1);
    expect(manifest[0]).toMatchObject({
      seq: 1,
      nodeId: "opening-image",
      capability: "@hypit/seedream@1#seedream-5-lite",
      status: "completed",
      dir: "nodes/0001-seedream-5-lite",
    });
    expect(typeof manifest[0]!.startedAt).toBe("string");
    expect(typeof manifest[0]!.endedAt).toBe("string");
  });

  it("findNode/allocateSeq 在进程重启（重新打开）后延续", async () => {
    const dataRoot = await freshDataRoot();
    const first = await openNodeJournals(dataRoot).beginBuild("build_02");
    const one = await first.beginNode({ seq: await first.allocateSeq(), capability: "@hypit/seedance@1#seedance-2-mini", nodeId: "take" });
    await one.recordRequest({ note: "submit 语义级请求" });
    // 异步节点停留在 running：进程退出，poll 在另一个进程里继续
    const two = await first.beginNode({ seq: await first.allocateSeq(), capability: "@hypit/media@1#normalize", nodeId: "take-media" });
    await two.recordOutputs([REF_B]);
    await two.endNode("completed");

    const reopened = await openNodeJournals(dataRoot).beginBuild("build_02");
    expect(await reopened.allocateSeq()).toBe(3);
    const found = await reopened.findNode("take");
    expect(found).toBeDefined();
    expect(found!.seq).toBe(1);
    expect(found!.dir).toBe(one.dir);
    await found!.recordReceipt({ status: "failed", message: "provider rejected" });
    await found!.endNode("failed");

    const meta = await readJson(join(one.dir, "meta.json"));
    expect(meta.status).toBe("failed");
    expect(typeof meta.endedAt).toBe("string");
    const manifest = (await readFile(join(reopened.dir, "manifest.jsonl"), "utf8")).trim().split("\n");
    expect(manifest).toHaveLength(2);
    expect(await reopened.findNode("missing")).toBeUndefined();
  });

  it("重复 beginNode 同一 nodeId 直接报错", async () => {
    const dataRoot = await freshDataRoot();
    const build = await openNodeJournals(dataRoot).beginBuild("build_03");
    await build.beginNode({ seq: 1, capability: "@hypit/seedream@1#seedream-5-lite", nodeId: "img" });
    await expect(build.beginNode({ seq: 2, capability: "@hypit/seedream@1#seedream-5-lite", nodeId: "img" })).rejects.toThrow(/img/);
  });
});
