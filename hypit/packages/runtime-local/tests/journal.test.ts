import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { EndpointRegistry, MemoryResourceStore } from "@hypit/driver-node";
import { nodeJournalContextKey } from "@hypit/node-journal";
import type { NodeJournalSession } from "@hypit/node-journal";
import { canonicalize } from "@hypit/protocol";
import type { CapabilityRef, Need, TypeRef } from "@hypit/protocol";
import type { EndpointPackage } from "@hypit/endpoint-kit";

import { wrapEndpointsWithNodeJournal } from "../src/journal.js";

const CAPABILITY: CapabilityRef = { module: { name: "@hypit/fake", version: "1" }, name: "fake-render" };
const RETURNS: TypeRef = { module: { name: "@hypit/fake", version: "1" }, name: "FakeOut" };

function need(id: string, constraints: unknown = { ports: { prompt: ["x"] } }): Need {
  return {
    id,
    result: `rec_${id}`,
    capability: CAPABILITY,
    returns: RETURNS,
    constraints: canonicalize(constraints),
  };
}

function contextFor(target: Need, resources: MemoryResourceStore) {
  return {
    command: { kind: "fulfill-need" as const, id: `command:${target.id}`, need: target },
    need: target,
    resources,
    credentials: {},
    operation: "op_1",
  };
}

async function install(packages: readonly EndpointPackage[]): Promise<EndpointRegistry> {
  const registry = new EndpointRegistry();
  for (const pkg of packages) await pkg.install(registry);
  return registry;
}

function resolveOne(registry: EndpointRegistry, target: Need) {
  const resolution = registry.resolve(target);
  if (resolution.status !== "resolved") throw new Error(`resolve failed: ${resolution.status}`);
  return resolution.registration;
}

async function readJson(path: string): Promise<Record<string, unknown>> {
  return JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
}

/** 假 immediate endpoint：记录是否拿到 journal 会话，产出一个视频 Blob。 */
function fakeImmediatePackage(captured: { journal?: NodeJournalSession }): EndpointPackage {
  return {
    instance: { id: "fake.local", pool: "fake" },
    offers: [{ capability: CAPABILITY, returns: RETURNS, endpoint: "fake.local" }],
    credentials: [],
    install(registry) {
      registry.registerImmediateEndpoint("fake.local", CAPABILITY, RETURNS, async (context) => {
        captured.journal = (context as Record<symbol, unknown>)[nodeJournalContextKey] as NodeJournalSession;
        const artifact = await context.resources.put(new Uint8Array([1, 2]), "video/mp4");
        return { value: { kind: "inline", value: canonicalize({ videos: [artifact] }) } };
      });
    },
  };
}

describe("runtime-local 包裹层产生 journal 记录", () => {
  it("immediate 节点：语义级请求/回执/输出引用/起止/状态全部落盘", async () => {
    const dataRoot = await mkdtemp(join(tmpdir(), "hypit-runtime-journal-"));
    const captured: { journal?: NodeJournalSession } = {};
    const wrapped = await wrapEndpointsWithNodeJournal([fakeImmediatePackage(captured)], dataRoot, "build_1");
    const registry = await install(wrapped);
    const target = need("render-main");
    const registration = resolveOne(registry, target);
    if (registration.kind !== "immediate") throw new Error("unreachable");

    const resources = new MemoryResourceStore();
    const result = await registration.handler(contextFor(target, resources) as never);
    if (result.value.kind !== "inline") throw new Error("unreachable");
    const produced = (result.value.value as { videos: { resource: string }[] }).videos[0]!;

    // provider 视角拿到了同一节点会话（provider-ark 用它写 wire payload）
    expect(captured.journal).toBeDefined();
    const dir = captured.journal!.dir;
    expect(dir).toBe(join(dataRoot, "builds", "build_1", "node-journal", "nodes", "0001-fake-render"));

    const request = await readJson(join(dir, "request.json"));
    expect(request).toMatchObject({
      capability: "@hypit/fake@1#fake-render",
      constraints: { ports: { prompt: ["x"] } },
    });
    const receipt = await readJson(join(dir, "receipt.json"));
    expect(receipt).toMatchObject({ status: "completed" });
    const meta = await readJson(join(dir, "meta.json"));
    expect(meta.status).toBe("completed");
    expect(meta.nodeId).toBe("render-main");
    expect(meta.endpoint).toBe("fake.local");
    expect(typeof meta.startedAt).toBe("string");
    expect(typeof meta.endedAt).toBe("string");
    expect(meta.outputs).toEqual([{ resource: produced.resource, size: 2, mediaType: "video/mp4" }]);

    const manifest = (await readFile(join(dataRoot, "builds", "build_1", "node-journal", "manifest.jsonl"), "utf8"))
      .trim().split("\n").map((line) => JSON.parse(line) as Record<string, unknown>);
    expect(manifest).toHaveLength(1);
    expect(manifest[0]).toMatchObject({ seq: 1, nodeId: "render-main", status: "completed", dir: "nodes/0001-fake-render" });
  });

  it("immediate 节点失败：状态 failed 且错误信息进 receipt，异常继续抛出", async () => {
    const dataRoot = await mkdtemp(join(tmpdir(), "hypit-runtime-journal-"));
    const failing: EndpointPackage = {
      instance: { id: "fake.local", pool: "fake" },
      offers: [{ capability: CAPABILITY, returns: RETURNS, endpoint: "fake.local" }],
      credentials: [],
      install(registry) {
        registry.registerImmediateEndpoint("fake.local", CAPABILITY, RETURNS, async () => {
          throw new Error("渲染进程崩溃");
        });
      },
    };
    const wrapped = await wrapEndpointsWithNodeJournal([failing], dataRoot, "build_2");
    const registry = await install(wrapped);
    const target = need("render-broken");
    const registration = resolveOne(registry, target);
    if (registration.kind !== "immediate") throw new Error("unreachable");
    await expect(registration.handler(contextFor(target, new MemoryResourceStore()) as never))
      .rejects.toThrow(/渲染进程崩溃/);

    const dir = join(dataRoot, "builds", "build_2", "node-journal", "nodes", "0001-fake-render");
    const meta = await readJson(join(dir, "meta.json"));
    expect(meta.status).toBe("failed");
    const receipt = await readJson(join(dir, "receipt.json"));
    expect(receipt).toMatchObject({ status: "failed" });
    expect(String(receipt.message)).toMatch(/渲染进程崩溃/);
  });

  it("async 节点：submit 记请求，poll 留 running，collect 后才终态", async () => {
    const dataRoot = await mkdtemp(join(tmpdir(), "hypit-runtime-journal-"));
    const seen: { sessions: (NodeJournalSession | undefined)[] } = { sessions: [] };
    const asyncPackage: EndpointPackage = {
      instance: { id: "fake.local", pool: "fake" },
      offers: [{ capability: CAPABILITY, returns: RETURNS, endpoint: "fake.local" }],
      credentials: [],
      install(registry) {
        registry.registerAsyncEndpoint("fake.local", CAPABILITY, RETURNS, {
          async start(context) {
            seen.sessions.push((context as Record<symbol, never>)[nodeJournalContextKey]);
            const handle = canonicalize({ id: "task-1" });
            await context.checkpoint?.({ handle, receipt: { id: "task-1" } });
            return { status: "pending", handle, wakeAt: Date.now() + 1000, receipt: { id: "task-1" } };
          },
          async poll(context) {
            seen.sessions.push((context as Record<symbol, never>)[nodeJournalContextKey]);
            return { status: "ready", handle: context.handle, receipt: { id: "task-1" } };
          },
          async collect(context) {
            seen.sessions.push((context as Record<symbol, never>)[nodeJournalContextKey]);
            const artifact = await context.resources.put(new Uint8Array([7]), "image/png");
            return {
              status: "completed" as const,
              result: { value: { kind: "inline" as const, value: canonicalize({ images: [artifact] }) } },
              receipt: { id: "task-1" },
            };
          },
        });
      },
    };
    const wrapped = await wrapEndpointsWithNodeJournal([asyncPackage], dataRoot, "build_3");
    const registry = await install(wrapped);
    const target = need("generate-shot");
    const registration = resolveOne(registry, target);
    if (registration.kind !== "asynchronous") throw new Error("unreachable");
    const base = contextFor(target, new MemoryResourceStore());

    const started = await registration.endpoint.start({ ...base, checkpoint: async () => undefined } as never);
    expect(started.status).toBe("pending");
    const dir = join(dataRoot, "builds", "build_3", "node-journal", "nodes", "0001-fake-render");
    // submit 后：请求已记、节点仍 running、manifest 还没有这一行
    expect(await readJson(join(dir, "request.json"))).toMatchObject({ constraints: { ports: { prompt: ["x"] } } });
    const metaAfterStart = await readJson(join(dir, "meta.json"));
    expect(metaAfterStart.status).toBe("running");

    const ready = await registration.endpoint.poll({ ...base, handle: (started as { handle: unknown }).handle } as never);
    expect(ready.status).toBe("ready");
    expect((await readJson(join(dir, "meta.json"))).status).toBe("running");

    if (registration.endpoint.collect === undefined) throw new Error("endpoint 缺少 collect");
    const done = await registration.endpoint.collect({ ...base, handle: (ready as { handle: unknown }).handle } as never);
    expect(done.status).toBe("completed");
    const meta = await readJson(join(dir, "meta.json"));
    expect(meta.status).toBe("completed");
    expect(meta.outputs).toEqual([{ resource: expect.any(String), size: 1, mediaType: "image/png" }]);
    const receipt = await readJson(join(dir, "receipt.json"));
    expect(receipt).toMatchObject({ status: "completed", receipt: { id: "task-1" } });

    // start/poll/collect 三次调用拿到的是同一个节点会话
    expect(seen.sessions).toHaveLength(3);
    expect(seen.sessions.every((session) => session?.dir === dir)).toBe(true);
  });

  it("async 节点 submit 即失败：节点直接终态 failed", async () => {
    const dataRoot = await mkdtemp(join(tmpdir(), "hypit-runtime-journal-"));
    const failing: EndpointPackage = {
      instance: { id: "fake.local", pool: "fake" },
      offers: [{ capability: CAPABILITY, returns: RETURNS, endpoint: "fake.local" }],
      credentials: [],
      install(registry) {
        registry.registerAsyncEndpoint("fake.local", CAPABILITY, RETURNS, {
          async start() {
            return { status: "failed" as const, failure: { code: "REJECTED", message: "供应商拒绝" } };
          },
          async poll() {
            throw new Error("不应到达 poll");
          },
        });
      },
    };
    const wrapped = await wrapEndpointsWithNodeJournal([failing], dataRoot, "build_4");
    const registry = await install(wrapped);
    const target = need("generate-rejected");
    const registration = resolveOne(registry, target);
    if (registration.kind !== "asynchronous") throw new Error("unreachable");
    const outcome = await registration.endpoint.start({ ...contextFor(target, new MemoryResourceStore()), checkpoint: async () => undefined } as never);
    expect(outcome.status).toBe("failed");
    const meta = await readJson(join(dataRoot, "builds", "build_4", "node-journal", "nodes", "0001-fake-render", "meta.json"));
    expect(meta.status).toBe("failed");
  });
});
