import { appendFile, mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

/**
 * 全节点产物留痕：每个 Build 的每个节点（生成、媒体规整、渲染）在
 * `{dataRoot}/builds/<buildId>/node-journal/` 下落一份语义级记录。
 *
 * 产物字节不复制，journal 只记 resource-store 引用；wire 级 payload 由
 * Provider（目前仅 provider-ark）通过 recordWireRequest/recordWireReceipt 写同一节点目录。
 */

export type NodeJournalStatus = "completed" | "failed";

/** resource-store 引用（BlobRef 去掉 kind 判别字段后的结构化投影）。 */
export type NodeJournalResourceRef = {
  readonly resource: string;
  readonly size: number;
  readonly mediaType: string;
};

export type NodeJournalBeginInput = {
  readonly seq: number;
  /** Capability 全名 `module@version#name`；目录名只取 `#` 后的短名。 */
  readonly capability: string;
  readonly nodeId: string;
  readonly endpoint?: string;
};

type NodeJournalMeta = {
  seq: number;
  nodeId: string;
  capability: string;
  endpoint?: string;
  status: "running" | NodeJournalStatus;
  startedAt: string;
  endedAt?: string;
  inputs: NodeJournalResourceRef[];
  outputs: NodeJournalResourceRef[];
};

export type NodeJournalSession = {
  readonly seq: number;
  readonly nodeId: string;
  readonly capability: string;
  /** 节点目录绝对路径。 */
  readonly dir: string;
  /** 语义级请求（runtime 层在 endpoint 调用前记录一次）。 */
  recordRequest(value: unknown): Promise<void>;
  /** 供应商回执 / 执行结果（终态时记录）。 */
  recordReceipt(value: unknown): Promise<void>;
  /** wire payload（provider 层，HTTP 发送前）。 */
  recordWireRequest(value: unknown): Promise<void>;
  /** wire 原始回执（provider 层，最近一次终态响应）。 */
  recordWireReceipt(value: unknown): Promise<void>;
  recordInputs(refs: readonly NodeJournalResourceRef[]): Promise<void>;
  recordOutputs(refs: readonly NodeJournalResourceRef[]): Promise<void>;
  /** 终态落盘：更新 meta.json 并追加 manifest.jsonl 一行。 */
  endNode(status: NodeJournalStatus): Promise<void>;
};

export type NodeJournalBuild = {
  readonly buildId: string;
  /** `<dataRoot>/builds/<buildId>/node-journal` 绝对路径。 */
  readonly dir: string;
  beginNode(input: NodeJournalBeginInput): Promise<NodeJournalSession>;
  /** 重启安全的序号分配：扫描 manifest 与已存在的节点目录取最大值 + 1。 */
  allocateSeq(): Promise<number>;
  /** 按 nodeId 找回已开始的节点（异步 poll/collect 或进程重启后续写）。 */
  findNode(nodeId: string): Promise<NodeJournalSession | undefined>;
};

export type NodeJournals = {
  beginBuild(buildId: string): Promise<NodeJournalBuild>;
};

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function shortCapabilityName(capability: string): string {
  const hash = capability.lastIndexOf("#");
  return hash === -1 ? capability : capability.slice(hash + 1);
}

/** 目录名只允许安全字符；capability 短名正常只含字母数字、点与横线。 */
function nodeDirName(seq: number, capability: string): string {
  const short = shortCapabilityName(capability).replace(/[^A-Za-z0-9._-]+/gu, "-");
  assert(short.length > 0, `capability ${capability} 没有可用的目录名`);
  return `${String(seq).padStart(4, "0")}-${short}`;
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function readJson(path: string): Promise<Record<string, unknown>> {
  const value = JSON.parse(await readFile(path, "utf8")) as unknown;
  assert(value !== null && typeof value === "object" && !Array.isArray(value), `${path} 不是 JSON 对象`);
  return value as Record<string, unknown>;
}

class JournalSession implements NodeJournalSession {
  readonly seq: number;
  readonly nodeId: string;
  readonly capability: string;
  readonly dir: string;
  readonly #metaPath: string;
  readonly #manifestPath: string;
  #meta: NodeJournalMeta;

  constructor(buildDir: string, manifestPath: string, meta: NodeJournalMeta) {
    this.#meta = meta;
    this.seq = meta.seq;
    this.nodeId = meta.nodeId;
    this.capability = meta.capability;
    this.dir = join(buildDir, "nodes", nodeDirName(meta.seq, meta.capability));
    this.#metaPath = join(this.dir, "meta.json");
    this.#manifestPath = manifestPath;
  }

  async recordRequest(value: unknown): Promise<void> {
    await writeJson(join(this.dir, "request.json"), value);
  }

  async recordReceipt(value: unknown): Promise<void> {
    await writeJson(join(this.dir, "receipt.json"), value);
  }

  async recordWireRequest(value: unknown): Promise<void> {
    await writeJson(join(this.dir, "wire-request.json"), value);
  }

  async recordWireReceipt(value: unknown): Promise<void> {
    await writeJson(join(this.dir, "wire-receipt.json"), value);
  }

  async recordInputs(refs: readonly NodeJournalResourceRef[]): Promise<void> {
    this.#meta = { ...this.#meta, inputs: [...refs] };
    await writeJson(this.#metaPath, this.#meta);
  }

  async recordOutputs(refs: readonly NodeJournalResourceRef[]): Promise<void> {
    this.#meta = { ...this.#meta, outputs: [...refs] };
    await writeJson(this.#metaPath, this.#meta);
  }

  async endNode(status: NodeJournalStatus): Promise<void> {
    assert(this.#meta.status === "running", `节点 ${this.nodeId} 已经结束（${this.#meta.status}）`);
    const endedAt = new Date().toISOString();
    this.#meta = { ...this.#meta, status, endedAt };
    await writeJson(this.#metaPath, this.#meta);
    const line = {
      seq: this.seq,
      nodeId: this.nodeId,
      capability: this.capability,
      status,
      startedAt: this.#meta.startedAt,
      endedAt,
      dir: `nodes/${nodeDirName(this.seq, this.capability)}`,
    };
    await appendFile(this.#manifestPath, `${JSON.stringify(line)}\n`, "utf8");
  }
}

class JournalBuild implements NodeJournalBuild {
  readonly buildId: string;
  readonly dir: string;
  readonly #nodesDir: string;
  readonly #manifestPath: string;

  constructor(buildId: string, dir: string) {
    this.buildId = buildId;
    this.dir = dir;
    this.#nodesDir = join(dir, "nodes");
    this.#manifestPath = join(dir, "manifest.jsonl");
  }

  static async open(buildId: string, dir: string): Promise<JournalBuild> {
    await mkdir(join(dir, "nodes"), { recursive: true });
    return new JournalBuild(buildId, dir);
  }

  async beginNode(input: NodeJournalBeginInput): Promise<NodeJournalSession> {
    assert(Number.isSafeInteger(input.seq) && input.seq > 0, `节点序号 ${input.seq} 必须是正整数`);
    assert(input.nodeId.trim().length > 0, "nodeId 不能为空");
    assert(await this.findNode(input.nodeId) === undefined, `节点 ${input.nodeId} 已有 journal 记录，不能重复 beginNode`);
    const meta: NodeJournalMeta = {
      seq: input.seq,
      nodeId: input.nodeId,
      capability: input.capability,
      ...(input.endpoint === undefined ? {} : { endpoint: input.endpoint }),
      status: "running",
      startedAt: new Date().toISOString(),
      inputs: [],
      outputs: [],
    };
    const session = new JournalSession(this.dir, this.#manifestPath, meta);
    await mkdir(session.dir, { recursive: false });
    await writeJson(join(session.dir, "meta.json"), meta);
    return session;
  }

  /** 已落盘的节点 meta（含 running 与终态），是重启后恢复序号的依据。 */
  async #persistedMetas(): Promise<NodeJournalMeta[]> {
    const entries = await readdir(this.#nodesDir, { withFileTypes: true });
    const metas: NodeJournalMeta[] = [];
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      try {
        metas.push((await readJson(join(this.#nodesDir, entry.name, "meta.json"))) as NodeJournalMeta);
      } catch {
        // 半截写坏的目录跳过，不阻塞其他节点的留痕。
      }
    }
    return metas;
  }

  async allocateSeq(): Promise<number> {
    const metas = await this.#persistedMetas();
    return metas.reduce((max, meta) => Math.max(max, meta.seq), 0) + 1;
  }

  async findNode(nodeId: string): Promise<NodeJournalSession | undefined> {
    const metas = await this.#persistedMetas();
    const meta = metas.find((item) => item.nodeId === nodeId);
    return meta === undefined ? undefined : new JournalSession(this.dir, this.#manifestPath, meta);
  }
}

/** 打开某个 dataRoot 的 journal 根（`<dataRoot>/builds`），不创建任何目录。 */
export function openNodeJournals(dataRoot: string): NodeJournals {
  return {
    beginBuild: async (buildId: string) => {
      assert(buildId.trim().length > 0, "buildId 不能为空");
      assert(!/[/\\]/u.test(buildId), `buildId ${buildId} 不能含路径分隔符`);
      return await JournalBuild.open(buildId, join(dataRoot, "builds", buildId, "node-journal"));
    },
  };
}

/**
 * runtime 层包裹 endpoint 时把节点会话挂到调用上下文上的键；provider-ark
 * 用它找到同一节点目录写 wire payload。符号键保证不与其他上下文字段冲突。
 */
export const nodeJournalContextKey: unique symbol = Symbol("hypit.node-journal.session");

export type NodeJournalCarrier = {
  readonly [nodeJournalContextKey]?: NodeJournalSession;
};
