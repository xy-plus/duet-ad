import { nodeJournalContextKey, openNodeJournals } from "@hypit/node-journal";
import type { NodeJournalBuild, NodeJournalResourceRef, NodeJournalSession } from "@hypit/node-journal";
import type {
  AsyncEndpoint,
  EndpointPackage,
  EndpointPollContext,
  EndpointRegistrar,
  EndpointStartContext,
  ImmediateEndpointHandler,
} from "@hypit/endpoint-kit";
import type { CapabilityRef, Need, StoredValue, TypeRef } from "@hypit/protocol";

/**
 * 全节点产物留痕的 runtime 挂接点（对上游的唯一 patch）。
 *
 * 包裹每个已激活 EndpointPackage 的 install：所有节点类型（生成、媒体规整、
 * 渲染）的 endpoint 调用在 invoke 前后记录语义级请求、回执、输入/输出资源
 * 引用、起止与状态到 `{dataRoot}/builds/<buildId>/node-journal/`。
 * 会话同时挂到调用上下文（nodeJournalContextKey），需要记 wire payload 的
 * Provider（provider-ark）自取，不向其他 Provider 暴露。
 */

function capabilityKey(capability: CapabilityRef): string {
  return `${capability.module.name}@${capability.module.version}#${capability.name}`;
}

function typeKey(type: TypeRef): string {
  return `${type.module.name}@${type.module.version}#${type.name}`;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** 从任意 canonical 值里收集 BlobRef 资源引用（输入约束与输出产物共用）。 */
function collectBlobRefs(value: unknown, found: NodeJournalResourceRef[] = []): NodeJournalResourceRef[] {
  if (value === null || typeof value !== "object") return found;
  if (Array.isArray(value)) {
    for (const item of value) collectBlobRefs(item, found);
    return found;
  }
  const record = value as Record<string, unknown>;
  if (record.kind === "blob" && typeof record.resource === "string") {
    found.push({
      resource: record.resource,
      size: typeof record.size === "number" ? record.size : -1,
      mediaType: typeof record.mediaType === "string" ? record.mediaType : "application/octet-stream",
    });
    return found;
  }
  for (const item of Object.values(record)) collectBlobRefs(item, found);
  return found;
}

function storedValueRefs(value: StoredValue): NodeJournalResourceRef[] {
  return collectBlobRefs(value.kind === "inline" ? value.value : value);
}

function requestPayload(need: Need): Record<string, unknown> {
  return {
    capability: capabilityKey(need.capability),
    returns: typeKey(need.returns),
    constraints: need.constraints,
  };
}

class NodeJournaling {
  readonly #journal: NodeJournalBuild;
  readonly #endpointId: string;

  constructor(journal: NodeJournalBuild, endpointId: string) {
    this.#journal = journal;
    this.#endpointId = endpointId;
  }

  /** 同一节点（need.id）复用同一会话：异步 submit/poll/collect 与进程重启都落到同一目录。 */
  async #sessionFor(need: Need): Promise<{ readonly session: NodeJournalSession; readonly created: boolean }> {
    const existing = await this.#journal.findNode(need.id);
    if (existing !== undefined) return { session: existing, created: false };
    const session = await this.#journal.beginNode({
      seq: await this.#journal.allocateSeq(),
      capability: capabilityKey(need.capability),
      nodeId: need.id,
      endpoint: this.#endpointId,
    });
    await session.recordRequest(requestPayload(need));
    await session.recordInputs(collectBlobRefs(need.constraints));
    return { session, created: true };
  }

  #attach<T extends object>(context: T, session: NodeJournalSession): T {
    return { ...context, [nodeJournalContextKey]: session };
  }

  async #finish(session: NodeJournalSession, outcome: {
    readonly status: "completed" | "failed";
    readonly receipt?: unknown;
    readonly failure?: unknown;
    readonly result?: { readonly value: StoredValue };
  }): Promise<void> {
    await session.recordReceipt({
      status: outcome.status,
      ...(outcome.receipt === undefined ? {} : { receipt: outcome.receipt }),
      ...(outcome.failure === undefined ? {} : { failure: outcome.failure }),
    });
    if (outcome.result !== undefined) await session.recordOutputs(storedValueRefs(outcome.result.value));
    await session.endNode(outcome.status);
  }

  wrapImmediate(handler: ImmediateEndpointHandler): ImmediateEndpointHandler {
    return async (context) => {
      const { session } = await this.#sessionFor(context.need);
      try {
        const result = await handler(this.#attach(context, session));
        await this.#finish(session, { status: "completed", result });
        return result;
      } catch (error) {
        await session.recordReceipt({ status: "failed", message: message(error) });
        await session.endNode("failed");
        throw error;
      }
    };
  }

  wrapAsync(endpoint: AsyncEndpoint): AsyncEndpoint {
    const run = async (
      action: "start" | "poll" | "collect",
      context: EndpointStartContext | EndpointPollContext,
      invoke: (context: never) => ReturnType<AsyncEndpoint["start"]>,
    ) => {
      const { session } = await this.#sessionFor(context.need);
      let outcome;
      try {
        outcome = await invoke(this.#attach(context, session) as never);
      } catch (error) {
        await this.#finish(session, { status: "failed", failure: { action, message: message(error) } });
        throw error;
      }
      // pending/ready 不是终态：节点保持 running，manifest 只在终态落行
      if (outcome.status === "completed" || outcome.status === "failed") {
        await this.#finish(session, {
          status: outcome.status,
          ...(outcome.receipt === undefined ? {} : { receipt: outcome.receipt }),
          ...(outcome.status === "failed" ? { failure: outcome.failure } : {}),
          ...(outcome.status === "completed" ? { result: outcome.result } : {}),
        });
      }
      return outcome;
    };
    return {
      start: (context) => run("start", context, (attached) => endpoint.start(attached)),
      poll: (context) => run("poll", context, (attached) => endpoint.poll(attached)),
      ...(endpoint.cancel === undefined ? {} : { cancel: (context: EndpointPollContext) => endpoint.cancel!(context) }),
      ...(endpoint.collect === undefined ? {} : {
        collect: (context: EndpointPollContext) => run("collect", context, (attached) => endpoint.collect!(attached)),
      }),
    };
  }
}

/** 包裹一组 EndpointPackage；返回值的 offers/credentials/pricing 与原包一致。 */
export async function wrapEndpointsWithNodeJournal(
  endpoints: readonly EndpointPackage[],
  dataRoot: string,
  build: string,
): Promise<EndpointPackage[]> {
  const journal = await openNodeJournals(dataRoot).beginBuild(build);
  return endpoints.map((endpoint) => {
    const journaling = new NodeJournaling(journal, endpoint.instance.id);
    return {
      ...endpoint,
      install: async (registrar: EndpointRegistrar) => await endpoint.install({
        registerImmediateEndpoint(id, capability, returns, handler, options) {
          registrar.registerImmediateEndpoint(id, capability, returns, journaling.wrapImmediate(handler), options);
        },
        registerAsyncEndpoint(id, capability, returns, asyncEndpoint, options) {
          registrar.registerAsyncEndpoint(id, capability, returns, journaling.wrapAsync(asyncEndpoint), options);
        },
      }),
    };
  });
}
