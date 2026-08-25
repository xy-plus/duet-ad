import type { CreateConversationIntent } from '../domain/createConversation';
import type {
  ConversationDetail,
  ConversationSummary,
  CreateConversationResponse,
  GenerationSubmitPayload,
  GenerationSubmitResponse,
  ImageOptimizationPrompt,
  ImageOptimizationPromptPatchPayload,
  LoginResponse,
  PostprocessPayload,
  PostprocessResponse,
  PostprocessSegmentRetryPayload,
  PromptPatchPayload,
  PromptPatchResponse,
} from '../domain/types';
import { ApiError, apiErrorFromPayload, networkApiError } from './errors';

export const TOKEN_STORAGE_KEY = 'cvs_token';
const SUBMISSION_RECONCILIATION_STORAGE_KEY = 'cvs_submission_reconciliations_v1';
const SUBMISSION_RECONCILIATION_VERSION = 1;

type FetchImplementation = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

interface GenerationProof {
  readonly requestId: string | null;
  readonly status: string | null;
  readonly stage: string | null;
}

interface SubmissionReconciliation {
  readonly requestId: string;
  readonly baselineKnown: boolean;
  readonly baseline: GenerationProof | null;
}

interface StoredSubmissionReconciliation extends SubmissionReconciliation {
  readonly conversationId: string;
}

const AUTHORITATIVE_GENERATION_STATUSES = new Set([
  'queued',
  'submitting',
  'running',
  'failed',
  'resume_required',
  'succeeded',
]);

export interface ApiClientOptions {
  readonly basePath?: string;
  readonly storage?: Storage | null;
  readonly fetchImplementation?: FetchImplementation;
  readonly xhrFactory?: () => XMLHttpRequest;
  readonly clearQueryCache?: () => void;
  readonly sessionKeyFactory?: () => string;
}

export interface ApiRequestOptions {
  readonly signal?: AbortSignal;
}

export interface CreateConversationRequestOptions extends ApiRequestOptions {
  readonly onProgress?: (ratio: number) => void;
}

function defaultStorage(): Storage | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage;
  } catch {
    return null;
  }
}

function defaultSessionKey(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `session-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function encodePath(value: string): string {
  return encodeURIComponent(value);
}

function encodeFilePath(value: string): string {
  return value.split('/').map(encodePath).join('/');
}

function record(value: unknown): Readonly<Record<string, unknown>> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : null;
}

function hasExactKeys(value: Readonly<Record<string, unknown>>, keys: readonly string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every((key) => Object.hasOwn(value, key));
}

function generationProofFromStored(value: unknown): GenerationProof | null | undefined {
  if (value === null) return null;
  const stored = record(value);
  if (!stored || !hasExactKeys(stored, ['requestId', 'status', 'stage'])) return undefined;
  if ((stored.requestId !== null && typeof stored.requestId !== 'string')
      || (stored.status !== null && typeof stored.status !== 'string')
      || (stored.stage !== null && typeof stored.stage !== 'string')) {
    return undefined;
  }
  return {
    requestId: stored.requestId,
    status: stored.status,
    stage: stored.stage,
  };
}

function readSubmissionReconciliations(storage: Storage | null): {
  readonly valid: boolean;
  readonly entries: Map<string, SubmissionReconciliation>;
} {
  const entries = new Map<string, SubmissionReconciliation>();
  if (!storage) return { valid: false, entries };
  try {
    const raw = storage.getItem(SUBMISSION_RECONCILIATION_STORAGE_KEY);
    if (raw === null) return { valid: true, entries };
    const envelope = record(JSON.parse(raw) as unknown);
    if (!envelope
        || !hasExactKeys(envelope, ['version', 'entries'])
        || envelope.version !== SUBMISSION_RECONCILIATION_VERSION
        || !Array.isArray(envelope.entries)) {
      return { valid: false, entries };
    }
    for (const value of envelope.entries) {
      const stored = record(value);
      if (!stored
          || !hasExactKeys(stored, ['conversationId', 'requestId', 'baselineKnown', 'baseline'])
          || typeof stored.conversationId !== 'string'
          || stored.conversationId.length === 0
          || typeof stored.requestId !== 'string'
          || stored.requestId.length === 0
          || typeof stored.baselineKnown !== 'boolean') {
        return { valid: false, entries: new Map() };
      }
      const baseline = generationProofFromStored(stored.baseline);
      if (baseline === undefined || (!stored.baselineKnown && baseline !== null)
          || entries.has(stored.conversationId)) {
        return { valid: false, entries: new Map() };
      }
      entries.set(stored.conversationId, {
        requestId: stored.requestId,
        baselineKnown: stored.baselineKnown,
        baseline,
      });
    }
    return { valid: true, entries };
  } catch {
    return { valid: false, entries };
  }
}

async function responsePayload(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export class ApiClient {
  readonly basePath: string;
  private readonly storage: Storage | null;
  private readonly fetchImplementation: FetchImplementation;
  private readonly xhrFactory: () => XMLHttpRequest;
  private readonly clearQueryCache: () => void;
  private readonly sessionKeyFactory: () => string;
  private readonly listeners = new Set<() => void>();
  private readonly submitFlights = new Set<string>();
  private readonly submissionReconciliations: Map<string, SubmissionReconciliation>;
  private readonly generationProofs = new Map<string, GenerationProof | null>();
  private reconciliationStorageValid: boolean;
  private token: string | null;
  private sessionEpoch = 0;
  private currentSessionKey = '';

  constructor(options: ApiClientOptions = {}) {
    this.basePath = (options.basePath ?? '/api').replace(/\/$/u, '');
    this.storage = options.storage === undefined ? defaultStorage() : options.storage;
    this.fetchImplementation = options.fetchImplementation ?? ((input, init) => fetch(input, init));
    this.xhrFactory = options.xhrFactory ?? (() => new XMLHttpRequest());
    this.clearQueryCache = options.clearQueryCache ?? (() => undefined);
    this.sessionKeyFactory = options.sessionKeyFactory ?? defaultSessionKey;
    this.token = this.storage?.getItem(TOKEN_STORAGE_KEY) ?? null;
    const reconciliations = readSubmissionReconciliations(this.storage);
    this.submissionReconciliations = reconciliations.entries;
    this.reconciliationStorageValid = reconciliations.valid;
    this.rotateSessionKey();
  }

  get sessionKey(): string {
    return this.currentSessionKey;
  }

  get hasToken(): boolean {
    return this.token !== null && this.token.length > 0;
  }

  readonly getSessionSnapshot = (): string => this.currentSessionKey;

  readonly subscribeSession = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  private rotateSessionKey(): void {
    this.sessionEpoch += 1;
    this.currentSessionKey = `${this.sessionKeyFactory()}:${this.sessionEpoch}`;
    for (const listener of this.listeners) listener();
  }

  private setToken(token: string): void {
    this.token = token;
    this.storage?.setItem(TOKEN_STORAGE_KEY, token);
    this.rotateSessionKey();
  }

  clearSession(): void {
    this.token = null;
    this.storage?.removeItem(TOKEN_STORAGE_KEY);
    this.clearQueryCache();
    this.submitFlights.clear();
    this.generationProofs.clear();
    this.rotateSessionKey();
  }

  private refreshSubmissionReconciliations(): void {
    const stored = readSubmissionReconciliations(this.storage);
    this.reconciliationStorageValid = stored.valid;
    if (!stored.valid) return;
    this.submissionReconciliations.clear();
    for (const [id, reconciliation] of stored.entries) {
      this.submissionReconciliations.set(id, reconciliation);
    }
  }

  private persistSubmissionReconciliations(
    entries: ReadonlyMap<string, SubmissionReconciliation>,
  ): boolean {
    if (!this.storage || !this.reconciliationStorageValid) return false;
    const storedEntries: StoredSubmissionReconciliation[] = [...entries].map(([
      conversationId,
      reconciliation,
    ]) => ({ conversationId, ...reconciliation }));
    try {
      this.storage.setItem(SUBMISSION_RECONCILIATION_STORAGE_KEY, JSON.stringify({
        version: SUBMISSION_RECONCILIATION_VERSION,
        entries: storedEntries,
      }));
      return true;
    } catch {
      return false;
    }
  }

  private beginSubmissionReconciliation(
    id: string,
    reconciliation: SubmissionReconciliation,
  ): void {
    const next = new Map(this.submissionReconciliations);
    next.set(id, reconciliation);
    if (!this.persistSubmissionReconciliations(next)) {
      throw new ApiError('无法安全记录生成请求，已阻止提交', {
        status: 0,
        code: 'reconciliation_persistence_failed',
      });
    }
    this.submissionReconciliations.set(id, reconciliation);
  }

  private clearSubmissionReconciliation(id: string): void {
    if (!this.submissionReconciliations.has(id)) return;
    const next = new Map(this.submissionReconciliations);
    next.delete(id);
    if (!this.persistSubmissionReconciliations(next)) return;
    this.submissionReconciliations.delete(id);
  }

  private path(path: string): string {
    return `${this.basePath}${path.startsWith('/') ? path : `/${path}`}`;
  }

  private async fetchResponse(
    path: string,
    init: RequestInit = {},
    authenticated = true,
  ): Promise<Response> {
    const headers = new Headers(init.headers);
    if (authenticated && this.token) headers.set('Authorization', `Bearer ${this.token}`);
    try {
      return await this.fetchImplementation(this.path(path), { ...init, headers });
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw networkApiError('无法连接服务器，请检查网络后重试', error);
    }
  }

  private async requestJson<T>(
    path: string,
    init: RequestInit = {},
    authenticated = true,
  ): Promise<T> {
    const response = await this.fetchResponse(path, init, authenticated);
    const payload = await responsePayload(response);
    if (response.status === 401) this.clearSession();
    if (!response.ok) {
      throw apiErrorFromPayload(payload, {
        status: response.status,
        fallback: `请求失败（${response.status}），请稍后重试`,
      });
    }
    if (payload === null) {
      throw new ApiError('服务器响应格式无效', {
        status: response.status,
        code: 'invalid_response',
      });
    }
    return payload as T;
  }

  async login(token: string, options: ApiRequestOptions = {}): Promise<LoginResponse> {
    const response = await this.requestJson<LoginResponse>('/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
      signal: options.signal,
    }, false);
    this.setToken(token);
    return response;
  }

  listConversations(options: ApiRequestOptions = {}): Promise<readonly ConversationSummary[]> {
    return this.requestJson('/conversations', { signal: options.signal });
  }

  async getConversation(id: string, options: ApiRequestOptions = {}): Promise<ConversationDetail> {
    const detail = await this.requestJson<ConversationDetail>(
      `/conversations/${encodePath(id)}`,
      { signal: options.signal },
    );
    const generation = detail.generation;
    const proof: GenerationProof | null = generation ? {
      requestId: typeof generation.client_request_id === 'string'
        ? generation.client_request_id
        : null,
      status: typeof generation.status === 'string' ? generation.status : null,
      stage: typeof generation.stage === 'string' ? generation.stage : null,
    } : null;
    this.refreshSubmissionReconciliations();
    const reconciliation = this.submissionReconciliations.get(id);
    const authoritativeProgress = reconciliation
      && proof?.requestId === reconciliation.requestId
      && proof.status !== null
      && AUTHORITATIVE_GENERATION_STATUSES.has(proof.status)
      && reconciliation.baselineKnown
      && (reconciliation.baseline?.requestId !== reconciliation.requestId
        || proof.status !== reconciliation.baseline.status
        || proof.stage !== reconciliation.baseline.stage);
    if (authoritativeProgress) {
      this.clearSubmissionReconciliation(id);
    }
    this.generationProofs.set(id, proof);
    return detail;
  }

  isSubmissionReconciling(id: string): boolean {
    this.refreshSubmissionReconciliations();
    return this.submissionReconciliations.has(id);
  }

  patchPrompt(
    id: string,
    payload: PromptPatchPayload,
    options: ApiRequestOptions = {},
  ): Promise<PromptPatchResponse> {
    return this.requestJson(`/conversations/${encodePath(id)}/prompt`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: options.signal,
    });
  }

  patchImageOptimizationPrompt(
    id: string,
    payload: ImageOptimizationPromptPatchPayload,
    options: ApiRequestOptions = {},
  ): Promise<ImageOptimizationPrompt> {
    return this.requestJson(`/conversations/${encodePath(id)}/image-optimization-prompt`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload), signal: options.signal,
    });
  }

  async submitConversation(
    id: string,
    payload: GenerationSubmitPayload,
    options: ApiRequestOptions = {},
  ): Promise<GenerationSubmitResponse> {
    this.refreshSubmissionReconciliations();
    if (this.submitFlights.has(id) || this.submissionReconciliations.has(id)) {
      throw new ApiError('生成请求正在提交，请等待详情更新', {
        status: 409,
        code: 'request_in_progress',
      });
    }
    const reconciliation: SubmissionReconciliation = {
      requestId: payload.client_request_id,
      baselineKnown: this.generationProofs.has(id),
      baseline: this.generationProofs.get(id) ?? null,
    };
    this.beginSubmissionReconciliation(id, reconciliation);
    this.submitFlights.add(id);
    try {
      const response = await this.requestJson<GenerationSubmitResponse>(`/conversations/${encodePath(id)}/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: options.signal,
      });
      if (response.status !== 'queued'
          && response.status !== 'submitting'
          && response.status !== 'running') {
        throw new ApiError('提交结果无法确认，正在核对服务端状态', {
          status: 0,
          code: 'submission_unknown',
        });
      }
      this.clearSubmissionReconciliation(id);
      return response;
    } catch (error) {
      const ambiguous = error instanceof ApiError
        && (error.code === 'network_error'
          || error.code === 'invalid_response'
          || error.code === 'submission_unknown'
          || error.code === 'submission_outcome_unknown'
          || error.status >= 500);
      if (ambiguous) {
        this.beginSubmissionReconciliation(id, reconciliation);
      } else {
        this.clearSubmissionReconciliation(id);
      }
      throw error;
    } finally {
      this.submitFlights.delete(id);
    }
  }

  postprocessConversation(
    id: string,
    payload: PostprocessPayload,
    options: ApiRequestOptions = {},
  ): Promise<PostprocessResponse> {
    return this.requestJson(`/conversations/${encodePath(id)}/postprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: options.signal,
    });
  }

  retryPostprocessSegment(
    id: string,
    index: number,
    payload: PostprocessSegmentRetryPayload,
    options: ApiRequestOptions = {},
  ): Promise<PostprocessResponse> {
    return this.requestJson(`/conversations/${encodePath(id)}/postprocess/segments/${index}/retry`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload), signal: options.signal,
    });
  }

  async getConversationFile(
    id: string,
    name: string,
    options: ApiRequestOptions = {},
  ): Promise<Blob> {
    const response = await this.fetchResponse(
      `/conversations/${encodePath(id)}/files/${encodeFilePath(name)}`,
      { signal: options.signal },
    );
    if (response.status === 401) this.clearSession();
    if (!response.ok) {
      throw apiErrorFromPayload(await responsePayload(response), {
        status: response.status,
        fallback: `文件加载失败（${response.status}）`,
      });
    }
    return response.blob();
  }

  createConversation(
    intent: CreateConversationIntent,
    options: CreateConversationRequestOptions = {},
  ): Promise<CreateConversationResponse> {
    return new Promise((resolve, reject) => {
      const xhr = this.xhrFactory();
      let settled = false;
      const finish = (callback: () => void) => {
        if (settled) return;
        settled = true;
        options.signal?.removeEventListener('abort', abort);
        callback();
      };
      const abort = () => {
        xhr.abort();
      };

      xhr.open('POST', this.path('/conversations'));
      if (this.token) xhr.setRequestHeader('Authorization', `Bearer ${this.token}`);
      xhr.responseType = 'json';
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && event.total > 0) {
          options.onProgress?.(event.loaded / event.total);
        }
      });
      xhr.addEventListener('load', () => finish(() => {
        if (xhr.status === 200 || xhr.status === 201) {
          resolve(xhr.response as CreateConversationResponse);
          return;
        }
        if (xhr.status === 401) this.clearSession();
        reject(apiErrorFromPayload(xhr.response, {
          status: xhr.status,
          fallback: `上传失败（${xhr.status}），请稍后重试`,
        }));
      }));
      xhr.addEventListener('error', () => finish(() => {
        reject(networkApiError('网络异常，上传未完成，请使用同一请求重试'));
      }));
      xhr.addEventListener('abort', () => finish(() => {
        reject(new ApiError('上传已中断，请重试', { status: 0, code: 'upload_aborted' }));
      }));

      const body = new FormData();
      if (intent.source.kind === 'file') {
        body.append('file', intent.source.file, intent.source.file.name);
      } else {
        body.append('reference_url', intent.source.url);
      }
      if (intent.note) body.append('note', intent.note);
      body.append('client_request_id', intent.clientRequestId);
      body.append('voice_mode', intent.voice.mode);
      if (intent.voice.mode === 'translate') {
        body.append('target_language', intent.voice.targetLanguage);
      }

      if (options.signal?.aborted) {
        reject(new ApiError('上传已中断，请重试', { status: 0, code: 'upload_aborted' }));
        return;
      }
      options.signal?.addEventListener('abort', abort, { once: true });
      xhr.send(body);
    });
  }
}
