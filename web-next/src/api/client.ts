import type { CreateConversationIntent } from '../domain/createConversation';
import type {
  ConversationDetail,
  ConversationSummary,
  CreateConversationResponse,
  GenerationSubmitPayload,
  GenerationSubmitResponse,
  LoginResponse,
  PostprocessPayload,
  PostprocessResponse,
  PromptPatchPayload,
  PromptPatchResponse,
} from '../domain/types';
import { ApiError, apiErrorFromPayload, networkApiError } from './errors';

export const TOKEN_STORAGE_KEY = 'cvs_token';

type FetchImplementation = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

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
  private token: string | null;
  private sessionEpoch = 0;
  private currentSessionKey = '';

  constructor(options: ApiClientOptions = {}) {
    this.basePath = (options.basePath ?? '/api').replace(/\/$/u, '');
    this.storage = options.storage === undefined ? defaultStorage() : options.storage;
    this.fetchImplementation = options.fetchImplementation ?? fetch.bind(globalThis);
    this.xhrFactory = options.xhrFactory ?? (() => new XMLHttpRequest());
    this.clearQueryCache = options.clearQueryCache ?? (() => undefined);
    this.sessionKeyFactory = options.sessionKeyFactory ?? defaultSessionKey;
    this.token = this.storage?.getItem(TOKEN_STORAGE_KEY) ?? null;
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
    this.rotateSessionKey();
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

  getConversation(id: string, options: ApiRequestOptions = {}): Promise<ConversationDetail> {
    return this.requestJson(`/conversations/${encodePath(id)}`, { signal: options.signal });
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

  async submitConversation(
    id: string,
    payload: GenerationSubmitPayload,
    options: ApiRequestOptions = {},
  ): Promise<GenerationSubmitResponse> {
    if (this.submitFlights.has(id)) {
      throw new ApiError('生成请求正在提交，请等待详情更新', {
        status: 409,
        code: 'request_in_progress',
      });
    }
    this.submitFlights.add(id);
    try {
      return await this.requestJson(`/conversations/${encodePath(id)}/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: options.signal,
      });
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
