import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiClient, TOKEN_STORAGE_KEY } from './client';
import { ApiError, apiErrorFromPayload } from './errors';
import { createConversationIntent } from '../domain/createConversation';

class MemoryStorage implements Storage {
  readonly values = new Map<string, string>();
  get length() { return this.values.size; }
  clear() { this.values.clear(); }
  getItem(key: string) { return this.values.get(key) ?? null; }
  key(index: number) { return [...this.values.keys()][index] ?? null; }
  removeItem(key: string) { this.values.delete(key); }
  setItem(key: string, value: string) { this.values.set(key, value); }
}

type XhrScenario = {
  readonly status?: number;
  readonly response?: unknown;
  readonly event?: 'load' | 'error' | 'abort';
};

class FakeXhr {
  static scenarios: XhrScenario[] = [];
  static sent: FakeXhr[] = [];

  status = 0;
  response: unknown = null;
  responseType: XMLHttpRequestResponseType = '';
  method = '';
  url = '';
  body: Document | XMLHttpRequestBodyInit | null = null;
  readonly headers = new Map<string, string>();
  readonly listeners = new Map<string, () => void>();
  readonly uploadListeners = new Map<string, (event: ProgressEvent) => void>();
  readonly upload = {
    addEventListener: (name: string, listener: (event: ProgressEvent) => void) => {
      this.uploadListeners.set(name, listener);
    },
  };

  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }

  setRequestHeader(name: string, value: string) {
    this.headers.set(name, value);
  }

  addEventListener(name: string, listener: () => void) {
    this.listeners.set(name, listener);
  }

  send(body: Document | XMLHttpRequestBodyInit | null) {
    this.body = body;
    FakeXhr.sent.push(this);
    this.uploadListeners.get('progress')?.({
      lengthComputable: true,
      loaded: 1,
      total: 2,
    } as ProgressEvent);
    const scenario = FakeXhr.scenarios.shift() ?? { status: 201, response: { id: 'created' } };
    this.status = scenario.status ?? 0;
    this.response = scenario.response;
    this.listeners.get(scenario.event ?? 'load')?.();
  }
}

describe('ApiClient', () => {
  beforeEach(() => {
    FakeXhr.scenarios = [];
    FakeXhr.sent = [];
  });

  it('normalizes structured and string API errors', () => {
    expect(apiErrorFromPayload(
      { detail: { code: 'prompt_changed', message: '提示词已更新' } },
      { status: 409, fallback: 'fallback' },
    )).toMatchObject({ status: 409, code: 'prompt_changed', message: '提示词已更新' });
    expect(apiErrorFromPayload(
      { detail: 'not found' },
      { status: 404, fallback: 'fallback' },
    )).toMatchObject({ status: 404, code: 'http_error', message: 'not found' });
  });

  it('stores the bearer token and implements the real JSON endpoints under /api', async () => {
    const storage = new MemoryStorage();
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const fetchImplementation = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      return new Response(JSON.stringify(
        url.endsWith('/login')
          ? { ok: true }
          : url.endsWith('/submit')
            ? { status: 'queued', attempt: 1 }
            : { id: 'c1', status: 'done' },
      ), { status: 200, headers: { 'Content-Type': 'application/json' } });
    });
    const client = new ApiClient({ storage, fetchImplementation, sessionKeyFactory: () => 'session' });

    await client.login('secret');
    await client.listConversations();
    await client.getConversation('c1');
    await client.patchPrompt('c1', {
      confirm: true,
      expected_sha256: 'a'.repeat(64),
      prompt: 'new prompt',
    });
    await client.submitConversation('c1', {
      confirm: true,
      client_request_id: 'request-1',
      dialogue_mode: 'auto',
      fit_mode: 'none',
      aspect_ratio: '16:9',
      resolution: '480p',
    });
    await client.postprocessConversation('c1', {
      confirm: true,
      options: { remove_subtitle: true, remove_brand: false },
    });

    expect(storage.getItem(TOKEN_STORAGE_KEY)).toBe('secret');
    expect(requests.map(({ url }) => url)).toEqual([
      '/api/login',
      '/api/conversations',
      '/api/conversations/c1',
      '/api/conversations/c1/prompt',
      '/api/conversations/c1/submit',
      '/api/conversations/c1/postprocess',
    ]);
    expect(new Headers(requests[1].init?.headers).get('Authorization')).toBe('Bearer secret');
    expect(requests.slice(2).map(({ init }) => init?.method ?? 'GET'))
      .toEqual(['GET', 'PATCH', 'POST', 'POST']);
  });

  it('clears token and query cache on any 401', async () => {
    const storage = new MemoryStorage();
    storage.setItem(TOKEN_STORAGE_KEY, 'expired');
    const clearQueryCache = vi.fn();
    const client = new ApiClient({
      storage,
      clearQueryCache,
      fetchImplementation: async () => new Response(
        JSON.stringify({ detail: 'invalid token' }),
        { status: 401, headers: { 'Content-Type': 'application/json' } },
      ),
    });

    await expect(client.listConversations()).rejects.toMatchObject({
      status: 401,
      code: 'unauthorized',
    });
    expect(storage.getItem(TOKEN_STORAGE_KEY)).toBeNull();
    expect(clearQueryCache).toHaveBeenCalledOnce();
  });

  it('keeps one upload intent id across a failed retry and preserves progress', async () => {
    const storage = new MemoryStorage();
    storage.setItem(TOKEN_STORAGE_KEY, 'secret');
    FakeXhr.scenarios = [
      { event: 'error' },
      { status: 201, response: { id: 'created', status: 'queued' } },
    ];
    const client = new ApiClient({
      storage,
      xhrFactory: () => new FakeXhr() as unknown as XMLHttpRequest,
    });
    const intent = createConversationIntent({
      source: { kind: 'url', url: 'https://example.com/video.mp4' },
      note: 'note',
      voice: { mode: 'translate', targetLanguage: '粤语（香港）' },
    }, () => 'upload-request-1');
    const progress = vi.fn();

    await expect(client.createConversation(intent, { onProgress: progress }))
      .rejects.toBeInstanceOf(ApiError);
    await expect(client.createConversation(intent, { onProgress: progress }))
      .resolves.toEqual({ id: 'created', status: 'queued' });

    const bodies = FakeXhr.sent.map(({ body }) => body as FormData);
    expect(bodies.map((body) => body.get('client_request_id')))
      .toEqual(['upload-request-1', 'upload-request-1']);
    expect(bodies[1].get('reference_url')).toBe('https://example.com/video.mp4');
    expect(bodies[1].get('voice_mode')).toBe('translate');
    expect(bodies[1].get('target_language')).toBe('粤语（香港）');
    expect(progress).toHaveBeenCalledWith(0.5);
  });

  it('supports keep, rewrite and arbitrary free-form translation targets', () => {
    const source = { kind: 'url' as const, url: 'https://example.com/video.mp4' };

    expect(createConversationIntent({ source }, () => 'request-keep').voice)
      .toEqual({ mode: 'keep' });
    expect(createConversationIntent({ source, voice: { mode: 'rewrite' } }, () => 'request-rewrite').voice)
      .toEqual({ mode: 'rewrite' });
    expect(createConversationIntent({
      source,
      voice: { mode: 'translate', targetLanguage: '  自由填写的方言目标  ' },
    }, () => 'request-translate').voice).toEqual({
      mode: 'translate',
      targetLanguage: '自由填写的方言目标',
    });
  });

  it('single-flights a conversation submit so a double action makes one POST', async () => {
    const storage = new MemoryStorage();
    storage.setItem(TOKEN_STORAGE_KEY, 'secret');
    let release: ((response: Response) => void) | undefined;
    const fetchImplementation = vi.fn((
      input: RequestInfo | URL,
      init?: RequestInit,
    ) => new Promise<Response>((resolve) => {
      void input;
      void init;
      release = resolve;
    }));
    const client = new ApiClient({ storage, fetchImplementation });
    const payload = {
      confirm: true as const,
      client_request_id: 'request-1',
      dialogue_mode: 'auto' as const,
      fit_mode: 'none' as const,
      aspect_ratio: '16:9' as const,
      resolution: '480p' as const,
    };

    const first = client.submitConversation('c1', payload);
    await expect(client.submitConversation('c1', payload)).rejects.toMatchObject({
      code: 'request_in_progress',
    });
    expect(fetchImplementation).toHaveBeenCalledOnce();
    release?.(new Response(JSON.stringify({ status: 'queued', attempt: 1 }), {
      status: 202,
      headers: { 'Content-Type': 'application/json' },
    }));
    await expect(first).resolves.toEqual({ status: 'queued', attempt: 1 });
  });

  it('keeps an ambiguous submit reconciliation lock until GET returns authoritative generation', async () => {
    const storage = new MemoryStorage();
    storage.setItem(TOKEN_STORAGE_KEY, 'secret');
    let detailGeneration: unknown = null;
    let submitCalls = 0;
    const fetchImplementation = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/submit')) {
        submitCalls += 1;
        throw new TypeError('connection reset');
      }
      if (url === '/api/conversations/c1') {
        return Response.json({ id: 'c1', generation: detailGeneration });
      }
      throw new Error(`unexpected request ${url}`);
    });
    const client = new ApiClient({ storage, fetchImplementation });
    const payload = {
      confirm: true as const,
      client_request_id: 'request-ambiguous',
      dialogue_mode: 'auto' as const,
      fit_mode: 'none' as const,
      aspect_ratio: '16:9' as const,
      resolution: '480p' as const,
    };

    await expect(client.submitConversation('c1', payload)).rejects.toMatchObject({ code: 'network_error' });
    await expect(client.submitConversation('c1', payload)).rejects.toMatchObject({ code: 'request_in_progress' });
    expect(submitCalls).toBe(1);

    await client.getConversation('c1');
    await expect(client.submitConversation('c1', payload)).rejects.toMatchObject({ code: 'request_in_progress' });
    expect(submitCalls).toBe(1);

    detailGeneration = { status: 'failed', client_request_id: 'request-from-older-attempt' };
    await client.getConversation('c1');
    await expect(client.submitConversation('c1', payload)).rejects.toMatchObject({ code: 'request_in_progress' });
    expect(submitCalls).toBe(1);

    detailGeneration = { status: 'queued', client_request_id: 'request-ambiguous' };
    await client.getConversation('c1');
    expect(client.isSubmissionReconciling('c1')).toBe(false);
  });

  it('fetches authenticated files as Blob and leaves URL ownership to the hook', async () => {
    const storage = new MemoryStorage();
    storage.setItem(TOKEN_STORAGE_KEY, 'secret');
    const fetchImplementation = vi.fn(async (
      input: RequestInfo | URL,
      init?: RequestInit,
    ) => {
      void input;
      void init;
      return new Response(new Blob(['video']));
    });
    const client = new ApiClient({ storage, fetchImplementation });

    const blob = await client.getConversationFile('c1', 'generated.mp4');

    expect(blob).toBeInstanceOf(Blob);
    expect(fetchImplementation).toHaveBeenCalledWith(
      '/api/conversations/c1/files/generated.mp4',
      expect.objectContaining({
        headers: expect.any(Headers),
      }),
    );
    const init = fetchImplementation.mock.calls[0][1];
    expect((init?.headers as Headers).get('Authorization')).toBe('Bearer secret');
  });
});
