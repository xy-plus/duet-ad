import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import App from './App';
import { createApiRuntime } from './state/runtime';
import { AppThemeProvider } from './ui/theme';

class TestIntersectionObserver implements IntersectionObserver {
  readonly root = null;
  readonly rootMargin = '';
  readonly thresholds = [];
  disconnect() {}
  observe() {}
  takeRecords() { return []; }
  unobserve() {}
}

class MemoryStorage implements Storage {
  private readonly values = new Map<string, string>();
  get length() { return this.values.size; }
  clear() { this.values.clear(); }
  getItem(key: string) { return this.values.get(key) ?? null; }
  key(index: number) { return [...this.values.keys()][index] ?? null; }
  removeItem(key: string) { this.values.delete(key); }
  setItem(key: string, value: string) { this.values.set(key, value); }
}

const baseDetail = {
  id: 'cid-1',
  title: '真实会话',
  note: '来自服务端的说明',
  status: 'done',
  navigation_status: 'generation_submission_unknown',
  error: null,
  created_at: '2026-08-25T00:00:00Z',
  updated_at: '2026-08-25T00:01:00Z',
  keyframes: [],
  prompt: '服务端提示词',
  source_prompt: '服务端提示词',
  source_prompt_sha256: 'a'.repeat(64),
  segments: [],
  voice_lines: [],
  read_only: false,
  duration_s: 8,
  fit_required: false,
  fit_mode: 'none',
  aspect_ratio: '16:9',
  resolution: '768p',
  fit_profiles: {
    '16:9': { fit_required: false, default_fit_mode: 'none' },
    '9:16': { fit_required: true, default_fit_mode: 'crop' },
  },
  dialogue: { mode: 'auto', lines: [], auto_lines: [] },
  receipt_version: 1,
  generation: {
    status: 'submission_unknown',
    error: '无法确认供应商是否已接单',
    attempt: 1,
    client_request_id: 'request-old',
    stage: 'h3',
  },
  has_source: false,
  has_video: false,
  submit_enabled: true,
  postprocess: null,
  postprocess_enabled: true,
} as const;

beforeAll(() => {
  globalThis.IntersectionObserver = TestIntersectionObserver;
});

afterEach(cleanup);

describe('production App integration', () => {
  it('uses token-only login, then loads the real list and selected detail', async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const fetchImplementation = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url === '/api/login') return Response.json({ ok: true });
      if (url === '/api/conversations') {
        return Response.json([{
          id: 'cid-1',
          title: '真实会话',
          note: '来自服务端的说明',
          status: 'done',
          navigation_status: 'generation_submission_unknown',
          created_at: '2026-08-25T00:00:00Z',
          has_video: false,
        }]);
      }
      if (url === '/api/conversations/cid-1') return Response.json(baseDetail);
      throw new Error(`unexpected request: ${url}`);
    });
    const { apiClient, queryClient } = createApiRuntime({
      fetchImplementation,
      storage: new MemoryStorage(),
      sessionKeyFactory: () => 'test-session',
    });

    render(
      <AppThemeProvider queryClient={queryClient}>
        <App apiClient={apiClient} />
      </AppThemeProvider>,
    );

    expect(screen.getByRole('heading', { name: '登录 Duet AI' })).toBeInTheDocument();
    expect(screen.queryByLabelText('用户名')).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.type(screen.getByLabelText('访问口令'), 'real-token');
    await user.click(screen.getByRole('button', { name: '登录' }));

    expect(await screen.findByText('真实会话')).toBeInTheDocument();
    expect(await screen.findByText('提交状态未知')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /重试|继续|确认生成/ })).not.toBeInTheDocument();
    await waitFor(() => expect(requests).toEqual(expect.arrayContaining([
      expect.objectContaining({ url: '/api/login' }),
      expect.objectContaining({ url: '/api/conversations' }),
      expect.objectContaining({ url: '/api/conversations/cid-1' }),
    ])));
    expect(requests.find(({ url }) => url === '/api/conversations')?.init?.headers)
      .toEqual(expect.objectContaining({}));
  });

  it.each([
    ['read_only', { read_only: true, submit_enabled: true }],
    ['submit_disabled', { read_only: false, submit_enabled: false }],
  ])('removes every provider-facing generation action for %s detail', async (_name, gate) => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    let submitCalls = 0;
    const gatedDetail = {
      ...baseDetail,
      ...gate,
      navigation_status: 'generation_resume_required',
      generation: {
        status: 'resume_required',
        error: '原任务等待继续',
        attempt: 1,
        client_request_id: 'request-resume',
        stage: 'h3',
      },
    };
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'gated-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([gatedDetail]);
        if (url === '/api/conversations/cid-1') return Response.json(gatedDetail);
        if (url === '/api/conversations/cid-1/submit') {
          submitCalls += 1;
          return Response.json({ status: 'queued' });
        }
        throw new Error(`unexpected request: ${url}`);
      }),
    });

    render(
      <AppThemeProvider queryClient={queryClient}>
        <App apiClient={apiClient} />
      </AppThemeProvider>,
    );

    expect(await screen.findByText('当前会话不可执行生成动作')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /确认生成|新建任务重试|继续原任务|继续拼接/ }))
      .not.toBeInTheDocument();
    expect(submitCalls).toBe(0);
  });
});
