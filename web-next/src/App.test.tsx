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

  it('requires one CAS-bound user confirmation before optimized images can be submitted', async () => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    let acceptanceCalls = 0;
    let acceptancePayload: unknown;
    let accepted = false;
    const acceptance = () => ({
      required: true,
      accepted,
      expected_meta_sha256: (accepted ? 'c' : 'b').repeat(64),
    });
    const currentDetail = () => ({
      ...baseDetail,
      navigation_status: 'analysis_complete',
      generation: null,
      image_acceptance: acceptance(),
      postprocess: {
        status: 'done',
        options: { remove_subtitle: false, remove_brand: false, optimize_image: true },
        frames: [],
        error: null,
        segments: [{
          index: 0, status: 'done', stage: 'done', completed_frames: 1,
          total_frames: 1, revision: 1, error: null,
        }],
      },
    });
    const fetchImplementation = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === '/api/conversations') return Response.json([currentDetail()]);
      if (url === '/api/conversations/cid-1') return Response.json(currentDetail());
      if (url === '/api/conversations/cid-1/image-acceptance') {
        acceptanceCalls += 1;
        acceptancePayload = JSON.parse(String(init?.body));
        accepted = true;
        return Response.json({ status: 'accepted', image_acceptance: acceptance() });
      }
      throw new Error(`unexpected request: ${url}`);
    });
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      fetchImplementation,
      sessionKeyFactory: () => 'acceptance-session',
    });

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);

    expect(await screen.findByText('请先确认使用当前优化图，再生成视频')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '确认生成' })).not.toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: '确认使用当前优化图生成视频' }));

    await waitFor(() => expect(acceptanceCalls).toBe(1));
    expect(acceptancePayload).toEqual({
      confirm: true,
      expected_meta_sha256: 'b'.repeat(64),
    });
    expect(await screen.findByText('已确认使用当前优化图生成视频')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '确认生成' })).toBeDisabled();
    await user.click(screen.getByText('画外', { exact: true }));
    expect(screen.getByRole('button', { name: '确认生成' })).toBeEnabled();
  });

  it('does not render image confirmation after video generation has started', async () => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    const detail = {
      ...baseDetail,
      navigation_status: 'generation_running',
      generation: {
        status: 'running', stage: 'h3', client_request_id: 'started-request', segments: [],
      },
      image_acceptance: {
        required: true, accepted: false, expected_meta_sha256: 'd'.repeat(64),
      },
      postprocess: {
        status: 'done',
        options: { remove_subtitle: false, remove_brand: false, optimize_image: true },
        frames: [], error: null,
        segments: [{
          index: 0, status: 'done', stage: 'done', completed_frames: 1,
          total_frames: 1, revision: 1, error: null,
        }],
      },
    };
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'generation-started-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([detail]);
        if (url === '/api/conversations/cid-1') return Response.json(detail);
        throw new Error(`unexpected request: ${url}`);
      }),
    });

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);

    expect(await screen.findByText('生成进行中')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '确认使用当前优化图生成视频' }))
      .not.toBeInTheDocument();
  });

  it('reports an image acceptance CAS conflict without automatically resending it', async () => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    let acceptanceCalls = 0;
    const detail = {
      ...baseDetail,
      navigation_status: 'analysis_complete',
      generation: null,
      image_acceptance: {
        required: true, accepted: false, expected_meta_sha256: 'b'.repeat(64),
      },
      postprocess: {
        status: 'done',
        options: { remove_subtitle: false, remove_brand: false, optimize_image: true },
        frames: [],
        error: null,
        segments: [{
          index: 0, status: 'done', stage: 'done', completed_frames: 1,
          total_frames: 1, revision: 1, error: null,
        }],
      },
    };
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'acceptance-conflict-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([detail]);
        if (url === '/api/conversations/cid-1') return Response.json(detail);
        if (url === '/api/conversations/cid-1/image-acceptance') {
          acceptanceCalls += 1;
          return Response.json(
            { detail: 'image_acceptance_meta_changed' },
            { status: 409 },
          );
        }
        throw new Error(`unexpected request: ${url}`);
      }),
    });

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);
    const user = userEvent.setup();
    await user.click(await screen.findByRole(
      'button',
      { name: '确认使用当前优化图生成视频' },
      { timeout: 5_000 },
    ));

    expect(await screen.findByText('优化图版本已变化，请刷新后重新确认。')).toBeInTheDocument();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(acceptanceCalls).toBe(1);
  });

  it('keeps an explicit off-screen choice across multimodal refresh without automatic resubmit', async () => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    const submitPayloads: unknown[] = [];
    const detail = {
      ...baseDetail,
      navigation_status: 'analysis_complete',
      generation: null,
      dialogue_delivery: null,
      image_acceptance: {
        required: true, accepted: true, expected_meta_sha256: 'c'.repeat(64),
      },
      postprocess: {
        status: 'done',
        options: { remove_subtitle: false, remove_brand: false, optimize_image: true },
        frames: [],
        error: null,
        segments: [{
          index: 0, status: 'done', stage: 'done', completed_frames: 1,
          total_frames: 1, revision: 1, error: null,
        }],
      },
    };
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'delivery-refresh-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([detail]);
        if (url === '/api/conversations/cid-1') return Response.json(detail);
        if (url === '/api/conversations/cid-1/submit') {
          submitPayloads.push(JSON.parse(String(init?.body)));
          if (submitPayloads.length === 1) {
            return Response.json({
              detail: {
                code: 'multimodal_input_refresh_required',
                message: '需要刷新多模态输入',
              },
            }, { status: 409 });
          }
          return Response.json({ status: 'queued', attempt: 1 });
        }
        throw new Error(`unexpected request: ${url}`);
      }),
    });

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);
    const user = userEvent.setup();
    expect(await screen.findByRole('button', { name: '确认生成' })).toBeDisabled();
    await user.click(screen.getByText('画外', { exact: true }));
    await user.click(screen.getByRole('button', { name: '确认生成' }));

    expect(await screen.findByText('音频与画面输入需要刷新，请等待页面更新后再次确认生成。'))
      .toBeInTheDocument();
    expect(submitPayloads).toHaveLength(1);
    expect(submitPayloads[0]).toMatchObject({ dialogue_delivery: 'off_screen' });
    expect(screen.getByRole('radio', { name: '画外' })).toBeChecked();

    await user.click(screen.getByRole('button', { name: '确认生成' }));
    await waitFor(() => expect(submitPayloads).toHaveLength(2));
    expect(submitPayloads[1]).toMatchObject({ dialogue_delivery: 'off_screen' });
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

  it('keeps a legacy read-only detail visible when frozen generation fields are null', async () => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    let submitCalls = 0;
    const legacyDetail = {
      ...baseDetail,
      read_only: true,
      aspect_ratio: null,
      resolution: null,
      fit_mode: null,
      fit_profiles: null,
      duration_s: null,
      generation: { ...baseDetail.generation, status: 'succeeded' },
    };
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'legacy-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([legacyDetail]);
        if (url === '/api/conversations/cid-1') return Response.json(legacyDetail);
        if (url.endsWith('/submit')) {
          submitCalls += 1;
          return Response.json({ status: 'queued' });
        }
        throw new Error(`unexpected request: ${url}`);
      }),
    });

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);

    expect(await screen.findByText('分析产物摘要')).toBeInTheDocument();
    expect(screen.getByText('已冻结生成参数')).toBeInTheDocument();
    expect(screen.getByText('视频生成完成')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /确认生成|新建任务重试|继续原任务|继续拼接/ }))
      .not.toBeInTheDocument();
    expect(submitCalls).toBe(0);
  });

  it.each([
    ['null status', null, 8],
    ['unknown status', 'future_status', 8],
    ['unknown duration', null, null],
  ])('fails closed in the App for malformed generation %s', async (_name, status, duration) => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    let submitCalls = 0;
    const malformedDetail = {
      ...baseDetail,
      duration_s: duration,
      generation: status === null && duration === null
        ? null
        : { ...baseDetail.generation, status },
    };
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'malformed-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([malformedDetail]);
        if (url === '/api/conversations/cid-1') return Response.json(malformedDetail);
        if (url.endsWith('/submit')) {
          submitCalls += 1;
          return Response.json({ status: 'queued' });
        }
        throw new Error(`unexpected request: ${url}`);
      }),
    });

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);

    expect(await screen.findByText('提交状态未知')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /确认生成|新建任务重试|继续原任务|继续拼接/ }))
      .not.toBeInTheDocument();
    expect(submitCalls).toBe(0);
  });

  it('locks the paid action while an ambiguous submit is reconciled through GET only', async () => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    let submitCalls = 0;
    let detailCalls = 0;
    const newDetail = { ...baseDetail, navigation_status: 'prompt_confirmed', generation: null };
    const unknownDetail = {
      ...newDetail,
      navigation_status: 'generation_submission_unknown',
      generation: { ...baseDetail.generation, status: null },
    };
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'reconciliation-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([newDetail]);
        if (url === '/api/conversations/cid-1') {
          detailCalls += 1;
          return Response.json(submitCalls > 0 ? unknownDetail : newDetail);
        }
        if (url.endsWith('/submit')) {
          submitCalls += 1;
          throw new TypeError('connection reset after send');
        }
        throw new Error(`unexpected request: ${url}`);
      }),
    });

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: '确认生成' }));

    expect(await screen.findByText('正在核对提交结果，已锁定再次提交')).toBeInTheDocument();
    await waitFor(() => expect(detailCalls).toBeGreaterThan(1));
    expect(screen.queryByRole('button', { name: '确认生成' })).not.toBeInTheDocument();
    expect(submitCalls).toBe(1);
  });

  it.each([
    [
      'resume',
      '继续原任务',
      {
        ...baseDetail,
        navigation_status: 'generation_resume_required',
        generation: {
          status: 'resume_required',
          error: '原任务等待继续',
          attempt: 1,
          client_request_id: 'request-resume',
          stage: 'h3',
        },
      },
    ],
    [
      'retry_stitch',
      '继续拼接',
      {
        ...baseDetail,
        duration_s: 30,
        segment_count: 3,
        plan_receipt: 'b'.repeat(64),
        navigation_status: 'generation_failed',
        generation: {
          status: 'failed',
          error: '拼接失败',
          attempt: 1,
          client_request_id: 'request-stitch',
          stage: 'stitch',
          fast_mode: false,
          segments: [],
        },
      },
    ],
  ])('keeps an ambiguous reused-id %s action locked when GET only returns its baseline proof', async (
    _name,
    actionLabel,
    baselineDetail,
  ) => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    let submitCalls = 0;
    let detailCalls = 0;
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'reused-id-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([baselineDetail]);
        if (url === '/api/conversations/cid-1') {
          detailCalls += 1;
          return Response.json(baselineDetail);
        }
        if (url.endsWith('/submit')) {
          submitCalls += 1;
          throw new TypeError('connection reset after reused-id submit');
        }
        throw new Error(`unexpected request: ${url}`);
      }),
    });

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);
    const user = userEvent.setup();
    await user.click(await screen.findByRole(
      'button',
      { name: actionLabel },
      { timeout: 4_000 },
    ));

    expect(await screen.findByText('正在核对提交结果，已锁定再次提交')).toBeInTheDocument();
    await waitFor(() => expect(detailCalls).toBeGreaterThan(1));
    expect(screen.queryByRole('button', { name: actionLabel })).not.toBeInTheDocument();
    expect(submitCalls).toBe(1);
  }, 10_000);

  it('keeps a string submission_outcome_unknown response locked in the App', async () => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    let submitCalls = 0;
    let detailCalls = 0;
    const newDetail = { ...baseDetail, navigation_status: 'prompt_confirmed', generation: null };
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'outcome-unknown-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([newDetail]);
        if (url === '/api/conversations/cid-1') {
          detailCalls += 1;
          return Response.json(newDetail);
        }
        if (url.endsWith('/submit')) {
          submitCalls += 1;
          return Response.json({ detail: 'submission_outcome_unknown' }, { status: 409 });
        }
        throw new Error(`unexpected request: ${url}`);
      }),
    });

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);
    const user = userEvent.setup();
    await user.click(await screen.findByRole(
      'button',
      { name: '确认生成' },
      { timeout: 4_000 },
    ));

    expect(await screen.findByText('正在核对提交结果，已锁定再次提交')).toBeInTheDocument();
    await waitFor(() => expect(detailCalls).toBeGreaterThan(1));
    expect(screen.queryByRole('button', { name: '确认生成' })).not.toBeInTheDocument();
    expect(submitCalls).toBe(1);
  }, 10_000);

  it('withholds postprocess retry authority from a read-only detail', async () => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    let postprocessCalls = 0;
    const readOnlyDetail = {
      ...baseDetail,
      read_only: true,
      postprocess: {
        status: 'failed',
        options: { remove_subtitle: true, remove_brand: false },
        frames: [],
        error: '服务端确认失败',
      },
    };
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'postprocess-gate-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([readOnlyDetail]);
        if (url === '/api/conversations/cid-1') return Response.json(readOnlyDetail);
        if (url.endsWith('/postprocess')) {
          postprocessCalls += 1;
          return Response.json({ status: 'running', frames: [] });
        }
        throw new Error(`unexpected request: ${url}`);
      }),
    });

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);

    expect(await screen.findByText('后处理失败')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '重试失败项' })).not.toBeInTheDocument();
    expect(postprocessCalls).toBe(0);
  });
});
