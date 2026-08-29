import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import App from './App';
import { queryKeys } from './state';
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

  it('shows partial prompt fusion through the same ordered segment model', async () => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    const detail = {
      ...baseDetail,
      duration_s: 20,
      segment_count: 2,
      plan_receipt: 'e'.repeat(64),
      navigation_status: 'analysis_complete',
      segments: [
        { index: 1, prompt: '片段一旧视频提示词', lines: [], keyframes: [] },
        { index: 2, prompt: '片段二旧视频提示词', lines: [], keyframes: [] },
      ],
      generation: null,
      image_acceptance: {
        required: true, accepted: true, expected_meta_sha256: 'f'.repeat(64),
      },
      postprocess: {
        status: 'done',
        options: { remove_subtitle: false, remove_brand: false, optimize_image: true },
        frames: [], error: null,
        segments: [1, 2].map((index) => ({
          index, status: 'done', stage: 'done', completed_frames: 9,
          total_frames: 9, revision: 1, error: null,
        })),
      },
      prompt_fusion: {
        status: 'running', error: null,
        segments: [
          { index: 2, status: 'running', final_prompt: '未完成提示词不得显示', error: null },
          { index: 1, status: 'done', final_prompt: '片段一最终融合提示词', error: null },
        ],
      },
    };
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'prompt-fusion-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([detail]);
        if (url === '/api/conversations/cid-1') return Response.json(detail);
        throw new Error(`unexpected request: ${url}`);
      }),
    });

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);

    const fusion = await screen.findByRole('region', { name: '最终提示词融合' });
    const generation = screen.getByRole('region', { name: '视频生成' });
    expect(fusion.compareDocumentPosition(generation) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0);
    expect(screen.getByText('片段 1 · 融合完成')).toBeInTheDocument();
    expect(screen.getByText('片段 2 · 融合中')).toBeInTheDocument();
    expect(screen.getByLabelText('片段 1 最终提示词')).toHaveTextContent('片段一最终融合提示词');
    expect(screen.queryByText('未完成提示词不得显示')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '确认生成' })).toBeDisabled();

    const user = userEvent.setup();
    await user.click(screen.getAllByRole('button', { name: '展开旧视频提示词（融合输入）' })[0]);
    expect(screen.getByRole('textbox', { name: '旧视频提示词（融合输入）' }))
      .toHaveValue('片段一旧视频提示词');
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

  it('continues a legacy multimodal refresh as one operation without another submit', async () => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    const submitPayloads: Array<{
      client_request_id: string;
      dialogue_delivery?: string;
    }> = [];
    const detailOperationRequestIds: string[] = [];
    let operationRequestId: string | null = null;
    const currentDetail = () => ({
      ...baseDetail,
      navigation_status: operationRequestId ? 'generation_running' : 'analysis_complete',
      generation: operationRequestId ? {
        status: 'running', stage: 'context_ir', attempt: 1,
        client_request_id: operationRequestId, segments: [],
      } : null,
      dialogue_delivery: operationRequestId ? 'off_screen' : null,
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
    });
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'delivery-refresh-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([currentDetail()]);
        if (url === '/api/conversations/cid-1') {
          if (operationRequestId) detailOperationRequestIds.push(operationRequestId);
          return Response.json(currentDetail());
        }
        if (url === '/api/conversations/cid-1/submit') {
          const payload = JSON.parse(String(init?.body)) as {
            client_request_id: string;
            dialogue_delivery?: string;
          };
          submitPayloads.push(payload);
          operationRequestId = payload.client_request_id;
          return Response.json({
            detail: {
              code: 'multimodal_input_refresh_required',
              message: '需要刷新多模态输入',
            },
          }, { status: 409 });
        }
        throw new Error(`unexpected request: ${url}`);
      }),
    });
    queryClient.setQueryData(queryKeys.list(apiClient.sessionKey), [currentDetail()]);
    queryClient.setQueryData(queryKeys.detail(apiClient.sessionKey, 'cid-1'), currentDetail());

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);
    expect(await screen.findByRole('button', { name: '确认生成' })).toBeDisabled();
    fireEvent.click(screen.getByText('画外', { exact: true }));
    fireEvent.click(screen.getByRole('button', { name: '确认生成' }));

    expect(await screen.findByText('音频与画面输入正在同一任务内刷新，完成后将自动继续生成。'))
      .toBeInTheDocument();
    expect(await screen.findByText('生成进行中')).toBeInTheDocument();
    await waitFor(() => expect(detailOperationRequestIds.length).toBeGreaterThanOrEqual(1));
    expect(submitPayloads).toHaveLength(1);
    expect(submitPayloads[0]).toMatchObject({ dialogue_delivery: 'off_screen' });
    expect(operationRequestId).toBe(submitPayloads[0]?.client_request_id);
    expect(new Set(detailOperationRequestIds)).toEqual(
      new Set([submitPayloads[0]!.client_request_id]),
    );
    expect(screen.queryByRole('button', { name: '确认生成' })).not.toBeInTheDocument();
  });

  it('polls prompt fusion inside the same client request without another submit', async () => {
    const storage = new MemoryStorage();
    storage.setItem('cvs_token', 'stored-token');
    const submitPayloads: Array<{
      client_request_id: string;
      dialogue_delivery?: string;
    }> = [];
    const detailOperationRequestIds: string[] = [];
    let operationRequestId: string | null = null;
    let fusionStatus: 'missing' | 'pending' | 'done' = 'missing';
    const currentDetail = () => ({
      ...baseDetail,
      navigation_status: operationRequestId ? 'generation_running' : 'analysis_complete',
      duration_s: 12,
      segment_count: 1,
      plan_receipt: '1'.repeat(64),
      segments: [{ index: 1, prompt: '旧视频提示词', lines: [], keyframes: [] }],
      generation: operationRequestId ? {
        status: 'running', stage: 'context_ir', attempt: 1,
        client_request_id: operationRequestId, segments: [],
      } : null,
      dialogue_delivery: operationRequestId ? 'off_screen' : null,
      image_acceptance: {
        required: true, accepted: true, expected_meta_sha256: '2'.repeat(64),
      },
      postprocess: {
        status: 'done',
        options: { remove_subtitle: false, remove_brand: false, optimize_image: true },
        frames: [], error: null,
        segments: [{
          index: 1, status: 'done', stage: 'done', completed_frames: 9,
          total_frames: 9, revision: 1, error: null,
        }],
      },
      ...(fusionStatus === 'missing' ? {} : {
        prompt_fusion: {
          status: fusionStatus,
          error: null,
          segments: [{
            index: 1,
            status: fusionStatus,
            final_prompt: fusionStatus === 'done' ? '最终融合提示词' : null,
            error: null,
          }],
        },
      }),
    });
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      sessionKeyFactory: () => 'prompt-fusion-refresh-session',
      fetchImplementation: vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url === '/api/conversations') return Response.json([currentDetail()]);
        if (url === '/api/conversations/cid-1') {
          if (operationRequestId) detailOperationRequestIds.push(operationRequestId);
          return Response.json(currentDetail());
        }
        if (url === '/api/conversations/cid-1/submit') {
          const payload = JSON.parse(String(init?.body)) as {
            client_request_id: string;
            dialogue_delivery?: string;
          };
          submitPayloads.push(payload);
          operationRequestId = payload.client_request_id;
          fusionStatus = 'pending';
          return Response.json({ status: 'running', attempt: 1 }, { status: 202 });
        }
        throw new Error(`unexpected request: ${url}`);
      }),
    });
    queryClient.setQueryData(queryKeys.list(apiClient.sessionKey), [currentDetail()]);
    queryClient.setQueryData(queryKeys.detail(apiClient.sessionKey, 'cid-1'), currentDetail());

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);
    fireEvent.click(await screen.findByText('画外', { exact: true }));
    fireEvent.click(screen.getByRole('button', { name: '确认生成' }));

    expect(await screen.findByText('系统正在融合最终提示词，完成后将沿同一任务自动继续生成。'))
      .toBeInTheDocument();
    expect(await screen.findByText('片段 1 · 等待融合')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '确认生成' })).not.toBeInTheDocument();
    expect(submitPayloads).toHaveLength(1);
    expect(submitPayloads[0]).toMatchObject({ dialogue_delivery: 'off_screen' });
    expect(detailOperationRequestIds.length).toBeGreaterThanOrEqual(1);
    expect(new Set(detailOperationRequestIds)).toEqual(
      new Set([submitPayloads[0]!.client_request_id]),
    );
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
    queryClient.setQueryData(queryKeys.list(apiClient.sessionKey), [newDetail]);
    queryClient.setQueryData(queryKeys.detail(apiClient.sessionKey, 'cid-1'), newDetail);

    render(<AppThemeProvider queryClient={queryClient}><App apiClient={apiClient} /></AppThemeProvider>);
    fireEvent.click(screen.getByRole('button', { name: '确认生成' }));

    expect(await screen.findByText('正在核对提交结果，已锁定再次提交')).toBeInTheDocument();
    await waitFor(() => expect(detailCalls).toBeGreaterThanOrEqual(1));
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
