import { expect, test, type Page, type Route } from '@playwright/test';

const receipt = 'b'.repeat(64);
const png = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64',
);

type JsonRecord = Record<string, unknown>;

function detail(id: string, overrides: JsonRecord = {}): JsonRecord {
  return {
    id,
    title: id,
    note: `说明 ${id}`,
    status: 'done',
    navigation_status: 'analysis_complete',
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
    fit_mode: null,
    aspect_ratio: '16:9',
    resolution: '768p',
    fit_profiles: {
      '16:9': { fit_required: false, default_fit_mode: 'none' },
      '9:16': { fit_required: true, default_fit_mode: 'crop' },
    },
    dialogue: { mode: 'auto', lines: [], auto_lines: [] },
    receipt_version: 1,
    generation: null,
    has_source: false,
    has_video: false,
    submit_enabled: true,
    postprocess: null,
    postprocess_enabled: true,
    ...overrides,
  };
}

function summary(value: JsonRecord): JsonRecord {
  return {
    id: value.id,
    title: value.title,
    note: value.note,
    status: value.status,
    navigation_status: value.navigation_status,
    created_at: value.created_at,
    has_video: value.has_video,
  };
}

interface ApiController {
  details: Record<string, JsonRecord>;
  order: string[];
  requests: Array<{ method: string; path: string; headers: Record<string, string>; body: string | null }>;
  create?: (route: Route, controller: ApiController) => Promise<void>;
  patchPrompt?: (route: Route, id: string, controller: ApiController) => Promise<void>;
  submit?: (route: Route, id: string, controller: ApiController) => Promise<void>;
  postprocess?: (route: Route, id: string, controller: ApiController) => Promise<void>;
}

async function installApi(page: Page, controller: ApiController) {
  await page.route(/^http:\/\/127\.0\.0\.1:4173\/api\//u, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    controller.requests.push({ method, path, headers: request.headers(), body: request.postData() });

    if (path === '/api/login' && method === 'POST') {
      await route.fulfill({ json: { ok: true } });
      return;
    }
    if (path === '/api/conversations' && method === 'GET') {
      await route.fulfill({ json: controller.order.map((id) => summary(controller.details[id])) });
      return;
    }
    if (path === '/api/conversations' && method === 'POST' && controller.create) {
      await controller.create(route, controller);
      return;
    }
    const fileMatch = path.match(/^\/api\/conversations\/([^/]+)\/files\/(.+)$/u);
    if (fileMatch && method === 'GET') {
      await route.fulfill({ body: png, contentType: 'image/png' });
      return;
    }
    const promptMatch = path.match(/^\/api\/conversations\/([^/]+)\/prompt$/u);
    if (promptMatch && method === 'PATCH' && controller.patchPrompt) {
      await controller.patchPrompt(route, decodeURIComponent(promptMatch[1]), controller);
      return;
    }
    const submitMatch = path.match(/^\/api\/conversations\/([^/]+)\/submit$/u);
    if (submitMatch && method === 'POST' && controller.submit) {
      await controller.submit(route, decodeURIComponent(submitMatch[1]), controller);
      return;
    }
    const postprocessMatch = path.match(/^\/api\/conversations\/([^/]+)\/postprocess$/u);
    if (postprocessMatch && method === 'POST' && controller.postprocess) {
      await controller.postprocess(route, decodeURIComponent(postprocessMatch[1]), controller);
      return;
    }
    const detailMatch = path.match(/^\/api\/conversations\/([^/]+)$/u);
    if (detailMatch && method === 'GET') {
      const found = controller.details[decodeURIComponent(detailMatch[1])];
      await route.fulfill(found ? { json: found } : { status: 404, json: { detail: 'not found' } });
      return;
    }
    await route.fulfill({ status: 500, json: { detail: `unhandled ${method} ${path}` } });
  });
}

async function stabilize(page: Page) {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.addStyleTag({
    content: '*, *::before, *::after { animation-duration: 0s !important; caret-color: transparent !important; font-family: Arial, sans-serif !important; transition-duration: 0s !important; }',
  });
}

async function login(page: Page) {
  await page.goto('/');
  await page.getByLabel('访问口令').fill('browser-token');
  await page.getByRole('button', { name: '登录' }).click();
}

function multipartField(body: string, name: string): string | null {
  const match = body.match(new RegExp(`name="${name}"\\r\\n\\r\\n([^\\r]+)`, 'u'));
  return match?.[1] ?? null;
}

test('login, real list/detail, bearer Blob URLs, revoke, URL creation progress and selection', async ({ page }) => {
  await page.addInitScript(() => {
    const tracked = { created: [] as string[], revoked: [] as string[] };
    Object.assign(window, { __duetObjectUrls: tracked });
    const create = URL.createObjectURL.bind(URL);
    const revoke = URL.revokeObjectURL.bind(URL);
    URL.createObjectURL = (blob) => {
      const url = create(blob);
      tracked.created.push(url);
      return url;
    };
    URL.revokeObjectURL = (url) => {
      tracked.revoked.push(url);
      revoke(url);
    };
  });
  const first = detail('material', {
    title: '素材会话',
    navigation_status: 'analysis_complete',
    keyframes: ['frame-a.png'],
    has_source: true,
  });
  const second = detail('other', { title: '另一个会话' });
  let releaseCreate = () => undefined;
  let signalCreate = () => undefined;
  const createGate = new Promise<void>((resolve) => { releaseCreate = resolve; });
  const createSeen = new Promise<void>((resolve) => { signalCreate = resolve; });
  const controller: ApiController = {
    details: { material: first, other: second },
    order: ['material', 'other'],
    requests: [],
    create: async (route, current) => {
      signalCreate();
      await createGate;
      const created = detail('created-url', {
        title: '新建 URL 会话',
        status: 'queued',
        navigation_status: 'analysis_queued',
        source_prompt: null,
        source_prompt_sha256: null,
      });
      current.details['created-url'] = created;
      current.order.unshift('created-url');
      await route.fulfill({ status: 201, json: { id: 'created-url', status: 'queued' } });
    },
  };
  await installApi(page, controller);
  await login(page);

  await expect(page.getByText('素材会话').first()).toBeVisible();
  await expect.poll(() => controller.requests.filter(({ path }) => path.includes('/files/')).length)
    .toBeGreaterThanOrEqual(2);
  const fileRequests = controller.requests.filter(({ path }) => path.includes('/files/'));
  expect(new Set(fileRequests.map(({ path }) => path))).toEqual(new Set([
    '/api/conversations/material/files/source.mp4',
    '/api/conversations/material/files/keyframes/frame-a.png',
  ]));
  for (const request of fileRequests) {
    expect(request.headers.authorization).toBe('Bearer browser-token');
  }
  await page.getByText('另一个会话').first().click();
  await expect.poll(() => page.evaluate(() => (
    (window as unknown as { __duetObjectUrls: { revoked: string[] } }).__duetObjectUrls.revoked.length
  ))).toBeGreaterThanOrEqual(2);

  await page.getByText('新建会话').click();
  await page.getByLabel('视频链接').fill('https://example.test/source.mp4');
  await page.getByText('翻译', { exact: true }).click();
  await page.getByLabel('目标语言').fill('日语');
  await page.getByPlaceholder('补充视频用途、受众或风格偏好…').fill('保留产品节奏');
  await page.getByRole('button', { name: '创建会话' }).click();
  await createSeen;
  await expect(page.getByRole('progressbar')).toBeVisible();
  releaseCreate();
  await expect(page.getByText('新建 URL 会话').first()).toBeVisible();

  const createRequest = controller.requests.find(({ method, path }) => method === 'POST' && path === '/api/conversations');
  expect(createRequest?.headers.authorization).toBe('Bearer browser-token');
  expect(createRequest?.body).toContain('https://example.test/source.mp4');
  expect(createRequest?.body).toContain('保留产品节奏');
  expect(multipartField(createRequest?.body ?? '', 'voice_mode')).toBe('translate');
  expect(multipartField(createRequest?.body ?? '', 'target_language')).toBe('日语');
  expect(multipartField(createRequest?.body ?? '', 'client_request_id')).toMatch(/^[\w-]{8,64}$/u);
});

test('file creation retry preserves its frozen intent and request id', async ({ page }) => {
  const base = detail('existing', { title: '既有会话' });
  const bodies: string[] = [];
  const controller: ApiController = {
    details: { existing: base },
    order: ['existing'],
    requests: [],
    create: async (route, current) => {
      bodies.push(route.request().postData() ?? '');
      if (bodies.length === 1) {
        await route.fulfill({ status: 503, json: { detail: '暂时不可用' } });
        return;
      }
      current.details.created = detail('created', { title: '文件创建成功', status: 'queued' });
      current.order.unshift('created');
      await route.fulfill({ status: 201, json: { id: 'created', status: 'queued' } });
    },
  };
  await installApi(page, controller);
  await login(page);
  await page.getByText('新建会话').click();
  await page.getByText('上传文件').click();
  await page.locator('input[accept="video/*"]').setInputFiles({
    name: 'fixture.mp4',
    mimeType: 'video/mp4',
    buffer: Buffer.from('offline-video'),
  });
  await page.getByText('改写', { exact: true }).click();
  await page.getByPlaceholder('补充视频用途、受众或风格偏好…').fill('文件备注');
  await page.getByRole('button', { name: '创建会话' }).click();
  await expect(page.getByRole('alert')).toContainText('暂时不可用');
  await page.getByRole('button', { name: '创建会话' }).click();
  await expect(page.getByText('文件创建成功').first()).toBeVisible();

  expect(bodies).toHaveLength(2);
  expect(bodies[0]).toContain('filename="fixture.mp4"');
  expect(multipartField(bodies[0], 'note')).toBe('文件备注');
  expect(multipartField(bodies[0], 'voice_mode')).toBe('rewrite');
  expect(multipartField(bodies[0], 'client_request_id')).toBe(multipartField(bodies[1], 'client_request_id'));
});

test('prompt CAS conflict sends exact PATCH, refetches, and displays the conflict', async ({ page }) => {
  const original = detail('prompt', { title: '提示词会话' });
  const controller: ApiController = {
    details: { prompt: original },
    order: ['prompt'],
    requests: [],
    patchPrompt: async (route, id, current) => {
      current.details[id] = {
        ...current.details[id],
        source_prompt: '服务端更新后的提示词',
        source_prompt_sha256: 'c'.repeat(64),
      };
      await route.fulfill({
        status: 409,
        json: { detail: { code: 'prompt_changed', message: '提示词已被其他请求修改' } },
      });
    },
  };
  await installApi(page, controller);
  await login(page);
  await page.getByRole('button', { name: /生成提示词/ }).click();
  await page.getByLabel('提示词草稿').fill('我的修改');
  await page.getByRole('button', { name: '确认保存' }).click();

  const conflict = page.getByRole('alert').filter({ hasText: 'prompt_changed' });
  await expect(conflict).toContainText('prompt_changed');
  await expect(conflict).toContainText('提示词已被其他请求修改');
  await expect(page.getByLabel('提示词草稿')).toHaveValue('服务端更新后的提示词');
  const patch = controller.requests.find(({ method }) => method === 'PATCH');
  expect(JSON.parse(patch?.body ?? '{}')).toEqual({
    confirm: true,
    expected_sha256: 'a'.repeat(64),
    prompt: '我的修改',
  });
  expect(controller.requests.filter(({ method, path }) => method === 'GET' && path === '/api/conversations/prompt').length)
    .toBeGreaterThanOrEqual(2);
});

test('generation evidence freezes drafts and every recovery action uses safe ids and parameters', async ({ page }) => {
  const frozenBase = {
    fit_mode: 'none',
    aspect_ratio: '16:9',
    resolution: '480p',
    dialogue: { mode: 'auto', lines: [], auto_lines: [] },
  };
  const details: Record<string, JsonRecord> = {
    'new-long': detail('new-long', {
      title: '新增长视频',
      duration_s: 30,
      segment_count: 2,
      plan_receipt: receipt,
      segments: [{ index: 1, start_s: 0, end_s: 10, keyframes: [] }, { index: 2, start_s: 10, end_s: 20, keyframes: [] }],
      ...frozenBase,
    }),
    'failed-short': detail('failed-short', {
      title: '短视频失败',
      ...frozenBase,
      generation: { status: 'failed', stage: 'h3', client_request_id: 'old-short', error: '明确失败' },
    }),
    'failed-long': detail('failed-long', {
      title: '长视频失败', duration_s: 30, segment_count: 2, plan_receipt: receipt,
      segments: [{ index: 1, keyframes: [] }, { index: 2, keyframes: [] }],
      ...frozenBase,
      generation: {
        status: 'failed', stage: 'h3', client_request_id: 'old-long', fast_mode: false,
        retry_paid_segment_count: 1, error: '一段明确失败', segments: [],
      },
    }),
    resume: detail('resume', {
      title: '继续原任务', duration_s: 30, segment_count: 2, plan_receipt: receipt,
      segments: [{ index: 1, keyframes: [] }, { index: 2, keyframes: [] }],
      ...frozenBase,
      generation: { status: 'resume_required', stage: 'h3', client_request_id: 'old-resume', fast_mode: false, segments: [] },
    }),
    stitch: detail('stitch', {
      title: '继续拼接任务', duration_s: 30, segment_count: 2, plan_receipt: receipt,
      segments: [{ index: 1, keyframes: [] }, { index: 2, keyframes: [] }],
      ...frozenBase,
      generation: { status: 'failed', stage: 'stitch', client_request_id: 'old-stitch', fast_mode: true, segments: [] },
    }),
    unknown: detail('unknown', {
      title: '提交未知任务',
      ...frozenBase,
      generation: { status: 'submission_unknown', stage: 'h3', client_request_id: 'old-unknown', error: '无法确认' },
    }),
  };
  const payloads: Record<string, JsonRecord> = {};
  const controller: ApiController = {
    details,
    order: Object.keys(details),
    requests: [],
    submit: async (route, id, current) => {
      payloads[id] = JSON.parse(route.request().postData() ?? '{}') as JsonRecord;
      current.details[id] = {
        ...current.details[id],
        generation: { ...(current.details[id].generation as JsonRecord ?? {}), status: 'queued' },
      };
      await route.fulfill({ json: { status: 'queued', attempt: 1 } });
    },
  };
  await installApi(page, controller);
  await login(page);

  await page.getByRole('button', { name: '确认生成' }).click();
  await expect.poll(() => payloads['new-long']).toBeTruthy();
  expect(payloads['new-long']).toMatchObject({
    confirm: true, expected_plan_receipt: receipt, fast_mode: true,
    dialogue_mode: 'auto', fit_mode: 'none', aspect_ratio: '16:9', resolution: '480p',
  });
  expect(payloads['new-long'].client_request_id).toMatch(/^[\w-]+$/u);
  await expect(page.getByText('已冻结生成参数')).toBeVisible();

  await page.getByText('短视频失败').first().click();
  await page.getByRole('button', { name: '新建任务重试' }).click();
  await expect.poll(() => payloads['failed-short']).toBeTruthy();
  expect(payloads['failed-short'].client_request_id).not.toBe('old-short');
  expect(payloads['failed-short']).not.toHaveProperty('fast_mode');

  await page.getByText('长视频失败').first().click();
  await page.getByRole('button', { name: '新建任务重试' }).click();
  await expect.poll(() => payloads['failed-long']).toBeTruthy();
  expect(payloads['failed-long']).toMatchObject({ expected_plan_receipt: receipt, fast_mode: false });
  expect(payloads['failed-long'].client_request_id).not.toBe('old-long');

  await page.getByText('继续原任务').first().click();
  await page.getByRole('button', { name: '继续原任务' }).click();
  await expect.poll(() => payloads.resume).toBeTruthy();
  expect(payloads.resume).toMatchObject({ client_request_id: 'old-resume', fast_mode: false });

  await page.getByText('继续拼接任务').first().click();
  await page.getByRole('button', { name: '继续拼接' }).click();
  await expect.poll(() => payloads.stitch).toBeTruthy();
  expect(payloads.stitch).toMatchObject({ client_request_id: 'old-stitch', fast_mode: true });

  await page.getByText('提交未知任务').first().click();
  await expect(page.getByText('提交状态未知')).toBeVisible();
  await expect(page.getByRole('button', { name: /确认生成|新建任务重试|继续原任务|继续拼接/u })).toHaveCount(0);
});

test('postprocess submits exact options, closes to a background card, and survives conversation switching', async ({ page }) => {
  const candidate = detail('post', {
    title: '后处理会话',
    navigation_status: 'completed',
    keyframes: ['one.png', 'two.png'],
    generation: { status: 'succeeded', stage: 'h3', client_request_id: 'generation-post' },
    fit_mode: 'none',
  });
  const other = detail('other', { title: '切换目标' });
  let release = () => undefined;
  let signal = () => undefined;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const seen = new Promise<void>((resolve) => { signal = resolve; });
  let postCount = 0;
  const controller: ApiController = {
    details: { post: candidate, other },
    order: ['post', 'other'],
    requests: [],
    postprocess: async (route, id, current) => {
      postCount += 1;
      signal();
      await gate;
      current.details[id] = {
        ...current.details[id],
        navigation_status: 'postprocessing',
        postprocess: {
          status: 'running',
          options: { remove_subtitle: true, remove_brand: false },
          frames: [],
          error: null,
        },
      };
      await route.fulfill({ json: { status: 'running', frames: [] } });
    },
  };
  await installApi(page, controller);
  await login(page);
  await page.getByRole('button', { name: '优化关键帧' }).click();
  const removeBrand = page.getByRole('checkbox', { name: '移除品牌标识' });
  await removeBrand.click();
  await expect(removeBrand).not.toBeChecked();
  await page.getByRole('button', { name: '开始后处理' }).click();
  await seen;
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByText('已提交，等待后台处理')).toBeVisible();
  await page.getByText('切换目标').first().click();
  await expect(page.getByText('切换目标').first()).toBeVisible();
  release();
  await page.getByText('后处理会话').first().click();
  await expect(page.getByText('后处理正在后台运行')).toBeVisible();

  const request = controller.requests.find(({ path, method }) => path.endsWith('/postprocess') && method === 'POST');
  expect(JSON.parse(request?.body ?? '{}')).toEqual({
    confirm: true,
    options: { remove_subtitle: true, remove_brand: false },
  });
  expect(postCount).toBe(1);
});

test('postprocess_options_locked refetches and displays server-frozen options without automatic resend', async ({ page }) => {
  const candidate = detail('locked', {
    title: '锁定选项会话',
    generation: { status: 'succeeded', client_request_id: 'generation-locked', stage: 'h3' },
    fit_mode: 'none',
    keyframes: ['one.png'],
  });
  let count = 0;
  const controller: ApiController = {
    details: { locked: candidate },
    order: ['locked'],
    requests: [],
    postprocess: async (route, id, current) => {
      count += 1;
      current.details[id] = {
        ...current.details[id],
        navigation_status: 'postprocessing',
        postprocess: {
          status: 'running',
          options: { remove_subtitle: false, remove_brand: true },
          frames: [],
          error: null,
        },
      };
      await route.fulfill({
        status: 409,
        json: { detail: { code: 'postprocess_options_locked', message: '选项已锁定' } },
      });
    },
  };
  await installApi(page, controller);
  await login(page);
  await page.getByRole('button', { name: '优化关键帧' }).click();
  await page.getByRole('button', { name: '开始后处理' }).click();
  await expect(page.getByRole('alert').filter({ hasText: '服务端已锁定后处理选项' }))
    .toContainText('服务端已锁定后处理选项');
  await expect(page.getByLabel('会话详情').getByText('移除品牌标识')).toBeVisible();
  expect(count).toBe(1);
});

test('desktop screenshot baseline', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  const workspace = detail('desktop', {
    title: '桌面视频会话',
    note: '复刻一支具有城市夜景氛围的宣传片',
    navigation_status: 'analysis_complete',
    duration_s: 8,
    has_source: true,
    keyframes: ['frame-1.png', 'frame-2.png', 'frame-3.png', 'frame-4.png'],
    voice_lines: [
      { start_s: 2.4, end_s: 5.8, text: '灯光点亮城市的夜晚。' },
      { start_s: 18.2, end_s: 22.6, text: '每一次出发，都有新的故事。' },
    ],
  });
  const controller: ApiController = { details: { desktop: workspace }, order: ['desktop'], requests: [] };
  await installApi(page, controller);
  await page.addInitScript(() => localStorage.setItem('cvs_token', 'browser-token'));
  await page.goto('/');
  await stabilize(page);
  await expect(page.getByRole('button', { name: '确认生成' })).toBeVisible();
  const geometry = await page.evaluate(() => {
    const detail = document.querySelector('.app-detail');
    const source = document.querySelector('.app-detail-intro__media');
    const content = document.querySelector('.workspace-shell__content');
    if (!(detail instanceof HTMLElement) || !(source instanceof HTMLElement) || !(content instanceof HTMLElement)) {
      return null;
    }
    return {
      bodyScrollHeight: document.documentElement.scrollHeight,
      contentClientHeight: content.clientHeight,
      contentScrollHeight: content.scrollHeight,
      detailWidth: detail.getBoundingClientRect().width,
      sourceWidth: source.getBoundingClientRect().width,
      viewportHeight: window.innerHeight,
    };
  });
  expect(geometry).not.toBeNull();
  expect(geometry?.bodyScrollHeight).toBeLessThanOrEqual(geometry?.viewportHeight ?? 0);
  expect(geometry?.contentScrollHeight).toBeGreaterThan(geometry?.contentClientHeight ?? Number.MAX_SAFE_INTEGER);
  expect(geometry?.detailWidth).toBeLessThanOrEqual(900);
  expect(geometry?.sourceWidth).toBeLessThanOrEqual(320);
  await expect(page).toHaveScreenshot('desktop-workspace.png', { animations: 'disabled', fullPage: true });
  await page.getByText('生成状态', { exact: true }).scrollIntoViewIfNeeded();
  await expect(page).toHaveScreenshot('desktop-generation.png', { animations: 'disabled', fullPage: true });
});

test('mobile Drawer and detail screenshot baselines', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const workspace = detail('mobile', {
    title: '移动端会话',
    note: 'Drawer 与正文契约',
    has_source: true,
    keyframes: ['mobile-1.png', 'mobile-2.png'],
    voice_lines: [{ start_s: 1.2, end_s: 3.8, text: '移动端也保持相同的信息层级。' }],
  });
  const controller: ApiController = { details: { mobile: workspace }, order: ['mobile'], requests: [] };
  await installApi(page, controller);
  await page.addInitScript(() => localStorage.setItem('cvs_token', 'browser-token'));
  await page.goto('/');
  await stabilize(page);
  await page.getByRole('button', { name: '打开会话导航' }).click();
  await expect(page.getByRole('dialog', { name: '会话导航' })).toBeVisible();
  await expect(page.getByText('移动端会话').first()).toBeVisible();
  await expect(page).toHaveScreenshot('mobile-drawer.png', { animations: 'disabled', fullPage: true });
  await page.getByRole('button', { name: 'Close' }).click();
  await expect(page.getByRole('dialog', { name: '会话导航' })).toHaveCount(0);
  const width = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(width.scroll).toBeLessThanOrEqual(width.client);
  await expect(page).toHaveScreenshot('mobile-detail.png', { animations: 'disabled', fullPage: true });
});
