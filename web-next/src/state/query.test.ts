import { describe, expect, it, vi } from 'vitest';
import { QueryClient } from '@tanstack/react-query';
import {
  DETAIL_POLL_INTERVAL_MS,
  conversationDetailQueryOptions,
  detailRefetchInterval,
  queryKeys,
} from './query';
import { ObjectUrlLease } from './useAuthenticatedFileUrl';
import { createApiRuntime } from './runtime';
import { TOKEN_STORAGE_KEY } from '../api/client';

function detail(id: string, status = 'done') {
  return { id, status, generation: null, postprocess: null };
}

describe('TanStack Query state contract', () => {
  it('isolates every query key by authenticated session and conversation', () => {
    expect(queryKeys.detail('session-a', 'c1')).not.toEqual(queryKeys.detail('session-b', 'c1'));
    expect(queryKeys.detail('session-a', 'c1')).not.toEqual(queryKeys.detail('session-a', 'c2'));
    expect(queryKeys.list('session-a')).not.toEqual(queryKeys.list('session-b'));
  });

  it('uses a two-second interval only while the current detail is running', () => {
    expect(detailRefetchInterval({ state: { data: detail('c1', 'processing') } }))
      .toBe(DETAIL_POLL_INTERVAL_MS);
    expect(detailRefetchInterval({
      state: { data: { ...detail('c1'), generation: { status: 'running' } } },
    })).toBe(DETAIL_POLL_INTERVAL_MS);
    expect(detailRefetchInterval({
      state: { data: { ...detail('c1'), generation: { status: 'submission_unknown' } } },
    })).toBe(false);
  });

  it('uses the existing detail observer for reconciliation without background polling', () => {
    const isSubmissionReconciling = vi.fn(() => true);
    const options = conversationDetailQueryOptions({
      sessionKey: 'session-reconciliation',
      getConversation: vi.fn(async () => detail('c1')),
      isSubmissionReconciling,
    }, 'c1');

    expect(options.refetchIntervalInBackground).toBe(false);
    expect(typeof options.refetchInterval).toBe('function');
    if (typeof options.refetchInterval !== 'function') throw new Error('缺少详情轮询策略');
    expect(options.refetchInterval({ state: { data: detail('c1') } } as never))
      .toBe(DETAIL_POLL_INTERVAL_MS);
    expect(isSubmissionReconciling).toHaveBeenCalledWith('c1');

    const source = readFileSync(resolve(
      dirname(fileURLToPath(import.meta.url)),
      '../app/ConversationDetailView.tsx',
    ), 'utf8');
    const generationSection = source.slice(
      source.indexOf('function GenerationSection'),
      source.indexOf('const initialPostprocessOptions'),
    );
    expect(generationSection).not.toContain('setInterval');
    expect(generationSection).not.toContain('getConversation(');
  });

  it('deduplicates an in-flight detail request and keeps stale sessions in old keys', async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    let releaseOld: ((value: ReturnType<typeof detail>) => void) | undefined;
    const oldApi = {
      sessionKey: 'session-old',
      getConversation: vi.fn(() => new Promise<ReturnType<typeof detail>>((resolve) => {
        releaseOld = resolve;
      })),
    };
    const newApi = {
      sessionKey: 'session-new',
      getConversation: vi.fn(async () => detail('same-id')),
    };
    const oldOptions = conversationDetailQueryOptions(oldApi, 'same-id');

    const oldOne = queryClient.fetchQuery(oldOptions);
    const oldTwo = queryClient.fetchQuery(oldOptions);
    await Promise.resolve();
    expect(oldApi.getConversation).toHaveBeenCalledOnce();

    await queryClient.fetchQuery(conversationDetailQueryOptions(newApi, 'same-id'));
    releaseOld?.(detail('same-id', 'failed'));
    await Promise.all([oldOne, oldTwo]);

    expect(queryClient.getQueryData(queryKeys.detail('session-new', 'same-id')))
      .toEqual(detail('same-id'));
    expect(queryClient.getQueryData(queryKeys.detail('session-old', 'same-id')))
      .toEqual(detail('same-id', 'failed'));
  });

  it('revokes Blob URLs on key replacement and disposal', () => {
    const createObjectURL = vi.fn()
      .mockReturnValueOnce('blob:first')
      .mockReturnValueOnce('blob:second');
    const revokeObjectURL = vi.fn();
    const lease = new ObjectUrlLease({ createObjectURL, revokeObjectURL });

    expect(lease.replace('session:c1:file', new Blob(['one']))).toBe('blob:first');
    expect(lease.replace('session:c2:file', new Blob(['two']))).toBe('blob:second');
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:first');

    lease.dispose();
    expect(revokeObjectURL).toHaveBeenLastCalledWith('blob:second');
    expect(revokeObjectURL).toHaveBeenCalledTimes(2);
  });

  it('wires 401 session expiry to the same QueryClient cache', async () => {
    const values = new Map([[TOKEN_STORAGE_KEY, 'expired']]);
    const storage: Storage = {
      get length() { return values.size; },
      clear: () => values.clear(),
      getItem: (key) => values.get(key) ?? null,
      key: (index) => [...values.keys()][index] ?? null,
      removeItem: (key) => { values.delete(key); },
      setItem: (key, value) => { values.set(key, value); },
    };
    const { apiClient, queryClient } = createApiRuntime({
      storage,
      fetchImplementation: async () => new Response(
        JSON.stringify({ detail: 'invalid token' }),
        { status: 401, headers: { 'Content-Type': 'application/json' } },
      ),
    });
    queryClient.setQueryData(['private'], { secret: true });

    await expect(apiClient.listConversations()).rejects.toMatchObject({ code: 'unauthorized' });

    expect(values.has(TOKEN_STORAGE_KEY)).toBe(false);
    expect(queryClient.getQueryCache().getAll()).toHaveLength(0);
  });
});
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
