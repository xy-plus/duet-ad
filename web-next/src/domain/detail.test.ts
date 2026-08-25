import { describe, expect, it, vi } from 'vitest';
import {
  canOperate,
  detailSignature,
  recoverLockedPostprocess,
  recoverPromptChanged,
  shouldPollDetail,
} from './detail';
import { ApiError } from '../api/errors';
import { conversationBadge } from './navigation';

describe('detail state contract', () => {
  it('fails closed when operation capability fields are missing', () => {
    expect(canOperate({ read_only: false, submit_enabled: true })).toBe(true);
    expect(canOperate({ read_only: true, submit_enabled: true })).toBe(false);
    expect(canOperate({ submit_enabled: true })).toBe(false);
    expect(canOperate({ read_only: false })).toBe(false);
  });

  it('maps authoritative navigation status without local inference', () => {
    expect(conversationBadge({ navigation_status: 'analysis_complete' }))
      .toEqual({ className: 'analyzed', text: '分析完成' });
    expect(conversationBadge({ navigation_status: 'generation_running' }))
      .toEqual({ className: 'processing', text: '生成中' });
    expect(conversationBadge({ navigation_status: 'completed' }))
      .toEqual({ className: 'done', text: '已完成' });
    expect(conversationBadge({ navigation_status: 'future' }))
      .toEqual({ className: 'failed', text: '状态异常' });
  });

  it('polls analysis, generation and postprocess running states only', () => {
    expect(shouldPollDetail({ status: 'queued' })).toBe(true);
    expect(shouldPollDetail({ status: 'processing' })).toBe(true);
    expect(shouldPollDetail({ status: 'done', generation: { status: 'running' } })).toBe(true);
    expect(shouldPollDetail({ status: 'done', generation: { status: 'queued' } })).toBe(true);
    expect(shouldPollDetail({ status: 'done', postprocess: { status: 'running' } })).toBe(true);
    expect(shouldPollDetail({ status: 'done', generation: { status: 'submission_unknown' } }))
      .toBe(false);
    expect(shouldPollDetail({ status: 'done', generation: { status: 'failed' } })).toBe(false);
  });

  it('separates stable, postprocess progress and generation signatures', () => {
    const base = {
      status: 'done',
      keyframes: [],
      segments: [],
      generation: { status: 'running', segments: [{ index: 1, status: 'running' }] },
      postprocess: { status: 'running', frames: [] },
    };
    const original = detailSignature(base);
    const frame = detailSignature({
      ...base,
      postprocess: { status: 'running', frames: ['frame.png'] },
    });
    const generation = detailSignature({
      ...base,
      generation: { status: 'running', segments: [{ index: 1, status: 'succeeded' }] },
    });

    expect(frame.stable).toBe(original.stable);
    expect(frame.dyn).not.toBe(original.dyn);
    expect(generation.stable).toBe(original.stable);
    expect(generation.generation).not.toBe(original.generation);
  });

  it('recovers prompt_changed from the latest validated detail', async () => {
    const latest = {
      id: 'c1',
      source_prompt: 'page-b',
      source_prompt_sha256: 'b'.repeat(64),
    };
    const fetchLatest = vi.fn(async () => latest);

    await expect(recoverPromptChanged(
      new ApiError('changed', { status: 409, code: 'prompt_changed' }),
      fetchLatest,
    )).resolves.toBe(latest);
    expect(fetchLatest).toHaveBeenCalledOnce();
    await expect(recoverPromptChanged(new Error('other'), fetchLatest)).resolves.toBeNull();
  });

  it('recovers postprocess_options_locked from server options', async () => {
    const latest = {
      id: 'c1',
      postprocess: {
        status: 'running',
        options: { remove_subtitle: false, remove_brand: true },
      },
    };

    await expect(recoverLockedPostprocess(
      new ApiError('locked', { status: 409, code: 'postprocess_options_locked' }),
      async () => latest,
    )).resolves.toEqual({ latest, options: latest.postprocess.options });
  });
});
