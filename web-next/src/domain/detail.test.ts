import { describe, expect, it, vi } from 'vitest';
import {
  canOperate,
  detailSignature,
  postprocessAllowsGeneration,
  recoverLockedPostprocess,
  recoverPromptChanged,
  shouldPollDetail,
  adaptConversationDetail,
} from './detail';
import { ApiError } from '../api/errors';
import { conversationBadge } from './navigation';

describe('detail state contract', () => {
  it('adapts image prompts, capabilities and segment postprocess into a canonical model', () => {
    const detail = adaptConversationDetail({
      image_optimization_prompt: { text: '优化', default_text: '默认', sha256: 'a'.repeat(64) },
      postprocess_capabilities: { remove_subtitle: true, remove_brand: false, optimize_image: true },
      postprocess: { segments: [{ index: 1, status: 'failed', stage: 'submission_unknown', completed_frames: 2, total_frames: 3, revision: 4, error: '未知' }] },
      segments: [{ index: 1, image_optimization_prompt: { text: '段优化', default_text: '段默认', sha256: 'b'.repeat(64) } }],
    });
    expect(detail.imageOptimizationPrompt?.text).toBe('优化');
    expect(detail.postprocessCapabilities).toEqual({ remove_subtitle: true, remove_brand: false, optimize_image: true });
    expect(detail.postprocessSegments[0]).toMatchObject({ index: 1, stage: 'submission_unknown', revision: 4 });
    expect(detail.segments[0].imageOptimizationPrompt?.defaultText).toBe('段默认');
  });

  it('accepts postprocess segment zero for short video but rejects it for long video', () => {
    const short = adaptConversationDetail({
      segments: [],
      postprocess: { segments: [{ index: 0, status: 'failed', stage: 'h3', completed_frames: 0, total_frames: 1, revision: 1, error: '失败' }] },
    });
    const long = adaptConversationDetail({
      segment_count: 2,
      plan_receipt: 'd'.repeat(64),
      segments: [{ index: 0, image_optimization_prompt: { text: '不可用', default_text: '默认', sha256: 'c'.repeat(64) } }, { index: 1 }],
      postprocess: { segments: [{ index: 0, status: 'failed', stage: 'h3', completed_frames: 0, total_frames: 1, revision: 1, error: '失败' }] },
    });
    expect(short.postprocessSegments).toHaveLength(1);
    expect(long.segments.map(({ index }) => index)).toEqual([1]);
    expect(long.postprocessSegments).toEqual([]);
    expect(adaptConversationDetail({ segment_count: 1, postprocess: short.postprocessSegments.map((segment) => ({ index: segment.index, status: segment.status, stage: segment.stage, completed_frames: segment.completedFrames, total_frames: segment.totalFrames, revision: segment.revision, error: segment.error })) }).postprocessSegments).toEqual([]);
    expect(adaptConversationDetail({ plan_receipt: 'receipt', postprocess: { segments: [{ index: 0, status: 'failed', stage: 'h3', completed_frames: 0, total_frames: 1, revision: 1, error: null }] } }).postprocessSegments).toEqual([]);
  });
  it('fails closed when operation capability fields are missing', () => {
    expect(canOperate({ read_only: false, submit_enabled: true })).toBe(true);
    expect(canOperate({ read_only: true, submit_enabled: true })).toBe(false);
    expect(canOperate({ submit_enabled: true })).toBe(false);
    expect(canOperate({ read_only: false })).toBe(false);
  });

  it('allows final generation only after every postprocess segment is done', () => {
    const segment = (index: number, status: string) => ({
      index, status, stage: status === 'done' ? 'done' : 'image', completed_frames: status === 'done' ? 1 : 0,
      total_frames: 1, revision: 1, error: null,
    });
    const long = {
      duration_s: 20, segment_count: 2, plan_receipt: 'e'.repeat(64),
      segments: [{ index: 1 }, { index: 2 }],
    };

    expect(postprocessAllowsGeneration({ postprocess: null })).toBe(true);
    expect(postprocessAllowsGeneration({
      ...long,
      postprocess: { status: 'running', segments: [segment(1, 'done'), segment(2, 'running')] },
    })).toBe(false);
    expect(postprocessAllowsGeneration({
      ...long,
      postprocess: { status: 'failed', segments: [segment(1, 'done'), segment(2, 'failed')] },
    })).toBe(false);
    expect(postprocessAllowsGeneration({
      ...long,
      postprocess: { status: 'done', segments: [segment(1, 'done'), segment(2, 'running')] },
    })).toBe(false);
    expect(postprocessAllowsGeneration({
      ...long,
      postprocess: { status: 'done', segments: [segment(1, 'done'), segment(2, 'done')] },
    })).toBe(true);
  });

  it('fails closed for done postprocess records with missing or malformed segments', () => {
    const segment = (index: number, revision = 1) => ({
      index, status: 'done', stage: 'image', completed_frames: 1,
      total_frames: 1, revision, error: null,
    });
    expect(postprocessAllowsGeneration({ postprocess: { status: 'done' } })).toBe(false);
    expect(postprocessAllowsGeneration({ postprocess: { status: 'done', segments: [] } })).toBe(false);
    expect(postprocessAllowsGeneration({
      segment_count: 2,
      segments: [{ index: 1 }, { index: 2 }],
      postprocess: { status: 'done', segments: [{ index: 1, status: 'done' }] },
    })).toBe(false);
    expect(postprocessAllowsGeneration({
      postprocess: { status: 'done', segments: [{ index: 1, status: 'done' }] },
    })).toBe(false);
    expect(postprocessAllowsGeneration({
      segments: [{ index: 1 }, { index: 3 }],
      postprocess: { status: 'done', segments: [segment(1), segment(3)] },
    })).toBe(false);
    expect(postprocessAllowsGeneration({
      segments: [{ index: 1 }, { index: 1 }],
      postprocess: { status: 'done', segments: [segment(1)] },
    })).toBe(false);
    expect(postprocessAllowsGeneration({
      postprocess: { status: 'done', segments: [segment(0, 0)] },
    })).toBe(false);
  });

  it('requires complete frame, terminal stage and empty error evidence for every done segment', () => {
    const candidate = {
      index: 0, status: 'done', stage: 'done', completed_frames: 1,
      total_frames: 1, revision: 1, error: null,
    };
    expect(postprocessAllowsGeneration({
      postprocess: { status: 'done', segments: [{ ...candidate, completed_frames: 0 }] },
    })).toBe(false);
    expect(postprocessAllowsGeneration({
      postprocess: { status: 'done', segments: [{ ...candidate, stage: 'seedream' }] },
    })).toBe(false);
    expect(postprocessAllowsGeneration({
      postprocess: { status: 'done', segments: [{ ...candidate, error: 'submission_unknown' }] },
    })).toBe(false);
  });

  it('does not synthesize missing long source segments from segment_count', () => {
    const postprocess = {
      status: 'done',
      segments: [1, 2].map((index) => ({
        index, status: 'done', stage: 'done', completed_frames: 1,
        total_frames: 1, revision: 1, error: null,
      })),
    };
    const contract = { duration_s: 20, segment_count: 2, plan_receipt: 'f'.repeat(64) };
    expect(postprocessAllowsGeneration({ ...contract, postprocess })).toBe(false);
    expect(postprocessAllowsGeneration({ ...contract, segments: [], postprocess })).toBe(false);
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
    )).resolves.toEqual({ latest, options: { ...latest.postprocess.options, optimize_image: false } });
  });
});
