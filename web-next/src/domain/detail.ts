import { isApiErrorCode } from '../api/errors';
import type { PostprocessOptions } from './types';

type UnknownRecord = Readonly<Record<string, unknown>>;

export interface DetailSignature {
  readonly stable: string;
  readonly dyn: number;
  readonly generation: string;
}

export interface AdaptedImageOptimizationPrompt {
  readonly text: string;
  readonly defaultText: string;
  readonly sha256: string;
}

export interface AdaptedConversationDetail {
  readonly imageOptimizationPrompt: AdaptedImageOptimizationPrompt | null;
  readonly postprocessCapabilities: PostprocessOptions;
  readonly postprocessSegments: readonly {
    readonly index: number;
    readonly status: string;
    readonly stage: string | null;
    readonly completedFrames: number;
    readonly totalFrames: number;
    readonly revision: number;
    readonly error: string | null;
  }[];
  readonly segments: readonly { readonly index: number; readonly imageOptimizationPrompt: AdaptedImageOptimizationPrompt | null }[];
}

function record(value: unknown): UnknownRecord | null {
  return typeof value === 'object' && value !== null ? value as UnknownRecord : null;
}

function array(value: unknown): readonly unknown[] {
  return Array.isArray(value) ? value : [];
}

export function adaptImageOptimizationPrompt(value: unknown): AdaptedImageOptimizationPrompt | null {
  const source = record(value);
  if (typeof source?.text !== 'string' || typeof source.default_text !== 'string'
      || typeof source.sha256 !== 'string' || !/^[0-9a-f]{64}$/u.test(source.sha256)) return null;
  return { text: source.text, defaultText: source.default_text, sha256: source.sha256 };
}

export function adaptConversationDetail(value: unknown): AdaptedConversationDetail {
  const source = record(value) ?? {};
  const sourceSegments = array(source.segments);
  const isLong = (Number.isInteger(source.segment_count) && Number(source.segment_count) > 0)
    || (typeof source.plan_receipt === 'string' && source.plan_receipt.length > 0)
    || sourceSegments.some((value) => {
    const segment = record(value);
    return segment && Number.isInteger(segment.index) && Number(segment.index) > 0;
  });
  const capabilities = record(source.postprocess_capabilities);
  const postprocess = record(source.postprocess);
  return {
    imageOptimizationPrompt: adaptImageOptimizationPrompt(source.image_optimization_prompt),
    postprocessCapabilities: {
      remove_subtitle: capabilities ? capabilities.remove_subtitle === true : source.postprocess_enabled === true,
      remove_brand: capabilities ? capabilities.remove_brand === true : source.postprocess_enabled === true,
      optimize_image: capabilities?.optimize_image === true,
    },
    postprocessSegments: array(postprocess?.segments).flatMap((value) => {
      const segment = record(value);
      if (!segment || !Number.isInteger(segment.index) || Number(segment.index) < 0
          || (isLong && Number(segment.index) === 0) || typeof segment.status !== 'string'
          || !Number.isInteger(segment.completed_frames) || !Number.isInteger(segment.total_frames)
          || !Number.isInteger(segment.revision)) return [];
      return [{
        index: Number(segment.index), status: segment.status,
        stage: typeof segment.stage === 'string' ? segment.stage : null,
        completedFrames: Number(segment.completed_frames), totalFrames: Number(segment.total_frames),
        revision: Number(segment.revision), error: typeof segment.error === 'string' ? segment.error : null,
      }];
    }),
    segments: sourceSegments.flatMap((value) => {
      const segment = record(value);
      return segment && Number.isInteger(segment.index) && Number(segment.index) > 0
        ? [{ index: Number(segment.index), imageOptimizationPrompt: adaptImageOptimizationPrompt(segment.image_optimization_prompt) }]
        : [];
    }),
  };
}

export function shouldPollDetail(detail: unknown): boolean {
  const source = record(detail);
  if (!source) return false;
  const generationStatus = record(source.generation)?.status;
  const postprocessStatus = record(source.postprocess)?.status;
  return source.status === 'queued'
    || source.status === 'processing'
    || generationStatus === 'queued'
    || generationStatus === 'running'
    || postprocessStatus === 'running';
}

export function canOperate(detail: unknown): boolean {
  const source = record(detail);
  return source?.read_only === false && source.submit_enabled === true;
}

export function detailSignature(detail: unknown): DetailSignature {
  const source = record(detail) ?? {};
  const postprocess = record(source.postprocess);
  const segments = array(source.segments);
  const stable = JSON.stringify([
    source.status,
    source.read_only === true,
    source.submit_enabled === true,
    source.fit_required === true,
    source.fit_mode,
    source.aspect_ratio,
    source.resolution,
    source.fit_profiles ?? null,
    source.duration_s,
    source.receipt_version,
    source.source_prompt ?? null,
    source.source_prompt_sha256 ?? null,
    source.image_optimization_prompt ?? null,
    source.postprocess_capabilities ?? null,
    source.dialogue ?? null,
    postprocess?.status ?? '',
    postprocess?.error ?? '',
    array(source.keyframes).join(','),
    source.prompt ?? '',
    segments.map((segment) => {
      const item = record(segment) ?? {};
      return [
        item.index,
        array(item.keyframes).join(','),
        item.prompt ?? '',
        item.image_optimization_prompt ?? null,
        array(item.lines).join('\n'),
      ];
    }),
    source.has_video ? 1 : 0,
  ]);
  const dyn = array(postprocess?.frames).length;
  const generation = JSON.stringify([
    source.plan_receipt ?? null,
    Number.isInteger(source.segment_count) ? source.segment_count : null,
    source.generation ?? null,
    source.has_video ? 1 : 0,
  ]);
  return { stable, dyn, generation };
}

export async function recoverPromptChanged<T>(
  error: unknown,
  fetchLatest: () => Promise<T>,
): Promise<T | null> {
  if (!isApiErrorCode(error, 'prompt_changed')) return null;
  const latest = await fetchLatest();
  const source = record(latest);
  if (typeof source?.source_prompt !== 'string'
      || typeof source.source_prompt_sha256 !== 'string'
      || !/^[0-9a-f]{64}$/u.test(source.source_prompt_sha256)) {
    throw new Error('最新提示词详情校验失败，请刷新页面后重试');
  }
  return latest;
}

export async function recoverLockedPostprocess<T>(
  error: unknown,
  fetchLatest: () => Promise<T>,
): Promise<{ readonly latest: T; readonly options: PostprocessOptions } | null> {
  if (!isApiErrorCode(error, 'postprocess_options_locked')) return null;
  const latest = await fetchLatest();
  const options = record(record(record(latest)?.postprocess)?.options);
  if (typeof options?.remove_subtitle !== 'boolean'
      || typeof options.remove_brand !== 'boolean') {
    throw new Error('服务端锁定选项校验失败，请刷新页面后重试');
  }
  return {
    latest,
    options: {
      remove_subtitle: options.remove_subtitle,
      remove_brand: options.remove_brand,
      optimize_image: options.optimize_image === true,
    },
  };
}
