import type { ConversationMessage } from '../features/conversation';
import type {
  GenerationDraft,
  GenerationRetryContract,
} from '../domain/generation';
import { generationRetryContract } from '../domain';
import { generationShapeIsOperable } from '../domain/generation';
import type {
  ConversationDetail,
  ConversationSegment,
  GenerationSegment as ApiGenerationSegment,
  PostprocessOptions as ApiPostprocessOptions,
} from '../domain/types';
import type {
  GenerationEvidence,
  GenerationSegment,
  GenerationSettingsValue,
  GenerationStatusModel,
} from '../features/generation';
import type { PostprocessTask, PostprocessTaskStatus } from '../features/postprocess';

function analysisStatus(status: string): ConversationMessage['status'] {
  if (status === 'queued' || status === 'processing' || status === 'done' || status === 'failed') {
    return status;
  }
  return 'failed';
}

export function conversationMessages(detail: ConversationDetail): ConversationMessage[] {
  const status = analysisStatus(detail.status);
  const content = status === 'failed'
    ? (detail.error ?? '分析失败')
    : status === 'queued'
      ? '分析排队中'
      : status === 'processing'
        ? '分析中'
        : '分析完成';
  const messages: ConversationMessage[] = [];
  if (detail.note || detail.title) {
    messages.push({
      id: `${detail.id}-request`,
      role: 'user',
      content: detail.note || detail.title,
      status: 'done',
      title: detail.title,
    });
  }
  messages.push({
    id: `${detail.id}-analysis`,
    role: 'assistant',
    content,
    status,
    error: status === 'failed' ? (detail.error ?? content) : undefined,
  });
  return messages;
}

export function generationSettingsValue(draft: GenerationDraft): GenerationSettingsValue {
  const dialogueText = draft.dialogueMode === 'edit'
    ? draft.editLinesText
    : draft.dialogueMode === 'custom'
      ? draft.customLinesText
      : '';
  return {
    dialogueMode: draft.dialogueMode,
    dialogueText,
    aspectRatio: draft.aspectRatio,
    resolution: draft.resolution,
    fitMode: draft.fitMode,
  };
}

export function generationEvidence(
  detail: ConversationDetail,
  parameters?: GenerationSettingsValue,
): GenerationEvidence | undefined {
  if (!detail.generation) return undefined;
  const requestId = detail.generation.client_request_id;
  return {
    id: typeof requestId === 'string' && requestId ? requestId : `${detail.id}-generation`,
    parameters,
    durationSeconds: typeof detail.duration_s === 'number' && Number.isFinite(detail.duration_s)
      ? detail.duration_s
      : null,
    segmentCount: Number.isInteger(detail.segment_count) && Number(detail.segment_count) > 0
      ? Number(detail.segment_count)
      : null,
  };
}

function segmentTitle(
  segment: ApiGenerationSegment,
  detailSegment: ConversationSegment | undefined,
): string {
  const prefix = `片段 ${segment.index}`;
  if (typeof detailSegment?.start_s === 'number' && typeof detailSegment.end_s === 'number') {
    return `${prefix} · ${detailSegment.start_s}—${detailSegment.end_s} 秒`;
  }
  if (typeof detailSegment?.duration_s === 'number') {
    return `${prefix} · ${detailSegment.duration_s} 秒`;
  }
  return prefix;
}

function segmentStatus(status: string | null | undefined): GenerationSegment['status'] {
  if (status === 'succeeded') return 'succeeded';
  if (status === 'failed') return 'failed';
  if (status === 'submission_unknown') return 'submission_unknown';
  if (status === 'queued' || status === 'submitting' || status === 'running') return 'running';
  return 'pending';
}

function generationSegments(detail: ConversationDetail): GenerationSegment[] {
  return (detail.generation?.segments ?? []).map((segment) => {
    const artifact = detail.segments.find(({ index }) => index === segment.index);
    return {
      id: segment.chain_id || `${detail.id}-segment-${segment.index}`,
      title: segmentTitle(segment, artifact),
      status: segmentStatus(segment.status),
      description: segment.error ?? undefined,
    };
  });
}

function generationPhase(detail: ConversationDetail): GenerationStatusModel['phase'] {
  const generation = detail.generation;
  if (!generation) return generationShapeIsOperable(detail) ? 'new' : 'submission_unknown';
  if (generation.status === 'queued'
      || generation.status === 'submitting'
      || generation.status === 'running') return 'running';
  if (generation.status === 'failed') {
    return generation.stage === 'stitch' ? 'stitch_required' : 'failed';
  }
  if (generation.status === 'resume_required') return 'resume_required';
  if (generation.status === 'succeeded') return 'succeeded';
  return 'submission_unknown';
}

export function generationStatusModel(
  detail: ConversationDetail,
  actionPending: boolean,
): GenerationStatusModel {
  const phase = generationPhase(detail);
  const contract: GenerationRetryContract = generationRetryContract(detail);
  const base = {
    paidTaskCount: contract.paidTaskCount,
    segments: generationSegments(detail),
    stageLabel: detail.generation?.stage ?? undefined,
    errorMessage: detail.generation?.error ?? undefined,
    actionPending,
  };
  if (phase === 'new') return { ...base, phase };
  const requestId = detail.generation?.client_request_id;
  return {
    ...base,
    phase,
    generationId: typeof requestId === 'string' && requestId ? requestId : detail.id,
  };
}

export function postprocessOptions(detail: ConversationDetail): ApiPostprocessOptions | null {
  const options = detail.postprocess?.options;
  if (!options
      || typeof options.remove_subtitle !== 'boolean'
      || typeof options.remove_brand !== 'boolean') return null;
  return {
    remove_subtitle: options.remove_subtitle,
    remove_brand: options.remove_brand,
  };
}

function postprocessStatus(
  status: string | null | undefined,
  completedFrames: number,
): PostprocessTaskStatus | null {
  if (status === 'queued') return 'queued';
  if (status === 'running') return 'running';
  if (status === 'failed') return completedFrames > 0 ? 'partial_success' : 'failed';
  if (status === 'done') return 'succeeded';
  return null;
}

export function postprocessTotalFrames(detail: ConversationDetail): number {
  if (detail.segments.length > 0) {
    return detail.segments.reduce((total, segment) => total + (segment.keyframes?.length ?? 0), 0);
  }
  return detail.keyframes.length;
}

export function postprocessTask(detail: ConversationDetail): PostprocessTask | null {
  const completedFrames = detail.postprocess?.frames?.length ?? 0;
  const status = postprocessStatus(detail.postprocess?.status, completedFrames);
  const options = postprocessOptions(detail);
  if (!status || !options) return null;
  return {
    id: `${detail.id}-postprocess`,
    status,
    options,
    processedCount: completedFrames,
    totalCount: postprocessTotalFrames(detail),
    results: [],
    errorMessage: detail.postprocess?.error ?? undefined,
  };
}

export function postprocessFileName(name: string): string {
  return name.startsWith('segments/') ? name : `postprocessed/${name}`;
}
