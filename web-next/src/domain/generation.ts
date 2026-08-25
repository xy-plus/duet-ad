import { formatDialogueLines, normalizeDialogueLines, parseDialogueLines } from './dialogue';
import type {
  AspectRatio,
  DialogueMode,
  FitMode,
  FitProfile,
  GenerationSubmitPayload,
  Resolution,
} from './types';

export const H3_ASPECT_RATIOS = Object.freeze(['16:9', '9:16'] as const);
export const H3_RESOLUTIONS = Object.freeze(['480p', '768p'] as const);

const DIALOGUE_MODES = Object.freeze(['auto', 'edit', 'custom', 'none'] as const);
const FIT_MODES = Object.freeze(['none', 'crop', 'pad'] as const);
const SHA256 = /^[0-9a-f]{64}$/u;

type UnknownRecord = Readonly<Record<string, unknown>>;

export interface LongVideoContract {
  readonly isLong: boolean;
  readonly ready: boolean;
  readonly segmentCount: number | null;
  readonly planReceipt: string | null;
}

export interface GenerationParameterDraft {
  readonly aspectRatio: AspectRatio;
  readonly resolution: Resolution;
  readonly fitMode: FitMode;
}

export interface GenerationDraft extends GenerationParameterDraft {
  readonly dialogueMode: DialogueMode;
  readonly editLinesText: string;
  readonly customLinesText: string;
  readonly fastMode: boolean;
  readonly parameterVersion: string;
  readonly receiptVersion: unknown;
  readonly parameterTouched: boolean;
  readonly editTouched: boolean;
  readonly frozen: boolean;
}

export type GenerationAction = 'new' | 'retry' | 'retry_stitch' | 'resume' | 'none';

export interface GenerationRetryContract {
  readonly action: GenerationAction;
  readonly paidTaskCount: number | null;
}

export interface BuildSubmitInput {
  readonly clientRequestId: string;
  readonly dialogueMode: DialogueMode;
  readonly linesText?: string;
  readonly fitRequired: boolean;
  readonly fitMode?: FitMode;
  readonly aspectRatio: AspectRatio;
  readonly resolution: Resolution;
  readonly isLong?: boolean;
  readonly fastMode?: boolean;
  readonly planReceipt?: string | null;
}

function record(value: unknown): UnknownRecord | null {
  return typeof value === 'object' && value !== null ? value as UnknownRecord : null;
}

function includes<T extends string>(values: readonly T[], value: unknown): value is T {
  return typeof value === 'string' && values.includes(value as T);
}

function requiredAspectRatio(value: unknown): AspectRatio {
  if (!includes(H3_ASPECT_RATIOS, value)) {
    throw new Error('服务端推荐画幅无效，请刷新页面后重试');
  }
  return value;
}

function requiredResolution(value: unknown): Resolution {
  if (!includes(H3_RESOLUTIONS, value)) {
    throw new Error('服务端推荐清晰度无效，请刷新页面后重试');
  }
  return value;
}

function generationRecord(detail: UnknownRecord | null): UnknownRecord | null {
  return record(detail?.generation);
}

function generationFastMode(detail: unknown): boolean {
  return generationRecord(record(detail))?.fast_mode === true;
}

export function longVideoContract(detail: unknown): LongVideoContract {
  const source = record(detail);
  const isLong = Number(source?.duration_s) > 10;
  const rawSegmentCount = source?.segment_count;
  const segmentCount = Number.isInteger(rawSegmentCount) && Number(rawSegmentCount) > 0
    ? Number(rawSegmentCount)
    : null;
  const rawReceipt = source?.plan_receipt;
  const planReceipt = typeof rawReceipt === 'string' && SHA256.test(rawReceipt)
    ? rawReceipt
    : null;
  return {
    isLong,
    ready: !isLong || (segmentCount !== null && planReceipt !== null),
    segmentCount,
    planReceipt,
  };
}

export function fitProfile(detail: unknown, aspectRatio: unknown): FitProfile {
  const profiles = record(record(detail)?.fit_profiles);
  const profile = record(typeof aspectRatio === 'string' ? profiles?.[aspectRatio] : null);
  const fitRequired = profile?.fit_required;
  const defaultFitMode = profile?.default_fit_mode;
  const expectedDefault = fitRequired === true ? 'crop' : 'none';
  if (typeof fitRequired !== 'boolean'
      || defaultFitMode !== expectedDefault) {
    throw new Error('服务端画幅适配建议无效，请刷新页面后重试');
  }
  return { fit_required: fitRequired, default_fit_mode: expectedDefault };
}

export function generationParameterDraft(detail: unknown): GenerationParameterDraft {
  const source = record(detail);
  const aspectRatio = requiredAspectRatio(source?.aspect_ratio);
  const resolution = requiredResolution(source?.resolution);
  const profile = fitProfile(source, aspectRatio);
  return {
    aspectRatio,
    resolution,
    fitMode: profile.default_fit_mode,
  };
}

function dialogueMode(detail: UnknownRecord): DialogueMode {
  const mode = record(detail.dialogue)?.mode;
  if (!includes(DIALOGUE_MODES, mode)) {
    throw new Error('服务端冻结台词模式无效，请刷新页面后重试');
  }
  return mode;
}

export function createGenerationDraft(
  detail: unknown,
  previous?: GenerationDraft,
): GenerationDraft {
  const source = record(detail);
  if (!source) throw new Error('会话详情无效，请刷新页面后重试');
  const parameters = generationParameterDraft(source);
  const parameterVersion = `${source.aspect_ratio}|${source.resolution}|${JSON.stringify(source.fit_profiles)}`;
  const generation = generationRecord(source);
  const base: GenerationDraft = previous ?? {
    dialogueMode: 'auto',
    editLinesText: formatDialogueLines(source.dialogue),
    customLinesText: '',
    fastMode: longVideoContract(source).isLong,
    ...parameters,
    parameterVersion,
    receiptVersion: source.receipt_version,
    parameterTouched: false,
    editTouched: false,
    frozen: false,
  };

  if (generation) {
    const frozenDialogueMode = dialogueMode(source);
    const frozenAspectRatio = requiredAspectRatio(source.aspect_ratio);
    const frozenResolution = requiredResolution(source.resolution);
    const profile = fitProfile(source, frozenAspectRatio);
    const frozenFitMode = source.fit_mode;
    if (!includes(FIT_MODES, frozenFitMode)
        || (profile.fit_required && !['crop', 'pad'].includes(frozenFitMode))
        || (!profile.fit_required && frozenFitMode !== 'none')) {
      throw new Error('服务端冻结适配方式无效，请刷新页面后重试');
    }
    const frozenLines = formatDialogueLines(source.dialogue);
    return {
      ...base,
      aspectRatio: frozenAspectRatio,
      resolution: frozenResolution,
      fitMode: frozenFitMode,
      dialogueMode: frozenDialogueMode,
      editLinesText: frozenDialogueMode === 'edit' ? frozenLines : base.editLinesText,
      customLinesText: frozenDialogueMode === 'custom' ? frozenLines : base.customLinesText,
      fastMode: generationFastMode(source),
      parameterVersion,
      receiptVersion: source.receipt_version,
      parameterTouched: false,
      editTouched: false,
      frozen: true,
    };
  }

  let next = { ...base, frozen: false };
  if (base.receiptVersion !== source.receipt_version && !base.editTouched) {
    next = {
      ...next,
      editLinesText: formatDialogueLines(source.dialogue),
      receiptVersion: source.receipt_version,
    };
  }
  if (base.parameterVersion !== parameterVersion && !base.parameterTouched) {
    next = { ...next, ...parameters, parameterVersion };
  }
  return next;
}

export function generationParameterSnapshot(detail: unknown) {
  const source = record(detail);
  if (!source) throw new Error('会话详情无效，请刷新页面后重试');
  const snapshot: Record<string, unknown> = {
    aspect_ratio: source.aspect_ratio,
    resolution: source.resolution,
    dialogue_mode: record(source.dialogue)?.mode,
    fit_mode: source.fit_mode,
    duration_s: source.duration_s,
    segment_count: source.segment_count,
  };
  if (longVideoContract(source).isLong) snapshot.fast_mode = generationFastMode(source);
  return snapshot;
}

export function buildSubmitPayload(input: BuildSubmitInput): GenerationSubmitPayload {
  if (!includes(DIALOGUE_MODES, input.dialogueMode)) throw new Error('请选择台词模式');
  const requestId = String(input.clientRequestId ?? '').trim();
  if (!requestId) throw new Error('缺少本次生成请求标识');
  if (input.isLong && !['auto', 'none'].includes(input.dialogueMode)) {
    throw new Error('长视频仅支持保留完整源音轨或静音');
  }
  if (input.isLong && (typeof input.planReceipt !== 'string' || !SHA256.test(input.planReceipt))) {
    throw new Error('长视频生成计划尚未就绪，请刷新后重试');
  }
  if (!includes(H3_ASPECT_RATIOS, input.aspectRatio)) throw new Error('请选择画幅');
  if (!includes(H3_RESOLUTIONS, input.resolution)) throw new Error('请选择清晰度');

  let fitMode: FitMode = 'none';
  if (input.fitRequired) {
    if (input.fitMode !== 'crop' && input.fitMode !== 'pad') {
      throw new Error('请选择裁切或留边以适配画幅');
    }
    fitMode = input.fitMode;
  }

  const body: {
    confirm: true;
    client_request_id: string;
    dialogue_mode: DialogueMode;
    fit_mode: FitMode;
    aspect_ratio: AspectRatio;
    resolution: Resolution;
    lines?: ReturnType<typeof parseDialogueLines>;
    expected_plan_receipt?: string;
    fast_mode?: boolean;
  } = {
    confirm: true,
    client_request_id: requestId,
    dialogue_mode: input.dialogueMode,
    fit_mode: fitMode,
    aspect_ratio: input.aspectRatio,
    resolution: input.resolution,
  };
  if (input.isLong) {
    body.expected_plan_receipt = input.planReceipt as string;
    body.fast_mode = input.fastMode === true;
  }
  if (input.dialogueMode === 'edit' || input.dialogueMode === 'custom') {
    const lines = parseDialogueLines(input.linesText);
    if (lines.length === 0) throw new Error('请至少填写一行台词');
    body.lines = lines;
  }
  return body;
}

export function generationAction(status: unknown, stage?: unknown): GenerationAction {
  if (status === null || status === undefined) return 'new';
  if (status === 'failed') return stage === 'stitch' ? 'retry_stitch' : 'retry';
  if (status === 'resume_required') return 'resume';
  return 'none';
}

export function generationRetryContract(detail: unknown): GenerationRetryContract {
  const source = record(detail);
  const generation = generationRecord(source);
  const action = generationAction(generation?.status, generation?.stage);
  const long = longVideoContract(source);
  if (!long.isLong) {
    return { action, paidTaskCount: action === 'new' || action === 'retry' ? 1 : 0 };
  }
  if (action === 'new') return { action, paidTaskCount: long.segmentCount };
  if (action === 'retry_stitch' || action === 'none' || action === 'resume') {
    return { action, paidTaskCount: 0 };
  }
  const serverCount = generation?.retry_paid_segment_count;
  const validServerCount = Number.isInteger(serverCount)
    && Number(serverCount) >= 0
    && long.segmentCount !== null
    && Number(serverCount) <= long.segmentCount;
  return { action, paidTaskCount: validServerCount ? Number(serverCount) : null };
}

function frozenGenerationInputs(detail: unknown) {
  const source = record(detail);
  const generation = generationRecord(source);
  const dialogue = record(source?.dialogue);
  if (!source || !generation || !dialogue) throw new Error('既有任务状态无效');
  const mode = dialogue.mode;
  if (!includes(DIALOGUE_MODES, mode)) throw new Error('既有任务台词模式无效');
  const fitMode = source.fit_mode;
  if (!includes(FIT_MODES, fitMode)) throw new Error('既有任务画幅模式无效');
  const aspectRatio = requiredAspectRatio(source.aspect_ratio);
  const resolution = requiredResolution(source.resolution);
  return { source, generation, dialogue, mode, fitMode, aspectRatio, resolution };
}

export function buildResumePayload(detail: unknown): GenerationSubmitPayload {
  const frozen = frozenGenerationInputs(detail);
  if (frozen.generation.status !== 'resume_required') throw new Error('当前任务无需继续');
  const requestId = frozen.generation.client_request_id;
  if (typeof requestId !== 'string' || !requestId.trim()) throw new Error('缺少既有任务请求标识');
  const long = longVideoContract(frozen.source);
  if (long.isLong && !long.ready) throw new Error('长视频生成计划尚未就绪，请刷新后重试');
  if (long.isLong && frozen.mode !== 'auto' && frozen.mode !== 'none') {
    throw new Error('长视频既有任务台词模式无效');
  }

  const body: GenerationSubmitPayload = {
    confirm: true,
    client_request_id: requestId,
    dialogue_mode: frozen.mode,
    fit_mode: frozen.fitMode,
    aspect_ratio: frozen.aspectRatio,
    resolution: frozen.resolution,
    ...(long.isLong ? {
      expected_plan_receipt: long.planReceipt as string,
      fast_mode: generationFastMode(frozen.source),
    } : {}),
    ...((frozen.mode === 'edit' || frozen.mode === 'custom') ? {
      lines: requiredFrozenLines(frozen.dialogue.lines),
    } : {}),
  };
  return body;
}

function requiredFrozenLines(value: unknown) {
  const raw = Array.isArray(value) ? value : [];
  const lines = normalizeDialogueLines(raw);
  if (lines.length === 0 || lines.length !== raw.length) throw new Error('既有任务台词缺失');
  return lines;
}

export function buildStitchRetryPayload(detail: unknown): GenerationSubmitPayload {
  const frozen = frozenGenerationInputs(detail);
  if (generationAction(frozen.generation.status, frozen.generation.stage) !== 'retry_stitch') {
    throw new Error('当前任务无需重试拼接');
  }
  const long = longVideoContract(frozen.source);
  if (!long.isLong || !long.ready) throw new Error('长视频生成计划尚未就绪，请刷新后重试');
  const requestId = frozen.generation.client_request_id;
  if (typeof requestId !== 'string' || !requestId.trim()) throw new Error('缺少既有任务请求标识');
  if (frozen.mode !== 'auto' && frozen.mode !== 'none') throw new Error('既有任务台词模式无效');
  return {
    confirm: true,
    client_request_id: requestId,
    dialogue_mode: frozen.mode,
    fit_mode: frozen.fitMode,
    aspect_ratio: frozen.aspectRatio,
    resolution: frozen.resolution,
    expected_plan_receipt: long.planReceipt as string,
    fast_mode: generationFastMode(frozen.source),
  };
}

export function buildLongFailedRetryPayload(
  detail: unknown,
  clientRequestId: string,
): GenerationSubmitPayload {
  const frozen = frozenGenerationInputs(detail);
  if (generationAction(frozen.generation.status, frozen.generation.stage) !== 'retry') {
    throw new Error('当前任务无需重试生成');
  }
  const long = longVideoContract(frozen.source);
  if (!long.isLong || !long.ready) throw new Error('长视频生成计划尚未就绪，请刷新后重试');
  if (frozen.mode !== 'auto' && frozen.mode !== 'none') throw new Error('既有任务台词模式无效');
  const profile = fitProfile(frozen.source, frozen.aspectRatio);
  return buildSubmitPayload({
    clientRequestId,
    dialogueMode: frozen.mode,
    fitRequired: profile.fit_required,
    fitMode: frozen.fitMode,
    aspectRatio: frozen.aspectRatio,
    resolution: frozen.resolution,
    isLong: true,
    fastMode: generationFastMode(frozen.source),
    planReceipt: long.planReceipt,
  });
}
