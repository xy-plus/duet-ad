export type DialogueMode = 'auto' | 'edit' | 'custom' | 'none';
export type AspectRatio = '16:9' | '9:16';
export type Resolution = '480p' | '768p';
export type FitMode = 'none' | 'crop' | 'pad';
export type VideoKind = 'short' | 'long';

export interface GenerationSettingsValue {
  dialogueMode: DialogueMode;
  dialogueText: string;
  aspectRatio: AspectRatio;
  resolution: Resolution;
  fitMode: FitMode;
}

export interface GenerationEvidence {
  id: string;
  parameters: GenerationSettingsValue;
}

export type GenerationSegmentStatus =
  | 'pending'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'submission_unknown';

export interface GenerationSegment {
  id: string;
  title: string;
  status: GenerationSegmentStatus;
  description?: string;
}

interface GenerationStatusBase {
  paidTaskCount: number | null;
  segments: GenerationSegment[];
  stageLabel?: string;
  errorMessage?: string;
  actionPending?: boolean;
}

export type GenerationStatusModel =
  | (GenerationStatusBase & { phase: 'new'; generationId?: never })
  | (GenerationStatusBase & {
      phase: 'running' | 'failed' | 'resume_required' | 'stitch_required' | 'submission_unknown' | 'succeeded';
      generationId: string;
    });

export type GenerationAction =
  | { type: 'new' }
  | { type: 'retry'; failedGenerationId: string; reuseGenerationId: false }
  | { type: 'resume'; generationId: string }
  | { type: 'retry_stitch'; generationId: string };
