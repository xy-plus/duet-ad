export type AspectRatio = '16:9' | '9:16';
export type Resolution = '480p' | '768p';
export type FitMode = 'none' | 'crop' | 'pad';
export type DialogueMode = 'auto' | 'edit' | 'custom' | 'none';

export interface DialogueLine {
  readonly start_s: number;
  readonly end_s: number;
  readonly text: string;
}

export interface DialogueDetail {
  readonly mode: DialogueMode;
  readonly lines: readonly DialogueLine[];
  readonly auto_lines: readonly DialogueLine[];
  readonly [compatibilityField: string]: unknown;
}

export interface FitProfile {
  readonly fit_required: boolean;
  readonly default_fit_mode: 'none' | 'crop';
  readonly [compatibilityField: string]: unknown;
}

export interface FitProfiles {
  readonly '16:9': FitProfile;
  readonly '9:16': FitProfile;
  readonly [compatibilityField: string]: unknown;
}

export interface ConversationSegment {
  readonly index: number;
  readonly start_s?: number;
  readonly end_s?: number;
  readonly duration_s?: number;
  readonly keyframes?: readonly string[];
  readonly prompt?: string | null;
  readonly lines?: readonly string[];
  readonly [compatibilityField: string]: unknown;
}

export interface GenerationSegment {
  readonly index: number;
  readonly chain_id?: string | null;
  readonly join_mode?: string | null;
  readonly status?: string | null;
  readonly attempt?: number | null;
  readonly error?: string | null;
  readonly [compatibilityField: string]: unknown;
}

export interface GenerationDetail {
  readonly status: string | null;
  readonly error?: string | null;
  readonly attempt?: number | null;
  readonly client_request_id?: string | null;
  readonly stage?: string | null;
  readonly fast_mode?: boolean;
  readonly segments?: readonly GenerationSegment[];
  readonly retry_paid_segment_count?: number;
  readonly [compatibilityField: string]: unknown;
}

export interface PostprocessOptions {
  readonly remove_subtitle: boolean;
  readonly remove_brand: boolean;
  readonly [compatibilityField: string]: unknown;
}

export interface PostprocessDetail {
  readonly status?: string | null;
  readonly options?: PostprocessOptions | null;
  readonly frames?: readonly string[];
  readonly error?: string | null;
  readonly [compatibilityField: string]: unknown;
}

export interface ConversationSummary {
  readonly id: string;
  readonly title: string;
  readonly note: string;
  readonly status: string;
  readonly navigation_status: string;
  readonly created_at: string;
  readonly has_video: boolean;
  readonly [compatibilityField: string]: unknown;
}

export interface ConversationDetail {
  readonly id: string;
  readonly title: string;
  readonly note: string;
  readonly status: string;
  readonly navigation_status: string;
  readonly error: string | null;
  readonly created_at: string;
  readonly updated_at: string;
  readonly keyframes: readonly string[];
  readonly prompt: string | null;
  readonly source_prompt: string | null;
  readonly source_prompt_sha256: string | null;
  readonly segments: readonly ConversationSegment[];
  readonly voice_lines: readonly DialogueLine[];
  readonly read_only: boolean;
  readonly duration_s: number | null;
  readonly fit_required: boolean | null;
  readonly fit_mode: FitMode | null;
  readonly aspect_ratio: AspectRatio | null;
  readonly resolution: Resolution | null;
  readonly fit_profiles: FitProfiles | null;
  readonly dialogue: DialogueDetail;
  readonly receipt_version: number | null;
  readonly generation: GenerationDetail | null;
  readonly has_source: boolean;
  readonly has_video: boolean;
  readonly submit_enabled: boolean;
  readonly postprocess: PostprocessDetail | null;
  readonly postprocess_enabled: boolean;
  readonly plan_receipt?: string | null;
  readonly segment_count?: number;
  readonly [compatibilityField: string]: unknown;
}

export interface LoginResponse {
  readonly ok: true;
}

export interface CreateConversationResponse {
  readonly id: string;
  readonly status: string;
}

export interface PromptPatchPayload {
  readonly confirm: true;
  readonly expected_sha256: string;
  readonly prompt: string;
}

export interface PromptPatchResponse {
  readonly prompt: string;
  readonly sha256: string;
  readonly final_prompt: string;
}

export interface GenerationSubmitPayload {
  readonly confirm: true;
  readonly client_request_id: string;
  readonly dialogue_mode: DialogueMode;
  readonly fit_mode: FitMode;
  readonly aspect_ratio: AspectRatio;
  readonly resolution: Resolution;
  readonly lines?: readonly DialogueLine[];
  readonly expected_plan_receipt?: string;
  readonly fast_mode?: boolean;
}

export interface GenerationSubmitResponse {
  readonly status: string;
  readonly attempt?: number | null;
}

export interface PostprocessPayload {
  readonly confirm: true;
  readonly options: PostprocessOptions;
}

export interface PostprocessResponse {
  readonly status: string;
  readonly frames: readonly string[];
}
