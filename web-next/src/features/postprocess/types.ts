export interface PostprocessOptions {
  remove_subtitle: boolean;
  remove_brand: boolean;
  optimize_image: boolean;
}

export type PostprocessCapabilities = PostprocessOptions;

export type PostprocessTaskStatus = 'queued' | 'running' | 'partial_success' | 'failed' | 'succeeded';

export type PostprocessResult =
  | { id: string; status: 'succeeded'; url: string; alt: string }
  | { id: string; status: 'failed'; errorMessage: string };

export interface PostprocessTask {
  id: string;
  status: PostprocessTaskStatus;
  options: PostprocessOptions;
  processedCount: number;
  totalCount: number;
  results: PostprocessResult[];
  errorMessage?: string;
  segments?: readonly PostprocessSegmentStatus[];
}

export interface PostprocessSegmentStatus {
  index: number;
  status: string;
  stage: string | null;
  completedFrames: number;
  totalFrames: number;
  revision: number;
  error: string | null;
}

export interface PostprocessSegmentRetryAction {
  index: number;
  expectedRevision: number;
}
