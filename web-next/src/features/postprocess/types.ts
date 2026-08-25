export interface PostprocessOptions {
  remove_subtitle: boolean;
  remove_brand: boolean;
}

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
}

export interface PostprocessRetryAction {
  taskId: string;
  options: PostprocessOptions;
}
