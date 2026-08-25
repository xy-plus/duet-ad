export interface ApiErrorOptions {
  readonly status: number;
  readonly code?: string;
  readonly detail?: unknown;
  readonly cause?: unknown;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: unknown;

  constructor(message: string, options: ApiErrorOptions) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = 'ApiError';
    this.status = options.status;
    this.code = options.code ?? (options.status === 401 ? 'unauthorized' : 'http_error');
    this.detail = options.detail;
  }
}

function record(value: unknown): Readonly<Record<string, unknown>> | null {
  return typeof value === 'object' && value !== null
    ? value as Readonly<Record<string, unknown>>
    : null;
}

export function apiErrorFromPayload(
  payload: unknown,
  options: { readonly status: number; readonly fallback: string },
): ApiError {
  const envelope = record(payload);
  const detail = envelope?.detail ?? envelope?.error ?? envelope?.message;
  const structured = record(detail);
  const message = structured && typeof structured.message === 'string'
    ? structured.message
    : (detail === undefined || detail === null ? options.fallback : String(detail));
  const code = structured && typeof structured.code === 'string'
    ? structured.code
    : (options.status === 401 ? 'unauthorized' : 'http_error');
  return new ApiError(message, { status: options.status, code, detail });
}

export function isApiErrorCode(error: unknown, code: string): error is ApiError {
  return error instanceof ApiError && error.code === code;
}

export function networkApiError(message: string, cause?: unknown): ApiError {
  return new ApiError(message, { status: 0, code: 'network_error', cause });
}
