import { ApiClient, type ApiClientOptions } from '../api/client';
import { createAppQueryClient } from './query';

export function createApiRuntime(options: Omit<ApiClientOptions, 'clearQueryCache'> = {}) {
  const queryClient = createAppQueryClient();
  const apiClient = new ApiClient({
    ...options,
    clearQueryCache: () => queryClient.clear(),
  });
  return { apiClient, queryClient } as const;
}
