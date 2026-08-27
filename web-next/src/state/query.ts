import { useEffect, useSyncExternalStore } from 'react';
import {
  QueryClient,
  queryOptions,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import type { ApiClient } from '../api/client';
import type { CreateConversationIntent } from '../domain/createConversation';
import { shouldPollDetail } from '../domain/detail';
import type {
  ConversationDetail,
  ConversationSummary,
  GenerationSubmitPayload,
  ImageAcceptancePayload,
  ImageOptimizationPromptPatchPayload,
  PostprocessPayload,
  PostprocessSegmentRetryPayload,
  PromptPatchPayload,
} from '../domain/types';

export const DETAIL_POLL_INTERVAL_MS = 2_000;

export const queryKeys = {
  session: (sessionKey: string) => ['session', sessionKey] as const,
  conversations: (sessionKey: string) => [...queryKeys.session(sessionKey), 'conversations'] as const,
  list: (sessionKey: string) => [...queryKeys.conversations(sessionKey), 'list'] as const,
  detail: (sessionKey: string, id: string) => [
    ...queryKeys.conversations(sessionKey),
    'detail',
    id,
  ] as const,
  file: (sessionKey: string, id: string, name: string) => [
    ...queryKeys.detail(sessionKey, id),
    'file',
    name,
  ] as const,
};

export function createAppQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        retry: false,
        staleTime: 30_000,
      },
      mutations: { retry: false },
    },
  });
}

interface DetailApi<T> {
  readonly sessionKey: string;
  getConversation(id: string, options?: { readonly signal?: AbortSignal }): Promise<T>;
  isSubmissionReconciling?(id: string): boolean;
}

interface ListApi<T> {
  readonly sessionKey: string;
  listConversations(options?: { readonly signal?: AbortSignal }): Promise<T>;
}

export function detailRefetchInterval(query: {
  readonly state: { readonly data?: unknown };
}): number | false {
  return shouldPollDetail(query.state.data) ? DETAIL_POLL_INTERVAL_MS : false;
}

export function conversationListQueryOptions<T>(api: ListApi<T>) {
  return queryOptions({
    queryKey: queryKeys.list(api.sessionKey),
    queryFn: ({ signal }) => api.listConversations({ signal }),
  });
}

export function conversationDetailQueryOptions<T>(api: DetailApi<T>, id: string) {
  return queryOptions({
    queryKey: queryKeys.detail(api.sessionKey, id),
    queryFn: ({ signal }) => api.getConversation(id, { signal }),
    refetchInterval: (query) => api.isSubmissionReconciling?.(id)
      ? DETAIL_POLL_INTERVAL_MS
      : detailRefetchInterval(query),
    refetchIntervalInBackground: false,
  });
}

export function useApiSessionKey(api: ApiClient): string {
  return useSyncExternalStore(
    api.subscribeSession,
    api.getSessionSnapshot,
    api.getSessionSnapshot,
  );
}

export function mergeConversationList(
  incoming: readonly ConversationSummary[],
  previous: readonly ConversationSummary[] = [],
): readonly ConversationSummary[] {
  const previousById = new Map(previous.map((item) => [item.id, item]));
  return incoming.map((item) => {
    const known = previousById.get(item.id);
    if (!known) return item;
    const merged: Record<string, unknown> = { ...item };
    for (const field of ['generation', 'navigation_status'] as const) {
      if (!Object.prototype.hasOwnProperty.call(item, field)
          && Object.prototype.hasOwnProperty.call(known, field)) {
        merged[field] = known[field];
      }
    }
    return merged as ConversationSummary;
  });
}

export function syncConversationDetail(
  conversations: readonly ConversationSummary[] | undefined,
  detail: ConversationDetail,
): readonly ConversationSummary[] | undefined {
  if (!conversations) return conversations;
  return conversations.map((summary) => summary.id === detail.id ? {
    ...summary,
    status: detail.status,
    has_video: detail.has_video,
    navigation_status: detail.navigation_status,
    generation: detail.generation,
  } : summary);
}

export function useConversationsQuery(api: ApiClient) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(api);
  const options = conversationListQueryOptions({
    sessionKey,
    listConversations: api.listConversations.bind(api),
  });
  return useQuery({
    ...options,
    queryFn: async ({ signal }) => mergeConversationList(
      await api.listConversations({ signal }),
      queryClient.getQueryData<readonly ConversationSummary[]>(queryKeys.list(sessionKey)),
    ),
    enabled: api.hasToken,
  });
}

export function useConversationDetailQuery(api: ApiClient, id: string | null) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(api);
  const conversationId = id ?? '';
  const options = conversationDetailQueryOptions({
    sessionKey,
    getConversation: api.getConversation.bind(api),
    isSubmissionReconciling: (conversationId) => api.isSubmissionReconciling(conversationId),
  }, conversationId);
  const query = useQuery({ ...options, enabled: api.hasToken && Boolean(id) });

  useEffect(() => {
    if (!query.data || !id) return;
    queryClient.setQueryData<readonly ConversationSummary[]>(
      queryKeys.list(sessionKey),
      (current) => syncConversationDetail(current, query.data as ConversationDetail),
    );
  }, [id, query.data, queryClient, sessionKey]);

  return query;
}

function invalidateConversation(queryClient: QueryClient, sessionKey: string, id?: string) {
  void queryClient.invalidateQueries({ queryKey: queryKeys.list(sessionKey) });
  if (id) void queryClient.invalidateQueries({ queryKey: queryKeys.detail(sessionKey, id) });
}

export function useLoginMutation(api: ApiClient) {
  return useMutation({
    mutationKey: ['login'],
    mutationFn: (token: string) => api.login(token),
    retry: false,
  });
}

export function useCreateConversationMutation(api: ApiClient) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(api);
  return useMutation({
    mutationKey: [...queryKeys.conversations(sessionKey), 'create'],
    mutationFn: ({
      intent,
      onProgress,
    }: {
      readonly intent: CreateConversationIntent;
      readonly onProgress?: (ratio: number) => void;
    }) => api.createConversation(intent, { onProgress }),
    retry: false,
    onSuccess: () => invalidateConversation(queryClient, sessionKey),
  });
}

export function usePatchPromptMutation(api: ApiClient, id: string) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(api);
  return useMutation({
    mutationKey: [...queryKeys.detail(sessionKey, id), 'patch-prompt'],
    mutationFn: (payload: PromptPatchPayload) => api.patchPrompt(id, payload),
    retry: false,
    onSuccess: () => invalidateConversation(queryClient, sessionKey, id),
  });
}

export function usePatchImageOptimizationPromptMutation(api: ApiClient, id: string) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(api);
  return useMutation({
    mutationKey: [...queryKeys.detail(sessionKey, id), 'patch-image-optimization-prompt'],
    mutationFn: (payload: ImageOptimizationPromptPatchPayload) => api.patchImageOptimizationPrompt(id, payload),
    retry: false,
    onSuccess: () => invalidateConversation(queryClient, sessionKey, id),
  });
}

export function useSubmitConversationMutation(api: ApiClient, id: string) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(api);
  return useMutation({
    mutationKey: [...queryKeys.detail(sessionKey, id), 'submit'],
    mutationFn: (payload: GenerationSubmitPayload) => api.submitConversation(id, payload),
    retry: false,
    onSettled: () => invalidateConversation(queryClient, sessionKey, id),
  });
}

export function usePostprocessConversationMutation(api: ApiClient, id: string) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(api);
  return useMutation({
    mutationKey: [...queryKeys.detail(sessionKey, id), 'postprocess'],
    mutationFn: (payload: PostprocessPayload) => api.postprocessConversation(id, payload),
    retry: false,
    onSettled: () => invalidateConversation(queryClient, sessionKey, id),
  });
}

export function useAcceptImageOptimizationMutation(api: ApiClient, id: string) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(api);
  return useMutation({
    mutationKey: [...queryKeys.detail(sessionKey, id), 'image-acceptance'],
    mutationFn: (payload: ImageAcceptancePayload) => api.acceptImageOptimization(id, payload),
    retry: false,
    onSuccess: () => invalidateConversation(queryClient, sessionKey, id),
  });
}

export function useRetryPostprocessSegmentMutation(api: ApiClient, id: string) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(api);
  return useMutation({
    mutationKey: [...queryKeys.detail(sessionKey, id), 'postprocess-segment-retry'],
    mutationFn: ({ index, payload }: { readonly index: number; readonly payload: PostprocessSegmentRetryPayload }) => api.retryPostprocessSegment(id, index, payload),
    retry: false,
    onSettled: () => invalidateConversation(queryClient, sessionKey, id),
  });
}
