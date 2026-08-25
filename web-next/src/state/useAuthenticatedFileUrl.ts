import { useEffect, useRef, useState } from 'react';
import { queryOptions, useQuery } from '@tanstack/react-query';
import type { ApiClient } from '../api/client';
import { queryKeys, useApiSessionKey } from './query';

export interface ObjectUrlApi {
  createObjectURL(blob: Blob): string;
  revokeObjectURL(url: string): void;
}

export class ObjectUrlLease {
  private current: { readonly key: string; readonly blob: Blob; readonly url: string } | null = null;

  constructor(private readonly urlApi: ObjectUrlApi = URL) {}

  replace(key: string, blob: Blob): string {
    if (this.current?.key === key && this.current.blob === blob) return this.current.url;
    this.clear();
    const url = this.urlApi.createObjectURL(blob);
    this.current = { key, blob, url };
    return url;
  }

  clear(): void {
    if (!this.current) return;
    this.urlApi.revokeObjectURL(this.current.url);
    this.current = null;
  }

  dispose(): void {
    this.clear();
  }
}

export function authenticatedFileQueryOptions(
  api: Pick<ApiClient, 'sessionKey' | 'getConversationFile'>,
  id: string,
  name: string,
) {
  return queryOptions({
    queryKey: queryKeys.file(api.sessionKey, id, name),
    queryFn: ({ signal }) => api.getConversationFile(id, name, { signal }),
    staleTime: Number.POSITIVE_INFINITY,
  });
}

export function useAuthenticatedFileUrl(
  api: ApiClient,
  id: string | null,
  name: string | null,
) {
  const sessionKey = useApiSessionKey(api);
  const leaseRef = useRef<ObjectUrlLease | null>(null);
  const [resource, setResource] = useState<{ readonly key: string; readonly url: string } | null>(null);
  if (!leaseRef.current) leaseRef.current = new ObjectUrlLease();
  const conversationId = id ?? '';
  const fileName = name ?? '';
  const resourceKey = `${sessionKey}:${conversationId}:${fileName}`;
  const query = useQuery({
    ...authenticatedFileQueryOptions({
      sessionKey,
      getConversationFile: api.getConversationFile.bind(api),
    }, conversationId, fileName),
    enabled: api.hasToken && Boolean(id) && Boolean(name),
  });

  useEffect(() => {
    const lease = leaseRef.current;
    if (!lease || !query.data || !id || !name) {
      lease?.clear();
      setResource(null);
      return;
    }
    const next = lease.replace(resourceKey, query.data);
    setResource({ key: resourceKey, url: next });
    return () => lease.clear();
  }, [id, name, query.data, resourceKey]);

  useEffect(() => () => leaseRef.current?.dispose(), []);

  return { ...query, url: resource?.key === resourceKey ? resource.url : null };
}
