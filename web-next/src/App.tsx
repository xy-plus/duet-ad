import { useEffect, useState } from 'react';
import type { ApiClient } from './api';
import { conversationBadge, type ConversationSummary } from './domain';
import {
  useApiSessionKey,
  useConversationDetailQuery,
  useConversationsQuery,
  useLoginMutation,
  UnsavedDraftProvider,
  useUnsavedDraftGuard,
} from './state';
import { ConversationSidebar, LoginView, WorkspaceShell } from './features/shell';
import { Alert, Spin, Tag } from './ui/antd';
import { ConversationDetailView } from './app/ConversationDetailView';
import { CreateConversationView } from './app/CreateConversationView';
import './app/app.css';

export interface AppProps {
  apiClient: ApiClient;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function Login({ apiClient }: AppProps) {
  const mutation = useLoginMutation(apiClient);
  const [error, setError] = useState<string>();
  return (
    <LoginView
      error={error}
      onErrorDismiss={() => {
        setError(undefined);
        mutation.reset();
      }}
      onSubmit={async ({ token }) => {
        setError(undefined);
        try {
          await mutation.mutateAsync(token);
        } catch (loginError) {
          setError(errorMessage(loginError, '登录失败'));
        }
      }}
      submitting={mutation.isPending}
    />
  );
}

const backgroundStatuses = new Set([
  'analysis_queued',
  'analysis_processing',
  'generation_queued',
  'generation_running',
  'postprocessing',
]);

function BackgroundConversationPoller({ apiClient, id }: { apiClient: ApiClient; id: string }) {
  useConversationDetailQuery(apiClient, id);
  return null;
}

function statusTag(conversation: ConversationSummary) {
  const badge = conversationBadge(conversation);
  const color = badge.className === 'failed'
    ? 'error'
    : badge.className === 'processing'
      ? 'processing'
      : badge.className === 'done'
        ? 'success'
        : 'default';
  return <Tag color={color}>{badge.text}</Tag>;
}

function Workspace({ apiClient }: AppProps) {
  const conversations = useConversationsQuery(apiClient);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const draftGuard = useUnsavedDraftGuard();

  useEffect(() => {
    if (!creating && selectedId === null && conversations.data?.length) {
      setSelectedId(conversations.data[0].id);
    }
  }, [conversations.data, creating, selectedId]);

  const selected = conversations.data?.find(({ id }) => id === selectedId);
  const sidebar = (
    <ConversationSidebar
      activeConversationId={creating ? undefined : selectedId ?? undefined}
      conversations={conversations.data ?? []}
      getNavigationStatus={statusTag}
      onConversationSelect={(id) => {
        draftGuard.run(() => { setCreating(false); setSelectedId(id); });
      }}
      onLogout={() => draftGuard.run(() => apiClient.clearSession())}
      onNewConversation={() => {
        draftGuard.run(() => { setSelectedId(null); setCreating(true); });
      }}
    />
  );

  let content;
  if (creating) {
    content = (
      <CreateConversationView
        apiClient={apiClient}
        onCreated={(id) => {
          setCreating(false);
          setSelectedId(id);
        }}
      />
    );
  } else if (selectedId) {
    content = <ConversationDetailView apiClient={apiClient} id={selectedId} key={selectedId} />;
  } else if (conversations.isPending) {
    content = <div className="app-centered"><Spin aria-label="正在加载会话列表" size="large" /></div>;
  } else if (conversations.error) {
    content = (
      <div className="app-page-inset">
        <Alert type="error" showIcon title={errorMessage(conversations.error, '会话列表加载失败')} />
      </div>
    );
  } else {
    content = (
      <CreateConversationView
        apiClient={apiClient}
        onCreated={(id) => setSelectedId(id)}
      />
    );
  }

  return (
    <>
      {(conversations.data ?? [])
        .filter(({ id, navigation_status: status }) => id !== selectedId && backgroundStatuses.has(status))
        .map(({ id }) => <BackgroundConversationPoller apiClient={apiClient} id={id} key={id} />)}
      <WorkspaceShell
        sidebar={sidebar}
        subtitle={selected?.note || undefined}
        title={creating ? '新建会话' : (selected?.title ?? 'Duet AI 视频工作台')}
      >
        {content}
      </WorkspaceShell>
    </>
  );
}

export default function App({ apiClient }: AppProps) {
  useApiSessionKey(apiClient);
  return apiClient.hasToken ? <UnsavedDraftProvider><Workspace apiClient={apiClient} /></UnsavedDraftProvider> : <Login apiClient={apiClient} />;
}
