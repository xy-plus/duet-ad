import { useRef, useState } from 'react';
import type { ApiClient } from '../api';
import {
  createConversationIntent,
  type CreateConversationIntent,
} from '../domain';
import { useCreateConversationMutation } from '../state';
import {
  CreateConversationComposer,
  type CreateConversationDraft,
  type UploadProgressState,
} from '../features/create';
import { Progress } from '../ui/antd';

const initialDraft: CreateConversationDraft = {
  source: { type: 'url', url: '' },
  note: '',
  transcript: { mode: 'keep' },
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '会话创建失败';
}

function freezeIntent(draft: CreateConversationDraft): CreateConversationIntent {
  const source = draft.source.type === 'file'
    ? { kind: 'file' as const, file: draft.source.file as File }
    : { kind: 'url' as const, url: draft.source.url };
  const voice = draft.transcript.mode === 'translate'
    ? { mode: 'translate' as const, targetLanguage: draft.transcript.targetLanguage }
    : { mode: draft.transcript.mode as 'keep' | 'rewrite' };
  return createConversationIntent({ source, note: draft.note, voice });
}

export function CreateConversationView({
  apiClient,
  onCreated,
}: {
  apiClient: ApiClient;
  onCreated: (id: string) => void;
}) {
  const mutation = useCreateConversationMutation(apiClient);
  const frozenIntent = useRef<CreateConversationIntent | undefined>(undefined);
  const [draft, setDraft] = useState(initialDraft);
  const [error, setError] = useState<string>();
  const [progress, setProgress] = useState<UploadProgressState>();

  const submit = async (submittedDraft: CreateConversationDraft) => {
    setError(undefined);
    let intent = frozenIntent.current;
    try {
      intent ??= freezeIntent(submittedDraft);
    } catch (intentError) {
      setError(errorMessage(intentError));
      return;
    }
    frozenIntent.current = intent;
    setProgress({ percent: 0, status: 'active' });
    try {
      const result = await mutation.mutateAsync({
        intent,
        onProgress: (ratio) => setProgress({
          percent: Math.max(0, Math.min(100, Math.round(ratio * 100))),
          status: 'active',
        }),
      });
      setProgress({ percent: 100, status: 'success' });
      frozenIntent.current = undefined;
      onCreated(result.id);
    } catch (createError) {
      setProgress((current) => ({ percent: current?.percent ?? 0, status: 'exception' }));
      setError(errorMessage(createError));
    }
  };

  return (
    <>
      <CreateConversationComposer
        error={error}
        languageOptions={[]}
        onChange={(value) => {
          setDraft(value);
          frozenIntent.current = undefined;
        }}
        onErrorDismiss={() => {
          setError(undefined);
          frozenIntent.current = undefined;
          mutation.reset();
        }}
        onSubmit={(value) => { void submit(value); }}
        submitting={mutation.isPending}
        uploadProgress={progress}
        value={draft}
      />
      {progress && draft.source.type === 'url' ? (
        <section aria-label="创建上传进度" className="app-create-progress">
          <Progress
            aria-label="上传进度"
            percent={progress.percent}
            status={progress.status}
          />
        </section>
      ) : null}
    </>
  );
}
