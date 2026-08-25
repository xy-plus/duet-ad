import { useEffect, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import type { ApiClient } from '../api';
import { isApiErrorCode } from '../api';
import {
  buildLongFailedRetryPayload,
  buildResumePayload,
  buildStitchRetryPayload,
  buildSubmitPayload,
  canOperate,
  createGenerationDraft,
  fitProfile,
  longVideoContract,
  newClientRequestId,
  recoverLockedPostprocess,
  recoverPromptChanged,
  type ConversationDetail,
  type GenerationDraft,
  type GenerationSubmitPayload,
  type PostprocessOptions as ApiPostprocessOptions,
} from '../domain';
import {
  queryKeys,
  useApiSessionKey,
  useConversationDetailQuery,
  usePatchPromptMutation,
  usePostprocessConversationMutation,
  useSubmitConversationMutation,
} from '../state';
import { ConversationOverview } from '../features/conversation';
import {
  GenerationSettings,
  GenerationStatus,
  type GenerationAction,
  type GenerationSettingsValue,
} from '../features/generation';
import { ArtifactSummary, PromptEditor } from '../features/media';
import {
  PostprocessConfig,
  PostprocessStatus,
  type PostprocessOptions,
  type PostprocessRetryAction,
  type PostprocessTask,
} from '../features/postprocess';
import { Alert, Button, Card, Space, Typography } from '../ui/antd';
import {
  AuthenticatedImageGrid,
  AuthenticatedSegments,
  AuthenticatedVideo,
} from './AuthenticatedMedia';
import {
  conversationMessages,
  generationEvidence,
  generationSettingsValue,
  generationStatusModel,
  postprocessFileName,
  postprocessOptions,
  postprocessTask,
  postprocessTotalFrames,
} from './model';
import './app.css';

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function PromptSection({ apiClient, detail }: { apiClient: ApiClient; detail: ConversationDetail }) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(apiClient);
  const mutation = usePatchPromptMutation(apiClient, detail.id);
  const sourcePrompt = detail.source_prompt;
  const [draft, setDraft] = useState(sourcePrompt ?? '');
  const [conflict, setConflict] = useState<{ code: 'prompt_changed'; message?: string }>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (!conflict) setDraft(sourcePrompt ?? '');
  }, [conflict, sourcePrompt]);

  if (typeof sourcePrompt !== 'string') return null;
  const editable = canOperate(detail)
    && detail.generation === null
    && typeof detail.source_prompt_sha256 === 'string'
    && /^[0-9a-f]{64}$/u.test(detail.source_prompt_sha256);

  const loadLatest = async () => {
    try {
      const latest = await apiClient.getConversation(detail.id);
      queryClient.setQueryData(queryKeys.detail(sessionKey, detail.id), latest);
      setDraft(latest.source_prompt ?? '');
      setConflict(undefined);
      setError(undefined);
    } catch (loadError) {
      setError(errorMessage(loadError, '最新提示词加载失败'));
    }
  };

  const save = async (prompt: string) => {
    if (!editable || typeof detail.source_prompt_sha256 !== 'string') return;
    setConflict(undefined);
    setError(undefined);
    try {
      const response = await mutation.mutateAsync({
        confirm: true,
        expected_sha256: detail.source_prompt_sha256,
        prompt,
      });
      setDraft(response.prompt);
    } catch (saveError) {
      try {
        const latest = await recoverPromptChanged(
          saveError,
          () => apiClient.getConversation(detail.id),
        );
        if (latest) {
          queryClient.setQueryData(queryKeys.detail(sessionKey, detail.id), latest);
          setDraft(latest.source_prompt ?? '');
          setConflict({ code: 'prompt_changed', message: errorMessage(saveError, '服务端提示词已变化') });
          return;
        }
      } catch (recoveryError) {
        setError(errorMessage(recoveryError, '最新提示词加载失败'));
        return;
      }
      setError(errorMessage(saveError, '提示词保存失败'));
    }
  };

  return (
    <div className="app-detail-stack">
      {error ? <Alert type="error" showIcon title={error} /> : null}
      <PromptEditor
        conflict={conflict}
        draft={draft}
        locked={!editable}
        lockReason={detail.generation ? '服务端已有生成记录，提示词已冻结' : '当前会话不可修改提示词'}
        onCopy={(value) => { void navigator.clipboard?.writeText(value); }}
        onDraftChange={(value) => {
          setDraft(value);
          setError(undefined);
        }}
        onReload={() => { void loadLatest(); }}
        onSave={(value) => { void save(value); }}
        pending={mutation.isPending}
        prompt={sourcePrompt}
      />
    </div>
  );
}

function updateGenerationDraft(
  previous: GenerationDraft,
  value: GenerationSettingsValue,
): GenerationDraft {
  const parameterTouched = previous.parameterTouched
    || previous.aspectRatio !== value.aspectRatio
    || previous.resolution !== value.resolution
    || previous.fitMode !== value.fitMode;
  const editTouched = previous.editTouched
    || ((value.dialogueMode === 'edit' && previous.editLinesText !== value.dialogueText)
      || (value.dialogueMode === 'custom' && previous.customLinesText !== value.dialogueText));
  return {
    ...previous,
    dialogueMode: value.dialogueMode,
    aspectRatio: value.aspectRatio,
    resolution: value.resolution,
    fitMode: value.fitMode,
    editLinesText: value.dialogueMode === 'edit' ? value.dialogueText : previous.editLinesText,
    customLinesText: value.dialogueMode === 'custom' ? value.dialogueText : previous.customLinesText,
    parameterTouched,
    editTouched,
  };
}

function newGenerationPayload(detail: ConversationDetail, draft: GenerationDraft) {
  const long = longVideoContract(detail);
  const profile = fitProfile(detail, draft.aspectRatio);
  return buildSubmitPayload({
    clientRequestId: newClientRequestId(),
    dialogueMode: draft.dialogueMode,
    linesText: draft.dialogueMode === 'edit' ? draft.editLinesText : draft.customLinesText,
    fitRequired: profile.fit_required,
    fitMode: draft.fitMode,
    aspectRatio: draft.aspectRatio,
    resolution: draft.resolution,
    isLong: long.isLong,
    fastMode: draft.fastMode,
    planReceipt: long.planReceipt,
  });
}

function payloadForAction(
  detail: ConversationDetail,
  draft: GenerationDraft,
  action: GenerationAction,
): GenerationSubmitPayload {
  if (action.type === 'resume') return buildResumePayload(detail);
  if (action.type === 'retry_stitch') return buildStitchRetryPayload(detail);
  if (action.type === 'retry' && longVideoContract(detail).isLong) {
    return buildLongFailedRetryPayload(detail, newClientRequestId());
  }
  return newGenerationPayload(detail, draft);
}

function GenerationSection({ apiClient, detail }: { apiClient: ApiClient; detail: ConversationDetail }) {
  const [draft, setDraft] = useState(() => createGenerationDraft(detail));
  const [error, setError] = useState<string>();
  const mutation = useSubmitConversationMutation(apiClient, detail.id);

  useEffect(() => {
    setDraft((previous) => createGenerationDraft(detail, previous));
  }, [detail]);

  const settings = generationSettingsValue(draft);
  const model = generationStatusModel(detail, mutation.isPending);
  const actionablePhase = ['new', 'failed', 'resume_required', 'stitch_required'].includes(model.phase);
  const operationsBlocked = !canOperate(detail) && actionablePhase;
  const submit = async (action: GenerationAction) => {
    setError(undefined);
    try {
      await mutation.mutateAsync(payloadForAction(detail, draft, action));
    } catch (submitError) {
      setError(errorMessage(submitError, '生成请求提交失败'));
    }
  };

  return (
    <section aria-label="视频生成" className="app-detail-stack">
      {error ? <Alert type="error" showIcon title={error} /> : null}
      <GenerationSettings
        disabled={mutation.isPending || draft.frozen || !canOperate(detail)}
        generation={generationEvidence(detail, draft)}
        initialValues={settings}
        onChange={(value) => setDraft((previous) => updateGenerationDraft(previous, value))}
        value={settings}
        videoKind={longVideoContract(detail).isLong ? 'long' : 'short'}
      />
      {operationsBlocked ? (
        <Card title="生成状态">
          <Alert
            type="warning"
            showIcon
            title="当前会话不可执行生成动作"
            description={detail.generation?.error ?? undefined}
          />
        </Card>
      ) : (
        <GenerationStatus
          model={model}
          onAction={(action) => { void submit(action); }}
        />
      )}
    </section>
  );
}

const initialPostprocessOptions: PostprocessOptions = {
  remove_subtitle: true,
  remove_brand: true,
};

function queuedPostprocessTask(
  detail: ConversationDetail,
  options: PostprocessOptions,
): PostprocessTask {
  return {
    id: `${detail.id}-postprocess-pending`,
    status: 'queued',
    options,
    processedCount: 0,
    totalCount: postprocessTotalFrames(detail),
    results: [],
  };
}

function PostprocessSection({ apiClient, detail }: { apiClient: ApiClient; detail: ConversationDetail }) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(apiClient);
  const mutation = usePostprocessConversationMutation(apiClient, detail.id);
  const serverOptions = postprocessOptions(detail) ?? undefined;
  const serverTask = postprocessTask(detail);
  const [open, setOpen] = useState(false);
  const [options, setOptions] = useState<PostprocessOptions>(serverOptions ?? initialPostprocessOptions);
  const [pendingOptions, setPendingOptions] = useState<PostprocessOptions>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (serverOptions) setOptions(serverOptions);
  }, [serverOptions?.remove_brand, serverOptions?.remove_subtitle]);

  const submit = async (submittedOptions: PostprocessOptions) => {
    const frozen: ApiPostprocessOptions = {
      remove_subtitle: submittedOptions.remove_subtitle,
      remove_brand: submittedOptions.remove_brand,
    };
    setPendingOptions(frozen);
    setError(undefined);
    try {
      await mutation.mutateAsync({ confirm: true, options: frozen });
    } catch (submitError) {
      setPendingOptions(undefined);
      try {
        const recovered = await recoverLockedPostprocess(
          submitError,
          () => apiClient.getConversation(detail.id),
        );
        if (recovered) {
          queryClient.setQueryData(queryKeys.detail(sessionKey, detail.id), recovered.latest);
          setOptions(recovered.options);
          setError('服务端已锁定后处理选项，已加载冻结值');
          return;
        }
      } catch (recoveryError) {
        setError(errorMessage(recoveryError, '服务端锁定选项加载失败'));
        return;
      }
      setError(errorMessage(submitError, '后处理请求提交失败'));
    }
  };

  const retry = ({ options: frozenOptions }: PostprocessRetryAction) => {
    void submit(frozenOptions);
  };
  const visibleTask = serverTask ?? (pendingOptions ? queuedPostprocessTask(detail, pendingOptions) : null);
  const frames = detail.postprocess?.frames ?? [];
  const unknownShape = Boolean(detail.postprocess?.status) && !serverTask;
  const canConfigure = detail.postprocess === null
    && detail.postprocess_enabled
    && canOperate(detail);

  return (
    <section aria-label="关键帧后处理" className="app-detail-stack">
      {error ? <Alert type={isApiErrorCode(mutation.error, 'postprocess_options_locked') ? 'warning' : 'error'} showIcon title={error} /> : null}
      {unknownShape ? <Alert type="error" showIcon title="服务端后处理状态无效" /> : null}
      {canConfigure ? (
        <Card>
          <Button onClick={() => setOpen(true)} type="primary">优化关键帧</Button>
        </Card>
      ) : null}
      <PostprocessConfig
        onCancel={() => setOpen(false)}
        onOptionsChange={setOptions}
        onSubmit={(submitted) => { void submit(submitted); }}
        open={open}
        options={options}
        serverOptions={serverOptions}
        submitting={mutation.isPending}
      />
      {visibleTask ? (
        <PostprocessStatus
          onRetry={retry}
          retrying={mutation.isPending}
          task={visibleTask}
        />
      ) : null}
      <AuthenticatedImageGrid
        apiClient={apiClient}
        conversationId={detail.id}
        files={frames.map((name, index) => ({
          fileName: postprocessFileName(name),
          alt: `优化后关键帧 ${index + 1}`,
        }))}
        title={frames.length > 0 ? '优化后关键帧' : undefined}
      />
    </section>
  );
}

function LoadedConversationDetail({ apiClient, detail }: { apiClient: ApiClient; detail: ConversationDetail }) {
  const dialogue = detail.voice_lines.map((line, index) => ({
    id: `${detail.id}-line-${index}`,
    text: line.text,
    startTime: `${line.start_s} 秒`,
    endTime: `${line.end_s} 秒`,
  }));

  return (
    <main className="app-detail" aria-label="会话详情">
      <ConversationOverview messages={conversationMessages(detail)} />
      <ArtifactSummary
        dialogue={dialogue}
        duration={detail.duration_s ?? undefined}
        keyframeCount={detail.keyframes.length}
      />
      {detail.has_source ? (
        <AuthenticatedVideo apiClient={apiClient} conversationId={detail.id} fileName="source.mp4" />
      ) : null}
      <AuthenticatedImageGrid
        apiClient={apiClient}
        conversationId={detail.id}
        files={detail.keyframes.map((name, index) => ({
          fileName: `keyframes/${name}`,
          alt: `关键帧 ${index + 1}`,
        }))}
        title={detail.keyframes.length > 0 ? '关键帧' : undefined}
      />
      <AuthenticatedSegments apiClient={apiClient} detail={detail} />
      {detail.status === 'done' ? (
        <>
          <PromptSection apiClient={apiClient} detail={detail} />
          <GenerationSection apiClient={apiClient} detail={detail} />
          <PostprocessSection apiClient={apiClient} detail={detail} />
        </>
      ) : null}
      {detail.has_video ? (
        <AuthenticatedVideo apiClient={apiClient} conversationId={detail.id} fileName="generated.mp4" />
      ) : null}
      <Card>
        <Space orientation="vertical">
          <Typography.Text type="secondary">创建时间：{detail.created_at}</Typography.Text>
          <Typography.Text type="secondary">更新时间：{detail.updated_at}</Typography.Text>
        </Space>
      </Card>
    </main>
  );
}

export function ConversationDetailView({ apiClient, id }: { apiClient: ApiClient; id: string }) {
  const query = useConversationDetailQuery(apiClient, id);
  if (query.isPending) return <ConversationOverview messages={[]} loading />;
  if (query.error) {
    return <ConversationOverview messages={[]} error={errorMessage(query.error, '会话详情加载失败')} />;
  }
  if (!query.data) return <Alert type="error" showIcon title="会话详情响应为空" />;
  return <LoadedConversationDetail apiClient={apiClient} detail={query.data} />;
}
