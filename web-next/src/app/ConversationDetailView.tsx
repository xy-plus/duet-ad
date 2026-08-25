import { useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import type { ApiClient } from '../api';
import { isApiErrorCode } from '../api';
import {
  buildLongFailedRetryPayload,
  buildResumePayload,
  buildStitchRetryPayload,
  buildSubmitPayload,
  canOperate,
  adaptConversationDetail,
  adaptImageOptimizationPrompt,
  fitProfile,
  generationRetryContract,
  longVideoContract,
  newClientRequestId,
  recoverLockedPostprocess,
  recoverPromptChanged,
  safeGenerationDraft,
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
  usePatchImageOptimizationPromptMutation,
  usePostprocessConversationMutation,
  useRetryPostprocessSegmentMutation,
  useSubmitConversationMutation,
  useUnsavedDraftGuard,
} from '../state';
import { ConversationOverview } from '../features/conversation';
import {
  GenerationSettings,
  GenerationStatus,
  type GenerationAction,
  type GenerationSettingsValue,
} from '../features/generation';
import { ArtifactSummary, ImageOptimizationPanel } from '../features/media';
import {
  PostprocessConfig,
  PostprocessStatus,
  type PostprocessOptions,
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

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function PromptSection({ apiClient, detail }: { apiClient: ApiClient; detail: ConversationDetail }) {
  const queryClient = useQueryClient();
  const sessionKey = useApiSessionKey(apiClient);
  const mutation = usePatchPromptMutation(apiClient, detail.id);
  const imageMutation = usePatchImageOptimizationPromptMutation(apiClient, detail.id);
  const adapted = adaptConversationDetail(detail);
  const imagePrompt = adapted.postprocessCapabilities.optimize_image
    ? adapted.imageOptimizationPrompt : null;
  const sourcePrompt = detail.source_prompt;
  const sourceSha = detail.source_prompt_sha256;
  const [promptSnapshot, setPromptSnapshot] = useState(() => ({
    text: sourcePrompt ?? '',
    sha256: typeof sourceSha === 'string' ? sourceSha : null,
  }));
  const lastDetailPromptKey = useRef(`${sourceSha ?? ''}:${sourcePrompt ?? ''}`);
  const [draft, setDraft] = useState(sourcePrompt ?? '');
  const [error, setError] = useState<string>();

  useEffect(() => {
    const key = `${sourceSha ?? ''}:${sourcePrompt ?? ''}`;
    if (key === lastDetailPromptKey.current) return;
    lastDetailPromptKey.current = key;
    setPromptSnapshot({ text: sourcePrompt ?? '', sha256: typeof sourceSha === 'string' ? sourceSha : null });
    setDraft(sourcePrompt ?? '');
  }, [sourcePrompt, sourceSha]);

  if (typeof sourcePrompt !== 'string') return null;
  const editable = canOperate(detail)
    && detail.generation === null
    && typeof promptSnapshot.sha256 === 'string'
    && /^[0-9a-f]{64}$/u.test(promptSnapshot.sha256);

  const save = async (prompt: string): Promise<string> => {
    if (!editable || typeof promptSnapshot.sha256 !== 'string') throw new Error('当前提示词不可保存');
    setError(undefined);
    try {
      const response = await mutation.mutateAsync({
        confirm: true,
        expected_sha256: promptSnapshot.sha256,
        prompt,
      });
      setPromptSnapshot({ text: response.prompt, sha256: response.sha256 });
      setDraft(response.prompt);
      return response.prompt;
    } catch (saveError) {
      let latest: ConversationDetail | null;
      try {
        latest = await recoverPromptChanged(
          saveError,
          () => apiClient.getConversation(detail.id),
        );
      } catch (recoveryError) {
        setError(errorMessage(recoveryError, '最新提示词加载失败'));
        throw recoveryError;
      }
      if (latest) {
        queryClient.setQueryData(queryKeys.detail(sessionKey, detail.id), latest);
        setPromptSnapshot({ text: latest.source_prompt ?? '', sha256: latest.source_prompt_sha256 });
        setDraft(latest.source_prompt ?? '');
        const conflict = new Error(`prompt_changed：${errorMessage(saveError, '服务端提示词已变化')}`);
        setError(conflict.message);
        throw conflict;
      }
      setError(errorMessage(saveError, '提示词保存失败'));
      throw saveError;
    }
  };

  return (
    <div className="app-detail-stack">
      {error ? <Alert type="error" showIcon title={error} /> : null}
      <ImageOptimizationPanel
        prompt={promptSnapshot.text}
        promptDraft={draft}
        dialogue={detail.voice_lines.map(({ text }) => text).join('\n')}
        imagePrompt={imagePrompt}
        draftId={`${detail.id}:0`}
        promptEditable={editable}
        promptPending={mutation.isPending}
        onPromptDraftChange={(value) => { setDraft(value); setError(undefined); }}
        onSavePrompt={save}
        onSaveImagePrompt={imagePrompt ? async ({ expected_sha256, prompt }) => {
          const latest = adaptImageOptimizationPrompt(await imageMutation.mutateAsync({ confirm: true, segment_index: 0, expected_sha256, prompt }));
          if (!latest) throw new Error('图片优化提示词响应无效');
          return latest;
        } : undefined}
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
  const [draft, setDraft] = useState(() => safeGenerationDraft(detail));
  const [error, setError] = useState<string>();
  const [reconciling, setReconciling] = useState(() => apiClient.isSubmissionReconciling(detail.id));
  const mutation = useSubmitConversationMutation(apiClient, detail.id);
  const draftGuard = useUnsavedDraftGuard();

  useEffect(() => {
    setDraft((previous) => safeGenerationDraft(detail, previous ?? undefined));
    setReconciling(apiClient.isSubmissionReconciling(detail.id));
  }, [apiClient, detail]);

  const settings = draft ? generationSettingsValue(draft) : undefined;
  const model = generationStatusModel(detail, mutation.isPending);
  const operationAllowed = canOperate(detail);
  const actionContract = generationRetryContract(detail);
  const actionAuthorized = operationAllowed
    && draft !== null
    && !reconciling
    && actionContract.action !== 'none';
  const submit = async (action: GenerationAction) => {
    if (!actionAuthorized || !draft) return;
    setError(undefined);
    try {
      await mutation.mutateAsync(payloadForAction(detail, draft, action));
    } catch (submitError) {
      if (apiClient.isSubmissionReconciling(detail.id)) {
        setReconciling(true);
      } else {
        setError(errorMessage(submitError, '生成请求提交失败'));
      }
    }
  };

  return (
    <section aria-label="视频生成" className="app-detail-stack">
      {error ? <Alert type="error" showIcon title={error} /> : null}
      {reconciling ? (
        <Alert type="warning" showIcon title="正在核对提交结果，已锁定再次提交" />
      ) : null}
      <GenerationSettings
        disabled={mutation.isPending || Boolean(draft?.frozen) || !operationAllowed}
        generation={generationEvidence(detail, settings)}
        initialValues={settings}
        onChange={(value) => setDraft((previous) => previous
          ? updateGenerationDraft(previous, value)
          : previous)}
        value={settings}
        videoKind={longVideoContract(detail).isLong ? 'long' : 'short'}
      />
      {!draft && !detail.generation ? (
        <Alert type="warning" showIcon title="服务端生成参数不完整，已禁止提交" />
      ) : null}
      {!operationAllowed ? (
        <Alert type="warning" showIcon title="当前会话不可执行生成动作" />
      ) : null}
      <GenerationStatus
        model={model}
        onAction={actionAuthorized ? (action) => { draftGuard.run(() => { void submit(action); }); } : undefined}
      />
    </section>
  );
}

const initialPostprocessOptions: PostprocessOptions = {
  remove_subtitle: true,
  remove_brand: true,
  optimize_image: false,
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
  const segmentRetryMutation = useRetryPostprocessSegmentMutation(apiClient, detail.id);
  const serverOptions = postprocessOptions(detail) ?? undefined;
  const serverTask = postprocessTask(detail);
  const [open, setOpen] = useState(false);
  const [optimizeDecision, setOptimizeDecision] = useState(false);
  const [options, setOptions] = useState<PostprocessOptions>(serverOptions ?? initialPostprocessOptions);
  const [pendingOptions, setPendingOptions] = useState<PostprocessOptions>();
  const [error, setError] = useState<string>();
  const [retryingSegment, setRetryingSegment] = useState<number>();
  const operationAllowed = canOperate(detail);
  const draftGuard = useUnsavedDraftGuard();
  const retrySegment = async (index: number, expectedRevision: number) => {
    if (retryingSegment !== undefined) return;
    setRetryingSegment(index); setError(undefined);
    try {
      await segmentRetryMutation.mutateAsync({ index, payload: { confirm: true, expected_revision: expectedRevision } });
    } catch (retryError) {
      setError(errorMessage(retryError, '分段后处理重试失败'));
    } finally { setRetryingSegment(undefined); }
  };

  useEffect(() => {
    if (serverOptions) setOptions(serverOptions);
  }, [serverOptions?.optimize_image, serverOptions?.remove_brand, serverOptions?.remove_subtitle]);

  const submit = async (submittedOptions: PostprocessOptions) => {
    if (!operationAllowed) return;
    const frozen: ApiPostprocessOptions = {
      remove_subtitle: submittedOptions.remove_subtitle,
      remove_brand: submittedOptions.remove_brand,
      optimize_image: submittedOptions.optimize_image,
    };
    setPendingOptions(frozen);
    setError(undefined);
    try {
      await mutation.mutateAsync({ confirm: true, options: frozen });
    } catch (submitError) {
      setOptimizeDecision(false);
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

  const visibleTask = serverTask ?? (pendingOptions ? queuedPostprocessTask(detail, pendingOptions) : null);
  const frames = detail.postprocess?.frames ?? [];
  const unknownShape = Boolean(detail.postprocess?.status) && !serverTask;
  const canConfigure = detail.postprocess === null
    && detail.postprocess_enabled
    && operationAllowed;

  return (
    <section aria-label="关键帧后处理" className="app-detail-stack">
      {error ? <Alert type={isApiErrorCode(mutation.error, 'postprocess_options_locked') ? 'warning' : 'error'} showIcon title={error} /> : null}
      {unknownShape ? <Alert type="error" showIcon title="服务端后处理状态无效" /> : null}
      {canConfigure ? (
        <Card>
          <Space orientation="vertical">
            <Typography.Text strong>是否优化素材？</Typography.Text>
            <Space>
              <Button aria-pressed={optimizeDecision} type={optimizeDecision ? 'primary' : 'default'} onClick={() => { draftGuard.run(() => { setOptimizeDecision(true); setOpen(true); }); }}>是</Button>
              <Button aria-pressed={!optimizeDecision} type={!optimizeDecision ? 'primary' : 'default'} onClick={() => { setOptimizeDecision(false); setOpen(false); }}>否</Button>
            </Space>
          </Space>
        </Card>
      ) : null}
      <PostprocessConfig
        onCancel={() => { setOpen(false); setOptimizeDecision(false); }}
        onOptionsChange={setOptions}
        onSubmit={(submitted) => { setOpen(false); void submit(submitted); }}
        open={open}
        options={options}
        serverOptions={serverOptions}
        submitting={mutation.isPending}
        capabilities={adaptConversationDetail(detail).postprocessCapabilities}
      />
      {visibleTask ? (
        <PostprocessStatus
          retrying={mutation.isPending || segmentRetryMutation.isPending || retryingSegment !== undefined}
          task={visibleTask}
          onRetrySegment={operationAllowed ? ({ index, expectedRevision }) => { void retrySegment(index, expectedRevision); } : undefined}
          onRefresh={() => { void queryClient.invalidateQueries({ queryKey: queryKeys.detail(sessionKey, detail.id) }); }}
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
          {detail.segments.length === 0 ? <PromptSection apiClient={apiClient} detail={detail} /> : null}
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
