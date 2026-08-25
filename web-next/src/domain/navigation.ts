interface NavigationSource {
  readonly navigation_status?: unknown;
  readonly generation?: unknown;
  readonly status?: unknown;
  readonly has_video?: unknown;
}

export interface ConversationBadge {
  readonly className: 'queued' | 'processing' | 'failed' | 'analyzed' | 'done';
  readonly text: string;
}

const NAVIGATION_BADGES: Readonly<Record<string, ConversationBadge>> = Object.freeze({
  analysis_queued: { className: 'queued', text: '分析排队中' },
  analysis_processing: { className: 'processing', text: '分析中' },
  analysis_failed: { className: 'failed', text: '分析失败' },
  analysis_unknown: { className: 'failed', text: '分析状态未知' },
  analysis_complete: { className: 'analyzed', text: '分析完成' },
  generation_queued: { className: 'processing', text: '生成排队中' },
  generation_running: { className: 'processing', text: '生成中' },
  generation_failed: { className: 'failed', text: '生成失败' },
  generation_submission_unknown: { className: 'failed', text: '提交结果未知' },
  generation_resume_required: { className: 'failed', text: '等待继续' },
  generation_unknown: { className: 'failed', text: '生成状态未知' },
  output_missing: { className: 'failed', text: '最终视频缺失' },
  completed: { className: 'done', text: '已完成' },
  postprocessing: { className: 'processing', text: '素材优化中' },
  postprocess_failed: { className: 'failed', text: '素材优化失败' },
  postprocess_done: { className: 'done', text: '已完成' },
});

function record(value: unknown): Readonly<Record<string, unknown>> | null {
  return typeof value === 'object' && value !== null
    ? value as Readonly<Record<string, unknown>>
    : null;
}

export function conversationBadge(conversation: NavigationSource): ConversationBadge {
  if (Object.prototype.hasOwnProperty.call(conversation, 'navigation_status')) {
    const status = conversation.navigation_status;
    return typeof status === 'string' && NAVIGATION_BADGES[status]
      ? NAVIGATION_BADGES[status]
      : { className: 'failed', text: '状态异常' };
  }

  const generationStatus = record(conversation.generation)?.status;
  if (generationStatus === 'queued') return { className: 'processing', text: '生成排队中' };
  if (generationStatus === 'running') return { className: 'processing', text: '生成中' };
  if (generationStatus === 'failed') return { className: 'failed', text: '生成失败' };
  if (generationStatus === 'submission_unknown') return { className: 'failed', text: '提交结果未知' };
  if (generationStatus === 'resume_required') return { className: 'failed', text: '等待继续' };
  if (generationStatus === 'succeeded') {
    return conversation.has_video === true
      ? { className: 'done', text: '已完成' }
      : { className: 'failed', text: '最终视频缺失' };
  }
  if (generationStatus) return { className: 'failed', text: '生成状态未知' };

  if (conversation.status === 'done') {
    return conversation.has_video === true
      ? { className: 'done', text: '已完成' }
      : { className: 'analyzed', text: '分析完成' };
  }
  if (conversation.status === 'failed') return { className: 'failed', text: '失败' };
  if (conversation.status === 'processing') return { className: 'processing', text: '处理中' };
  return { className: 'queued', text: '排队中' };
}
