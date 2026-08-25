import {
  Alert,
  Button,
  Card,
  Progress,
  Result,
  Space,
  ThoughtChain,
  Typography,
} from '../../ui/antd';
import './generation.css';
import type {
  GenerationAction,
  GenerationSegment,
  GenerationStatusModel,
} from './types';

interface GenerationStatusProps {
  model: GenerationStatusModel;
  onAction?: (action: GenerationAction) => void;
}

function segmentStatus(status: GenerationSegment['status']) {
  switch (status) {
    case 'running':
      return 'loading' as const;
    case 'succeeded':
      return 'success' as const;
    case 'failed':
      return 'error' as const;
    case 'submission_unknown':
      return 'abort' as const;
    case 'pending':
      return undefined;
  }
}

function progressPercent(model: GenerationStatusModel) {
  if (model.phase === 'succeeded') return 100;
  if (model.segments.length === 0) return 0;
  const completed = model.segments.filter(({ status }) => status === 'succeeded').length;
  return Math.round((completed / model.segments.length) * 100);
}

function actionFor(model: GenerationStatusModel): { label: string; command: GenerationAction; paid: boolean } | null {
  switch (model.phase) {
    case 'new':
      return { label: '确认生成', command: { type: 'new' }, paid: true };
    case 'failed':
      return {
        label: '新建任务重试',
        command: {
          type: 'retry',
          failedGenerationId: model.generationId,
          reuseGenerationId: false,
        },
        paid: true,
      };
    case 'resume_required':
      return {
        label: '继续原任务',
        command: { type: 'resume', generationId: model.generationId },
        paid: false,
      };
    case 'stitch_required':
      return {
        label: '继续拼接',
        command: { type: 'retry_stitch', generationId: model.generationId },
        paid: false,
      };
    case 'running':
    case 'submission_unknown':
    case 'succeeded':
      return null;
  }
}

function stateSurface(model: GenerationStatusModel, actionButton: React.ReactNode) {
  switch (model.phase) {
    case 'new':
      return (
        <Alert
          type="info"
          showIcon
          title="准备生成"
          description="确认付费任务数后再提交。"
          action={actionButton}
        />
      );
    case 'running':
      return (
        <Alert
          type="info"
          showIcon
          title="生成进行中"
          description={model.stageLabel ?? '正在等待服务端更新任务状态。'}
        />
      );
    case 'failed':
      return (
        <Result
          status="error"
          title="生成失败"
          subTitle={model.errorMessage ?? '服务端已确认任务失败。重试将创建新的任务标识。'}
          extra={actionButton}
        />
      );
    case 'resume_required':
      return (
        <Alert
          type="warning"
          showIcon
          title="原任务可继续"
          description="继续操作会复用服务端冻结的任务标识。"
          action={actionButton}
        />
      );
    case 'stitch_required':
      return (
        <Alert
          type="warning"
          showIcon
          title="分片已生成，等待拼接"
          description="继续操作只恢复原任务的拼接阶段。"
          action={actionButton}
        />
      );
    case 'submission_unknown':
      return (
        <Alert
          type="error"
          showIcon
          title="提交状态未知"
          description={model.errorMessage ?? '无法确认供应商是否已接单，为避免重复付费，不提供重提操作。'}
        />
      );
    case 'succeeded':
      return (
        <Result
          status="success"
          title="视频生成完成"
          subTitle="所有分片与拼接状态均来自服务端。"
        />
      );
  }
}

export function GenerationStatus({ model, onAction }: GenerationStatusProps) {
  const action = actionFor(model);
  const paidCountKnown = model.paidTaskCount !== null;
  const actionButton = action && onAction ? (
    <Button
      type="primary"
      aria-label={model.actionPending ? '提交中' : action.label}
      loading={model.actionPending}
      disabled={model.actionPending || (action.paid && !paidCountKnown)}
      onClick={() => onAction(action.command)}
    >
      {model.actionPending ? '提交中' : action.label}
    </Button>
  ) : null;
  const percent = progressPercent(model);
  const progressStatus = model.phase === 'failed' || model.phase === 'submission_unknown'
    ? 'exception'
    : model.phase === 'succeeded'
      ? 'success'
      : model.phase === 'running'
        ? 'active'
        : 'normal';

  return (
    <Card title="生成状态">
      <Space orientation="vertical" size="middle" className="generation-status-stack">
        <Typography.Text strong>
          {model.paidTaskCount === null
            ? '付费任务数：未知'
            : `付费任务数：${model.paidTaskCount} 个`}
        </Typography.Text>
        {!paidCountKnown && (model.phase === 'new' || model.phase === 'failed') && (
          <Alert
            type="warning"
            showIcon
            title="无法确认付费任务数"
            description="已禁用可能创建新付费任务的确认操作。"
          />
        )}
        {model.segments.length > 0 && (
          <>
            <Progress percent={percent} status={progressStatus} aria-label="生成进度" />
            <ThoughtChain
              items={model.segments.map((segment) => ({
                key: segment.id,
                title: segment.title,
                description: segment.description,
                status: segmentStatus(segment.status),
              }))}
            />
          </>
        )}
        {stateSurface(model, actionButton)}
      </Space>
    </Card>
  );
}
