import {
  Alert,
  Button,
  Card,
  Descriptions,
  Image,
  Progress,
  Space,
  Tag,
  Typography,
} from '../../ui/antd';
import './postprocess.css';
import type {
  PostprocessOptions,
  PostprocessRetryAction,
  PostprocessTask,
} from './types';

interface PostprocessStatusProps {
  task: PostprocessTask;
  retrying?: boolean;
  onRetry: (action: PostprocessRetryAction) => void;
}

const statusLabels: Record<PostprocessTask['status'], string> = {
  queued: '等待处理',
  running: '后台处理中',
  partial_success: '部分处理成功',
  failed: '处理失败',
  succeeded: '处理完成',
};

function optionsLabel(options: PostprocessOptions) {
  const labels = [
    options.remove_subtitle ? '移除字幕' : null,
    options.remove_brand ? '移除品牌标识' : null,
  ].filter((label): label is string => label !== null);
  return labels.length > 0 ? labels.join('、') : '未选择处理项';
}

function taskAlert(task: PostprocessTask) {
  switch (task.status) {
    case 'queued':
      return <Alert type="info" showIcon title="已提交，等待后台处理" />;
    case 'running':
      return <Alert type="info" showIcon title="后处理正在后台运行" />;
    case 'partial_success':
      return (
        <Alert
          type="warning"
          showIcon
          title="部分处理成功"
          description={task.errorMessage ?? '部分关键帧仍需重试。'}
        />
      );
    case 'failed':
      return (
        <Alert
          type="error"
          showIcon
          title="后处理失败"
          description={task.errorMessage ?? '服务端已确认处理失败。'}
        />
      );
    case 'succeeded':
      return <Alert type="success" showIcon title="后处理已完成" />;
  }
}

export function PostprocessStatus({ task, retrying = false, onRetry }: PostprocessStatusProps) {
  const percent = task.totalCount > 0
    ? Math.min(100, Math.round((task.processedCount / task.totalCount) * 100))
    : 0;
  const retryable = task.status === 'failed' || task.status === 'partial_success';
  const successfulResults = task.results.filter((result) => result.status === 'succeeded');
  const failedResults = task.results.filter((result) => result.status === 'failed');
  const progressStatus = task.status === 'failed'
    ? 'exception'
    : task.status === 'succeeded'
      ? 'success'
      : task.status === 'running'
        ? 'active'
        : 'normal';

  return (
    <Card
      title="关键帧后处理"
      extra={<Tag>{statusLabels[task.status]}</Tag>}
    >
      <Space orientation="vertical" size="middle" className="postprocess-status-stack">
        <Descriptions
          size="small"
          column={1}
          items={[
            { key: 'options', label: '服务端锁定选项', children: optionsLabel(task.options) },
            {
              key: 'progress',
              label: '处理进度',
              children: `${task.processedCount}/${task.totalCount}`,
            },
          ]}
        />
        <Progress percent={percent} status={progressStatus} aria-label="后处理进度" />
        {taskAlert(task)}
        {retryable && (
          <Button
            aria-label={retrying ? '重试中' : '重试失败项'}
            loading={retrying}
            disabled={retrying}
            onClick={() => onRetry({ taskId: task.id, options: task.options })}
          >
            {retrying ? '重试中' : '重试失败项'}
          </Button>
        )}
        {successfulResults.length > 0 && (
          <Image.PreviewGroup>
            <div className="postprocess-result-grid">
              {successfulResults.map((result) => (
                <Image key={result.id} src={result.url} alt={result.alt} />
              ))}
            </div>
          </Image.PreviewGroup>
        )}
        {failedResults.map((result) => (
          <Typography.Text key={result.id} type="danger">
            {result.errorMessage}
          </Typography.Text>
        ))}
      </Space>
    </Card>
  );
}
