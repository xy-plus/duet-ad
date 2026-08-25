import { useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Image,
  Modal,
  Progress,
  Space,
  Tag,
  Typography,
} from '../../ui/antd';
import './postprocess.css';
import type {
  PostprocessOptions,
  PostprocessSegmentRetryAction,
  PostprocessTask,
} from './types';

interface PostprocessStatusProps {
  task: PostprocessTask;
  retrying?: boolean;
  onRetrySegment?: (action: PostprocessSegmentRetryAction) => void;
  onRefresh?: () => void;
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
    options.remove_subtitle ? '移除文字/字幕' : null,
    options.remove_brand ? '移除常见 Logo/图标' : null,
    options.optimize_image ? '进行图片优化' : null,
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

function stageLabel(stage: string | null, status: string): string {
  if (stage === 'queued' || stage === 'pending') return '等待处理';
  if (stage === 'prepare' || stage === 'extract') return '准备素材';
  if (stage === 'remove_subtitle' || stage === 'text') return '文字处理';
  if (stage === 'remove_brand' || stage === 'logo') return '标识处理';
  if (stage === 'optimize_image' || stage === 'seedream' || stage === 'image') return '图片优化';
  if (stage === 'finalize' || stage === 'write' || stage === 'publishing') return '整理结果';
  if (stage === 'done') return '处理完成';
  if (status === 'failed') return '处理失败';
  if (status === 'succeeded' || status === 'done') return '处理完成';
  return '处理中';
}

export function PostprocessStatus({ task, retrying = false, onRetrySegment, onRefresh }: PostprocessStatusProps) {
  const [unknownRetry, setUnknownRetry] = useState<PostprocessSegmentRetryAction>();
  const percent = task.totalCount > 0
    ? Math.min(100, Math.round((task.processedCount / task.totalCount) * 100))
    : 0;
  const retryable = task.status === 'failed' || task.status === 'partial_success';
  const failedSegments = task.segments?.filter((segment) => segment.status === 'failed') ?? [];
  const successfulResults = task.results.filter((result) => result.status === 'succeeded');
  const failedResults = task.results.filter((result) => result.status === 'failed');
  const progressStatus = task.status === 'failed'
    ? 'exception'
    : task.status === 'succeeded'
      ? 'success'
      : task.status === 'running'
        ? 'active'
        : 'normal';

  return (<>
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
        {retryable && failedSegments.length === 0 ? (
          <Alert
            type="warning"
            showIcon
            title="没有可安全重试的分段 revision，已禁止整体重发"
            description={onRefresh ? <Button onClick={onRefresh}>刷新状态</Button> : '请刷新详情后再检查。'}
          />
        ) : null}
        {task.segments?.map((segment) => {
          const segmentPercent = segment.totalFrames > 0
            ? Math.min(100, Math.round((segment.completedFrames / segment.totalFrames) * 100)) : 0;
          const retryableSegment = segment.status === 'failed';
          const submissionUnknown = segment.error === 'submission_unknown';
          return (
            <Card key={segment.index} size="small" title={segment.index === 0 ? '当前视频' : `第 ${segment.index} 段`} extra={<Tag>{stageLabel(segment.stage, segment.status)}</Tag>}>
              <Space orientation="vertical" className="postprocess-status-stack">
                <Progress aria-label={`第 ${segment.index} 段后处理进度`} percent={segmentPercent} status={retryableSegment ? 'exception' : undefined} />
                {segment.error ? <Alert type={submissionUnknown ? 'warning' : 'error'} showIcon title={submissionUnknown ? '提交状态未知' : segment.error} description={submissionUnknown ? '重试可能重复计费。' : undefined} /> : null}
                {retryableSegment && onRetrySegment ? <Button disabled={retrying} loading={retrying} onClick={() => {
                  const action = { index: segment.index, expectedRevision: segment.revision };
                  if (submissionUnknown) setUnknownRetry(action); else onRetrySegment(action);
                }}>重试本段</Button> : null}
              </Space>
            </Card>
          );
        })}
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
    <Modal
      title="确认重试未知提交段"
      open={unknownRetry !== undefined}
      onCancel={() => setUnknownRetry(undefined)}
      footer={<Space><Button onClick={() => setUnknownRetry(undefined)}>取消</Button><Button danger disabled={retrying} onClick={() => { if (unknownRetry) onRetrySegment?.(unknownRetry); setUnknownRetry(undefined); }}>仍要重试本段</Button></Space>}
    >当前提交结果未知，继续重试可能重复计费。仅在确认需要人工重试时继续。</Modal>
  </>);
}
