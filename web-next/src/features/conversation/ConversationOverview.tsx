import type { ReactNode } from 'react';
import {
  Alert,
  Bubble,
  Empty,
  Skeleton,
  Space,
  Tag,
  Typography,
} from '../../ui/antd';
import './conversation.css';

export type ConversationMessageStatus = 'queued' | 'processing' | 'done' | 'failed';
export type ConversationMessageRole = 'user' | 'assistant' | 'system';

export interface ConversationMessage {
  id: string;
  role: ConversationMessageRole;
  content: string;
  status: ConversationMessageStatus;
  error?: string;
  title?: string;
}

export interface ConversationOverviewProps {
  messages: readonly ConversationMessage[];
  appearance?: 'thread' | 'summary';
  loading?: boolean;
  error?: string;
  emptyDescription?: ReactNode;
}

const statusPresentation: Record<
  ConversationMessageStatus,
  { label: string; color?: 'default' | 'processing' | 'success' | 'error' }
> = {
  queued: { label: '排队中', color: 'default' },
  processing: { label: '处理中', color: 'processing' },
  done: { label: '已完成', color: 'success' },
  failed: { label: '失败', color: 'error' },
};

function MessageContent({ message }: { message: ConversationMessage }) {
  if (message.status === 'failed') {
    return (
      <Alert
        type="error"
        showIcon
        title={message.error ?? '处理失败'}
        description={message.content}
      />
    );
  }

  if (message.status === 'processing' && !message.content) {
    return <Skeleton active paragraph={{ rows: 2 }} title={false} />;
  }

  return <Typography.Paragraph>{message.content}</Typography.Paragraph>;
}

export function ConversationOverview({
  messages,
  appearance = 'thread',
  loading = false,
  error,
  emptyDescription = '暂无会话记录',
}: ConversationOverviewProps) {
  if (loading) {
    return (
      <div aria-label="正在加载会话">
        <Skeleton active paragraph={{ rows: 4 }} />
      </div>
    );
  }

  if (error) {
    return <Alert type="error" showIcon title={error} />;
  }

  if (messages.length === 0) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyDescription} />;
  }

  if (appearance === 'summary') {
    const message = messages[messages.length - 1];
    const presentation = statusPresentation[message.status];
    if (message.status === 'failed') {
      return <Alert type="error" showIcon title={message.error ?? '处理失败'} description={message.content} />;
    }
    return (
      <section className="conversation-analysis-summary" aria-label="会话分析进度">
        <div className="conversation-analysis-summary__heading">
          <div>
            <Typography.Title level={3}>
              {message.status === 'done' ? '视频分析完成' : message.status === 'processing' ? '正在分析视频' : '分析任务已排队'}
            </Typography.Title>
            <Typography.Paragraph type="secondary">
              {message.status === 'done'
                ? '已整理真实关键帧、识别台词与生成输入，可以继续调整生成策略。'
                : message.content}
            </Typography.Paragraph>
          </div>
          <Tag color={presentation.color}>{message.status === 'done' ? '可生成' : presentation.label}</Tag>
        </div>
        {message.status === 'processing' ? <Skeleton active paragraph={{ rows: 2 }} title={false} /> : null}
      </section>
    );
  }

  return (
    <section className="conversation-overview" aria-label="会话分析进度">
      <Bubble.List
        className="conversation-overview-list"
        role={{
          user: { placement: 'end', variant: 'filled' },
          assistant: { placement: 'start', variant: 'outlined' },
          system: { placement: 'start', variant: 'borderless' },
        }}
        items={messages.map((message) => {
          const presentation = statusPresentation[message.status];
          return {
            key: message.id,
            role: message.role,
            content: <MessageContent message={message} />,
            header: (
              <Space size="small">
                {message.title ? <Typography.Text strong>{message.title}</Typography.Text> : null}
                <Tag color={presentation.color}>{presentation.label}</Tag>
              </Space>
            ),
          };
        })}
      />
    </section>
  );
}
