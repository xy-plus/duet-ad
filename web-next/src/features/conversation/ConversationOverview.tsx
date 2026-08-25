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
