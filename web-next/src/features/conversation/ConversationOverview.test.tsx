import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { ConversationOverview, type ConversationMessage } from './ConversationOverview';

class TestIntersectionObserver implements IntersectionObserver {
  readonly root = null;
  readonly rootMargin = '';
  readonly thresholds = [];
  disconnect() {}
  observe() {}
  takeRecords() { return []; }
  unobserve() {}
}

beforeAll(() => {
  globalThis.IntersectionObserver = TestIntersectionObserver;
});

afterEach(cleanup);

const messages: ConversationMessage[] = [
  { id: 'queued', role: 'user', content: '等待上传', status: 'queued' },
  { id: 'processing', role: 'assistant', content: '正在分析视频', status: 'processing' },
  { id: 'done', role: 'assistant', content: '分析完成', status: 'done' },
  {
    id: 'failed',
    role: 'assistant',
    content: '分析未完成',
    status: 'failed',
    error: '媒体读取失败',
  },
];

describe('ConversationOverview', () => {
  it('renders the real queued, processing, done and failed states in the bubble flow', () => {
    render(<ConversationOverview messages={messages} />);

    expect(screen.getByText('排队中')).toBeInTheDocument();
    expect(screen.getByText('处理中')).toBeInTheDocument();
    expect(screen.getByText('已完成')).toBeInTheDocument();
    expect(screen.getByText('失败')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('媒体读取失败');
  });

  it('updates a bubble when its server-owned status changes', () => {
    const { rerender } = render(
      <ConversationOverview messages={[messages[0]]} />,
    );

    expect(screen.getByText('排队中')).toBeInTheDocument();

    rerender(
      <ConversationOverview
        messages={[{ ...messages[0], content: '上传完成', status: 'done' }]}
      />,
    );

    expect(screen.queryByText('排队中')).not.toBeInTheDocument();
    expect(screen.getByText('已完成')).toBeInTheDocument();
    expect(screen.getByText('上传完成')).toBeInTheDocument();
  });

  it('uses explicit loading, failure and empty states', () => {
    const { rerender } = render(<ConversationOverview messages={[]} loading />);
    expect(screen.getByLabelText('正在加载会话')).toBeInTheDocument();

    rerender(<ConversationOverview messages={[]} error="会话加载失败" />);
    expect(screen.getByRole('alert')).toHaveTextContent('会话加载失败');

    rerender(<ConversationOverview messages={[]} />);
    expect(screen.getByText('暂无会话记录')).toBeInTheDocument();
  });

  it('uses one compact analysis conclusion in the task detail instead of repeating bubbles', () => {
    render(<ConversationOverview appearance="summary" messages={messages.slice(0, 3)} />);

    expect(screen.getByRole('heading', { name: '视频分析完成' })).toBeInTheDocument();
    expect(screen.getByText('可生成')).toBeInTheDocument();
    expect(screen.queryByText('等待上传')).not.toBeInTheDocument();
  });
});
