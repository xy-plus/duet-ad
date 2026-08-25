import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { Button } from '../../ui/antd';
import { PostprocessConfig } from './PostprocessConfig';
import { PostprocessStatus } from './PostprocessStatus';
import type { PostprocessOptions, PostprocessTask } from './types';

const options: PostprocessOptions = {
  remove_subtitle: true,
  remove_brand: false,
};

const runningTask: PostprocessTask = {
  id: 'post-running',
  status: 'running',
  options,
  processedCount: 2,
  totalCount: 5,
  results: [],
};

afterEach(cleanup);

describe('PostprocessConfig', () => {
  it('uses only the server contract option names before submission', () => {
    render(
      <PostprocessConfig
        open
        options={options}
        onOptionsChange={vi.fn()}
        onCancel={vi.fn()}
        onSubmit={vi.fn()}
      />,
    );

    expect(screen.getByRole('checkbox', { name: '移除字幕' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: '移除品牌标识' })).not.toBeChecked();
    expect(screen.queryByText(/remove_subtitles|remove_copyrighted_objects/)).not.toBeInTheDocument();
  });

  it('emits the exact server option snapshot', async () => {
    const user = userEvent.setup();
    const onOptionsChange = vi.fn();
    render(
      <PostprocessConfig
        open
        options={options}
        onOptionsChange={onOptionsChange}
        onCancel={vi.fn()}
        onSubmit={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('checkbox', { name: '移除品牌标识' }));
    expect(onOptionsChange).toHaveBeenCalledWith({
      remove_subtitle: true,
      remove_brand: true,
    });
  });

  it('locks duplicate submission immediately', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <PostprocessConfig
        open
        options={options}
        onOptionsChange={vi.fn()}
        onCancel={vi.fn()}
        onSubmit={onSubmit}
      />,
    );

    await user.dblClick(screen.getByRole('button', { name: '开始后处理' }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: '提交中' })).toBeDisabled();
  });

  it('closes the modal as soon as server options prove submission', () => {
    render(
      <PostprocessConfig
        open
        options={options}
        serverOptions={options}
        onOptionsChange={vi.fn()}
        onCancel={vi.fn()}
        onSubmit={vi.fn()}
      />,
    );

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
});

describe('PostprocessStatus', () => {
  it('is a non-blocking card, so navigation remains available during background work', async () => {
    const user = userEvent.setup();
    const onNavigate = vi.fn();
    render(
      <>
        <Button onClick={onNavigate}>切换会话</Button>
        <PostprocessStatus task={runningTask} onRetry={vi.fn()} />
      </>,
    );

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '40');
    await user.click(screen.getByRole('button', { name: '切换会话' }));
    expect(onNavigate).toHaveBeenCalledOnce();
  });

  it('shows locked server options, partial success, retry and previewable successful images', async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const task: PostprocessTask = {
      id: 'post-partial',
      status: 'partial_success',
      options: { remove_subtitle: true, remove_brand: true },
      processedCount: 3,
      totalCount: 3,
      errorMessage: '1 张关键帧处理失败',
      results: [
        { id: 'frame-ok', status: 'succeeded', url: '/api/frames/ok.png', alt: '处理后关键帧 1' },
        { id: 'frame-failed', status: 'failed', errorMessage: '局部修复失败' },
      ],
    };

    render(<PostprocessStatus task={task} onRetry={onRetry} />);

    expect(screen.getAllByText('部分处理成功')).toHaveLength(2);
    expect(screen.getByText('移除字幕、移除品牌标识')).toBeInTheDocument();
    expect(screen.getByRole('img', { name: '处理后关键帧 1' })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '重试失败项' }));
    expect(onRetry).toHaveBeenCalledWith({ taskId: 'post-partial', options: task.options });
  });

  it('locks repeated retry while the retry request is pending', () => {
    const failedTask: PostprocessTask = {
      ...runningTask,
      id: 'post-failed',
      status: 'failed',
      errorMessage: '处理失败',
    };

    render(<PostprocessStatus task={failedTask} retrying onRetry={vi.fn()} />);

    expect(screen.getByRole('button', { name: '重试中' })).toBeDisabled();
  });

  it('retries a confirmed failed task with its server-frozen options', async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const failedTask: PostprocessTask = {
      ...runningTask,
      id: 'post-failed',
      status: 'failed',
      errorMessage: '处理失败',
    };

    render(<PostprocessStatus task={failedTask} onRetry={onRetry} />);
    await user.click(screen.getByRole('button', { name: '重试失败项' }));

    expect(onRetry).toHaveBeenCalledWith({ taskId: 'post-failed', options: failedTask.options });
  });
});
