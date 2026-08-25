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
  optimize_image: false,
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

    expect(screen.getByRole('checkbox', { name: '移除文字/字幕' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: '移除常见 Logo/图标' })).not.toBeChecked();
    expect(screen.getByRole('checkbox', { name: '进行图片优化' })).not.toBeChecked();
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

    await user.click(screen.getByRole('checkbox', { name: '移除常见 Logo/图标' }));
    expect(onOptionsChange).toHaveBeenCalledWith({
      remove_subtitle: true,
      remove_brand: true,
      optimize_image: false,
    });
  });

  it('normalizes hidden unsupported options to false before submit', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<PostprocessConfig open options={{ remove_subtitle: true, remove_brand: true, optimize_image: true }} capabilities={{ remove_subtitle: true, remove_brand: false, optimize_image: false }} onOptionsChange={vi.fn()} onCancel={vi.fn()} onSubmit={onSubmit} />);
    await user.click(screen.getByRole('button', { name: '开始后处理' }));
    expect(onSubmit).toHaveBeenCalledWith({ remove_subtitle: true, remove_brand: false, optimize_image: false });
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
  it('does not render retry when the caller withholds authority', () => {
    render(
      <PostprocessStatus
        task={{ ...runningTask, status: 'failed', errorMessage: '处理失败' }}
      />,
    );

    expect(screen.queryByRole('button', { name: /重试/u })).not.toBeInTheDocument();
  });

  it('is a non-blocking card, so navigation remains available during background work', async () => {
    const user = userEvent.setup();
    const onNavigate = vi.fn();
    render(
      <>
        <Button onClick={onNavigate}>切换会话</Button>
        <PostprocessStatus task={runningTask} />
      </>,
    );

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '40');
    await user.click(screen.getByRole('button', { name: '切换会话' }));
    expect(onNavigate).toHaveBeenCalledOnce();
  });

  it('shows locked options and previews, but only offers refresh when failed segments are absent', async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();
    const task: PostprocessTask = {
      id: 'post-partial',
      status: 'partial_success',
      options: { remove_subtitle: true, remove_brand: true, optimize_image: false },
      processedCount: 3,
      totalCount: 3,
      errorMessage: '1 张关键帧处理失败',
      results: [
        { id: 'frame-ok', status: 'succeeded', url: '/api/frames/ok.png', alt: '处理后关键帧 1' },
        { id: 'frame-failed', status: 'failed', errorMessage: '局部修复失败' },
      ],
    };

    render(<PostprocessStatus task={task} onRefresh={onRefresh} />);

    expect(screen.getAllByText('部分处理成功')).toHaveLength(2);
    expect(screen.getByText('移除文字/字幕、移除常见 Logo/图标')).toBeInTheDocument();
    expect(screen.getByRole('img', { name: '处理后关键帧 1' })).toBeInTheDocument();
    expect(screen.getByText(/没有可安全重试的分段 revision/u)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '刷新状态' }));
    expect(onRefresh).toHaveBeenCalledOnce();
    expect(screen.queryByRole('button', { name: /重试/u })).not.toBeInTheDocument();
  });

  it('locks repeated retry while the retry request is pending', () => {
    const failedTask: PostprocessTask = {
      ...runningTask,
      id: 'post-failed',
      status: 'failed',
      errorMessage: '处理失败',
      segments: [{ index: 1, status: 'failed', stage: 'optimize_image', completedFrames: 0, totalFrames: 1, revision: 2, error: '处理失败' }],
    };

    render(<PostprocessStatus task={failedTask} retrying onRetrySegment={vi.fn()} />);

    expect(screen.getByRole('button', { name: /重试本段/u })).toBeDisabled();
  });

  it('retries only a confirmed failed segment with revision CAS and neutral stage text', async () => {
    const user = userEvent.setup();
    const onRetrySegment = vi.fn();
    const failedTask: PostprocessTask = {
      ...runningTask,
      id: 'post-failed',
      status: 'failed',
      errorMessage: '处理失败',
      segments: [{ index: 0, status: 'failed', stage: 'seedream', completedFrames: 0, totalFrames: 1, revision: 5, error: '局部失败' }],
    };

    render(<PostprocessStatus task={failedTask} onRetrySegment={onRetrySegment} />);
    expect(screen.getByText('当前视频')).toBeInTheDocument();
    expect(screen.getByText('图片优化')).toBeInTheDocument();
    expect(screen.queryByText('seedream')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '重试本段' }));

    expect(onRetrySegment).toHaveBeenCalledWith({ index: 0, expectedRevision: 5 });
  });

  it('renders per-segment progress and retries only with index and expected revision', async () => {
    const user = userEvent.setup();
    const onRetrySegment = vi.fn();
    render(<PostprocessStatus task={{ ...runningTask, segments: [{ index: 2, status: 'failed', stage: 'seedream', completedFrames: 1, totalFrames: 4, revision: 7, error: 'submission_unknown' }] }} onRetrySegment={onRetrySegment} />);
    expect(screen.getByRole('progressbar', { name: '第 2 段后处理进度' })).toHaveAttribute('aria-valuenow', '25');
    expect(screen.getByText(/重试可能重复计费/u)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '重试本段' }));
    expect(onRetrySegment).not.toHaveBeenCalled();
    expect(screen.getByRole('dialog', { name: '确认重试未知提交段' })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '仍要重试本段' }));
    expect(onRetrySegment).toHaveBeenCalledWith({ index: 2, expectedRevision: 7 });
  });
});
