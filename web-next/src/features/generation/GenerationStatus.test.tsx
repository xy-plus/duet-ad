import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GenerationStatus } from './GenerationStatus';
import type { GenerationStatusModel } from './types';

const segments = [
  { id: 'segment-1', title: '片段 1 · 0—8 秒', status: 'succeeded' as const },
  { id: 'segment-2', title: '片段 2 · 8—17 秒', status: 'succeeded' as const },
  { id: 'segment-3', title: '片段 3 · 17—24 秒', status: 'running' as const },
];

afterEach(cleanup);

describe('GenerationStatus', () => {
  it('renders the actual segment list and progress without inventing retry counts', () => {
    const model: GenerationStatusModel = {
      phase: 'running',
      generationId: 'generation-running',
      paidTaskCount: 3,
      segments,
      stageLabel: '正在生成第 3 个片段',
    };

    render(<GenerationStatus model={model} onAction={vi.fn()} />);

    expect(screen.getByText('片段 1 · 0—8 秒')).toBeInTheDocument();
    expect(screen.getByText('片段 2 · 8—17 秒')).toBeInTheDocument();
    expect(screen.getByText('片段 3 · 17—24 秒')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '67');
    expect(screen.getByText('付费任务数：3 个')).toBeInTheDocument();
    expect(screen.queryByText(/自动重试|重试次数/)).not.toBeInTheDocument();
  });

  it.each([
    ['context_ir_native', 'Context IR 正在优化最终提示词'],
    ['h3', 'H3 视频生成中'],
    ['stitch', '正在合成最终视频'],
  ])('renders the server stage %s as a user-visible pipeline step', (stageLabel, expected) => {
    render(
      <GenerationStatus
        model={{
          phase: 'running',
          generationId: 'generation-running',
          paidTaskCount: 1,
          segments: [],
          stageLabel,
        }}
      />,
    );

    expect(screen.getByText(expected)).toBeInTheDocument();
  });

  it.each([
    [
      'new',
      { phase: 'new', paidTaskCount: 2, segments: [] } satisfies GenerationStatusModel,
      '确认生成',
      { type: 'new' },
    ],
    [
      'failed',
      {
        phase: 'failed',
        generationId: 'generation-failed',
        paidTaskCount: 1,
        segments,
        errorMessage: '供应商明确失败',
      } satisfies GenerationStatusModel,
      '新建任务重试',
      { type: 'retry', failedGenerationId: 'generation-failed', reuseGenerationId: false },
    ],
    [
      'resume',
      {
        phase: 'resume_required',
        generationId: 'generation-resume',
        paidTaskCount: 2,
        segments,
      } satisfies GenerationStatusModel,
      '继续原任务',
      { type: 'resume', generationId: 'generation-resume' },
    ],
    [
      'stitch',
      {
        phase: 'stitch_required',
        generationId: 'generation-stitch',
        paidTaskCount: 2,
        segments,
      } satisfies GenerationStatusModel,
      '继续拼接',
      { type: 'retry_stitch', generationId: 'generation-stitch' },
    ],
  ])('emits the financially safe %s action', async (_name, model, label, expected) => {
    const user = userEvent.setup();
    const onAction = vi.fn();
    render(<GenerationStatus model={model} onAction={onAction} />);

    await user.click(screen.getByRole('button', { name: label }));

    expect(onAction).toHaveBeenCalledWith(expected);
  });

  it('shows submission_unknown as a terminal warning with zero actions', () => {
    render(
      <GenerationStatus
        model={{
          phase: 'submission_unknown',
          generationId: 'generation-unknown',
          paidTaskCount: null,
          segments,
          errorMessage: '无法确认供应商是否已接单',
        }}
        onAction={vi.fn()}
      />,
    );

    expect(screen.getByText('提交状态未知')).toBeInTheDocument();
    expect(screen.getByText('付费任务数：未知')).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('disables paid confirmation when the exact task count is unknown', () => {
    render(
      <GenerationStatus
        model={{ phase: 'new', paidTaskCount: null, segments: [] }}
        onAction={vi.fn()}
      />,
    );

    expect(screen.getByText('付费任务数：未知')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '确认生成' })).toBeDisabled();
  });

  it('keeps the paid action visible but disabled while required settings are missing', () => {
    render(
      <GenerationStatus
        model={{ phase: 'new', paidTaskCount: 1, segments: [] }}
        actionDisabled
        onAction={vi.fn()}
      />,
    );

    expect(screen.getByRole('button', { name: '确认生成' })).toBeDisabled();
  });

  it('uses a result surface for completed generation', () => {
    render(
      <GenerationStatus
        model={{
          phase: 'succeeded',
          generationId: 'generation-done',
          paidTaskCount: 3,
          segments: segments.map((segment) => ({ ...segment, status: 'succeeded' as const })),
        }}
        onAction={vi.fn()}
      />,
    );

    expect(screen.getByText('视频生成完成')).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });
});
