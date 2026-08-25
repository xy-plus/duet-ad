import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GenerationSettings } from './GenerationSettings';
import type { GenerationSettingsValue } from './types';

const recommended: GenerationSettingsValue = {
  dialogueMode: 'auto',
  dialogueText: '',
  aspectRatio: '16:9',
  resolution: '768p',
  fitMode: 'crop',
};

afterEach(cleanup);

describe('GenerationSettings', () => {
  it('selects the server recommendation and exposes every short-video dialogue mode', () => {
    render(
      <GenerationSettings
        videoKind="short"
        initialValues={recommended}
        value={recommended}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByRole('radio', { name: '自动台词' })).toBeChecked();
    expect(screen.getByRole('radio', { name: '画幅 16:9' })).toBeChecked();
    expect(screen.getByRole('radio', { name: '编辑识别台词' })).toBeEnabled();
    expect(screen.getByRole('radio', { name: '自定义台词' })).toBeEnabled();
    expect(screen.getByRole('radio', { name: '无台词' })).toBeEnabled();
    expect(screen.getByText('768p')).toBeInTheDocument();
  });

  it('shows the dialogue editor for edit and custom values', () => {
    const { rerender } = render(
      <GenerationSettings
        videoKind="short"
        initialValues={recommended}
        value={{ ...recommended, dialogueMode: 'edit', dialogueText: '识别台词' }}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByRole('textbox', { name: '识别台词内容' })).toHaveValue('识别台词');

    rerender(
      <GenerationSettings
        videoKind="short"
        initialValues={recommended}
        value={{ ...recommended, dialogueMode: 'custom', dialogueText: '自定义台词' }}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByRole('textbox', { name: '自定义台词内容' })).toHaveValue('自定义台词');
  });

  it('offers none, crop and pad without hiding either fit tradeoff', async () => {
    const user = userEvent.setup();
    render(
      <GenerationSettings
        videoKind="short"
        initialValues={recommended}
        value={recommended}
        onChange={vi.fn()}
      />,
    );

    await user.click(screen.getByLabelText('画面适配'));

    expect(screen.getByRole('option', { name: '无需适配' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: '裁切铺满' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: '完整留白' })).toBeInTheDocument();
  });

  it('limits long video to auto/none and never renders a fast-mode control or summary', () => {
    render(
      <GenerationSettings
        videoKind="long"
        initialValues={recommended}
        value={recommended}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByRole('radio', { name: '自动台词' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: '无台词' })).toBeInTheDocument();
    expect(screen.queryByRole('radio', { name: '编辑识别台词' })).not.toBeInTheDocument();
    expect(screen.queryByRole('radio', { name: '自定义台词' })).not.toBeInTheDocument();
    expect(screen.queryByText(/fast|快速模式|快速生成/iu)).not.toBeInTheDocument();
  });

  it('replaces the draft with server-frozen generation evidence and removes all editors', () => {
    const evidence = {
      id: 'generation-frozen',
      durationSeconds: 30,
      segmentCount: 3,
      parameters: {
        dialogueMode: 'custom' as const,
        dialogueText: '服务端冻结台词',
        aspectRatio: '9:16' as const,
        resolution: '480p' as const,
        fitMode: 'pad' as const,
      },
    };
    render(
      <GenerationSettings
        videoKind="short"
        initialValues={recommended}
        value={recommended}
        generation={evidence}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByText('已冻结生成参数')).toBeInTheDocument();
    expect(screen.getByText('服务端冻结台词')).toBeInTheDocument();
    expect(screen.getByText('9:16')).toBeInTheDocument();
    expect(screen.getByText('480p')).toBeInTheDocument();
    expect(screen.getByText('完整留白')).toBeInTheDocument();
    expect(screen.getByText('30 秒')).toBeInTheDocument();
    expect(screen.getByText('3 段')).toBeInTheDocument();
    expect(screen.queryByRole('radio')).not.toBeInTheDocument();
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
  });

  it('shows safe placeholders for missing legacy frozen fields', () => {
    const evidence = {
      id: 'legacy-generation',
      durationSeconds: null,
      segmentCount: null,
    };

    render(
      <GenerationSettings
        videoKind="short"
        initialValues={recommended}
        generation={evidence}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByText('已冻结生成参数')).toBeInTheDocument();
    expect(screen.getAllByText('未提供').length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });
});
