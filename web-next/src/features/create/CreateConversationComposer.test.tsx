import { useState } from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  CreateConversationComposer,
  type CreateConversationDraft,
} from './CreateConversationComposer';

afterEach(cleanup);

const urlDraft: CreateConversationDraft = {
  source: { type: 'url', url: '' },
  note: '',
  transcript: { mode: 'keep' },
};

function ComposerHarness({
  initialValue = urlDraft,
  onSubmit = () => undefined,
  submitting = false,
  uploadProgress,
}: {
  initialValue?: CreateConversationDraft;
  onSubmit?: (draft: CreateConversationDraft) => void;
  submitting?: boolean;
  uploadProgress?: { percent: number; status?: 'normal' | 'active' | 'success' | 'exception' };
}) {
  const [value, setValue] = useState(initialValue);

  return (
    <CreateConversationComposer
      languageOptions={[{ value: '英语' }, { value: '日语' }]}
      onChange={setValue}
      onSubmit={onSubmit}
      submitting={submitting}
      uploadProgress={uploadProgress}
      value={value}
    />
  );
}

describe('CreateConversationComposer', () => {
  it('clears the other source whenever URL and file modes are switched', async () => {
    const user = userEvent.setup();
    const { container } = render(<ComposerHarness />);

    await user.type(screen.getByLabelText('视频链接'), 'https://example.com/source.mp4');
    await user.click(screen.getByText('上传文件'));

    const fileInput = container.querySelector<HTMLInputElement>('input[accept="video/*"]');
    expect(fileInput).not.toBeNull();
    const file = new File(['video'], 'source.mp4', { type: 'video/mp4' });
    fireEvent.change(fileInput!, { target: { files: [file] } });
    await waitFor(() => expect(screen.getByText('source.mp4')).toBeInTheDocument());

    await user.click(screen.getByText('链接输入'));
    expect(screen.getByLabelText('视频链接')).toHaveValue('');
    await user.click(screen.getByText('上传文件'));
    expect(screen.queryByText('source.mp4')).not.toBeInTheDocument();
  });

  it('locks every creation control while submission is in flight', () => {
    render(
      <ComposerHarness
        initialValue={{
          source: { type: 'url', url: 'https://example.com/source.mp4' },
          note: '做成一支简洁广告',
          transcript: { mode: 'translate', targetLanguage: '法语' },
        }}
        submitting
      />,
    );

    expect(screen.getByLabelText('视频来源')).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByLabelText('视频链接')).toBeDisabled();
    expect(screen.getByLabelText('台词处理')).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByLabelText('目标语言')).toBeDisabled();
    expect(screen.getByPlaceholderText('补充视频用途、受众或风格偏好…')).toBeDisabled();
    expect(screen.getByRole('button', { name: '创建中' })).toBeDisabled();
  });

  it('accepts a free-form translation language and submits the controlled draft', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn<(draft: CreateConversationDraft) => void>();
    render(
      <ComposerHarness
        initialValue={{
          source: { type: 'url', url: 'https://example.com/source.mp4' },
          note: '保留画面节奏',
          transcript: { mode: 'translate', targetLanguage: '' },
        }}
        onSubmit={onSubmit}
      />,
    );

    await user.type(screen.getByLabelText('目标语言'), '克林贡语');
    await user.click(screen.getByRole('button', { name: '创建会话' }));

    expect(onSubmit).toHaveBeenCalledWith({
      source: { type: 'url', url: 'https://example.com/source.mp4' },
      note: '保留画面节奏',
      transcript: { mode: 'translate', targetLanguage: '克林贡语' },
    });
  });

  it('renders caller-owned upload progress without manufacturing completion', () => {
    const file = new File(['video'], 'uploading.mp4', { type: 'video/mp4' });
    render(
      <ComposerHarness
        initialValue={{
          source: { type: 'file', file },
          note: '',
          transcript: { mode: 'rewrite' },
        }}
        uploadProgress={{ percent: 42, status: 'active' }}
      />,
    );

    expect(screen.getByText('uploading.mp4')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '42');
  });

  it('reports backend errors and never invents a follow-up message after submit', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn<(draft: CreateConversationDraft) => void>();
    render(
      <CreateConversationComposer
        error="创建失败，请检查来源后重试"
        languageOptions={[]}
        onChange={() => undefined}
        onSubmit={onSubmit}
        submitting={false}
        value={{
          source: { type: 'url', url: 'https://example.com/source.mp4' },
          note: '真实创建请求',
          transcript: { mode: 'keep' },
        }}
      />,
    );

    expect(screen.getByRole('alert')).toHaveTextContent('创建失败');
    await user.click(screen.getByRole('button', { name: '创建会话' }));
    expect(onSubmit).toHaveBeenCalledOnce();
    expect(screen.queryByText(/已记录|已发送|follow-up/i)).not.toBeInTheDocument();
  });
});
