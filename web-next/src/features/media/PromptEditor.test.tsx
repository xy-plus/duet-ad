import { useState } from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { PromptEditor, type PromptEditorProps } from './PromptEditor';

afterEach(cleanup);

function ControlledPromptEditor(props: Omit<PromptEditorProps, 'draft' | 'onDraftChange'>) {
  const [draft, setDraft] = useState(props.prompt);
  return <PromptEditor {...props} draft={draft} onDraftChange={setDraft} />;
}

describe('PromptEditor', () => {
  it('supports controlled editing, copy and confirmed save', async () => {
    const user = userEvent.setup();
    const onCopy = vi.fn();
    const onSave = vi.fn();
    render(
      <ControlledPromptEditor prompt="原始提示词" onCopy={onCopy} onSave={onSave} />,
    );

    await user.click(screen.getByRole('button', { name: '复制提示词' }));
    expect(onCopy).toHaveBeenCalledWith('原始提示词');

    expect(screen.queryByLabelText('提示词草稿')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /生成提示词/ }));
    const editor = screen.getByLabelText('提示词草稿');
    await user.clear(editor);
    await user.type(editor, '修改后的提示词');
    await user.click(screen.getByRole('button', { name: '确认保存' }));
    expect(onSave).toHaveBeenCalledWith('修改后的提示词');
  });

  it('locks the prompt when generation evidence exists and respects pending state', async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <PromptEditor
        prompt="已冻结提示词"
        draft="试图修改"
        locked
        lockReason="已生成分段，提示词不可再修改"
        onDraftChange={vi.fn()}
        onCopy={vi.fn()}
        onSave={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('button', { name: /生成提示词/ }));
    expect(screen.getByRole('alert')).toHaveTextContent('已生成分段，提示词不可再修改');
    expect(screen.getByLabelText('提示词草稿')).toBeDisabled();
    expect(screen.getByRole('button', { name: '确认保存' })).toBeDisabled();

    rerender(
      <PromptEditor
        prompt="服务端提示词"
        draft="服务端提示词"
        pending
        onDraftChange={vi.fn()}
        onCopy={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    expect(screen.getByRole('button', { name: '正在保存' })).toBeDisabled();
    expect(screen.getByLabelText('提示词草稿')).toBeDisabled();
  });

  it('shows prompt_changed CAS conflict and delegates the reload action', async () => {
    const user = userEvent.setup();
    const onReload = vi.fn();
    render(
      <PromptEditor
        prompt="旧提示词"
        draft="本地修改"
        conflict={{ code: 'prompt_changed', message: '提示词已被其他请求修改' }}
        onDraftChange={vi.fn()}
        onCopy={vi.fn()}
        onSave={vi.fn()}
        onReload={onReload}
      />,
    );

    await user.click(screen.getByRole('button', { name: /生成提示词/ }));
    expect(screen.getByRole('alert')).toHaveTextContent('prompt_changed');
    expect(screen.getByRole('alert')).toHaveTextContent('提示词已被其他请求修改');
    await user.click(screen.getByRole('button', { name: '重新加载最新提示词' }));
    expect(onReload).toHaveBeenCalledOnce();
  });
});
