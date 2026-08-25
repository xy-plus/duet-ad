import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ImageOptimizationPanel } from './ImageOptimizationPanel';

afterEach(cleanup);

describe('ImageOptimizationPanel', () => {
  it('uses one text area for three text modes and keeps long prompt and dialogue read only', async () => {
    const user = userEvent.setup();
    render(<ImageOptimizationPanel prompt="长段生成提示词" dialogue="长段台词" imagePrompt={{ text: '优化稿', defaultText: '默认优化稿', sha256: 'a'.repeat(64) }} promptEditable={false} onSaveImagePrompt={vi.fn()} />);
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '展开生成提示词' }));
    expect(screen.getByRole('textbox', { name: '生成提示词' })).toHaveValue('长段生成提示词');
    expect(screen.getByRole('textbox', { name: '生成提示词' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: '展开段台词' }));
    expect(screen.getByRole('textbox', { name: '段台词' })).toHaveValue('长段台词');
    await user.click(screen.getByRole('button', { name: '展开图片优化' }));
    expect(screen.getByRole('textbox', { name: '图片优化' })).toBeEnabled();
  });

  it('restores the default only into a dirty draft and saves with CAS evidence', async () => {
    const user = userEvent.setup();
    const onSave = vi.fn().mockResolvedValue({ text: '默认优化稿', defaultText: '默认优化稿', sha256: 'c'.repeat(64) });
    render(<ImageOptimizationPanel prompt="生成提示词" dialogue="台词" imagePrompt={{ text: '已保存优化稿', defaultText: '默认优化稿', sha256: 'b'.repeat(64) }} promptEditable onSaveImagePrompt={onSave} />);
    await user.click(screen.getByRole('button', { name: '展开图片优化' }));
    await user.click(screen.getByRole('button', { name: '恢复默认' }));
    expect(screen.getByRole('textbox', { name: '图片优化' })).toHaveValue('默认优化稿');
    expect(onSave).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: '保存图片优化' }));
    expect(onSave).toHaveBeenCalledWith({ expected_sha256: 'b'.repeat(64), prompt: '默认优化稿' });
  });

  it('does not open image editing without a capability-approved prompt', () => {
    render(<ImageOptimizationPanel prompt="生成" dialogue="台词" imagePrompt={null} promptEditable onSaveImagePrompt={vi.fn()} />);
    expect(screen.getByRole('button', { name: '展开图片优化' })).toBeDisabled();
  });

  it('guards a dirty short prompt before switching text and can discard it', async () => {
    const user = userEvent.setup();
    function Harness() {
      const [promptDraft, setPromptDraft] = useState('已保存生成稿');
      return <ImageOptimizationPanel prompt="已保存生成稿" promptDraft={promptDraft} dialogue="台词" imagePrompt={{ text: '图片', defaultText: '默认', sha256: 'd'.repeat(64) }} promptEditable onPromptDraftChange={setPromptDraft} onSavePrompt={vi.fn()} onSaveImagePrompt={vi.fn()} />;
    }
    render(<Harness />);
    await user.click(screen.getByRole('button', { name: '展开生成提示词' }));
    await user.type(screen.getByRole('textbox', { name: '提示词草稿' }), '未保存');
    await user.click(screen.getByRole('button', { name: '展开段台词' }));
    expect(screen.getByRole('dialog', { name: '生成提示词尚未保存' })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /丢\s*弃/u }));
    expect(screen.getByRole('textbox', { name: '段台词' })).toHaveValue('台词');
  });

  it('uses the latest returned sha for consecutive image saves', async () => {
    const user = userEvent.setup();
    const onSave = vi.fn()
      .mockResolvedValueOnce({ text: '第一次', defaultText: '默认二', sha256: '2'.repeat(64) })
      .mockResolvedValueOnce({ text: '第二次', defaultText: '默认三', sha256: '3'.repeat(64) });
    render(<ImageOptimizationPanel prompt="生成" dialogue="台词" imagePrompt={{ text: '初始', defaultText: '默认一', sha256: '1'.repeat(64) }} promptEditable onSaveImagePrompt={onSave} />);
    await user.click(screen.getByRole('button', { name: '展开图片优化' }));
    await user.clear(screen.getByRole('textbox', { name: '图片优化' }));
    await user.type(screen.getByRole('textbox', { name: '图片优化' }), '第一次');
    await user.click(screen.getByRole('button', { name: '保存图片优化' }));
    await user.clear(screen.getByRole('textbox', { name: '图片优化' }));
    await user.type(screen.getByRole('textbox', { name: '图片优化' }), '第二次');
    await user.click(screen.getByRole('button', { name: '保存图片优化' }));
    expect(onSave.mock.calls.map(([payload]) => payload.expected_sha256)).toEqual(['1'.repeat(64), '2'.repeat(64)]);
  });
});
