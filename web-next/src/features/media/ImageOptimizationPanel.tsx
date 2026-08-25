import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, Button, Card, CopyOutlined, Input, Modal, Space, subscribeBeforeUnload } from '../../ui/antd';
import type { AdaptedImageOptimizationPrompt } from '../../domain';
import { useUnsavedDraftGuard } from '../../state';
import './media.css';

type TextMode = 'prompt' | 'dialogue' | 'image';

export interface ImagePromptSave {
  readonly expected_sha256: string;
  readonly prompt: string;
}

interface Props {
  prompt: string;
  dialogue: string;
  imagePrompt: AdaptedImageOptimizationPrompt | null;
  promptEditable: boolean;
  promptDraft?: string;
  promptPending?: boolean;
  onPromptDraftChange?: (value: string) => void;
  onSavePrompt?: (value: string) => Promise<string>;
  onSaveImagePrompt?: (payload: ImagePromptSave) => Promise<AdaptedImageOptimizationPrompt>;
  onDirtyChange?: (dirty: boolean) => void;
  draftId?: string;
}

const modeLabels: Record<TextMode, string> = {
  prompt: '生成提示词', dialogue: '段台词', image: '图片优化',
};

export function ImageOptimizationPanel({
  prompt, dialogue, imagePrompt, promptEditable, promptDraft, promptPending = false,
  onPromptDraftChange, onSavePrompt, onSaveImagePrompt, onDirtyChange, draftId,
}: Props) {
  const [mode, setMode] = useState<TextMode>();
  const [savedPrompt, setSavedPrompt] = useState(prompt);
  const [currentImagePrompt, setCurrentImagePrompt] = useState(imagePrompt);
  const [draft, setDraft] = useState(imagePrompt?.text ?? '');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string>();
  const [pendingMode, setPendingMode] = useState<TextMode>();
  const lastImagePropSha = useRef(imagePrompt?.sha256);
  const guard = useUnsavedDraftGuard();
  const imageDirty = currentImagePrompt !== null && draft !== currentImagePrompt.text;
  const currentPromptDraft = promptDraft ?? savedPrompt;
  const promptDirty = promptEditable && currentPromptDraft !== savedPrompt;
  const dirty = imageDirty || promptDirty;

  useEffect(() => {
    if (!promptDirty || currentPromptDraft === prompt) setSavedPrompt(prompt);
  }, [currentPromptDraft, prompt, promptDirty]);
  useEffect(() => {
    if (!imageDirty && imagePrompt?.sha256 !== lastImagePropSha.current) {
      lastImagePropSha.current = imagePrompt?.sha256;
      setCurrentImagePrompt(imagePrompt); setDraft(imagePrompt?.text ?? '');
    }
  }, [imageDirty, imagePrompt]);
  useEffect(() => { onDirtyChange?.(dirty); }, [dirty, onDirtyChange]);
  useEffect(() => {
    if (!dirty) return;
    const beforeUnload = (event: BeforeUnloadEvent) => event.preventDefault();
    return subscribeBeforeUnload(beforeUnload);
  }, [dirty]);

  const saveImage = useCallback(async (): Promise<boolean> => {
    if (!currentImagePrompt || !onSaveImagePrompt || !imageDirty || saving || !draft.trim()) return !imageDirty;
    setSaving(true); setError(undefined);
    try {
      const latest = await onSaveImagePrompt({ expected_sha256: currentImagePrompt.sha256, prompt: draft.trim() });
      setCurrentImagePrompt(latest); setDraft(latest.text);
      return true;
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : '图片优化提示词保存失败');
      return false;
    } finally { setSaving(false); }
  }, [currentImagePrompt, draft, imageDirty, onSaveImagePrompt, saving]);
  const savePrompt = useCallback(async (): Promise<boolean> => {
    if (!onSavePrompt || !promptDirty || saving || !currentPromptDraft.trim()) return !promptDirty;
    setSaving(true); setError(undefined);
    try {
      const latest = await onSavePrompt(currentPromptDraft.trim());
      setSavedPrompt(latest); onPromptDraftChange?.(latest);
      return true;
    } catch {
      return false;
    } finally { setSaving(false); }
  }, [currentPromptDraft, onPromptDraftChange, onSavePrompt, promptDirty, saving]);
  const discardImage = useCallback(() => setDraft(currentImagePrompt?.text ?? ''), [currentImagePrompt]);
  const discardPrompt = useCallback(() => onPromptDraftChange?.(savedPrompt), [onPromptDraftChange, savedPrompt]);
  useEffect(() => {
    const id = currentImagePrompt ? `${draftId ?? currentImagePrompt.sha256}:image` : undefined;
    if (id) guard.register(id, imageDirty ? { save: async () => { if (!await saveImage()) throw new Error('图片优化提示词保存失败'); }, discard: discardImage } : null);
    return () => { if (id) guard.register(id, null); };
  }, [currentImagePrompt, discardImage, draftId, guard, imageDirty, saveImage]);
  useEffect(() => {
    const id = `${draftId ?? 'prompt'}:prompt`;
    guard.register(id, promptDirty ? { save: async () => { if (!await savePrompt()) throw new Error('生成提示词保存失败'); }, discard: discardPrompt } : null);
    return () => guard.register(id, null);
  }, [discardPrompt, draftId, guard, promptDirty, savePrompt]);
  const chooseMode = (next: TextMode) => {
    const currentDirty = mode === 'image' ? imageDirty : mode === 'prompt' ? promptDirty : false;
    if (currentDirty && next !== mode) setPendingMode(next);
    else setMode((current) => current === next ? undefined : next);
  };
  const value = mode === 'image' ? draft : mode === 'prompt' ? currentPromptDraft : dialogue;
  const editable = mode === 'image' || (mode === 'prompt' && promptEditable);

  return (
    <Card title="文本与图片优化">
      <Space orientation="vertical" className="prompt-editor-layout">
        {error ? <Alert type="error" showIcon title={error} /> : null}
        <div className="image-prompt-mode-grid">
          {(Object.keys(modeLabels) as TextMode[]).map((key) => (
            <Button key={key} type={mode === key ? 'primary' : 'default'} disabled={key === 'image' && !imagePrompt} aria-label={`展开${modeLabels[key]}`} onClick={() => chooseMode(key)}>
              {`展开${modeLabels[key]}`}
            </Button>
          ))}
        </div>
        {mode ? (
          <>
            <Input.TextArea
              rows={6}
              aria-label={mode === 'prompt' && promptEditable ? '提示词草稿' : modeLabels[mode]}
              value={value}
              disabled={!editable || saving || promptPending}
              onChange={(event) => mode === 'image' ? setDraft(event.target.value) : onPromptDraftChange?.(event.target.value)}
            />
            {mode === 'prompt' && promptEditable ? (
              <Button type="primary" aria-label={promptPending ? '正在保存' : '确认保存'} loading={promptPending || saving} disabled={!onSavePrompt || !promptDirty} onClick={() => { void savePrompt(); }}>确认保存</Button>
            ) : null}
            {mode === 'image' ? (
              <Space wrap>
                <Button type="primary" loading={saving} disabled={!imageDirty || !draft.trim()} onClick={() => { void saveImage(); }}>保存图片优化</Button>
                <Button disabled={saving || !currentImagePrompt || draft === currentImagePrompt.defaultText} onClick={() => { if (currentImagePrompt) setDraft(currentImagePrompt.defaultText); }}>恢复默认</Button>
                <Button icon={<CopyOutlined />} onClick={() => { void navigator.clipboard?.writeText(draft); }}>复制</Button>
              </Space>
            ) : null}
          </>
        ) : null}
      </Space>
      <Modal
        title={`${mode === 'prompt' ? '生成提示词' : '图片优化提示词'}尚未保存`}
        open={pendingMode !== undefined}
        closable={false}
        footer={<Space><Button onClick={() => setPendingMode(undefined)}>取消</Button><Button danger onClick={() => { if (mode === 'prompt') discardPrompt(); else discardImage(); setMode(pendingMode); setPendingMode(undefined); }}>丢弃</Button><Button type="primary" loading={saving} onClick={() => { const save = mode === 'prompt' ? savePrompt : saveImage; void save().then((saved) => { if (saved) { setMode(pendingMode); setPendingMode(undefined); } }); }}>保存</Button></Space>}
      >请选择保存、丢弃或取消。</Modal>
    </Card>
  );
}
