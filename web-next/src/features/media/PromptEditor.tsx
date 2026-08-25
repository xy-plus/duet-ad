import {
  Alert,
  Button,
  Card,
  Collapse,
  CopyOutlined,
  Input,
  LockOutlined,
  ReloadOutlined,
  Space,
  Typography,
} from '../../ui/antd';

export interface PromptConflict {
  code: 'prompt_changed';
  message?: string;
}

export interface PromptEditorProps {
  prompt: string;
  draft: string;
  onDraftChange: (value: string) => void;
  onCopy: (value: string) => void;
  onSave: (value: string) => void;
  onReload?: () => void;
  pending?: boolean;
  locked?: boolean;
  lockReason?: string;
  conflict?: PromptConflict;
}

export function PromptEditor({
  prompt,
  draft,
  onDraftChange,
  onCopy,
  onSave,
  onReload,
  pending = false,
  locked = false,
  lockReason = '已存在生成记录，提示词不可再修改',
  conflict,
}: PromptEditorProps) {
  const editingBlocked = locked || pending || conflict !== undefined;
  const saveDisabled = editingBlocked || draft.trim().length === 0 || draft === prompt;

  return (
    <Card
      title="提示词编辑"
      extra={
        <Button
          icon={<CopyOutlined />}
          aria-label="复制提示词"
          onClick={() => onCopy(prompt)}
        >
          复制
        </Button>
      }
    >
      <Space orientation="vertical" className="prompt-editor-layout">
        {locked ? (
          <Alert type="info" showIcon icon={<LockOutlined />} title={lockReason} />
        ) : null}
        {conflict ? (
          <Alert
            type="warning"
            showIcon
            title={`${conflict.code}：${conflict.message ?? '服务端提示词已变化'}`}
            action={
              <Button
                icon={<ReloadOutlined />}
                aria-label="重新加载最新提示词"
                onClick={onReload}
                disabled={!onReload}
              >
                重新加载
              </Button>
            }
          />
        ) : null}
        <Collapse
          defaultActiveKey={['prompt']}
          items={[
            {
              key: 'prompt',
              label: <Typography.Text strong>生成提示词</Typography.Text>,
              children: (
                <Space orientation="vertical" className="prompt-editor-layout">
                  <Input.TextArea
                    aria-label="提示词草稿"
                    value={draft}
                    disabled={editingBlocked}
                    rows={5}
                    onChange={(event) => onDraftChange(event.target.value)}
                  />
                  <Button
                    type="primary"
                    loading={pending}
                    disabled={saveDisabled}
                    aria-label={pending ? '正在保存' : '确认保存'}
                    onClick={() => onSave(draft.trim())}
                  >
                    {pending ? '正在保存' : '确认保存'}
                  </Button>
                </Space>
              ),
            },
          ]}
        />
      </Space>
    </Card>
  );
}
