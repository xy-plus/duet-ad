import { useCallback, type ReactNode } from 'react';
import {
  Alert,
  AutoComplete,
  Button,
  Card,
  DeleteOutlined,
  Form,
  Input,
  LinkOutlined,
  Progress,
  Radio,
  Segmented,
  SendOutlined,
  Sender,
  Space,
  Typography,
  Upload,
  UploadOutlined,
} from '../../ui/antd';
import './create.css';

export type CreateConversationSource =
  | { type: 'url'; url: string }
  | { type: 'file'; file?: File };

export type TranscriptHandling =
  | { mode: 'keep' }
  | { mode: 'rewrite' }
  | { mode: 'translate'; targetLanguage: string };

export interface CreateConversationDraft {
  note: string;
  source: CreateConversationSource;
  transcript: TranscriptHandling;
}

export interface LanguageOption {
  label?: ReactNode;
  value: string;
}

export interface UploadProgressState {
  percent: number;
  status?: 'normal' | 'active' | 'success' | 'exception';
}

export interface CreateConversationComposerProps {
  error?: string;
  languageOptions: readonly LanguageOption[];
  onChange: (value: CreateConversationDraft) => void;
  onErrorDismiss?: () => void;
  onSubmit: (value: CreateConversationDraft) => void | Promise<void>;
  submitting: boolean;
  uploadProgress?: UploadProgressState;
  value: CreateConversationDraft;
}

export function CreateConversationComposer({
  error,
  languageOptions,
  onChange,
  onErrorDismiss,
  onSubmit,
  submitting,
  uploadProgress,
  value,
}: CreateConversationComposerProps) {
  const update = useCallback((next: CreateConversationDraft) => {
    if (submitting) return;
    if (error) onErrorDismiss?.();
    onChange(next);
  }, [error, onChange, onErrorDismiss, submitting]);

  const changeSourceType = (type: CreateConversationSource['type']) => {
    update({
      ...value,
      source: type === 'url' ? { type: 'url', url: '' } : { type: 'file' },
    });
  };

  const changeTranscriptMode = (mode: TranscriptHandling['mode']) => {
    update({
      ...value,
      transcript: mode === 'translate' ? { mode, targetLanguage: '' } : { mode },
    });
  };

  const sourceReady = value.source.type === 'url'
    ? Boolean(value.source.url.trim())
    : Boolean(value.source.file);
  const transcriptReady = value.transcript.mode !== 'translate'
    || Boolean(value.transcript.targetLanguage.trim());
  const canSubmit = sourceReady && transcriptReady && !submitting;
  const selectedFile = value.source.type === 'file' ? value.source.file : undefined;

  const submit = () => {
    if (canSubmit) void onSubmit(value);
  };

  const configuration = (
    <div className="create-composer__configuration">
      <Form.Item className="create-composer__field" label="视频来源">
        <Segmented<CreateConversationSource['type']>
          aria-disabled={submitting}
          aria-label="视频来源"
          block
          disabled={submitting}
          onChange={changeSourceType}
          options={[
            { label: '链接输入', value: 'url' },
            { label: '上传文件', value: 'file' },
          ]}
          value={value.source.type}
        />
      </Form.Item>

      {value.source.type === 'url' ? (
        <Form.Item className="create-composer__field" label="视频链接" required>
          <Input
            aria-label="视频链接"
            disabled={submitting}
            onChange={(event) => update({
              ...value,
              source: { type: 'url', url: event.target.value },
            })}
            placeholder="https://example.com/video.mp4"
            prefix={<LinkOutlined />}
            value={value.source.url}
          />
        </Form.Item>
      ) : (
        <Form.Item className="create-composer__field" label="视频文件" required>
          <Space className="create-composer__file-controls" orientation="vertical" size="small">
            <Upload
              accept="video/*"
              beforeUpload={(file) => {
                update({ ...value, source: { type: 'file', file } });
                return false;
              }}
              disabled={submitting}
              fileList={[]}
              maxCount={1}
              showUploadList={false}
            >
              <Button
                disabled={submitting}
                icon={<UploadOutlined />}
              >
                选择视频文件
              </Button>
            </Upload>
            {selectedFile ? (
              <Card
                extra={(
                  <Button
                    aria-label="移除视频文件"
                    disabled={submitting}
                    icon={<DeleteOutlined />}
                    onClick={() => update({ ...value, source: { type: 'file' } })}
                    type="text"
                  />
                )}
                size="small"
                title={selectedFile.name}
              />
            ) : null}
            {selectedFile && uploadProgress ? (
              <Progress
                aria-label="上传进度"
                percent={uploadProgress.percent}
                status={uploadProgress.status}
              />
            ) : null}
          </Space>
        </Form.Item>
      )}

      <Form.Item className="create-composer__field" label="台词处理">
        <div className="create-composer__transcript-controls">
          <Radio.Group
            aria-disabled={submitting}
            aria-label="台词处理"
            disabled={submitting}
            onChange={(event) => changeTranscriptMode(event.target.value as TranscriptHandling['mode'])}
            value={value.transcript.mode}
          >
            <Radio.Button value="keep">保持原文</Radio.Button>
            <Radio.Button value="rewrite">改写</Radio.Button>
            <Radio.Button value="translate">翻译</Radio.Button>
          </Radio.Group>
          {value.transcript.mode === 'translate' ? (
            <AutoComplete
              aria-label="目标语言"
              className="create-composer__language"
              disabled={submitting}
              filterOption={(inputValue, option) => (
                String(option?.value ?? '').toLocaleLowerCase().includes(inputValue.toLocaleLowerCase())
              )}
              onChange={(targetLanguage) => update({
                ...value,
                transcript: { mode: 'translate', targetLanguage },
              })}
              options={languageOptions.map((option) => ({
                label: option.label ?? option.value,
                value: option.value,
              }))}
              placeholder="选择或输入任意语言"
              value={value.transcript.targetLanguage}
            />
          ) : null}
        </div>
      </Form.Item>
    </div>
  );

  return (
    <section aria-label="创建会话" className="create-composer">
      <header className="create-composer__heading">
        <Typography.Title className="create-composer__title" level={3}>创建视频会话</Typography.Title>
        <Typography.Text type="secondary">
          选择一个视频来源，并告诉 Duet 这次创作的目标。
        </Typography.Text>
      </header>

      {error ? (
        <Alert
          className="create-composer__alert"
          closable={Boolean(onErrorDismiss)}
          title={error}
          onClose={onErrorDismiss}
          showIcon
          type="error"
        />
      ) : null}

      <Sender
        autoSize={{ minRows: 2, maxRows: 6 }}
        disabled={submitting}
        header={configuration}
        onChange={(note) => update({ ...value, note })}
        onSubmit={submit}
        placeholder="补充视频用途、受众或风格偏好…"
        rootClassName="create-composer__sender"
        suffix={(
          <Button
            aria-label={submitting ? '创建中' : '创建会话'}
            disabled={!canSubmit}
            icon={<SendOutlined />}
            loading={submitting}
            onClick={submit}
            type="primary"
          >
            {submitting ? '创建中' : '创建会话'}
          </Button>
        )}
        value={value.note}
      />
    </section>
  );
}
