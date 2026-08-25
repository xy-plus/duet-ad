import { Card, Descriptions, Form, Input, Radio, Select, Typography } from '../../ui/antd';
import './generation.css';
import type {
  DialogueMode,
  FitMode,
  GenerationEvidence,
  GenerationSettingsValue,
  VideoKind,
} from './types';

const dialogueLabels: Record<DialogueMode, string> = {
  auto: '自动台词',
  edit: '编辑识别台词',
  custom: '自定义台词',
  none: '无台词',
};

const fitLabels: Record<FitMode, string> = {
  none: '无需适配',
  crop: '裁切铺满',
  pad: '完整留白',
};

const fitOptions = (Object.entries(fitLabels) as Array<[FitMode, string]>).map(([value, label]) => ({
  value,
  label,
}));

interface GenerationSettingsProps {
  videoKind: VideoKind;
  initialValues: GenerationSettingsValue;
  value?: GenerationSettingsValue;
  generation?: GenerationEvidence;
  disabled?: boolean;
  onChange: (value: GenerationSettingsValue) => void;
}

function frozenItems(values: GenerationSettingsValue) {
  const items = [
    { key: 'dialogue', label: '台词模式', children: dialogueLabels[values.dialogueMode] },
    { key: 'aspect', label: '画幅', children: values.aspectRatio },
    { key: 'resolution', label: '清晰度', children: values.resolution },
    { key: 'fit', label: '画面适配', children: fitLabels[values.fitMode] },
  ];

  if ((values.dialogueMode === 'edit' || values.dialogueMode === 'custom') && values.dialogueText) {
    items.push({ key: 'dialogue-text', label: '台词内容', children: values.dialogueText });
  }

  return items;
}

export function GenerationSettings({
  videoKind,
  initialValues,
  value,
  generation,
  disabled = false,
  onChange,
}: GenerationSettingsProps) {
  if (generation) {
    return (
      <Card title="已冻结生成参数">
        <Descriptions
          bordered
          column={{ xs: 1, sm: 2 }}
          items={frozenItems(generation.parameters)}
          size="small"
        />
      </Card>
    );
  }

  const current = value ?? initialValues;
  const emit = <Key extends keyof GenerationSettingsValue>(
    key: Key,
    nextValue: GenerationSettingsValue[Key],
  ) => onChange({ ...current, [key]: nextValue });
  const dialogueModes: DialogueMode[] = videoKind === 'long'
    ? ['auto', 'none']
    : ['auto', 'edit', 'custom', 'none'];
  const dialogueEditable = current.dialogueMode === 'edit' || current.dialogueMode === 'custom';

  return (
    <Card title="生成设置">
      <Form layout="vertical" disabled={disabled}>
        <div className="generation-settings-grid">
          <Form.Item label="台词模式" className="generation-settings-wide">
            <Radio.Group
              aria-label="台词模式"
              name="generation-dialogue-mode"
              value={current.dialogueMode}
              onChange={(event) => emit('dialogueMode', event.target.value as DialogueMode)}
            >
              {dialogueModes.map((mode) => (
                <Radio.Button key={mode} value={mode} aria-label={dialogueLabels[mode]}>
                  {dialogueLabels[mode]}
                </Radio.Button>
              ))}
            </Radio.Group>
          </Form.Item>

          {dialogueEditable && videoKind === 'short' && (
            <Form.Item
              label={current.dialogueMode === 'edit' ? '识别台词内容' : '自定义台词内容'}
              className="generation-settings-wide"
            >
              <Input.TextArea
                aria-label={current.dialogueMode === 'edit' ? '识别台词内容' : '自定义台词内容'}
                rows={3}
                value={current.dialogueText}
                onChange={(event) => emit('dialogueText', event.target.value)}
              />
            </Form.Item>
          )}

          <Form.Item label="画幅">
            <Radio.Group
              aria-label="画幅"
              name="generation-aspect-ratio"
              value={current.aspectRatio}
              onChange={(event) => emit('aspectRatio', event.target.value as GenerationSettingsValue['aspectRatio'])}
            >
              <Radio.Button value="16:9" aria-label="画幅 16:9">16:9</Radio.Button>
              <Radio.Button value="9:16" aria-label="画幅 9:16">9:16</Radio.Button>
            </Radio.Group>
          </Form.Item>

          <Form.Item label="清晰度">
            <Select
              aria-label="清晰度"
              value={current.resolution}
              options={[
                { value: '480p', label: '480p' },
                { value: '768p', label: '768p' },
              ]}
              onChange={(nextValue) => emit('resolution', nextValue)}
            />
          </Form.Item>

          <Form.Item label="画面适配">
            <Select
              aria-label="画面适配"
              value={current.fitMode}
              options={fitOptions}
              onChange={(nextValue) => emit('fitMode', nextValue)}
            />
          </Form.Item>
        </div>
        <Typography.Text type="secondary">
          推荐值由服务端给出；生成开始后以服务端冻结参数为准。
        </Typography.Text>
      </Form>
    </Card>
  );
}
