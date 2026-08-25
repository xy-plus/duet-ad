import { useEffect, useState } from 'react';

import { Button, Checkbox, Modal, Space, Typography } from '../../ui/antd';
import type { PostprocessOptions } from './types';

interface PostprocessConfigProps {
  open: boolean;
  options: PostprocessOptions;
  serverOptions?: PostprocessOptions;
  submitting?: boolean;
  onOptionsChange: (options: PostprocessOptions) => void;
  onCancel: () => void;
  onSubmit: (options: PostprocessOptions) => void;
}

type PostprocessOptionKey = keyof PostprocessOptions;

const optionLabels: Record<PostprocessOptionKey, string> = {
  remove_subtitle: '移除字幕',
  remove_brand: '移除品牌标识',
};

const optionKeys = Object.keys(optionLabels) as PostprocessOptionKey[];

export function PostprocessConfig({
  open,
  options,
  serverOptions,
  submitting = false,
  onOptionsChange,
  onCancel,
  onSubmit,
}: PostprocessConfigProps) {
  const [submitRequested, setSubmitRequested] = useState(false);
  const submitted = serverOptions !== undefined;
  const locked = submitting || submitRequested;
  const selectedOptions = optionKeys.filter((key) => options[key]);

  useEffect(() => {
    if (!open) setSubmitRequested(false);
  }, [open]);

  const submit = () => {
    if (locked || selectedOptions.length === 0 || submitted) return;
    setSubmitRequested(true);
    onSubmit(options);
    onCancel();
  };

  return (
    <Modal
      title="关键帧后处理"
      open={open && !submitted}
      destroyOnHidden
      onCancel={onCancel}
      footer={(
        <Space>
          <Button onClick={onCancel}>取消</Button>
          <Button
            type="primary"
            aria-label={locked ? '提交中' : '开始后处理'}
            loading={locked}
            disabled={locked || selectedOptions.length === 0}
            onClick={submit}
          >
            {locked ? '提交中' : '开始后处理'}
          </Button>
        </Space>
      )}
    >
      <Space orientation="vertical" size="middle">
        <Typography.Paragraph type="secondary">
          选项只在提交前配置；提交后由服务端冻结，并在会话内后台处理。
        </Typography.Paragraph>
        <Checkbox.Group
          value={selectedOptions}
          disabled={locked}
          onChange={(values) => {
            const selected = new Set(values as PostprocessOptionKey[]);
            onOptionsChange({
              remove_subtitle: selected.has('remove_subtitle'),
              remove_brand: selected.has('remove_brand'),
            });
          }}
        >
          <Space orientation="vertical">
            {optionKeys.map((key) => (
              <Checkbox key={key} value={key}>{optionLabels[key]}</Checkbox>
            ))}
          </Space>
        </Checkbox.Group>
      </Space>
    </Modal>
  );
}
