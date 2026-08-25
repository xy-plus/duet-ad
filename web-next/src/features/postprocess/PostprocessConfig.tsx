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
  capabilities?: PostprocessOptions;
}

type PostprocessOptionKey = keyof PostprocessOptions;

const optionLabels: Record<PostprocessOptionKey, string> = {
  remove_subtitle: '移除文字/字幕',
  remove_brand: '移除常见 Logo/图标',
  optimize_image: '进行图片优化',
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
  capabilities = { remove_subtitle: true, remove_brand: true, optimize_image: true },
}: PostprocessConfigProps) {
  const [submitRequested, setSubmitRequested] = useState(false);
  const submitted = serverOptions !== undefined;
  const locked = submitting || submitRequested;
  const normalizedOptions: PostprocessOptions = {
    remove_subtitle: capabilities.remove_subtitle && options.remove_subtitle,
    remove_brand: capabilities.remove_brand && options.remove_brand,
    optimize_image: capabilities.optimize_image && options.optimize_image,
  };
  const selectedOptions = optionKeys.filter((key) => normalizedOptions[key]);

  useEffect(() => {
    if (!open) setSubmitRequested(false);
  }, [open]);

  const submit = () => {
    if (locked || selectedOptions.length === 0 || submitted) return;
    setSubmitRequested(true);
    onSubmit(normalizedOptions);
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
          服务端会按固定阶段顺序处理所选项目；提交后选项冻结，并在会话内后台处理。
        </Typography.Paragraph>
        <Checkbox.Group
          value={selectedOptions}
          disabled={locked}
          onChange={(values) => {
            const selected = new Set(values as PostprocessOptionKey[]);
            onOptionsChange({
              remove_subtitle: capabilities.remove_subtitle && selected.has('remove_subtitle'),
              remove_brand: capabilities.remove_brand && selected.has('remove_brand'),
              optimize_image: capabilities.optimize_image && selected.has('optimize_image'),
            });
          }}
        >
          <Space orientation="vertical">
            {optionKeys.filter((key) => capabilities[key]).map((key) => (
              <Checkbox key={key} value={key}>{optionLabels[key]}</Checkbox>
            ))}
          </Space>
        </Checkbox.Group>
      </Space>
    </Modal>
  );
}
