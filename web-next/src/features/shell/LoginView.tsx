import { useCallback } from 'react';
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  LockOutlined,
  Typography,
} from '../../ui/antd';
import './shell.css';

export interface LoginCredentials {
  token: string;
}

export interface LoginViewProps {
  error?: string;
  onErrorDismiss?: () => void;
  onSubmit: (credentials: LoginCredentials) => void | Promise<void>;
  submitting: boolean;
}

export function LoginView({
  error,
  onErrorDismiss,
  onSubmit,
  submitting,
}: LoginViewProps) {
  const dismissError = useCallback(() => {
    if (error) onErrorDismiss?.();
  }, [error, onErrorDismiss]);

  return (
    <main className="login-view">
      <Card className="login-view__card">
        <header className="login-view__heading">
          <Typography.Title className="login-view__title" level={2}>登录 Duet AI</Typography.Title>
          <Typography.Text type="secondary">进入你的视频创作工作台</Typography.Text>
        </header>

        {error ? (
          <Alert
            className="login-view__alert"
            closable={Boolean(onErrorDismiss)}
            title={error}
            onClose={onErrorDismiss}
            showIcon
            type="error"
          />
        ) : null}

        <Form<LoginCredentials>
          autoComplete="on"
          disabled={submitting}
          layout="vertical"
          onFinish={onSubmit}
          onValuesChange={dismissError}
          requiredMark={false}
        >
          <Form.Item
            label="访问口令"
            name="token"
            rules={[{ required: true, message: '请输入访问口令' }]}
          >
            <Input.Password
              autoComplete="current-password"
              placeholder="请输入访问口令"
              prefix={<LockOutlined />}
            />
          </Form.Item>
          <Button
            aria-label="登录"
            block
            htmlType="submit"
            loading={submitting}
            type="primary"
          >
            登录
          </Button>
        </Form>
      </Card>
    </main>
  );
}
