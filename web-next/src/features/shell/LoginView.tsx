import { useCallback } from 'react';
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  LockOutlined,
  Typography,
  UserOutlined,
} from '../../ui/antd';
import './shell.css';

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface LoginViewProps {
  error?: string;
  initialUsername?: string;
  onErrorDismiss?: () => void;
  onSubmit: (credentials: LoginCredentials) => void | Promise<void>;
  submitting: boolean;
}

export function LoginView({
  error,
  initialUsername,
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
          initialValues={{ username: initialUsername }}
          layout="vertical"
          onFinish={onSubmit}
          onValuesChange={dismissError}
          requiredMark={false}
        >
          <Form.Item
            label="用户名"
            name="username"
            rules={[{ required: true, message: '请输入用户名' }]}
          >
            <Input
              autoComplete="username"
              placeholder="请输入用户名"
              prefix={<UserOutlined />}
            />
          </Form.Item>
          <Form.Item
            label="密码"
            name="password"
            rules={[{ required: true, message: '请输入密码' }]}
          >
            <Input.Password
              autoComplete="current-password"
              placeholder="请输入密码"
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
