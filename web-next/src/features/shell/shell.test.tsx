import { useState, type ReactNode } from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ConversationSidebar } from './ConversationSidebar';
import { LoginView, type LoginCredentials } from './LoginView';
import { WorkspaceShell } from './WorkspaceShell';

afterEach(cleanup);

describe('LoginView', () => {
  it('lets a user recover from an expired-login error and retry', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn<(credentials: LoginCredentials) => void>();

    function Harness() {
      const [error, setError] = useState<string | undefined>('登录状态已过期，请重新登录');

      return (
        <LoginView
          error={error}
          onErrorDismiss={() => setError(undefined)}
          onSubmit={onSubmit}
          submitting={false}
        />
      );
    }

    render(<Harness />);

    expect(screen.getByRole('alert')).toHaveTextContent('登录状态已过期');
    await user.type(screen.getByLabelText('用户名'), 'alice');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    await user.type(screen.getByLabelText('密码'), 'correct horse battery staple');
    await user.click(screen.getByRole('button', { name: '登录' }));

    expect(onSubmit).toHaveBeenCalledWith({
      username: 'alice',
      password: 'correct horse battery staple',
    });
  });
});

describe('WorkspaceShell', () => {
  it('uses a drawer for navigation on mobile screens', async () => {
    const user = userEvent.setup();

    render(
      <WorkspaceShell
        sidebar={<div>移动会话导航</div>}
        title="当前会话"
      >
        <div>工作区内容</div>
      </WorkspaceShell>,
    );

    expect(screen.queryByText('移动会话导航')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '打开会话导航' }));

    expect(screen.getByRole('dialog', { name: '会话导航' })).toBeInTheDocument();
    expect(screen.getByText('移动会话导航')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Close' }));
    expect(screen.queryByRole('dialog', { name: '会话导航' })).not.toBeInTheDocument();
  });
});

describe('ConversationSidebar', () => {
  it('renders navigation status supplied by the authoritative caller', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn<(id: string) => void>();
    const onNew = vi.fn();
    const onLogout = vi.fn();
    const statuses: Record<string, ReactNode> = {
      alpha: <span>服务端：生成中 37%</span>,
      beta: <span>服务端：等待确认</span>,
    };

    render(
      <ConversationSidebar
        activeConversationId="alpha"
        conversations={[
          { id: 'alpha', title: '春日广告' },
          { id: 'beta', title: '夜市长视频' },
        ]}
        getNavigationStatus={(conversation) => statuses[conversation.id]}
        onConversationSelect={onSelect}
        onLogout={onLogout}
        onNewConversation={onNew}
      />,
    );

    expect(screen.getByText('服务端：生成中 37%')).toBeInTheDocument();
    await user.click(screen.getByText('夜市长视频'));
    expect(onSelect).toHaveBeenCalledWith('beta');
    await user.click(screen.getByRole('button', { name: /新建会话/ }));
    await user.click(screen.getByRole('button', { name: '退出登录' }));
    expect(onNew).toHaveBeenCalledOnce();
    expect(onLogout).toHaveBeenCalledOnce();
  });
});
