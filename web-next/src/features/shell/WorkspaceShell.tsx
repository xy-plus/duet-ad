import { useEffect, useState, type ReactNode } from 'react';
import {
  Button,
  Drawer,
  Grid,
  Layout,
  MenuOutlined,
  Space,
  Typography,
} from '../../ui/antd';
import './shell.css';

export interface WorkspaceShellProps {
  children: ReactNode;
  headerExtra?: ReactNode;
  sidebar: ReactNode;
  subtitle?: ReactNode;
  title: ReactNode;
}

export function WorkspaceShell({
  children,
  headerExtra,
  sidebar,
  subtitle,
  title,
}: WorkspaceShellProps) {
  const screens = Grid.useBreakpoint();
  const desktop = Boolean(screens.md);
  const [navigationOpen, setNavigationOpen] = useState(false);

  useEffect(() => {
    if (desktop) setNavigationOpen(false);
  }, [desktop]);

  return (
    <Layout className="workspace-shell">
      {desktop ? (
        <Layout.Sider className="workspace-shell__sider" theme="light" width={272}>
          {sidebar}
        </Layout.Sider>
      ) : (
        <Drawer
          destroyOnHidden
          onClose={() => setNavigationOpen(false)}
          open={navigationOpen}
          placement="left"
          title="会话导航"
          size={272}
        >
          {sidebar}
        </Drawer>
      )}

      <Layout className="workspace-shell__workspace">
        <Layout.Header className="workspace-shell__header">
          <Space align="center" size="middle">
            {!desktop ? (
              <Button
                aria-label="打开会话导航"
                icon={<MenuOutlined />}
                onClick={() => setNavigationOpen(true)}
                type="text"
              />
            ) : null}
            <div className="workspace-shell__title">
              <Typography.Text strong>{title}</Typography.Text>
              {subtitle ? <Typography.Text type="secondary">{subtitle}</Typography.Text> : null}
            </div>
          </Space>
          {headerExtra}
        </Layout.Header>
        <Layout.Content className="workspace-shell__content">{children}</Layout.Content>
      </Layout>
    </Layout>
  );
}
