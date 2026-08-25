import type { ReactNode } from 'react';
import {
  Button,
  Conversations,
  Divider,
  Flex,
  LogoutOutlined,
  PlusOutlined,
  ProductOutlined,
  Typography,
} from '../../ui/antd';
import './shell.css';

export interface ConversationNavigationItem {
  group?: string;
  id: string;
  title: string;
}

export interface ConversationSidebarProps<T extends ConversationNavigationItem = ConversationNavigationItem> {
  activeConversationId?: string;
  brand?: ReactNode;
  conversations: readonly T[];
  getNavigationStatus: (conversation: T) => ReactNode;
  onConversationSelect: (id: string) => void;
  onLogout: () => void;
  onNewConversation: () => void;
  userLabel?: ReactNode;
}

export function ConversationSidebar<T extends ConversationNavigationItem>({
  activeConversationId,
  brand = 'Duet AI',
  conversations,
  getNavigationStatus,
  onConversationSelect,
  onLogout,
  onNewConversation,
  userLabel,
}: ConversationSidebarProps<T>) {
  const items = conversations.map((conversation) => ({
    group: conversation.group,
    icon: <ProductOutlined />,
    key: conversation.id,
    label: (
      <div className="conversation-sidebar__item-label">
        <Typography.Text ellipsis>{conversation.title}</Typography.Text>
        <div className="conversation-sidebar__status">
          {getNavigationStatus(conversation)}
        </div>
      </div>
    ),
  }));
  const hasGroups = conversations.some((conversation) => Boolean(conversation.group));

  return (
    <aside aria-label="会话导航" className="conversation-sidebar">
      <Flex align="center" className="conversation-sidebar__brand" gap="small">
        <span className="conversation-sidebar__brand-mark"><ProductOutlined /></span>
        <span className="conversation-sidebar__brand-copy">
          <Typography.Text strong>{brand}</Typography.Text>
          <Typography.Text type="secondary">视频工作台</Typography.Text>
        </span>
      </Flex>

      <Conversations
        activeKey={activeConversationId}
        className="conversation-sidebar__conversations"
        creation={{
          align: 'center',
          icon: <PlusOutlined />,
          label: '新建会话',
          onClick: onNewConversation,
        }}
        groupable={hasGroups}
        items={items}
        onActiveChange={(key) => onConversationSelect(key)}
      />

      <footer className="conversation-sidebar__footer">
        <Divider />
        {userLabel ? <Typography.Text type="secondary">{userLabel}</Typography.Text> : null}
        <Button
          aria-label="退出登录"
          block
          icon={<LogoutOutlined />}
          onClick={onLogout}
          type="text"
        >
          退出登录
        </Button>
      </footer>
    </aside>
  );
}
