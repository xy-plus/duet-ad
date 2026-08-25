import type { PropsWithChildren } from 'react';
import { QueryClientProvider, type QueryClient } from '@tanstack/react-query';
import { AntApp, XProvider, zhCN, type XProviderProps } from './antd';
import './baseline.css';

type AppTheme = NonNullable<XProviderProps['theme']>;

export const appTheme: AppTheme = {
  cssVar: { key: 'duet-next' },
  hashed: false,
  token: {
    colorPrimary: '#202123',
    colorInfo: '#202123',
    colorInfoBg: '#f3f3f4',
    colorInfoBorder: '#dedee1',
    colorInfoText: '#202123',
    colorSuccess: '#23855b',
    colorWarning: '#a66a13',
    colorError: '#c2413a',
    colorBgLayout: '#f7f7f8',
    colorBgContainer: '#ffffff',
    colorBgElevated: '#ffffff',
    colorFillTertiary: '#ececee',
    colorFillQuaternary: '#f3f3f4',
    colorText: '#202123',
    colorTextSecondary: '#66666b',
    colorBorder: '#dedee1',
    colorBorderSecondary: '#e9e9eb',
    borderRadius: 12,
    borderRadiusLG: 18,
    fontFamily:
      'Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
    boxShadowSecondary: '0 12px 36px rgba(20, 20, 22, 0.10)',
  },
  components: {
    Button: {
      borderRadius: 10,
      controlHeight: 38,
      primaryShadow: 'none',
    },
    Modal: { borderRadiusLG: 20 },
    Conversations: {
      creationBgColor: '#ffffff',
      creationBorderColor: '#dedee1',
      creationHoverColor: '#f3f3f4',
    },
    Layout: {
      bodyBg: '#ffffff',
      headerBg: '#ffffff',
      headerHeight: 64,
      headerPadding: '0 24px',
      lightSiderBg: '#f3f3f4',
    },
  },
};

interface AppThemeProviderProps extends PropsWithChildren {
  queryClient: QueryClient;
}

export function AppThemeProvider({ children, queryClient }: AppThemeProviderProps) {
  return (
    <QueryClientProvider client={queryClient}>
      <XProvider theme={appTheme} locale={zhCN}>
        <AntApp>{children}</AntApp>
      </XProvider>
    </QueryClientProvider>
  );
}
