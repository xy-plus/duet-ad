import type { ThemeConfig } from 'antd';

export const appTheme: ThemeConfig = {
  cssVar: { prefix: 'ant' },
  hashed: false,
  token: {
    colorPrimary: '#202123',
    colorInfo: '#202123',
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
    Drawer: {
      colorBgElevated: '#f3f3f4',
    },
    Modal: {
      borderRadiusLG: 20,
    },
    Segmented: {
      itemSelectedBg: '#ffffff',
    },
  },
};
