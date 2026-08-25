import '@testing-library/jest-dom/vitest';

class TestNotification {
  static permission: NotificationPermission = 'granted';
  static requestPermission = async () => 'granted' as NotificationPermission;
  onclick: ((event: Event) => void) | null = null;
  onshow: ((event: Event) => void) | null = null;
  onclose: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  close() {}
}

globalThis.Notification = TestNotification as unknown as typeof Notification;

const getComputedStyle = window.getComputedStyle.bind(window);
Object.defineProperty(window, 'getComputedStyle', {
  configurable: true,
  value: (element: Element) => {
    const style = getComputedStyle(element);
    const getPropertyValue = style.getPropertyValue.bind(style);
    style.getPropertyValue = (property: string) => {
      const value = getPropertyValue(property);
      if (/^(?:border-(?:bottom|top)-width|padding-(?:bottom|top))$/u.test(property)) {
        return Number.isFinite(Number.parseFloat(value)) ? value : '0px';
      }
      if (value) return value;
      if (property === 'box-sizing') return 'border-box';
      if (property === 'line-height') return '20px';
      return value;
    };
    return style;
  },
});

Object.defineProperty(window, 'matchMedia', {
  configurable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    addListener: () => undefined,
    removeListener: () => undefined,
    dispatchEvent: () => false,
  }),
});

class TestResizeObserver implements ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

globalThis.ResizeObserver = TestResizeObserver;
