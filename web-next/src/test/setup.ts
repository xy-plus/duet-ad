import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => cleanup());

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  }),
});

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

Object.defineProperty(window, 'ResizeObserver', { value: ResizeObserverStub });
Object.defineProperty(globalThis, 'ResizeObserver', { value: ResizeObserverStub });
Object.defineProperty(window, 'scrollTo', { value: () => undefined });
Object.defineProperty(HTMLElement.prototype, 'scrollTo', { value: () => undefined });

const nativeGetComputedStyle = window.getComputedStyle.bind(window);
Object.defineProperty(window, 'getComputedStyle', {
  value: (element: Element) => nativeGetComputedStyle(element),
});

class NotificationStub {
  static permission: NotificationPermission = 'granted';
  static requestPermission = async () => 'granted' as NotificationPermission;
  close() {}
}

Object.defineProperty(globalThis, 'Notification', { value: NotificationStub });
