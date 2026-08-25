import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { createApiRuntime } from './state/runtime';
import { AppThemeProvider } from './ui/theme';

const root = document.getElementById('root');

if (!root) {
  throw new Error('Missing root application mount point.');
}

const runtime = createApiRuntime();

createRoot(root).render(
  <StrictMode>
    <AppThemeProvider queryClient={runtime.queryClient}>
      <App apiClient={runtime.apiClient} />
    </AppThemeProvider>
  </StrictMode>,
);
