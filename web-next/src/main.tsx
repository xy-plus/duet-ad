import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { AppThemeProvider } from './ui/theme';

const root = document.getElementById('root');

if (!root) {
  throw new Error('Missing #root application mount point.');
}

createRoot(root).render(
  <StrictMode>
    <AppThemeProvider>
      <App />
    </AppThemeProvider>
  </StrictMode>,
);
