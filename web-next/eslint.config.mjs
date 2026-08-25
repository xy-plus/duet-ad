import js from '@eslint/js';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import globals from 'globals';
import tseslint from 'typescript-eslint';

const UI_FACADE_MESSAGE = 'Import UI components from src/ui/antd instead.';
const NATIVE_ELEMENT_MESSAGE = 'Use the UI facade instead of native visual elements.';
const COLOR_LITERAL = /^(?:#[\da-f]{3,8}|(?:rgb|hsl|hwb|lab|lch|oklab|oklch|color)\s*\()/iu;
const RESTRICTED_NATIVE_ELEMENTS = new Set([
  'audio',
  'button',
  'canvas',
  'fieldset',
  'form',
  'img',
  'input',
  'meter',
  'option',
  'picture',
  'progress',
  'select',
  'svg',
  'textarea',
  'video',
]);

const governancePlugin = {
  rules: {
    'no-color-literals': {
      meta: {
        type: 'problem',
        schema: [],
        messages: { forbidden: 'Color literals belong in src/ui/theme.ts Tokens.' },
      },
      create(context) {
        return {
          Literal(node) {
            if (typeof node.value === 'string' && COLOR_LITERAL.test(node.value.trim())) {
              context.report({ node, messageId: 'forbidden' });
            }
          },
          TemplateElement(node) {
            if (COLOR_LITERAL.test(node.value.raw.trim())) {
              context.report({ node, messageId: 'forbidden' });
            }
          },
        };
      },
    },
    'no-inline-styles': {
      meta: {
        type: 'problem',
        schema: [],
        messages: { forbidden: 'Inline styles are forbidden; use token-driven stylesheets.' },
      },
      create(context) {
        return {
          JSXAttribute(node) {
            if (node.name.type === 'JSXIdentifier' && node.name.name === 'style') {
              context.report({ node, messageId: 'forbidden' });
            }
          },
        };
      },
    },
    'no-native-visual-elements': {
      meta: {
        type: 'problem',
        schema: [],
        messages: { forbidden: NATIVE_ELEMENT_MESSAGE },
      },
      create(context) {
        return {
          JSXOpeningElement(node) {
            const name = node.name.type === 'JSXIdentifier' ? node.name.name : undefined;
            if (name && RESTRICTED_NATIVE_ELEMENTS.has(name)) {
              context.report({ node, messageId: 'forbidden' });
            }
          },
        };
      },
    },
  },
};

export default tseslint.config(
  { ignores: ['dist/**', 'coverage/**', 'node_modules/**', 'test-results/**'] },
  js.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    extends: [tseslint.configs.recommended],
    languageOptions: {
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    plugins: {
      'jsx-a11y': jsxA11y,
      governance: governancePlugin,
    },
    rules: {
      ...jsxA11y.flatConfigs.recommended.rules,
    },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/**/*.test.{ts,tsx}', 'src/ui/antd.ts'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          paths: [
            { name: 'antd', message: UI_FACADE_MESSAGE },
            { name: '@ant-design/x', message: UI_FACADE_MESSAGE },
            { name: '@ant-design/icons', message: UI_FACADE_MESSAGE },
          ],
          patterns: [
            {
              group: ['antd/*', '@ant-design/x/*', '@ant-design/icons/*'],
              message: UI_FACADE_MESSAGE,
            },
            {
              group: ['*.svg', '**/*.svg'],
              message: 'Hand-written SVG assets are forbidden; use the governed icon facade.',
            },
          ],
        },
      ],
    },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/**/*.test.{ts,tsx}', 'src/ui/video.tsx'],
    rules: {
      'governance/no-inline-styles': 'error',
      'governance/no-native-visual-elements': 'error',
    },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/**/*.test.{ts,tsx}', 'src/ui/**'],
    rules: {
      'governance/no-color-literals': 'error',
    },
  },
  {
    files: ['src/ui/video.tsx'],
    rules: {
      'jsx-a11y/media-has-caption': 'off',
    },
  },
  {
    files: ['vite.config.ts', 'playwright.config.ts', 'eslint.config.mjs', 'stylelint.config.mjs'],
    languageOptions: { globals: globals.node },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    languageOptions: { globals: globals.browser },
  },
);
