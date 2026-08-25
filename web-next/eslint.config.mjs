import js from '@eslint/js';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import globals from 'globals';
import tseslint from 'typescript-eslint';

const UI_FACADE_MESSAGE = 'Import UI components from src/ui/antd instead.';
const NATIVE_ELEMENT_MESSAGE = 'Use the UI facade instead of native visual elements.';
const COLOR_LITERAL = /(?:#|(?:rgb|hsl|hwb|lab|lch|oklab|oklch|color)\s*\()/iu;
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
    'no-direct-ui-imports': {
      meta: {
        type: 'problem',
        schema: [],
        messages: {
          forbidden: UI_FACADE_MESSAGE,
          reactFactory: 'Use named React hooks/types and JSX through the UI facade.',
          unsafeDynamic: 'Dynamic imports must use a statically known local path.',
        },
      },
      create(context) {
        const reportRestrictedSource = (node, value) => {
          if (typeof value === 'string'
              && /^(?:antd|@ant-design\/x|@ant-design\/icons)(?:\/|$)/u.test(value)) {
            context.report({ node, messageId: 'forbidden' });
          }
        };
        const staticImportSource = (source) => {
          if (source.type === 'Literal') {
            return typeof source.value === 'string' ? source.value : undefined;
          }
          if (source.type === 'TemplateLiteral' && source.expressions.length === 0) {
            return source.quasis[0]?.value.cooked ?? source.quasis[0]?.value.raw;
          }
          return undefined;
        };
        return {
          ImportDeclaration(node) {
            reportRestrictedSource(node.source, node.source.value);
            if (node.source.value === 'react') {
              for (const specifier of node.specifiers) {
                const importsElementFactory = specifier.type === 'ImportDefaultSpecifier'
                  || specifier.type === 'ImportNamespaceSpecifier'
                  || (specifier.type === 'ImportSpecifier'
                    && specifier.imported.type === 'Identifier'
                    && specifier.imported.name === 'createElement');
                if (importsElementFactory) {
                  context.report({ node: specifier, messageId: 'reactFactory' });
                }
              }
            }
          },
          ImportExpression(node) {
            const source = staticImportSource(node.source);
            if (source === undefined || (!source.startsWith('./') && !source.startsWith('../'))) {
              if (typeof source === 'string'
                  && /^(?:antd|@ant-design\/x|@ant-design\/icons)(?:\/|$)/u.test(source)) {
                reportRestrictedSource(node.source, source);
              } else {
                context.report({ node: node.source, messageId: 'unsafeDynamic' });
              }
            }
          },
        };
      },
    },
    'no-color-literals': {
      meta: {
        type: 'problem',
        schema: [],
        messages: { forbidden: 'Color literals belong in src/ui/theme.tsx Tokens.' },
      },
      create(context) {
        return {
          Literal(node) {
            if (typeof node.value === 'string' && COLOR_LITERAL.test(node.value)) {
              context.report({ node, messageId: 'forbidden' });
            }
          },
          TemplateElement(node) {
            if (COLOR_LITERAL.test(node.value.raw)) {
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
        const nativeVideoWrapper = context.filename.replaceAll('\\', '/').endsWith('/src/ui/video.tsx');
        return {
          JSXOpeningElement(node) {
            const name = node.name.type === 'JSXIdentifier' ? node.name.name : undefined;
            if (name
                && RESTRICTED_NATIVE_ELEMENTS.has(name)
                && !(nativeVideoWrapper && name === 'video')) {
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
    linterOptions: { noInlineConfig: true },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/**/*.test.{ts,tsx}', 'src/ui/antd.ts'],
    rules: {
      'governance/no-direct-ui-imports': 'error',
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
    ignores: ['src/**/*.test.{ts,tsx}'],
    rules: {
      'governance/no-inline-styles': 'error',
    },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/**/*.test.{ts,tsx}'],
    rules: {
      'governance/no-native-visual-elements': 'error',
    },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/**/*.test.{ts,tsx}', 'src/ui/theme.tsx'],
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
