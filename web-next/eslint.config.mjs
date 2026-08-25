import js from '@eslint/js';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import globals from 'globals';
import tseslint from 'typescript-eslint';

const UI_FACADE_MESSAGE = 'Import UI components from src/ui/antd instead.';
const NATIVE_ELEMENT_MESSAGE = 'Use the UI facade instead of native visual elements.';
const COLOR_LITERAL = /(?:#[\da-f]{3,8}|(?:rgb|hsl|hwb|lab|lch|oklab|oklch|color)\s*\()/iu;
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
        const sourceCode = context.sourceCode;
        const unknown = Symbol('unknown static value');
        const findVariable = (node, name) => {
          let scope = sourceCode.getScope(node);
          while (scope) {
            const variable = scope.set.get(name);
            if (variable) return variable;
            scope = scope.upper;
          }
          return undefined;
        };
        const staticValue = (node, seen = new Set()) => {
          if (node.type === 'Literal') {
            return node.value === null
                || typeof node.value === 'string'
                || typeof node.value === 'number'
                || typeof node.value === 'boolean'
              ? node.value
              : unknown;
          }
          if (node.type === 'TemplateLiteral') {
            let value = node.quasis[0]?.value.cooked ?? node.quasis[0]?.value.raw ?? '';
            for (const [index, expression] of node.expressions.entries()) {
              const expressionValue = staticValue(expression, seen);
              if (expressionValue === unknown) return unknown;
              value += String(expressionValue);
              value += node.quasis[index + 1]?.value.cooked
                ?? node.quasis[index + 1]?.value.raw
                ?? '';
            }
            return value;
          }
          if (node.type === 'BinaryExpression' && node.operator === '+') {
            const left = staticValue(node.left, seen);
            const right = staticValue(node.right, seen);
            if (left === unknown || right === unknown) return unknown;
            if (typeof left === 'string' || typeof right === 'string') {
              return String(left) + String(right);
            }
            return typeof left === 'number' && typeof right === 'number' ? left + right : unknown;
          }
          if (node.type === 'TSAsExpression'
              || node.type === 'TSTypeAssertion'
              || node.type === 'TSNonNullExpression') {
            return staticValue(node.expression, seen);
          }
          if (node.type !== 'Identifier') return unknown;
          const variable = findVariable(node, node.name);
          if (!variable || seen.has(variable)) return unknown;
          const definition = variable.defs.find((candidate) => candidate.type === 'Variable');
          if (!definition
              || definition.parent?.kind !== 'const'
              || definition.node.id.type !== 'Identifier'
              || !definition.node.init) {
            return unknown;
          }
          const nextSeen = new Set(seen);
          nextSeen.add(variable);
          return staticValue(definition.node.init, nextSeen);
        };
        const reportStaticColor = (node) => {
          const value = staticValue(node);
          if (typeof value === 'string' && COLOR_LITERAL.test(value.trim())) {
            context.report({ node, messageId: 'forbidden' });
          }
        };
        return {
          Literal(node) {
            reportStaticColor(node);
          },
          TemplateElement(node) {
            if (COLOR_LITERAL.test(node.value.raw.trim())) {
              context.report({ node, messageId: 'forbidden' });
            }
          },
          TemplateLiteral(node) {
            reportStaticColor(node);
          },
          BinaryExpression(node) {
            if (node.operator === '+') reportStaticColor(node);
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
        const sourceCode = context.sourceCode;
        const nativeVideoWrapper = context.filename.replaceAll('\\', '/').endsWith('/src/ui/video.tsx');
        const staticPropertyName = (node) => {
          if (!node.computed && node.property.type === 'Identifier') return node.property.name;
          if (node.computed && node.property.type === 'Literal') return node.property.value;
          return undefined;
        };
        const findVariable = (node, name) => {
          let scope = sourceCode.getScope(node);
          while (scope) {
            const variable = scope.set.get(name);
            if (variable) return variable;
            scope = scope.upper;
          }
          return undefined;
        };
        const isReactReference = (node, seen = new Set()) => {
          if (node?.type !== 'Identifier') return false;
          const variable = findVariable(node, node.name);
          if (!variable || seen.has(variable)) return false;
          const importDefinition = variable.defs.find((definition) => (
            definition.type === 'ImportBinding'
            && (definition.node.type === 'ImportDefaultSpecifier'
              || definition.node.type === 'ImportNamespaceSpecifier')
            && definition.parent.source.value === 'react'
          ));
          if (importDefinition) return true;
          const variableDefinition = variable.defs.find((definition) => definition.type === 'Variable');
          if (!variableDefinition
              || variableDefinition.node.id.type !== 'Identifier'
              || !variableDefinition.node.init) {
            return false;
          }
          const nextSeen = new Set(seen);
          nextSeen.add(variable);
          return isReactReference(variableDefinition.node.init, nextSeen);
        };
        const destructuresCreateElement = (declarator, localName) => declarator.id.type === 'ObjectPattern'
          && isReactReference(declarator.init)
          && declarator.id.properties.some((property) => property.type === 'Property'
            && ((property.key.type === 'Identifier' && property.key.name === 'createElement')
              || (property.key.type === 'Literal' && property.key.value === 'createElement'))
            && ((property.value.type === 'Identifier' && property.value.name === localName)
              || (property.value.type === 'AssignmentPattern'
                && property.value.left.type === 'Identifier'
                && property.value.left.name === localName)));
        const isCreateElementReference = (node, seen = new Set()) => {
          if (node.type === 'MemberExpression') return staticPropertyName(node) === 'createElement';
          if (node.type !== 'Identifier') return false;
          if (node.name === 'createElement') return true;
          const variable = findVariable(node, node.name);
          if (!variable || seen.has(variable)) return false;
          const importAlias = variable.defs.some((definition) => definition.type === 'ImportBinding'
            && definition.node.type === 'ImportSpecifier'
            && definition.node.imported.type === 'Identifier'
            && definition.node.imported.name === 'createElement'
            && definition.parent.source.value === 'react');
          if (importAlias) return true;
          const variableDefinition = variable.defs.find((definition) => definition.type === 'Variable');
          if (!variableDefinition) return false;
          if (destructuresCreateElement(variableDefinition.node, node.name)) return true;
          if (variableDefinition.node.id.type !== 'Identifier' || !variableDefinition.node.init) {
            return false;
          }
          const nextSeen = new Set(seen);
          nextSeen.add(variable);
          return isCreateElementReference(variableDefinition.node.init, nextSeen);
        };
        return {
          JSXOpeningElement(node) {
            const name = node.name.type === 'JSXIdentifier' ? node.name.name : undefined;
            if (name
                && RESTRICTED_NATIVE_ELEMENTS.has(name)
                && !(nativeVideoWrapper && name === 'video')) {
              context.report({ node, messageId: 'forbidden' });
            }
          },
          CallExpression(node) {
            if (isCreateElementReference(node.callee)) {
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
