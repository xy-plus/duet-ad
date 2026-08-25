import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { ESLint } from 'eslint';
import stylelint from 'stylelint';
import { describe, expect, it } from 'vitest';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const eslint = new ESLint({ cwd: projectRoot });

async function lintBusinessSource(code: string, relativePath = 'src/features/example.tsx') {
  const [result] = await eslint.lintText(code, {
    filePath: resolve(projectRoot, relativePath),
  });

  return result.messages.map(({ message }) => message);
}

async function lintStyles(code: string) {
  const result = await stylelint.lint({
    code,
    codeFilename: resolve(projectRoot, 'src/features/example.css'),
  });

  return result.results.flatMap(({ warnings }) => warnings.map(({ text }) => text));
}

describe('UI foundation contract', () => {
  it('ships scoped instructions for future frontend agents', () => {
    const instructionsPath = resolve(projectRoot, 'AGENTS.md');

    expect(existsSync(instructionsPath)).toBe(true);
    const instructions = readFileSync(instructionsPath, 'utf8');
    expect(instructions).toContain('src/ui/antd');
    expect(instructions).toContain('原生 `<video controls>`');
    expect(instructions).toContain('npm run check');
  });

  it('provides one component facade and one token/provider module', () => {
    expect(existsSync(resolve(projectRoot, 'src/ui/antd.ts'))).toBe(true);
    expect(existsSync(resolve(projectRoot, 'src/ui/theme.tsx'))).toBe(true);
  });

  it('locks the approved AI workspace geometry and visual regression states', () => {
    const shellStyles = readFileSync(resolve(projectRoot, 'src/features/shell/shell.css'), 'utf8');
    const appStyles = readFileSync(resolve(projectRoot, 'src/app/app.css'), 'utf8');
    const promptSource = readFileSync(resolve(projectRoot, 'src/features/media/PromptEditor.tsx'), 'utf8');
    const shellSource = readFileSync(resolve(projectRoot, 'src/features/shell/WorkspaceShell.tsx'), 'utf8');
    const browserContract = readFileSync(resolve(projectRoot, 'tests/app.spec.ts'), 'utf8');

    expect(shellStyles).toMatch(/\.workspace-shell\s*\{[^}]*height:\s*100dvh;[^}]*overflow:\s*hidden;/su);
    expect(shellSource).toContain('width={272}');
    expect(shellSource).toContain('size={272}');
    expect(appStyles).toContain('width: min(100%, 56.25rem);');
    expect(appStyles).toContain('grid-template-columns: minmax(0, 1fr) minmax(15rem, 18rem);');
    expect(promptSource).toContain('className="prompt-editor-surface"');
    expect(promptSource).not.toContain('<Card');
    expect(readFileSync(resolve(projectRoot, 'src/ui/theme.tsx'), 'utf8'))
      .toContain('Conversations:');
    expect(browserContract).toContain("'desktop-workspace.png'");
    expect(browserContract).toContain("'desktop-generation.png'");
    expect(browserContract).toContain("'mobile-drawer.png'");
    expect(browserContract).toContain("'mobile-detail.png'");
  });

  it('lets XProvider own the single Ant theme boundary', () => {
    const facade = readFileSync(resolve(projectRoot, 'src/ui/antd.ts'), 'utf8');
    const provider = readFileSync(resolve(projectRoot, 'src/ui/theme.tsx'), 'utf8');

    expect(facade).not.toMatch(/\bConfigProvider\b/);
    expect(provider).toContain('<XProvider theme={appTheme} locale={zhCN}>');
    expect(provider).not.toContain('createElement');
    expect(provider).not.toMatch(/\bConfigProvider\b/);
  });

  it.each(['antd', '@ant-design/x', '@ant-design/icons'])(
    'rejects direct %s imports from business source',
    async (moduleName) => {
      const messages = await lintBusinessSource(`import { Button } from '${moduleName}';\nvoid Button;`);

      expect(messages).toContainEqual(expect.stringContaining('src/ui/antd'));
    },
  );

  it.each([
    'antd/es/button',
    '@ant-design/x/es/bubble',
    '@ant-design/icons/es/icons/PlusOutlined',
  ])('rejects dynamic imports from %s subpaths', async (moduleName) => {
    const messages = await lintBusinessSource(
      `export const load = () => import('${moduleName}');`,
    );

    expect(messages).toContainEqual(expect.stringContaining('src/ui/antd'));
  });

  it.each([
    'antd/es/button',
    '@ant-design/x/es/bubble',
    '@ant-design/icons/es/icons/PlusOutlined',
  ])('rejects template-literal dynamic imports from %s', async (moduleName) => {
    const messages = await lintBusinessSource(
      `export const load = () => import(\`${moduleName}\`);`,
    );

    expect(messages).toContainEqual(expect.stringContaining('src/ui/antd'));
  });

  it.each([
    "const target = '../local'; export const load = () => import(target);",
    "const part = 'local'; export const load = () => import('../' + part);",
  ])('rejects dynamic imports that are not statically provable local paths', async (source) => {
    const messages = await lintBusinessSource(source);

    expect(messages).toContainEqual(expect.stringContaining('statically known local path'));
  });

  it('accepts UI imports through the facade', async () => {
    const messages = await lintBusinessSource(
      "import { Button } from '../ui/antd';\nexport const Example = () => <Button>Run</Button>;",
    );

    expect(messages).toEqual([]);
  });

  it.each([
    "import React from 'react'; export const fragment = React.Fragment;",
    "import * as ReactNamespace from 'react'; export const fragment = ReactNamespace.Fragment;",
    "import { createElement as h } from 'react'; export { h };",
  ])('rejects React import forms that reopen the native element escape hatch', async (source) => {
    const messages = await lintBusinessSource(source);

    expect(messages).toContainEqual(expect.stringContaining('named React hooks/types'));
  });

  it('rejects React element factory re-exports', async () => {
    const messages = await lintBusinessSource(
      "export { createElement as h } from 'react';",
    );

    expect(messages).toContainEqual(expect.stringContaining('named React hooks/types'));
  });

  it.each([
    "import { jsx as h } from 'react/jsx-runtime'; export { h };",
    "import { jsxDEV as h } from 'react/jsx-dev-runtime'; export { h };",
    "export { jsx as h } from 'react/jsx-runtime';",
    "export { jsxDEV as h } from 'react/jsx-dev-runtime';",
  ])('rejects direct jsx runtime imports and re-exports', async (source) => {
    const messages = await lintBusinessSource(source);

    expect(messages).toContainEqual(expect.stringContaining('named React hooks/types'));
  });

  it.each(['button', 'form', 'input', 'select', 'textarea', 'img', 'svg', 'video'])(
    'rejects native <%s> from business source',
    async (element) => {
      const messages = await lintBusinessSource(
        `export const Example = () => <${element} aria-label="example" />;`,
      );

      expect(messages).toContainEqual(expect.stringContaining('UI facade'));
    },
  );

  it('allows the single native video wrapper', async () => {
    const messages = await lintBusinessSource(
      'export const NativeVideo = () => <video />;',
      'src/ui/video.tsx',
    );

    expect(messages).toEqual([]);
  });

  it('still rejects other native controls inside the video wrapper module', async () => {
    const messages = await lintBusinessSource(
      'export const BadControl = () => <button>bad</button>;',
      'src/ui/video.tsx',
    );

    expect(messages).toContainEqual(expect.stringContaining('UI facade'));
  });

  it('does not exempt inline styles in the native video wrapper', async () => {
    const messages = await lintBusinessSource(
      'export const NativeVideo = () => <video style={{ width: 10 }} />;',
      'src/ui/video.tsx',
    );

    expect(messages).toContainEqual(expect.stringContaining('Inline styles'));
  });

  it.each([
    "import { createElement } from 'react'; export const Bad = () => createElement('button');",
    "import React from 'react'; export const Bad = () => React.createElement('input');",
    "import { createElement } from 'react'; const tag = 'button'; export const Bad = () => createElement(tag);",
    "import React from 'react'; export const Bad = () => React['createElement']('button');",
    "import { createElement } from 'react'; const h = createElement; export const Bad = () => h('button');",
    "import { createElement as h } from 'react'; export const Bad = () => h('button');",
    "import React from 'react'; const { createElement: h } = React; export const Bad = () => h('button');",
  ])('rejects restricted native elements constructed through createElement', async (source) => {
    const messages = await lintBusinessSource(source);

    expect(messages).toContainEqual(expect.stringContaining('UI facade'));
  });

  it('does not confuse unrelated local helpers with createElement aliases', async () => {
    const messages = await lintBusinessSource(
      "const h = (value: string) => value; export const safe = () => h('button');",
    );

    expect(messages).toEqual([]);
  });

  it('allows an unrelated local function named createElement', async () => {
    const messages = await lintBusinessSource(
      "const createElement = (value: string) => value; export const safe = createElement('safe');",
    );

    expect(messages).toEqual([]);
  });

  it.each([
    "export const bad = document.createElement('button');",
    "const dom = window.document; export const bad = dom.createElement('button');",
    "const dom = globalThis.document; export const bad = dom.createElement('button');",
    "const dom = document; export const bad = dom.createElement('button');",
  ])('rejects DOM element factory access from business source', async (source) => {
    const messages = await lintBusinessSource(source);

    expect(messages).toContainEqual(expect.stringContaining('DOM element factory'));
  });

  it.each([
    "const browser = window; const dom = browser.document; export const bad = dom.createElement('button');",
    "const browser = globalThis; const dom = browser.document; export const bad = dom.createElement('button');",
  ])('rejects browser root aliases before they can expose a DOM factory', async (source) => {
    const messages = await lintBusinessSource(source);

    expect(messages).toContainEqual(expect.stringContaining('DOM element factory'));
  });

  it('reports one diagnostic for direct document factory access', async () => {
    const messages = await lintBusinessSource(
      "export const bad = document.createElement('button');",
    );

    expect(messages.filter((message) => message.includes('DOM element factory'))).toHaveLength(1);
  });

  it('leaves browser globals available to the governed test setup only', async () => {
    const messages = await lintBusinessSource(
      'export const browser = window; export const runtime = globalThis;',
      'src/test/governance-helper.ts',
    );

    expect(messages).toEqual([]);
  });

  it('allows only the root lookup document access in the production entrypoint', async () => {
    const messages = await lintBusinessSource(
      "export const root = document.getElementById('root');",
      'src/main.tsx',
    );

    expect(messages).toEqual([]);
  });

  it('allows color literals only in the theme module', async () => {
    const facadeMessages = await lintBusinessSource(
      "export const color = '#fff';",
      'src/ui/video.tsx',
    );
    const themeMessages = await lintBusinessSource(
      "export const color = '#fff';",
      'src/ui/theme.tsx',
    );

    expect(facadeMessages).toContainEqual(expect.stringContaining('Color literals'));
    expect(themeMessages).toEqual([]);
  });

  it('rejects inline ESLint disable directives', async () => {
    const messages = await lintBusinessSource(
      '/* eslint-disable governance/no-inline-styles */\nexport const Example = () => <div />;',
    );

    expect(messages).toContainEqual(expect.stringContaining('eslint-disable'));
  });

  it('rejects inline styles and color constants from business source', async () => {
    const inlineMessages = await lintBusinessSource(
      'export const Example = () => <div style={{ color: "var(--ant-color-text)" }} />;',
    );
    const colorMessages = await lintBusinessSource("export const brandColor = '#1677ff';");

    expect(inlineMessages).toContainEqual(expect.stringContaining('Inline styles'));
    expect(colorMessages).toContainEqual(expect.stringContaining('Color literals'));
  });

  it('rejects color literals embedded inside longer strings', async () => {
    const messages = await lintBusinessSource("export const border = '1px solid #fff';");

    expect(messages).toContainEqual(expect.stringContaining('Color literals'));
  });

  it.each([
    "export const color = '#' + 'fff';",
    "const hash = '#'; const tail = 'ff' + 'f'; export const color = hash + tail;",
    "export const color = `${'#'}${'fff'}`;",
  ])('rejects statically evaluable color expressions', async (source) => {
    const messages = await lintBusinessSource(source);

    expect(messages).toContainEqual(expect.stringContaining('Color literals'));
  });

  it('rejects hash-based colors assembled by otherwise static helper calls', async () => {
    const messages = await lintBusinessSource(
      "export const color = ['#', 'fff'].join('');",
    );

    expect(messages).toContainEqual(expect.stringContaining('Color literals'));
  });

  it('reports one diagnostic for one statically concatenated color expression', async () => {
    const messages = await lintBusinessSource("export const color = '#fff' + '00';");

    expect(messages.filter((message) => message.includes('Color literals'))).toHaveLength(1);
  });

  it('accepts semantic structure without visual escape hatches', async () => {
    const messages = await lintBusinessSource(
      'export const Example = () => <main><section><div>Content</div></section></main>;',
    );

    expect(messages).toEqual([]);
  });

  it('contains no hand-written SVG assets', () => {
    const sourceFiles = readdirSync(resolve(projectRoot, 'src'), { recursive: true });

    expect(sourceFiles.filter((file) => String(file).endsWith('.svg'))).toEqual([]);
  });

  it('rejects Ant internals, literal colors, and important declarations in CSS', async () => {
    const messages = await lintStyles('.ant-btn { color: #fff !important; }');

    expect(messages).toContainEqual(expect.stringContaining('.ant-'));
    expect(messages).toContainEqual(expect.stringContaining('hex'));
    expect(messages).toContainEqual(expect.stringContaining('important'));
  });

  it('accepts token-driven CSS', async () => {
    const messages = await lintStyles('.shell { display: grid; gap: var(--ant-margin-sm); }');

    expect(messages).toEqual([]);
  });

  it('rejects hand-written visual surfaces and secondary visual variables in feature CSS', async () => {
    const messages = await lintStyles(`
      .surface {
        --surface-color: var(--ant-color-bg-container);
        background: var(--ant-color-bg-container);
        border: 1px solid var(--ant-color-border);
        border-radius: var(--ant-border-radius-lg);
        box-shadow: var(--ant-box-shadow-tertiary);
      }
    `);

    expect(messages.filter((message) => message.includes('geometry-only'))).toHaveLength(5);
  });

  it('rejects workspace geometry that can recreate a wide or document-scrolling page', async () => {
    const messages = await lintStyles('.page { width: 72rem; min-height: 200dvh; }');

    expect(messages.filter((message) => message.includes('Token-driven'))).toHaveLength(2);
  });

  it('rejects raw spacing values from feature CSS', async () => {
    const messages = await lintStyles('.feature { padding: 13px; gap: 1rem; }');

    expect(messages).toContainEqual(expect.stringContaining('Token'));
  });

  it.each([
    ['border-radius', '12px'],
    ['border-top-left-radius', '12px'],
    ['box-shadow', '0 1px 2px #000'],
    ['text-shadow', '0 1px var(--ant-color-text)'],
    ['font', '14px Arial'],
    ['font-size', '14px'],
    ['font-family', 'Arial'],
  ])('rejects raw %s values from feature CSS', async (property, value) => {
    const messages = await lintStyles(`.feature { ${property}: ${value}; }`);

    expect(messages).toContainEqual(expect.stringContaining('Token'));
  });

  it.each(['filter', '--feature-shadow'])(
    'rejects drop-shadow functions in %s declarations',
    async (property) => {
      const messages = await lintStyles(
        `.feature { ${property}: drop-shadow(0 1px 2px var(--ant-color-text)); }`,
      );

      expect(messages).toContainEqual(expect.stringContaining('drop-shadow'));
    },
  );

  it('rejects case-variant drop-shadow functions', async () => {
    const messages = await lintStyles(
      '.feature { --feature-shadow: DrOp-ShAdOw(0 1px 2px var(--ant-color-text)); }',
    );

    expect(messages).toContainEqual(expect.stringContaining('DrOp-ShAdOw'));
  });

  it('rejects inline Stylelint disable directives', async () => {
    const messages = await lintStyles(
      '/* stylelint-disable declaration-property-value-disallowed-list */ .feature { padding: 13px; }',
    );

    expect(messages).toContainEqual(expect.stringContaining('stylelint-disable'));
  });

  it('does not expose unused facade components', () => {
    const facade = readFileSync(resolve(projectRoot, 'src/ui/antd.ts'), 'utf8');
    const exported = [...facade.matchAll(/export(?:\s+type)?\s*\{([\s\S]*?)\}\s*from/gu)]
      .flatMap(([, block]) => block.split(','))
      .map((entry) => entry.trim().match(/(?:\bas\s+)?([A-Za-z_$][\w$]*)$/u)?.[1])
      .filter((name): name is string => Boolean(name));
    const sourceFiles = readdirSync(resolve(projectRoot, 'src'), { recursive: true })
      .map(String)
      .filter((file) => /\.[cm]?[jt]sx?$/u.test(file) && file !== 'ui/antd.ts');
    const imported = sourceFiles
      .map((file) => readFileSync(resolve(projectRoot, 'src', file), 'utf8'))
      .flatMap((source) => [...source.matchAll(
        /import(?:\s+type)?\s*\{([^}]*)\}\s*from\s*['"](?:\.{1,2}\/)+(?:ui\/)?antd['"]/gu,
      )])
      .flatMap(([, block]) => block.split(','))
      .map((entry) => entry.trim().replace(/^type\s+/u, '').match(/^([A-Za-z_$][\w$]*)/u)?.[1])
      .filter((name): name is string => Boolean(name));
    const unused = exported.filter((name) => !imported.includes(name));

    expect(unused).toEqual([]);
  });
});
