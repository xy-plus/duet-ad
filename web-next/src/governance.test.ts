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
  it('provides one component facade and one token/provider module', () => {
    expect(existsSync(resolve(projectRoot, 'src/ui/antd.ts'))).toBe(true);
    expect(existsSync(resolve(projectRoot, 'src/ui/theme.ts'))).toBe(true);
  });

  it('lets XProvider own the single Ant theme boundary', () => {
    const facade = readFileSync(resolve(projectRoot, 'src/ui/antd.ts'), 'utf8');
    const provider = readFileSync(resolve(projectRoot, 'src/ui/theme.ts'), 'utf8');

    expect(facade).not.toMatch(/\bConfigProvider\b/);
    expect(provider).toMatch(/createElement\(\s*XProvider/);
    expect(provider).toContain('{ theme: appTheme, locale: zhCN }');
    expect(provider).not.toMatch(/\bConfigProvider\b/);
  });

  it.each(['antd', '@ant-design/x', '@ant-design/icons'])(
    'rejects direct %s imports from business source',
    async (moduleName) => {
      const messages = await lintBusinessSource(`import { Button } from '${moduleName}';\nvoid Button;`);

      expect(messages).toContainEqual(expect.stringContaining('src/ui/antd'));
    },
  );

  it('accepts UI imports through the facade', async () => {
    const messages = await lintBusinessSource(
      "import { Button } from '../ui/antd';\nexport const Example = () => <Button>Run</Button>;",
    );

    expect(messages).toEqual([]);
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

  it('rejects inline styles and color constants from business source', async () => {
    const inlineMessages = await lintBusinessSource(
      'export const Example = () => <div style={{ color: "var(--ant-color-text)" }} />;',
    );
    const colorMessages = await lintBusinessSource("export const brandColor = '#1677ff';");

    expect(inlineMessages).toContainEqual(expect.stringContaining('Inline styles'));
    expect(colorMessages).toContainEqual(expect.stringContaining('Color literals'));
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
    const messages = await lintStyles('.shell { color: var(--ant-color-text); }');

    expect(messages).toEqual([]);
  });
});
