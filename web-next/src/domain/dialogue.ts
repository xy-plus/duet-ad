import type { DialogueLine } from './types';

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null;
}

export function normalizeDialogueLines(dialogue: unknown): DialogueLine[] {
  let raw = dialogue;
  if (!Array.isArray(raw) && isRecord(raw)) {
    raw = Array.isArray(raw.lines)
      ? raw.lines
      : (Array.isArray(raw.auto_lines) ? raw.auto_lines : []);
  }
  if (!Array.isArray(raw)) return [];

  return raw.flatMap((line): DialogueLine[] => {
    if (!isRecord(line)) return [];
    const start = Number(line.start_s ?? line.start);
    const end = Number(line.end_s ?? line.end);
    const text = String(line.text ?? '').trim();
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || start >= end || !text) {
      return [];
    }
    return [{ start_s: start, end_s: end, text }];
  });
}

export function formatDialogueLines(dialogue: unknown): string {
  return normalizeDialogueLines(dialogue)
    .map(({ start_s: start, end_s: end, text }) => `${start} - ${end} | ${text}`)
    .join('\n');
}

export function parseDialogueLines(text: unknown): DialogueLine[] {
  const rows = String(text ?? '')
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean);

  return rows.map((line, index) => {
    const match = line.match(/^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\|\s*(.+)$/u);
    if (!match) throw new Error(`第 ${index + 1} 行格式应为：开始 - 结束 | 台词`);
    const start = Number(match[1]);
    const end = Number(match[2]);
    const value = match[3].trim();
    if (start >= end) throw new Error(`第 ${index + 1} 行结束时间必须晚于开始时间`);
    if (!value) throw new Error(`第 ${index + 1} 行台词不能为空`);
    return { start_s: start, end_s: end, text: value };
  });
}
