import { describe, expect, it } from 'vitest';
import {
  formatDialogueLines,
  normalizeDialogueLines,
  parseDialogueLines,
} from './dialogue';

describe('dialogue contract', () => {
  it('normalizes current and historical dialogue envelopes without inventing lines', () => {
    expect(normalizeDialogueLines({
      mode: 'auto',
      lines: [
        { start_s: 0, end_s: 1.5, text: ' first ' },
        { start: 1.5, end: 3, text: 'second' },
        { start_s: 3, end_s: 3, text: 'invalid' },
        null,
      ],
    })).toEqual([
      { start_s: 0, end_s: 1.5, text: 'first' },
      { start_s: 1.5, end_s: 3, text: 'second' },
    ]);

    expect(normalizeDialogueLines({ auto_lines: [{ start_s: 0, end_s: 1, text: 'auto' }] }))
      .toEqual([{ start_s: 0, end_s: 1, text: 'auto' }]);
    expect(normalizeDialogueLines({ unknown: [] })).toEqual([]);
  });

  it('round-trips the editable line format', () => {
    const text = formatDialogueLines([
      { start_s: 0, end_s: 1.5, text: 'first' },
      { start_s: 1.5, end_s: 3, text: 'second' },
    ]);

    expect(text).toBe('0 - 1.5 | first\n1.5 - 3 | second');
    expect(parseDialogueLines(text)).toEqual([
      { start_s: 0, end_s: 1.5, text: 'first' },
      { start_s: 1.5, end_s: 3, text: 'second' },
    ]);
  });

  it.each([
    ['0 to 1 | bad', '第 1 行格式应为'],
    ['1 - 1 | bad', '第 1 行结束时间必须晚于开始时间'],
  ])('rejects invalid editable text', (text, message) => {
    expect(() => parseDialogueLines(text)).toThrow(message);
  });
});
