import { describe, expect, it } from 'vitest';
import { postprocessCompletedFrames } from './model';
import type { ConversationDetail } from '../domain/types';

describe('postprocess live progress', () => {
  it('renders the receipt-projected 0 through 9 sequence before canonical publication', () => {
    const observed = Array.from({ length: 10 }, (_, completed) =>
      postprocessCompletedFrames({
        segments: [],
        keyframes: Array.from({ length: 9 }, (_value, index) => `${index + 1}.png`),
        postprocess: {
          status: 'running',
          frames: [],
          segments: [{
            index: 0,
            status: 'running',
            stage: 'seedream',
            completed_frames: completed,
            total_frames: 9,
            revision: 1,
            error: null,
          }],
        },
      } as unknown as ConversationDetail),
    );

    expect(observed).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
});
