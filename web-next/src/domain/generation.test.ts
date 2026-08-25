import { describe, expect, it } from 'vitest';
import {
  buildLongFailedRetryPayload,
  buildResumePayload,
  buildStitchRetryPayload,
  buildSubmitPayload,
  createGenerationDraft,
  fitProfile,
  generationAction,
  generationParameterDraft,
  generationParameterSnapshot,
  generationRetryContract,
  longVideoContract,
} from './generation';

const receipt = 'a'.repeat(64);
const profiles = {
  '16:9': { fit_required: false, default_fit_mode: 'none' as const },
  '9:16': { fit_required: true, default_fit_mode: 'crop' as const },
};

function longDetail(overrides: Record<string, unknown> = {}) {
  return {
    id: 'cid-long',
    duration_s: 30,
    segment_count: 3,
    plan_receipt: receipt,
    aspect_ratio: '16:9',
    resolution: '480p',
    fit_required: false,
    fit_mode: 'none',
    fit_profiles: profiles,
    receipt_version: 1,
    dialogue: { mode: 'auto', lines: [], auto_lines: [] },
    generation: null,
    ...overrides,
  };
}

describe('generation contract', () => {
  it('fails closed when a long-video frozen plan is incomplete', () => {
    expect(longVideoContract(longDetail())).toEqual({
      isLong: true,
      ready: true,
      segmentCount: 3,
      planReceipt: receipt,
    });
    expect(longVideoContract(longDetail({ plan_receipt: undefined }))).toMatchObject({
      isLong: true,
      ready: false,
      planReceipt: null,
    });
    expect(longVideoContract({ duration_s: 10 })).toEqual({
      isLong: false,
      ready: true,
      segmentCount: null,
      planReceipt: null,
    });
  });

  it('uses only valid server fit recommendations', () => {
    expect(fitProfile(longDetail(), '9:16')).toEqual({
      fit_required: true,
      default_fit_mode: 'crop',
    });
    expect(generationParameterDraft(longDetail())).toEqual({
      aspectRatio: '16:9',
      resolution: '480p',
      fitMode: 'none',
    });
    expect(() => fitProfile({ fit_profiles: {} }, '16:9')).toThrow('服务端画幅适配建议无效');
  });

  it('silently defaults only a new long-video draft to fast mode', () => {
    expect(createGenerationDraft(longDetail()).fastMode).toBe(true);
    expect(createGenerationDraft(longDetail({ duration_s: 10 })).fastMode).toBe(false);
    expect(createGenerationDraft(longDetail({
      generation: { status: 'failed', fast_mode: true },
    })).fastMode).toBe(true);
    expect(createGenerationDraft(longDetail({
      generation: { status: 'failed', fast_mode: false },
    })).fastMode).toBe(false);
    expect(createGenerationDraft(longDetail({
      generation: { status: 'succeeded' },
    })).fastMode).toBe(false);
  });

  it('server-frozen values replace a touched local draft', () => {
    const touched = {
      ...createGenerationDraft(longDetail()),
      aspectRatio: '9:16' as const,
      resolution: '768p' as const,
      fitMode: 'pad' as const,
      dialogueMode: 'none' as const,
      parameterTouched: true,
    };
    const frozen = createGenerationDraft(longDetail({
      aspect_ratio: '9:16',
      resolution: '768p',
      fit_mode: 'crop',
      dialogue: { mode: 'auto', lines: [], auto_lines: [] },
      generation: { status: 'failed', fast_mode: false },
    }), touched);

    expect(frozen).toMatchObject({
      aspectRatio: '9:16',
      resolution: '768p',
      fitMode: 'crop',
      dialogueMode: 'auto',
      fastMode: false,
      frozen: true,
      parameterTouched: false,
    });
  });

  it('builds short and new long payloads from explicit frozen inputs', () => {
    expect(buildSubmitPayload({
      clientRequestId: 'request-short',
      dialogueMode: 'custom',
      linesText: '0 - 1 | hello',
      fitRequired: false,
      aspectRatio: '9:16',
      resolution: '768p',
      isLong: false,
    })).toEqual({
      confirm: true,
      client_request_id: 'request-short',
      dialogue_mode: 'custom',
      fit_mode: 'none',
      aspect_ratio: '9:16',
      resolution: '768p',
      lines: [{ start_s: 0, end_s: 1, text: 'hello' }],
    });
    expect(buildSubmitPayload({
      clientRequestId: 'request-long',
      dialogueMode: 'auto',
      fitRequired: false,
      aspectRatio: '16:9',
      resolution: '480p',
      isLong: true,
      fastMode: true,
      planReceipt: receipt,
    })).toMatchObject({
      client_request_id: 'request-long',
      expected_plan_receipt: receipt,
      fast_mode: true,
    });
  });

  it('reuses the old id for resume and preserves a historical false fast mode', () => {
    expect(buildResumePayload(longDetail({
      generation: { status: 'resume_required', client_request_id: 'request-old' },
      dialogue: { mode: 'none', lines: [], auto_lines: [] },
    }))).toEqual({
      confirm: true,
      client_request_id: 'request-old',
      dialogue_mode: 'none',
      fit_mode: 'none',
      aspect_ratio: '16:9',
      resolution: '480p',
      expected_plan_receipt: receipt,
      fast_mode: false,
    });
  });

  it('uses a new id for failed paid segments and the old id for free stitch retry', () => {
    expect(buildLongFailedRetryPayload(longDetail({
      generation: { status: 'failed', stage: 'h3', fast_mode: false },
    }), 'request-new')).toMatchObject({
      client_request_id: 'request-new',
      fast_mode: false,
    });
    expect(buildStitchRetryPayload(longDetail({
      generation: {
        status: 'failed',
        stage: 'stitch',
        client_request_id: 'request-old',
        fast_mode: true,
        retry_paid_segment_count: 0,
      },
    }))).toMatchObject({
      client_request_id: 'request-old',
      fast_mode: true,
    });
    expect(generationRetryContract(longDetail({
      generation: {
        status: 'failed',
        stage: 'stitch',
        retry_paid_segment_count: 0,
      },
    }))).toEqual({ action: 'retry_stitch', paidTaskCount: 0 });
  });

  it('never exposes an action for running or submission-unknown provider states', () => {
    expect([
      generationAction('running'),
      generationAction('queued'),
      generationAction('submission_unknown'),
      generationAction('failed', 'h3'),
      generationAction('failed', 'stitch'),
    ]).toEqual(['none', 'none', 'none', 'retry', 'retry_stitch']);
    expect(generationRetryContract(longDetail({
      generation: { status: 'submission_unknown', stage: 'h3' },
    }))).toEqual({ action: 'none', paidTaskCount: 0 });
  });

  it('snapshots only server-frozen values and keeps historical fast mode false', () => {
    expect(generationParameterSnapshot(longDetail({
      fit_mode: 'crop',
      generation: { status: 'succeeded' },
    }))).toEqual({
      aspect_ratio: '16:9',
      resolution: '480p',
      dialogue_mode: 'auto',
      fit_mode: 'crop',
      duration_s: 30,
      segment_count: 3,
      fast_mode: false,
    });
  });
});
