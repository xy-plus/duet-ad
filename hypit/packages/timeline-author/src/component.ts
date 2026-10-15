import type { ProgramClock } from "@hypit/program-space";
import type { ComponentPackage } from "@hypit/component-kit";
import type { StoredValue } from "@hypit/protocol";
import { canonicalize } from "@hypit/protocol";
import type { SemanticTake } from "@hypit/speech";

import { timelineAuthorProducers } from "./manifest.js";
import { appendTimelineAuthorTake, assembleTimelineAuthor, createTimelineAuthorSet } from "./program.js";
import type { TimelineAuthorHeader, TimelineAuthorSet } from "./types.js";

function inline<T>(value: StoredValue | undefined, subject: string): T {
  if (value?.kind !== "inline") throw new Error(`${subject} must be inline`);
  return value.value as unknown as T;
}

export const timelineAuthorComponent = {
  producers: [
    {
      producer: timelineAuthorProducers.createSet,
      handler: () => ({
        outputs: { set: { kind: "inline", value: canonicalize(createTimelineAuthorSet()) } },
        needs: {},
      }),
    },
    {
      producer: timelineAuthorProducers.appendTake,
      handler: ({ inputs }) => ({
        outputs: { set: { kind: "inline", value: canonicalize(appendTimelineAuthorTake(
          inline<TimelineAuthorSet>(inputs.set?.value, "TimelineAuthorSet"),
          inline<SemanticTake>(inputs.take?.value, "SemanticTake"),
        )) } },
        needs: {},
      }),
    },
    {
      producer: timelineAuthorProducers.assembleTrack,
      handler: ({ inputs }) => ({
        outputs: { track: { kind: "inline", value: canonicalize(assembleTimelineAuthor(
          inline<TimelineAuthorHeader>(inputs.header?.value, "TimelineAuthorHeader"),
          inline<TimelineAuthorSet>(inputs.set?.value, "TimelineAuthorSet"),
          inline<ProgramClock>(inputs.clock?.value, "ProgramClock"),
        )) } },
        needs: {},
      }),
    },
  ],
} satisfies ComponentPackage;
