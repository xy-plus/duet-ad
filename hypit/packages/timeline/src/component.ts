import type { ComponentPackage } from "@hypit/component-kit";
import type { StoredValue } from "@hypit/protocol";
import { canonicalize } from "@hypit/protocol";

import { timelineProducers, timelineTypes } from "./manifest.js";
import { assertTimelineIdentity } from "./identity.js";
import { projectTimelineAudio, projectTimelineSpace } from "./projection.js";
import type { Timeline } from "./types.js";

function track(value: StoredValue | undefined): Timeline {
  if (value?.kind !== "inline") throw new Error("Timeline must be inline.");
  return value.value as unknown as Timeline;
}

export const timelineComponent = {
  producers: [
    { producer: timelineProducers.projectProgramSpace, handler: ({ inputs }) => ({ outputs: { space: { kind: "inline", value: canonicalize(projectTimelineSpace(track(inputs.track?.value))) } }, needs: {} }) },
    { producer: timelineProducers.projectAudio, handler: ({ inputs }) => ({ outputs: { audio: { kind: "inline", value: canonicalize(projectTimelineAudio(track(inputs.track?.value))) } }, needs: {} }) },
  ],
  validators: [{
    type: timelineTypes.track,
    handler: ({ value }) => assertTimelineIdentity(track(value)),
  }],
} satisfies ComponentPackage;
