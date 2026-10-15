import { createStudioCompanionHostFacet } from "@hypit/studio-adapter";
import { soundStudioTrackCompanions, soundStudioParameterCompanions } from "./index.js";
export default {
  format: "hypit.node-package@1" as const,
  hostFacets: [createStudioCompanionHostFacet({ tracks: soundStudioTrackCompanions, parameters: soundStudioParameterCompanions })],
};
