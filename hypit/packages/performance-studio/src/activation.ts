import { createStudioCompanionHostFacet } from "@hypit/studio-adapter";
import { performanceStudioTrackCompanions, performanceStudioParameterCompanions } from "./index.js";
export default {
  format: "hypit.node-package@1" as const,
  hostFacets: [createStudioCompanionHostFacet({ tracks: performanceStudioTrackCompanions, parameters: performanceStudioParameterCompanions })],
};
