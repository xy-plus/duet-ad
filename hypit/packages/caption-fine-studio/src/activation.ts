import { createStudioCompanionHostFacet } from "@hypit/studio-adapter";
import { captionFineStudioTrackCompanions, captionFineStudioParameterCompanions } from "./index.js";

export default {
  format: "hypit.node-package@1" as const,
  hostFacets: [createStudioCompanionHostFacet({ tracks: captionFineStudioTrackCompanions, parameters: captionFineStudioParameterCompanions })],
};
