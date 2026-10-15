import { artifactManifest } from "@hypit/artifact";
import { compositionComponent, compositionManifest } from "@hypit/composition";
import { mediaComponent, mediaManifest } from "@hypit/media";
import { narrativeManifest } from "@hypit/narrative";
import { programSpaceManifest } from "@hypit/program-space";
import { timelineManifest } from "@hypit/timeline";
import { spatialComponent, spatialManifest } from "@hypit/spatial";
import { speechManifest } from "@hypit/speech";
import { speechEvidenceManifest } from "@hypit/speech-evidence";
import { svsManifest } from "@hypit/svs";
import { temporalManifest } from "@hypit/temporal";
import { visualIrManifest } from "@hypit/visual-ir";

/** Shared test fixture only; production packages import only the contracts they use. */
export const videoContractManifests = [
  artifactManifest,
  narrativeManifest,
  mediaManifest,
  programSpaceManifest,
  speechManifest,
  speechEvidenceManifest,
  svsManifest,
  timelineManifest,
  spatialManifest,
  temporalManifest,
  visualIrManifest,
  compositionManifest,
] as const;

export { compositionComponent, mediaComponent, spatialComponent };
