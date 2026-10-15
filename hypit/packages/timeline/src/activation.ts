import { timelineComponent, timelineManifest } from "./index.js";

export const hypitPackage = {
  format: "hypit.node-package@1" as const,
  modules: [{ manifest: timelineManifest }],
  components: [timelineComponent],
};
export default hypitPackage;
