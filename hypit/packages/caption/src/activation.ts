import { createMarkupSurfaceHostFacet } from "@hypit/markup";
import {
  captionComponent, captionManifest, captionModuleRef,
  decodeHiddenCaptionStyleSurface,
  captionMarkupSurfaces,
} from "./index.js";

export const hypitPackage = {
  format: "hypit.node-package@1" as const,
  modules: [{ manifest: captionManifest }],
  components: [captionComponent],
  hostFacets: [
    createMarkupSurfaceHostFacet({ module: captionModuleRef,
    declaration: captionMarkupSurfaces.find((item) => item.name === "hidden")!, handler: decodeHiddenCaptionStyleSurface }),
  ],
};
export default hypitPackage;
