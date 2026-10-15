import { createMarkupSurfaceHostFacet } from "@hypit/markup";
import { soundManifest, soundModuleRef, soundMarkupSurfaces } from "./manifest.js";
import { soundComponent } from "./component.js";
import { decodeSoundStyleSurface, decodeSoundTrackSurface } from "./surface.js";
export const hypitPackage = {
  format: "hypit.node-package@1" as const,
  modules: [{ manifest: soundManifest }], components: [soundComponent],
  hostFacets: soundMarkupSurfaces.map(declaration => createMarkupSurfaceHostFacet({ module: soundModuleRef, declaration,
    handler: declaration.name === "style" ? decodeSoundStyleSurface : decodeSoundTrackSurface })),
};
export default hypitPackage;
