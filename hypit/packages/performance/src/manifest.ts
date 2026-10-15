import { compositionDependency, compositionTypes } from "@hypit/composition";
import { timelineDependency, timelineTypes } from "@hypit/timeline";
import { spatialDependency, spatialTypes } from "@hypit/spatial";
import { temporalDependency, temporalTypes } from "@hypit/temporal";
import { mediaTrackDependency, mediaTrackTypes } from "@hypit/media-track";
import { svsRecipeType } from "@hypit/svs";
import { temporalWindowAttributeVocabulary } from "@hypit/temporal-markup";
import type { ModuleManifest, ProducerRef, TypeRef } from "@hypit/protocol";

export const performanceModuleRef = { name: "@hypit/performance", version: "1" } as const;
export const performanceTypes = Object.fromEntries(["Style", "Set", "Header"].map(name => [name.toLowerCase(), { module: performanceModuleRef, name: `Performance${name}` }])) as Record<"style" | "set" | "header", TypeRef>;
export const performanceProducers = Object.fromEntries(["create", "append", "resolve", "ordinary"].map(name => [name, { module: performanceModuleRef, name }])) as Record<"create" | "append" | "resolve" | "ordinary", ProducerRef>;
export const ordinaryInputs = [
  { name: "timeline", type: timelineTypes.track }, { name: "canvas", type: spatialTypes.canvas },
  { name: "window", type: temporalTypes.window }, { name: "frame", type: spatialTypes.frame },
  { name: "fit", type: spatialTypes.fit }, { name: "sample", type: mediaTrackTypes.sampleLayerSpec },
  { name: "spec", type: mediaTrackTypes.itemSpec },
];
export const performanceManifest: ModuleManifest = {
  format: "hypit.module@1", ...performanceModuleRef,
  dependencies: [compositionDependency, timelineDependency, spatialDependency, temporalDependency, mediaTrackDependency],
  types: Object.values(performanceTypes).map(type => ({ name: type.name })), capabilities: [],
  producers: [
    { name: "create", inputs: [], outputs: [{ name: "set", type: performanceTypes.set }], needs: [] },
    { name: "append", inputs: [{ name: "set", type: performanceTypes.set }, { name: "window", type: temporalTypes.window }, { name: "visual", type: compositionTypes.visualTrack }], outputs: [{ name: "set", type: performanceTypes.set }], needs: [] },
    { name: "resolve", inputs: [{ name: "set", type: performanceTypes.set }, { name: "timeline", type: timelineTypes.track }, { name: "header", type: performanceTypes.header }], outputs: [{ name: "visual", type: compositionTypes.visualTrack }], needs: [] },
    { name: "ordinary", inputs: ordinaryInputs, outputs: [{ name: "visual", type: compositionTypes.visualTrack }], needs: [] },
  ],
};
export const performanceMarkupSurfaces = [
  { name: "style", tag: "Style", mode: "structured", outputs: [performanceTypes.style, spatialTypes.fit, mediaTrackTypes.sampleLayerSpec, mediaTrackTypes.itemSpec], vocabulary: {
    summary: "Defines ordinary A-roll framing using Media appearance properties.",
    example: `<performance:Style id="full" frame={layout.full} appearance={recipes.media.presenter}/>`,
    attributes: [
      { name: "id", kind: "identifier", required: true, summary: "Style name." },
      { name: "frame", kind: "reference", required: true, accepts: [spatialTypes.frame], summary: "Destination Frame." },
      { name: "appearance", kind: "reference", required: true, accepts: [svsRecipeType], summary: "Media appearance Recipe: fitting, clipping, border and color." },
    ],
  } },
  { name: "track", tag: "Track", mode: "structured", outputs: [compositionTypes.visualTrack, performanceTypes.set, performanceTypes.header, temporalTypes.instantSpec, temporalTypes.windowSpec, temporalTypes.instant, temporalTypes.window], vocabulary: {
    summary: "Presents Timeline footage through ordered Use rules, retaining source playback time.",
    example: `<performance:Track id="presenter" timeline={program.timeline} canvas={canvas}><performance:Use style={full}/></performance:Track>`,
    attributes: [
      { name: "id", kind: "identifier", required: true, summary: "Presentation name." },
      { name: "timeline", kind: "reference", required: true, accepts: [timelineTypes.track], summary: "Placed material and film time." },
      { name: "canvas", kind: "reference", required: true, accepts: [spatialTypes.canvas], summary: "Composition Canvas." },
    ], children: [{ tag: "Use", cardinality: "many", summary: "Last matching Use wins. No time selector means the whole Timeline.", attributes: [
      { name: "id", kind: "identifier", required: false, summary: "Optional occurrence identity." },
      { name: "style", kind: "reference", required: true, accepts: [performanceTypes.style], summary: "Ordinary or project-authored presentation Style." },
      ...temporalWindowAttributeVocabulary,
    ] }],
  } },
] as const;
