import {
  createRuntimeEndpointAdapterFacet,
  runtimeConfigCredentialRef,
  runtimeConfigExact,
  runtimeConfigObject,
  runtimeConfigString,
} from "@hypit/runtime-kit";

import { createArkProvider } from "./provider.js";

/**
 * ark endpoint config：capability（"seedance" | "seedream"）、baseUrl、model、
 * apiKey（CredentialRef）四个显式必填键，不设缺省值；model 必须与 ark 控制台
 * 为该账号开通的模型 ID 完全一致。
 * assetPublishDir / assetPublicBaseUrl 是可选的公网资产发布通道（ark 的参考
 * 视频/音频与超大图片只接受公网 URL）：两个键必须同时提供或同时不提供。
 */
export const hypitPackage = {
  format: "hypit.node-package@1" as const,
  hostFacets: [createRuntimeEndpointAdapterFacet({
    use: "@hypit/provider-ark",
    async activate(context) {
      if (context.pool === undefined) throw new Error("ark Provider 需要 pool");
      const config = runtimeConfigObject(context.config, "ark");
      runtimeConfigExact(config, ["capability", "baseUrl", "model", "apiKey", "assetPublishDir", "assetPublicBaseUrl"], "ark");
      const capability = runtimeConfigString(config.capability, "ark capability");
      if (capability !== "seedance" && capability !== "seedream") {
        throw new Error('ark capability 必须是 "seedance" 或 "seedream"');
      }
      const baseUrl = runtimeConfigString(config.baseUrl, "ark baseUrl");
      const model = runtimeConfigString(config.model, "ark model");
      const apiKey = runtimeConfigCredentialRef(config.apiKey, "ark apiKey");
      if (baseUrl === undefined || model === undefined || apiKey === undefined) {
        throw new Error("ark endpoint config 必须显式提供 baseUrl、model 和 apiKey");
      }
      const assetPublishDir = runtimeConfigString(config.assetPublishDir, "ark assetPublishDir");
      const assetPublicBaseUrl = runtimeConfigString(config.assetPublicBaseUrl, "ark assetPublicBaseUrl");
      if ((assetPublishDir === undefined) !== (assetPublicBaseUrl === undefined)) {
        throw new Error("ark assetPublishDir 与 assetPublicBaseUrl 必须同时配置或同时省略");
      }
      return {
        endpoint: createArkProvider({
          instance: context.instance,
          pool: context.pool,
          capability,
          baseUrl,
          model,
          apiKey,
          ...(assetPublishDir === undefined || assetPublicBaseUrl === undefined ? {} : {
            assetPublish: { dir: assetPublishDir, publicBaseUrl: assetPublicBaseUrl },
          }),
        }),
      };
    },
  })],
};

export default hypitPackage;
