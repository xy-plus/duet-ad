# Prompt and compression rules

## Score candidate scenes

Rate every source scene on five observable dimensions:

1. Hook strength: the initial defect or dirt is obvious within one second.
2. Action readability: the spray, foam, sponge, brush, or towel can be seen causing the change.
3. Result contrast: the final state is visibly different from the initial state.
4. Frame cleanliness: subtitles, watermarks, logos, blur, and transition blending are absent or minimal.
5. Continuity: the same surface, geometry, tool, camera side, and lighting remain stable.

Keep the highest-scoring complete loops. A dramatic action without a clean final result is weaker than a simpler action with an obvious before-and-after.

## Allocate nine images

### One continuous scene

Use nine states when the source already contains a full 15-second action:

1. establish the surface;
2. create or show the stain;
3. intensify or finish the stain;
4. introduce the cleaner;
5. spray or apply;
6. reaction or dwell;
7. scrub;
8. wipe dry;
9. final result.

### Two related scenes

Use a 5+4 or 4+5 allocation. Complete the first result before the single direct cut. The two subjects should share product, environment, material family, or camera style.

### Three high-contrast scenes

Use 3+3+3 only when each group has a complete mini-loop:

1. before or action setup;
2. peak application or wipe;
3. final result.

Use exactly two direct cuts. Do not cross-dissolve or merge objects.

## Write a frame-bound prompt

Use this skeleton and replace every bracketed field with inspected evidence:

```text
生成一支 15 秒、9:16 竖屏、720p、写实手机实拍风格的[清洁主题]短视频。只保留[场景数量]个场景，并在[切镜时间]进行直接切镜。

参考图片顺序：
图片1：[主体、状态、构图和该帧唯一作用]
...
图片9：[最终状态和保持不变的结构]

主体与环境连续性：[车辆/表面/工具/手/背景/光线/机位的不变量]

拍摄风格：普通手机近景，轻微手持漂移、小幅自动曝光变化、自然散射光、真实材料形变。无字幕、Logo、贴纸或水印。

时间线：
0.0–X.X 秒：[可观察动作]，对应图片1。
...
X.X–15.0 秒：[最终展示]，对应图片9。

物理连续性：[原因先于结果；变化边界随工具移动；未接触区域保持原状]

避免：[只列最相关的失真风险]
```

## Preserve physical causality

- A mark appears only where a finger, pen, or contaminant touches the surface.
- Foam grows outward from a visible nozzle or applicator.
- Dirt may loosen under foam, but it must not disappear before wiping when the intended mechanism is wipe removal.
- Clean boundaries follow the towel, sponge, or brush edge.
- Foam and dirt remain on untouched regions.
- Hands, tools, panels, wheels, lamps, trim, and reflections do not change identity mid-scene.

## Keep timing legible

Reserve about 0.7–1.2 seconds for the initial defect and about 0.7–1.3 seconds for the final result. Give the longest interval to the main cleaning action. Budget 0.1–0.2 seconds for each direct cut. Ensure all intervals are contiguous and end at 15.0 seconds.

The model may shift cut timing. Treat prompt timing as a strong guide, then report actual timing drift during generated-video review.
