"""Skill 定稿条款断言（2026-09-28 多版本对照定稿：V2 产品突出版为正式版；
2026-09-28 吸收 hypit 思想增补功能归因/措辞禁令条款；2026-09-29 摘除
增强机制（措辞库五分类 + CAPS 字面形式），只保留复刻与全链路禁止编造）。

只断言条款存在/不存在与关键约束词，不断言全文措辞，避免 skill 文本
微调即红。
"""

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


def _read(skill: str) -> str:
    return (SKILLS_DIR / skill / "SKILL.md").read_text(encoding="utf-8")


def test_skills_carry_no_variant_experiment_markers() -> None:
    # V1 人物表演 / V2 产品突出 / V3 完整对照已定稿为 V2，VARIANT 行级
    # 标记块是实验脚手架，定稿版一个不留。
    for skill in ("video-maker", "h3-max-r2v-prompt"):
        assert "VARIANT:" not in _read(skill)


def test_video_maker_requires_product_presentation_block() -> None:
    text = _read("video-maker")
    assert "产品呈现" in text
    assert "展示手法" in text


def test_video_maker_drops_person_performance_block() -> None:
    # V2 定稿摘除人物表演段（表情零件/微动作/时序渐变）。
    text = _read("video-maker")
    assert "人物表演" not in text
    assert "时序渐变" not in text


def test_video_maker_performance_blocks_stay_pixel_verifiable() -> None:
    # 分工约束：video-maker 只记录像素可证实的事实，猜测只发生在 fusion。
    text = _read("video-maker")
    assert "像素可证实" in text
    assert "禁止猜测" in text


def test_video_maker_function_attribution_clause() -> None:
    # 功能归因：关键镜头变化/产品镜头追加"作用："前缀的归因句，措辞取自
    # 中文化功能词表；节奏写"事实→感受"双句。
    text = _read("video-maker")
    assert "作用：" in text
    for word in ("建立人物", "对比主张", "积累证据", "放大反应", "让过程可懂", "交接注意力"):
        assert word in text
    assert "事实→感受" in text
    # 多小节段必须另起独立的整体节奏双句（2026-09-28 复验：九小节段
    # 缺少该句时节奏归因退化，只分散在小节标题里）。
    assert "另起一句独立的整体节奏" in text


def test_video_maker_fact_attribution_separation() -> None:
    # 事实/归因分离：归因必须锚定到已陈述的事实；"像素可证实""禁止猜测"
    # 原字样保留（见上方存量断言）。
    text = _read("video-maker")
    assert "事实与归因分离" in text
    assert "锚定" in text


def test_video_maker_attribution_stays_out_of_project_index() -> None:
    # 功能归因只进逐段 visual_prompt；项目级索引 relations 的
    # "不因常识补造功能"条款不受影响。
    text = _read("video-maker")
    assert "不因常识补造功能" in text
    assert "不进入项目级索引" in text


def test_video_maker_product_presentation_role_x_technique() -> None:
    # "产品呈现"段升级为"叙事角色 × 展示手法"双维。
    text = _read("video-maker")
    assert "叙事角色" in text
    assert "展示手法" in text
    for role in ("被演示的证据", "被声称的对象", "展示的材质", "CTA 落点"):
        assert role in text


def test_h3_skill_segment1_covers_action_only_not_identity() -> None:
    # 段 1 边界维持"只覆盖动作，不含身份外观"；表情条款随 V2 定稿摘除。
    text = _read("h3-max-r2v-prompt")
    assert "只覆盖动作层面信息" in text
    assert "身份外观" in text
    # 表情条款残留清零。
    assert "表情" not in text
    assert "微动作" not in text
    assert "眉毛上扬" not in text


def test_h3_skill_does_not_consume_audio_analysis() -> None:
    # V2 正式版不消费 audio_analysis；fusion payload 注入代码保留供表演类
    # 版本迭代（取舍记录见 app/pipeline.py 注入点注释）。
    text = _read("h3-max-r2v-prompt")
    assert "audio_analysis" not in text
    assert "仅凭 visual_prompt 判断语气" not in text


def test_h3_skill_product_prominence_replication_only() -> None:
    # 2026-09-29 用户决策：②增强机制（措辞库五分类 + CAPS 字面形式）实测
    # 会编造原视频不存在的镜头（项目 a564dbb9 段1），整体摘除；第 6 条只
    # 保留复刻原片手法，全链路禁止编造原片没有的内容。
    text = _read("h3-max-r2v-prompt")
    assert "产品突出" in text
    assert "复刻" in text
    assert "不得编造" in text
    assert "措辞库" not in text
    assert "增强（" not in text
    for dim in ("REVEAL", "DEMO", "BENEFIT", "PROBLEM-SOLUTION", "UGC"):
        assert dim not in text


def test_h3_skill_web_search_clause_marked_inactive() -> None:
    # 2026-09-28 冒烟未过：条款保留但必须显式标注未生效，并转为
    # 禁止编造检索结果的护栏。
    text = _read("h3-max-r2v-prompt")
    assert "联网搜索" in text
    assert "未生效" in text


def test_h3_skill_pre_delivery_checklist() -> None:
    text = _read("h3-max-r2v-prompt")
    assert "交付前自检" in text
    assert "源外观" in text
    # 增强机制摘除后，自检清单改查编造原视频不存在的内容。
    assert "原视频不存在" in text
    # 表情自检条目随 V2 定稿摘除。
    assert "表情与微动作" not in text


def test_h3_skill_causal_trigger_anchor() -> None:
    # 第 1 条补因果触发锚定：允许 ≤6 字台词关键词片段作时序锚点，
    # 不整句复述、不输出台词指令；对口型风险保持 UNVERIFIED 标注。
    # 关键词强制【】包裹、禁用引号（2026-09-28 用户决策：引号是视频模型
    # "角色说出此话"的通行语法，【】是关键词强调标记）。
    text = _read("h3-max-r2v-prompt")
    assert "当台词提到【关键词】时" in text
    assert "必须用【】包裹" in text
    assert "不整句" in text
    assert "不输出" in text
    assert "UNVERIFIED" in text


def test_h3_skill_wording_bans() -> None:
    # 措辞禁令：抽象评价词必须落到可见事实；全面禁比喻标注为 HOUSE RULE。
    text = _read("h3-max-r2v-prompt")
    assert "抽象评价词" in text
    assert "禁止比喻" in text
    assert "HOUSE RULE" in text


def test_h3_skill_attribution_sentence_isolation() -> None:
    # 归因句隔离：visual_prompt 中"作用："前缀句不得复述进第 1 条动作描述。
    text = _read("h3-max-r2v-prompt")
    assert "作用：" in text
    assert "不得复述" in text


def test_h3_skill_checklist_covers_new_clauses() -> None:
    # 自检清单同步保留条款（措辞库条目随增强机制摘除）。
    text = _read("h3-max-r2v-prompt")
    assert "落到可见事实" in text
    assert "触发锚点" in text


def test_object_replace_target_description_person_appearance_ban() -> None:
    # 2026-09-29 用户决策：被替换人物的形象由图版 tile（头肩面部特写）
    # 与绑定用户图逐像素传递，target_visual_description 禁止写 tile 可见
    # 形象细节，只保留身份锚点、一行服装简述与动作/关系/时序；
    # appearance_description 的完整形象条款不受影响。
    text = _read("object-replace")
    assert "头肩面部特写" in text
    assert "身份锚点" in text
    assert "服装简述" in text
    assert "由图版 tile" in text
    for word in ("脸型", "五官", "肤色", "肤质", "发型发色", "痣", "手部皮肤纹理"):
        assert word in text
    # 正反例钉死生产实测语义。
    assert "反例" in text
    assert "正例" in text


def test_object_replace_partial_person_non_empty_fallback() -> None:
    # 2026-09-29 实测：部分可见人物（只露眼部局部，0fb88d02 person-03）在
    # 人物瘦身条款下被反复输出空字符串，validator 拒绝、节点重试全败；
    # 兜底条款钉死：禁止空字符串 + 只基于可见部分写最小非空描述。
    text = _read("object-replace")
    assert "禁止输出空字符串" in text
    assert "部分可见人物" in text
    assert "可见什么写什么" in text


def test_h3_skill_person_appearance_not_restated() -> None:
    # 第 2 条：人物 stable key 不复述形象，以"权威块指定的新人物"指代，
    # 只写身份锚点与服装简述；自检清单（第 45 条）覆盖该禁令。
    text = _read("h3-max-r2v-prompt")
    assert "不复述形象" in text
    assert "权威块指定的新人物" in text
    assert "复述了被替换人物" in text
