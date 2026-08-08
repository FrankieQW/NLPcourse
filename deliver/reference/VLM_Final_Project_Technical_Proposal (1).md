# 视觉语言大模型课程设计 · 技术方案

**项目代号：忆见 AskAlbum —— 面向真实场景相册的「结构化标注 + 回译自验证 + 混合检索 + 叙事 Agent」系统**

课程：人工智能课程 —— 视觉与自然语言处理（华中科技大学 人工智能与自动化学院）

---

## 目录

- [一、作业要求的精确拆解](#一作业要求的精确拆解)
- [二、资源盘算与可行性边界](#二资源盘算与可行性边界)
- [三、方案总览](#三方案总览)
- [四、方案 A 的系统架构](#四方案-a-的系统架构)
- [五、核心模块技术细节](#五核心模块技术细节)
- [六、实验设计](#六实验设计)
- [七、分工、时间线与交付](#七分工时间线与交付)
- [八、备选方案](#八备选方案)
- [九、风险清单与兜底](#九风险清单与兜底)
- [十、参考资料清单](#十参考资料清单)

---

# 一、作业要求的精确拆解

## 1.1 硬性要求（原文）

| 项 | 内容 |
|---|---|
| 数据 | 训练集 2000 张真实场景图，验证集 369 张，**测试集 200 张不公开** |
| 前置任务 | 图片**无标注**，必须先用"各种免费视觉语言大模型工具"生成**多维度**描述信息；必须体现 prompt engineering |
| 作业主体 | 含图文理解或生成任务：以图生文 / 以文生图 / 图文问答 / 图文搜索 / 多模态交互，鼓励创意 |
| 报告 | LaTeX，≤20 页正文，学术论文结构：Abstract / Introduction / Related Work / Method / Experiment / Result / Conclusion |
| 代码 | Prompt Engineering、模型训练和测试；完整 ReadMe、环境依赖、部署文档，Demo 可选 |
| 其它 | 个人报告（每人一份）；PPT 可选；≤3 分钟演示视频（加分）；4～5 人一组 |
| 拓展 | 可自采领域图片（抽象画 / 古典画作 / 医学影像）测迁移能力并讨论 |

## 1.2 隐含约束（决定方案天花板的五条）

1. **"测试集 200 张不公开"意味着交付的是一条可复跑的流水线，不是一堆结果文件。**
   代码必须支持 `python run.py --input_dir <任意新目录>` 一键完成"标注 → 建索引 → 可检索/可问答"。这一条很多组会栽跟头。

2. **"多维度"是明确的评分锚点。**
   出一段自由文本的 caption 拿不到分，必须是**结构化 schema**（对象 / 属性 / 空间关系 / OCR / 光照 / 情绪 / 美学 / 关键词……）。

3. **课程第 4 章整章讲评估**（CIDEr、MRR/NDCG、WUPS、MM-Vet/MME/SEED-Bench、TouchStone 的 LLM-as-judge、Do-Not-Answer 的人工评估、LVLM-eHub 的模型竞技场）。
   这是老师的偏好指纹：报告里出现这些指标 + 一张自建 Arena 的投票表，性价比极高。

4. **课程第 2 章整章讲多 Agent 编排**（顺序 / 并行 / 主管-下属 / 路由 / 辩论）。
   方案里显式用上 2～3 种编排模式并画图，直接呼应。

5. **第 110 页专门讲"Agent 会在任何步骤幻觉"。**
   谁能把**幻觉量化并抑制**，谁就在"讨论"部分赢。

> **结论**：本作业的真正难点不是模型，是**在无标注条件下如何证明你的标注是好的**。方案的学术内核应放在这里。

---

# 二、资源盘算与可行性边界

| 资源 | 可做 | 不可做 |
|---|---|---|
| RTX 4090 48G（间歇） | vLLM 批量跑 Qwen3-VL-8B / Qwen2.5-VL-7B-AWQ 做全量标注；32B AWQ 也能塞下；FLUX.1-dev FP8 文生图 | 全参微调 7B+（时间不划算） |
| RTX 4070 Laptop 8G | Chinese-CLIP ViT-B/16 编码、BGE-M3 检索、DINOv2 特征、SDXL-Lightning 4-step 文生图、GroundingDINO/OWLv2 检测、Web Demo | FLUX FP16、SD3.5 Large、任何 7B VLM 的 FP16 推理 |
| 免费 API | GLM-4V-Flash（智谱，免费）、阿里百炼 Qwen-VL 新用户额度、Gemini Flash 免费层、硅基流动免费模型 | 无限量高并发；需做限流与重试 |

**数据量级的关键事实**：2569 张图 × 1024 维 × 4 字节 ≈ **10.5 MB**。这个规模下 FAISS 用 `IndexFlatIP` 暴力精确检索即可，单次查询 < 5 ms，**不需要 HNSW/IVF**。

报告里诚实写这一点比硬上向量数据库更显专业，可在 Discussion 补一段："若扩到 \(10^6\) 量级应改用 HNSW，复杂度从 \(O(N)\) 降到 \(O(\log N)\)。"

**标注吞吐估算**（需自行实测，此处给量级）：
Qwen2.5-VL / Qwen3-VL 用 `max_pixels = 768 × 28 × 28 ≈ 6.0×10^5`（约 896×672），每图视觉 token ≤ 768；输出约 500 token。vLLM 在 4090 上开 batch 16～32，2569 张图单轮大约 **0.5～1.5 小时**。跑 3 轮自洽采样也在一晚内完成。这是本方案完全不需要训练的底气。

---

# 三、方案总览

推荐一个主方案，把作业要求的五种形式（以图生文 / 图文搜索 / 图文问答 / 以文生图 / 多模态交互）**全部收进同一条技术主线**，而不是拼四个互不相干的小 demo —— 后者是这类课设最常见的失分点。

## 主方案：忆见 AskAlbum

> **一句话定位**：Google Photos "Ask Photos" 的开源可复现简化版，但补上了它没有公开的东西 —— **一套无需人工标注就能量化"描述好不好"的自验证机制**。

**对标物（放 Related Work 很好用）**：

- **Google Photos Ask Photos**：Gemini 驱动的对话式相册检索。先返回一页相关结果，对于"哪些照片适合当手机壁纸""我在巴塞罗那吃了什么"这类复杂问题，再由 Gemini 在后台继续缩小结果集并抽取信息 —— **这就是 Agent 层**。
- **Immich Smart Search**：图像入库时编码成向量存进 PostgreSQL 的 `smart_search` 表，查询文本用 CLIP 编码，用 pgvector 的 `<=>` 算子做余弦距离排序 —— **纯 CLIP 单塔检索**，这正是你们要超越的 baseline。

**两个备选**（详见第八节）：

- **方案 B**：图像情绪/美学 → 中文诗歌与风格化再创作（创意强，学术性弱，可作为主方案的一个"生成头"）
- **方案 C**：纯评估向 —— 「无参考图像描述质量度量」研究（学术性最强，交互性弱）

主方案实际上已经把 B 和 C 都吸收为子模块。

---

# 四、方案 A 的系统架构

```
                 ┌──────────── 离线阶段（一次性，可对新目录重跑）────────────┐
raw images ──▶ M0 路由分类 ──▶ M1 结构化标注引擎 ──▶ M2 可验证性校验层 ──▶ M3 多路索引
             (CLIP+规则弱路由)  (VLM×N, JSON约束,       (检测器 grounding      (CLIP-I / 文本向量
                                自洽采样+跨模型投票)      + 回译 EchoBack)       / BM25 倒排)
                                                             │
                 ┌──────────────────── 在线阶段 ────────────────────┐
user query ──▶ M4 查询理解 Agent ──▶ M5 混合召回+RRF ──▶ M6 VLM Reranker ──▶ M7 输出 Agent
             (槽位抽取/改写/路由)      (三路并行)          (listwise 打分)      ├ 检索结果页
                                                                              ├ 图文问答（带引用）
                                                                              ├ 图文游记/日记生成
                                                                              └ 缺图补全（文生图）
```

三种课上讲过的编排模式都用到了：

- **顺序流水线**：M0 → M1 → M2
- **并行 fan-out / fan-in**：M5 三路召回、M1 多模型采样
- **路由分发**：M0 粗粒度 Prompt 路由、M4 查询类型路由
- （M2 的跨模型比对本质是**辩论**模式的简化）

---

# 五、核心模块技术细节

## M0 · 轻量粗粒度路由

真实数据审查表明，图片同时包含街景、人物、交通、文字等内容，还混有插画、表情图、屏幕、教材公式和展品。因此 M0 不能把“街景 / 美食 / 夜景 / 文字密集”当成互斥单标签：夜景和文字密集是跨内容属性，不是与街景平级的内容类别。

M0 复用后续索引所需的 CLIP 类图像向量，输出一个主内容路由、最多两个辅助路由和若干 modifier：

```json
{
  "primary_route": "street_urban",
  "secondary_routes": ["people_activity", "transport"],
  "modifiers": ["text_rich", "low_light"],
  "fallback": false,
  "fallback_reason": null
}
```

内容路由包括 `indoor / street_urban / nature / people_activity / food / transport / animal_plant / object_exhibit / illustration_meme / document_screen`；`general` 只用于 M0 未运行、失败或低置信回退。`text_rich / low_light / crowd / close_up` 等 modifier 独立判定，`low_resolution` 必须由原图尺寸规则产生，不能交给 CLIP 猜。

**用途**：为 M1 组合观察补丁，而不是给 M1 提供真值。M1 必须独立看图生成 scene；路由与图像冲突时以图像为准。当前共享开发阶段已延期 M0，先使用统一 Prompt 跑通 M1 双模型标注。

---

## M1 · 结构化标注引擎（本作业的正题）

### 输出契约

VLM 只生成 `annotation_payload`，不生成 `image_id`、哈希、模型版本、调用成本、关键词或融合置信度。客户端在候选运行记录中保存这些元数据和 M0 `route_context`，融合器再生成 canonical annotation。

M1 Schema v1.2 的主要字段为：

```text
scene(primary_type, secondary_types, media_type, sub_type_zh, environment)
capture_visual(time_of_day, weather, lighting, viewpoint, shot_scale, blur_level)
entities[](entity_id, type, name, count, bbox, position, visibility, attributes)
ocr[](text_id, text_raw, language, bbox, legibility)
relations[](subject_id, predicate, object_id)
event(summary, evidence_entity_ids)
subjective(mood, palette, aesthetic)
captions(short_zh, dense_zh)
uncertainties[](JSON Pointer, reason, note)
```

可执行的唯一 Schema 来源是 [`annotation_payload.schema.json`](../annotation_payload.schema.json)，共享 Prompt 是 [`prompt_v1.md`](../prompt_v1.md)。文档中的简写不能替代这两个版本化文件。

### 为什么这样设计

- `entities` / `relations` / `ocr` 是可被检测器与 OCR 工具验证的**硬字段**；
- `subjective` 是单独保存的**软字段**，不与硬事实共同投票；
- `media_type` 区分真实照片、插画、截图和文档，避免给非摄影内容编造天气或光照；
- `keywords_zh/en` 在融合后从 accepted facts 统一派生，避免两个 VLM 的翻译差异污染字段比较；
- `uncertainties` 使用 JSON Pointer 指向具体字段并记录原因，不能用一个全局自报置信度代替。

### Prompt Engineering 的六个可写进报告的技巧

1. **JSON Schema 强约束解码**
   两端尽量使用所选版本支持的 server-side structured outputs，并始终再做本地 canonical Schema 和语义校验。vLLM 的具体参数随版本变化，不能把旧 `guided_json` 写死；adapter 按锁定版本转换 Schema 子集。

2. **先证据后综合的输出组织**
   Prompt 先要求实体、OCR 和关系，再生成 event 与 caption。这样便于检查 caption 是否引入结构化字段之外的新事实；它是输出约束，不应宣称为可观测的思维链。

3. **负面约束清单**
   禁止推测人物身份/职业/民族；禁止用"可能/似乎"含糊其辞（改为写进 `uncertain_fields`）；禁止描述画面中不存在的物体；数量必须是确切整数或写 `null`。

4. **少样本锚定**
   默认先做 zero-shot P0-P3 对比。只有粒度仍不稳定时才加入一个 few-shot，且只能来自训练集 `prompt_dev_50`，不能使用 369 张验证图。

5. **自洽采样 + 字段级投票**
   全量阶段每个 backend 默认一次低温/确定性调用。\(M=3\) 只用于 prompt-dev 或争议样本，并逐字段计算一致度：

   \[
   \mathrm{Agr}(f) \;=\; \frac{2}{M(M-1)} \sum_{a<b} \mathbb{1}\!\left[\mathrm{match}\big(y_f^{(a)},\, y_f^{(b)}\big)\right]
   \]

   数值字段用相等判定；实体和文本必须先做 ID、同义词与位置对齐。`Agr(f)` 低只能触发复核，不能直接把事实改写为模型自报的不确定性。

6. **跨模型交叉验证**
   本地 Qwen3-VL 与 API 端优先选择不同模型家族。两个候选互不看对方输出；一致字段自动接受，冲突字段经过定向调用、M2 工具或人工复核，不把 API 默认当真值。

> **工程门禁**：不要依据二手资料假定 model ID、base64/URL、JSON Schema、免费额度或并发能力。API 负责人先对账号实际可用模型做 12 图能力探针；认证信息和临时签名 URL 不得进入日志。数据含可识别人物，上传前还要核对服务商的数据留存和训练政策。

### Prompt 版本实验

从训练集划出 `prompt_dev_50`，固定图片、预处理、模型、Schema 和解码参数，依次比较 P0 自由描述、P1 Schema、P2 可见性/隐私/OCR 约束、P3 M0 路由补丁。另设与其不重叠的 `m1_audit_50` 做发布门禁；369 张验证集不参与 Prompt 选择或 few-shot。

选择 Prompt 时优先比较 Schema 合法率、硬实体 precision/recall、计数、OCR 捏造、关系 precision、caption 新增事实和隐私违规。CIDEr、CLIPScore 等只作补充，不能把图文相似度当作细粒度事实正确。详细实验和双人审计流程见 M1 Agenda。

---

## M2 · 可验证性校验层（方案的学术内核）

### (a) 对象级幻觉率 CHAIR

从 `objects[].name` 抽出名词集合，用开放词汇检测器（GroundingDINO / OWLv2 / YOLO-World，8G 显存足够）在原图上做 grounding。若某名词的最高置信度 < 阈值 \(\tau\)（建议 0.3，做敏感性分析），判为幻觉。沿用 Rohrbach 等人的定义：

\[
\mathrm{CHAIR}_i = \frac{\big|\{\text{提及但未被检出的对象}\}\big|}{\big|\{\text{caption 中提及的全部对象}\}\big|}
\]

\[
\mathrm{CHAIR}_s = \frac{\big|\{\text{含} \ge 1 \text{ 个幻觉对象的图}\}\big|}{|\{\text{全部图}\}|}
\]

同时报告**覆盖率**（检测器检出但 caption 未提及的显著对象占比）。因为 caption 可以通过变得过度保守来避免幻觉，也可以通过增加细节来提升覆盖但引入无根据的断言 —— 只报幻觉率是可以被"少说话"刷分的，必须成对报告。这个"保真–信息量"权衡是 Discussion 的绝佳素材。

### (b) 回译自验证 EchoBack（建议作为方法创新点）

**思想**：一段好的描述应该包含足够重建原图的信息。把 `caption_dense` 喂给文生图模型 \(G\) 生成 \(N\) 张图，再用一个**与标注/检索都无关的第三方视觉编码器** \(f\)（DINOv2，纯视觉自监督，不受 CLIP 文本对齐偏置影响）算与原图的相似度：

\[
S_{\mathrm{RT}}(c, v) \;=\; \frac{1}{N}\sum_{k=1}^{N} \cos\!\Big( f(v),\; f\big(G(c;\,\epsilon_k)\big) \Big),
\qquad \hat v_k = G(c;\epsilon_k)
\]

**为什么原理上站得住**：这是把描述质量转成一个信息瓶颈问题 —— \(v \to c \to \hat v\)，\(S_{\mathrm{RT}}\) 度量的是经过语言瓶颈后保留的视觉信息量。它**完全无需人工参考**，且与 CLIPScore 的失效模式互补：CLIPScore、UMIC、PAC-S 这类无参考指标虽然与人类判断相关性高，但难以识别细粒度错误，尤其对"描述不合理"类错误敏感度有限；而回译对"缺失细节"和"数量/空间关系错误"很敏感（生成图会明显长歪）。

**必须诚实交代的局限**（写进 Limitations 反而加分）：
\(S_{\mathrm{RT}}\) 会受 \(G\) 自身能力上限的污染，也会奖励"符合 T2I 训练分布的描述风格"。缓解措施：

1. 用固定 seed 集合，只做**相对比较**（prompt A vs prompt B），不做绝对值解读；
2. 报告 \(S_{\mathrm{RT}}\) 与人工评分的 Kendall's \(\tau\) 相关性（在 50 张人工打分的子集上），证明它确实有效。

**顺带**：这个模块**一次性满足了作业里"以文生图"的形式要求**，而且是有目的的生成，不是硬凑。用 SDXL-Lightning 4-step（4070 上约 0.5～1 s/张 @768px）或 FLUX.1-schnell（Apache 2.0，4 步）都行；50 张图 × 4 samples ≈ 几分钟。

---

## M3–M6 · 混合检索

### 三路召回并行

\[
s(q,v) \;=\; \lambda_1 \cos\!\big(E^{\text{clip}}_T(q),\, E^{\text{clip}}_I(v)\big)
\;+\; \lambda_2 \cos\!\big(E^{\text{txt}}(q),\, E^{\text{txt}}(c_v)\big)
\;+\; \lambda_3 \,\widetilde{\mathrm{BM25}}(q,\, k_v)
\]

- **第一项 · 图像塔**：中文查询建议用 **Chinese-CLIP ViT-B/16**（达摩院，2 亿中文原生图文对训练）或 **jina-clip-v2**（支持 89 种语言的多语言图像检索，输入分辨率提升到 512×512，且支持 Matryoshka 表示，可把输出维度从 1024 截断到 64）。做一组对比实验本身就是一个 ablation。
- **第二项 · 结构化 caption 的文本塔**（BGE-M3 / Qwen3-Embedding）：这是本方案对 Immich 式纯 CLIP baseline 的核心增益来源。CLIP 文本塔通常只有 77 token 上下文，处理"傍晚、红伞、店招写着面馆、有两个人"这种**长复合查询**时会退化，而文本塔能吃下完整长句。
- **第三项 · BM25** over `keywords_zh/en + ocr.text`：负责精确词命中（尤其是店招文字）。

### 融合用 RRF

避免三路分数量纲不可比，且无需调参：

\[
\mathrm{RRF}(v) \;=\; \sum_{r \in \{\text{clip},\,\text{txt},\,\text{bm25}\}} \frac{1}{k + \mathrm{rank}_r(v)}, \qquad k = 60
\]

同时做一组"RRF vs 归一化加权和（调 \(\lambda\)）"的对比。

### VLM Reranker

对 top-20 用 Qwen3-VL 做 **listwise** 打分（一次把 20 张缩略图 + 查询喂进去，输出排序），或用 pointwise 的 VQAScore 思路：

\[
s_{\text{rerank}}(q, v) \;=\; P_\theta\big(\text{"yes"} \mid v,\; \text{"这张图符合描述：} q \text{ 吗？"}\big)
\]

pointwise 更稳定、可并行、显存友好；listwise 效果通常更好但受位置偏置影响。**两个都做，写进消融表。** 这类"MLLM 作为强 reranker"的思路有现成工作可引（RagVL）。

### 查询理解 Agent（M4）

把自然语言查询拆成：

- `semantic_text` → 送稠密检索
- `hard_filters` → 场景类型、时间、颜色、数量、OCR 关键词，走结构化过滤
- `query_type` → 简单 / 组合 / 否定 / 计数

这一步用免费 LLM API 即可，是**路由分发**编排模式的实例。

否定查询（"没有人的街景"）纯向量检索几乎必挂，靠结构化字段过滤能救 —— 这是一个非常好的 case study。

---

## M7 · 输出 Agent（交互与创意层）

三个输出头共享检索结果：

1. **带引用的图文问答**
   对 top-k 图做 VLM 阅读，回答必须标注 `[img_0731]` 形式的引用；无法回答时显式说"检索结果中没有依据"。直接对标 Ask Photos 的行为，也是抗幻觉的产品级设计。

2. **图文游记 / 日记生成**
   用户选 3～8 张图 → 按 `capture.time_of_day` 与场景相似度做序列编排 → 生成带小标题的图文并茂游记。可引 Visual Storytelling (VIST) 作为 Related Work。

3. **缺图补全**
   游记里若某个叙事节点没有对应实拍图，用 SDXL / FLUX 按风格一致的 prompt 生成插图，并**明确标注"AI 生成"**。

---

# 六、实验设计

对应报告的 Experiment / Result 章。

## 6.1 评测集怎么造（有个陷阱，务必按此破解）

**陷阱**：如果检索评测的 query 是由你们自己的标注生成的，那"用标注做检索"当然赢 —— **循环论证**，会被一眼看穿。

**破解（三管齐下，写进报告会显著加分）**：

| 评测集 | 构造方式 | 规模 | 用途 |
|---|---|---|---|
| **人工 query 集** | 由**未参与标注模块**的 2 名成员对着原图直接写查询，每人 50 条 | 100 | 主结果，无污染 |
| **异源 query 集** | 用 GLM-4V-Flash（与主标注模型 Qwen3-VL 不同源）看图生成查询，人工过滤 | 200 | 规模化补充 |
| **困难 query 集** | 组合条件（颜色+数量+空间关系）、否定、OCR 精确匹配 | 100 | 分层分析 |
| **人工参考描述集** | 4 人 × 每人 25 张 × 2 条 = 200 条 ref，覆盖 100 张 val 图（每张图由 2 人各写 1 条，交叉覆盖） | 100 图 | CIDEr 计算 |
| **VQA 集** | 由标注自动生成四选一，**全部人工校验并改错** | 200 题 | Accuracy + WUPS |
| **Arena 对比集** | 60 组 pairwise（不同 prompt / 不同模型的描述），4 人全员盲投 | 4 × 60 = 240 票 | 人工评估 |

人工工作量：每人约 50 条 ref + 60 票 + 50 条 query 校验，约一个下午。

> 4 人打分时 Fleiss' \(\kappa\) 的评分者数 \(n=4\)，仍在可报告范围内；但**每组 pairwise 必须四人全投**（而非分摊），否则 \(\kappa\) 无法计算。投票界面须随机化左右顺序并隐藏来源，避免位置偏置与身份泄露。

## 6.2 指标与公式（全部是课上讲过的，刻意对齐）

### 图像描述 · CIDEr（课程 60–68 页详讲）

\[
\mathrm{CIDEr}_n(c_i, S_i) \;=\; \frac{1}{M}\sum_{j=1}^{M} \frac{\mathbf{g}^n(c_i)^{\top}\,\mathbf{g}^n(s_{ij})}{\lVert \mathbf{g}^n(c_i)\rVert \; \lVert \mathbf{g}^n(s_{ij})\rVert}
\]

\[
\mathrm{CIDEr} = \sum_{n=1}^{4} w_n\,\mathrm{CIDEr}_n
\]

其中 \(\mathbf{g}^n(\cdot)\) 是 n-gram 的 TF-IDF 向量，\(w_n = 1/4\)。

> **注意**：CIDEr 是为短 caption 设计的，对 100+ 字的 dense caption 会失真，所以只在 `caption_short` 上算，并在报告里说明这一点。这体现你们真懂指标，而不是套公式。

### 无参考补充 · CLIPScore（Hessel et al., 2021）

\[
\mathrm{CLIP\text{-}S}(c, v) \;=\; w \cdot \max\!\big(\cos(E_T(c), E_I(v)),\, 0\big), \qquad w = 2.5
\]

### 幻觉

\(\mathrm{CHAIR}_i,\ \mathrm{CHAIR}_s\)（见 M2a）

### 回译

\(S_{\mathrm{RT}}\)（见 M2b）

### 检索 · MRR 与 NDCG（课程 69–70 页）

\[
\mathrm{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\mathrm{rank}_q}
\]

\[
\mathrm{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_2(i+1)},
\qquad
\mathrm{NDCG@K} = \frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}}
\]

单正例查询报 R@1/5/10 与 MRR；对 50 条人工标了三级相关性（0/1/2）的查询报 NDCG@10。

### VQA · Accuracy + WUPS（课程 71–77 页）

\[
\mathrm{WUP}(a,b) = \frac{2\,\mathrm{depth}\big(\mathrm{LCS}(a,b)\big)}{\mathrm{depth}(a) + \mathrm{depth}(b)}
\]

WUPS 需要 WordNet，所以只对英文关键词答案算；中文部分用 embedding 相似度替代并说明理由。

### 生成质量

LLM-as-judge（TouchStone 范式，课程 94–95 页）+ 人工 Arena（LVLM-eHub 范式，课程 100–102 页）。Arena 用 Bradley-Terry 拟合排名：

\[
P(i \succ j) = \frac{e^{\theta_i}}{e^{\theta_i} + e^{\theta_j}}
\]

并报告评分者间一致性 Fleiss' \(\kappa\)。

## 6.3 必做的消融实验（这张表就是 Result 章的骨架）

| # | 消融维度 | 对照组 |
|---|---|---|
| A1 | 标注 prompt | 朴素 "describe this image" / 单段 dense caption / **本文结构化 schema** |
| A2 | M0 弱路由 | 统一模板 / **主辅内容路由 + modifier 观察补丁** |
| A3 | 自洽采样 | 单次 \(T=0\) / \(M=3\) 投票 / +跨模型交叉验证 |
| A4 | 标注模型 | Qwen3-VL-4B / 8B / GLM-4V-Flash / 多模型融合 |
| A5 | 检索路数 | 仅 CLIP（≈Immich baseline）/ 仅文本 / 仅 BM25 / **RRF 三路** |
| A6 | Reranker | 无 / pointwise VQAScore / listwise VLM |
| A7 | 图像编码器 | Chinese-CLIP ViT-B/16 vs jina-clip-v2 |
| A8 | 幻觉阈值 \(\tau\) | 0.2 / 0.3 / 0.4 敏感性 |
| A9 | 领域迁移 | 真实场景 / 抽象画 / 中国古画 / 医学影像 |

**A5 是核心结果**：预期混合检索在困难查询上大幅超越纯 CLIP，在简单查询上差距很小 —— 这个"何时有用、何时没用"的结论比"我们更好"有价值得多。

**A9 的讨论方向**：抽象画上 `objects` 字段大量失效但 `affect` / `palette` 仍可用；古画上 OCR 会把题跋识别成乱码；医学影像上 VLM 会给出**自信但错误**的解读 —— 这正是课程 108–110 页"经验法则而非世界模型"的绝佳实证。

> ⚠️ 医学影像部分必须声明**不用于任何诊断用途**。

---

# 七、分工、时间线与交付

## 7.1 四人分工

原五人方案中的「系统与实验组织」角色不单独设人，而是按**天然邻接关系**拆解：Docker / 依赖 / 部署文档并入检索岗（他管着索引构建，本来就要处理环境），Demo 前后端并入 Agent 岗（Demo 就是 Agent 层的可视化外壳），Arena 平台与人工评估组织并入评估岗（他定义指标，理应组织打分）。

| 角色 | 核心模块 | 附加职责 | 个人报告主题 |
|---|---|---|---|
| **A｜标注与 Prompt** | M0 粗粒度路由、M1 结构化标注引擎 | 数据下载/去重（hash+pHash）、schema 定稿、vLLM 服务部署、API 客户端与限流重试、P0-P3 Prompt 对比 | Prompt Engineering 的量化方法论：从 schema 设计到版本化实验 |
| **B｜验证与评估** | M2 CHAIR + EchoBack | 全部指标实现（CIDEr / CLIPScore / WUPS / MRR / NDCG 的统一 `eval` 模块）、T2I 推理封装（`t2i.py`，D 复用）、Arena 平台与人工打分组织、A9 领域迁移实验 | 无参考图像描述质量度量与幻觉量化 |
| **C｜检索与索引** | M3 多路索引、M5 混合召回+RRF、M6 Reranker | 检索评测集构造与相关性标注规范、A5–A7 消融、Dockerfile / requirements / `DEPLOY.md`、`run_all.sh` 一键流水线 | 混合多模态检索：三路召回与重排的消融分析 |
| **D｜Agent、生成与系统** | M4 查询理解 Agent、M7 三个输出头 | Gradio/Streamlit + FastAPI Demo、README、系统架构图、演示视频、多 Agent 编排图 | 多 Agent 编排模式与检索增强下的幻觉抑制 |

**四人共同承担**（不计入上表，须写进日程）：

- 人工标注众包：参考描述 ref、query 校验、VQA 校验、Arena 投票（详见 6.1，每人约半天）
- LaTeX 报告：按 §7.4 的章节映射各写各的，最后由 D 统稿排版
- 每周一次 30 分钟同步 + 接口约定（`annotations.jsonl` 的 schema 一旦定稿即冻结，任何改动需全组确认）

**关键接口契约**（四人并行开发的前提，务必 W1 就定死）：

| 生产者 | 产物 | 消费者 |
|---|---|---|
| A | `annotations.jsonl`（每行一个 §M1 的 JSON） | B、C、D |
| B | `t2i.generate(prompt, seed) -> PIL.Image` | D（缺图补全） |
| B | `verify_report.json`（每图 CHAIR / \(S_{\mathrm{RT}}\) / 字段一致度） | A（作为 prompt 进化的 reward） |
| C | `search(query, topk) -> List[image_id, score]` | D（Agent 调用） |

> ⚠️ A 与 B 之间是**闭环依赖**（A 产标注 → B 算分 → A 调 prompt）。W1 必须先用 200 张小样把这个环跑通，否则 W3 的 prompt 进化会卡死。

## 7.2 四周时间线（按人分解）

| 周 | A｜标注 | B｜评估 | C｜检索 | D｜Agent |
|---|---|---|---|---|
| **W1** | 数据去重、schema 定稿、vLLM 起服务、200 张小样跑通 | CLIPScore/CIDEr 先行实现、检测器选型验证 | CLIP/BGE-M3 编码打通、FAISS 建索引 | Demo 框架搭壳、免费 LLM API 打通、系统图初稿 |
| **W2** | 全量标注（3 轮采样 + 跨模型交叉） | CHAIR + EchoBack 落地、领域迁移数据采集 | 三路召回 + RRF、检索评测集构造 | 查询理解 Agent、叙事生成头 |
| **W3** | prompt 进化搜索（8–12 代）、A1–A4 消融 | A8/A9 实验、指标与人工评分相关性分析 | A5–A7 消融、Reranker、Docker | 输出三头联调、缺图补全、Demo 打磨 |
| **W4** | 写 Method §标注、补附录 prompt 全文 | 组织 Arena 投票、出全部图表 | 写 Experiment §检索、部署文档 | 统稿排版、录视频、README |

**里程碑**：
- W1 末 —— 200 张图的**端到端最小闭环**（标注 → 索引 → 检索 → 出一个分数）
- W2 末 —— 全量 2569 张标注完成，检索可用
- W3 末 —— **实验冻结**，此后只写作不跑新实验

> **留出 W4 全周写报告** —— LaTeX 排版与绘图明确是评分项，别压到最后两天。图表建议统一用 matplotlib + 一套配色，系统图用 draw.io 或 TikZ。

## 7.3 报告章节与人员映射（避免四人写出四种文风）

| 章节 | 主笔 | 说明 |
|---|---|---|
| Abstract / Introduction / Conclusion | D | 统稿人写头尾，保证叙事一致 |
| Related Work | 四人各供 1 段（标注/评估/检索/Agent），D 合并 | 每人只写自己熟的那 8–10 篇 |
| Method | A（M0/M1）+ B（M2）+ C（M3–M6）+ D（M4/M7） | 按模块切，最自然 |
| Experiment / Result | B 定指标口径与图表模板，C 写检索部分，A 写标注部分 | **图表模板必须先统一** |
| Discussion / Limitations | B | 幻觉、迁移失效、指标局限都在他手里 |
| 排版、参考文献、附录 | D | 统一 BibTeX，附录放大表与 prompt 全文 |

## 7.4 仓库结构（对应"完整 ReadMe / 环境依赖 / 部署文档"）

```
askalbum/
├── README.md              # 快速开始 / 系统图 / 结果表
├── docs/DEPLOY.md         # Docker + 无 GPU 降级路径（纯免费 API 模式）
├── environment.yml, requirements.txt, Dockerfile
├── configs/               # 模型、路径、超参全部外置为 yaml
├── prompts/               # 版本化的 prompt 模板 + 进化搜索日志
├── src/
│   ├── annotate/   route.py  vlm_client.py  schema.py  vote.py
│   ├── verify/     chair.py  echoback.py
│   ├── index/      embed.py  bm25.py  build.py
│   ├── search/     fuse.py   rerank.py  agent.py
│   ├── generate/   narrate.py  t2i.py
│   └── eval/       cider.py  clipscore.py  retrieval.py  vqa.py  arena.py
├── scripts/run_all.sh     # ★ 对新目录一键跑通（应对 200 张隐藏测试集）
└── data/                  # .gitignore，只留 sample
```

---

# 八、备选方案

## 方案 B｜「见景生情」图像 → 中文诗歌 / 微散文

**主线**：情绪与美学多维标注 → 检索式诗歌 RAG（建一个古诗词/现代诗库，用意象关键词检索，避免 LLM 硬写打油诗）→ 风格化配图。

**评估**：LLM-as-judge + 人工 Arena + 意象一致性（CLIPScore of 诗中意象 vs 原图）。

- 优点：轻、有趣、演示效果好
- 缺点：学术厚度不足，Experiment 章会很空
- 建议：作为主方案的第四个输出头

## 方案 C｜无参考图像描述评价指标研究

只做 M2，把 EchoBack 与 CLIPScore / CHAIR / LLM-judge 在人工打分上做相关性对比，写成一篇小论文。

- 优点：最像学术论文
- 缺点：不满足"具备实用性或趣味性的小型系统/应用"的期待，且没有交互 Demo，演示吃亏

## 结论

**主推 A，把 B 作为输出头，把 C 作为 Method 的创新点。** 三者合一，形式覆盖最全、学术性最强。

**四人规模下的取舍**：若 W2 结束时进度落后，按以下优先级砍功能，**不要砍评估**（评估是本方案的差异化所在，也是老师的评分重心）：

1. 先砍 **B 方案的诗歌输出头**（M7 的第四头，纯锦上添花）
2. 再砍 **M7 的缺图补全**（EchoBack 已经覆盖了"以文生图"的形式要求）
3. 再把 **M6 Reranker 从 listwise 简化为 pointwise**（少写一半代码，效果损失有限，且仍能进消融表）
4. 最后把 **prompt 进化搜索从自动进化降级为手工 3 版对比**（仍能画出"prompt 版本 vs 指标"的图）

核心闭环 **M1 结构化标注 → M2 验证 → M3–M5 混合检索 → M7 问答** 无论如何要保住，这是报告的主干。

---

# 九、风险清单与兜底

| 风险 | 概率 | 兜底 |
|---|---|---|
| 4090 长时间不可用 | 中 | 降级到 Qwen3-VL-4B AWQ 在 4070 上跑（约 3.5 GB @Q4）+ 免费 API 分担；标注是离线一次性任务，可分批断点续跑（`--resume`） |
| 免费 API 限流 / 政策变动 | 高 | 多平台轮询 + 指数退避 + 本地缓存（按 image_hash 落盘，绝不重复调用）；准备纯本地全离线路径 |
| GLM-4V-Flash 不收 base64 | 中 | 临时静态图床或对象存储；或整条交叉验证换用其它免费 VLM |
| 中文 CLIP 检索效果不佳 | 中 | 混合检索的文本塔本来就是主力，CLIP 只是一路召回；再不行让 M4 把中文查询翻译成英文走英文 CLIP |
| 自造评测集被质疑循环论证 | 高 | 6.1 的三管齐下；主结果只用人工 query 集 |
| SDXL 在 8G 上 OOM | 中 | `--medvram`，或用 SDXL-Lightning 4-step @768px；实在不行 FLUX GGUF Q4_K_S（约 6.8 GB）+ `--lowvram`，并注意用**量化版 T5 编码器**（fp16 T5 单独就约 9 GB，8G 卡塞不下） |
| 报告超 20 页 | 高 | 消融大表和 prompt 全文放附录（不计页数） |

---

# 十、参考资料清单

## 官方文档 / 免费 API

1. 智谱 BigModel 开放平台（GLM-4V-Flash 免费多模态）— https://bigmodel.cn
   说明文：https://zhuanlan.zhihu.com/p/12036071605
2. 阿里云百炼 / DashScope（Qwen-VL 系列免费额度）— https://bailian.console.aliyun.com
3. Google AI Studio（Gemini Flash 免费层）— https://aistudio.google.com
4. 硅基流动 SiliconFlow（部分模型免费）— https://siliconflow.cn
5. 免费 LLM API 追踪 — https://freellm.net
   2026 汇总：https://www.17you.com/freeresources/free-llm-api-guide-2026

## 模型

6. Qwen3-VL（2B/4B/8B/32B，Apache 2.0）— https://github.com/QwenLM/Qwen3-VL
7. Qwen2.5-VL — https://github.com/QwenLM/Qwen2.5-VL
8. Chinese-CLIP（达摩院，2 亿中文图文对）— https://github.com/OFA-Sys/Chinese-CLIP
   论文：https://arxiv.org/abs/2211.01335
9. jina-clip-v2（89 语言、512×512、Matryoshka）— https://jina.ai/news/jina-clip-v2-multilingual-multimodal-embeddings-for-text-and-images/
   论文：https://arxiv.org/abs/2412.08802
10. BGE-M3 — https://github.com/FlagOpen/FlagEmbedding
11. DINOv2（回译评估的第三方视觉编码器）— https://github.com/facebookresearch/dinov2
12. GroundingDINO — https://github.com/IDEA-Research/GroundingDINO
    YOLO-World — https://github.com/AILab-CVC/YOLO-World
13. SDXL-Lightning — https://huggingface.co/ByteDance/SDXL-Lightning
    FLUX.1-schnell（Apache 2.0）— https://huggingface.co/black-forest-labs/FLUX.1-schnell
14. ComfyUI-GGUF（低显存跑 FLUX）— https://github.com/city96/ComfyUI-GGUF
15. vLLM（guided JSON 解码、批量 VLM 推理）— https://github.com/vllm-project/vllm

## 对标产品 / 系统

16. Google Photos "Ask Photos" 官方说明 — https://support.google.com/photos/answer/15318661
17. Google 官方博客（Ask Photos 发布）— https://blog.google/products-and-platforms/products/photos/google-ask-photos-early-access/
18. Immich（自托管 Google Photos 替代，CLIP + pgvector 智能搜索）— https://github.com/immich-app/immich
    架构解析：https://deepwiki.com/immich-app/immich/4.4-search-and-discovery
19. immich-go-analyze（用本地 Ollama VLM 给相册自动打标签，与 M1 思路一致的轻量实现）— https://github.com/seconion/immich-go-analyze
20. PhotoPrism（另一个自托管相册，可做对比）— https://github.com/photoprism/photoprism

## 论文（Related Work 直接可引）

21. CLIPScore: A Reference-free Evaluation Metric for Image Captioning — https://aclanthology.org/2021.emnlp-main.595
22. CHAIR（Object Hallucination in Image Captioning）— https://arxiv.org/abs/1809.02156
23. CapArena: Benchmarking and Analyzing Detailed Image Captioning — https://aclanthology.org/2025.findings-acl.724.pdf
24. SPECS: Specificity-Enhanced CLIPScore for Long Image Caption Evaluation — https://arxiv.org/abs/2509.03897
25. An Examination of the Robustness of Reference-Free Image Captioning Evaluation Metrics — https://arxiv.org/abs/2305.14998
26. ALOHa: A New Measure for Hallucination in Captioning Models — https://arxiv.org/abs/2404.02904
27. ShareGPT4V — https://arxiv.org/abs/2311.12793
28. DenseFusion-1M（用视觉专家模型辅助 VLM 产出超详细描述，与 M2 验证思路同源）— https://arxiv.org/abs/2407.08303
29. From Pixels to Prose / PixelProse — https://arxiv.org/abs/2406.10328
30. RagVL: MLLM Is a Strong Reranker — https://github.com/DataArcTech/RagVL
31. Multimodal RAG Survey — https://github.com/llm-lab-org/Multimodal-RAG-Survey
32. Visual Storytelling (VIST) 数据集与任务 — http://visionandlanguage.net/VIST/
33. WUPS 原始定义（VQA 评价，课程 71 页引用）— https://arxiv.org/abs/1410.0210
34. MM-Vet — https://github.com/yuweihao/MM-Vet
    MME — https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models
    SEED-Bench — https://github.com/AILab-CVC/SEED-Bench
35. TouchStone — https://github.com/OFA-Sys/TouchStone
    LVLM-eHub — https://github.com/OpenGVLab/Multi-Modality-Arena

## 工程参考

36. 8GB 显存跑 Stable Diffusion / FLUX 实操 — https://localaimaster.com/blog/run-flux-on-low-vram-gpu
37. 本地小 VLM 选型（VRAM 分档）— https://tinyweights.dev/posts/best-local-vision-language-models-2026/
38. 视觉故事生成参考实现 — https://github.com/Pendulibrium/ai-visual-storytelling-seq2seq
    https://github.com/SartajBhuvaji/PictureTales

---

## 后续可继续深化的两件事

1. **把 M1 的完整 prompt 模板逐字写出来**（含 JSON Schema、few-shot、负面约束清单）—— W1 就要用的东西。
2. **把 6.3 的消融表填成一张空白 LaTeX 表格骨架** —— 让实验一开跑就有地方填数。
