# Intelligent Image-Text Search and Content Generation Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 2000 张无标注训练图片和 369 张验证图片，构建一个支持复杂中文自然语言搜图、视觉重排、匹配解释、图片问答、文案生成和 Stable Diffusion 补图的可复现实验系统。

**Architecture:** 使用 Qwen3-VL-2B 为图片生成多维结构化伪标注；以 BM25、中文文本向量和结构化字段完成多路召回，再通过 RRF 融合和 Qwen3-VL 视觉重排得到最终结果。训练一个轻量文本检索模型，Stable Diffusion 仅在用户主动请求或真实图库匹配不足时生成补充图片，所有离线产物均带版本和哈希信息。

**Tech Stack:** Python 3.11、PyTorch/CUDA、Transformers、Qwen3-VL-2B-Instruct、Sentence Transformers、BGE Small Chinese、FAISS、BM25、Pydantic、SQLite、Diffusers、Stable Diffusion、Gradio、Pytest、LaTeX。

---

## 1. 已知条件与固定决策

- 工作目录：`E:\programs\Anima`
- 训练图片：`Train/*.jpg`，共 2000 张。
- 验证图片：`Val/*.jpg`，共 369 张。
- 视觉语言模型：`Qwen--Qwen3-VL-2B-Instruct/snapshots/master`。
- 图像生成模型：`stablediffusion/`。
- GPU：NVIDIA GeForce RTX 4050 Laptop，6 GB 显存。
- `Train` 可用于伪标注生成、困难负样本构造和模型训练。
- `Val` 仅用于最终评测，不参与模型训练或阈值拟合。
- 不微调 Qwen3-VL；训练对象为小型文本向量模型或轻量排序模型。
- Qwen3-VL 与 Stable Diffusion 不同时驻留显存。
- 真实图片和 AI 生成图片在界面、索引和评测数据中必须明确区分。
- 当前目录不是 Git 仓库；计划中的提交命令在初始化 Git 后执行。

## 2. 最终用户流程

```text
用户查询
  -> 查询结构化解析
  -> BM25 / 稠密向量 / 字段过滤并行召回
  -> RRF 融合候选
  -> Qwen3-VL 查看 Top-N 图片并重排
  -> 返回真实图片、匹配证据和不匹配项
  -> 可选：图片问答、生成文案、生成相似图片
```

当最高相关度低于设定阈值时，系统只提示“图库匹配较弱”，由用户决定是否调用 Stable Diffusion，不自动用生成图冒充检索结果。

## 3. 目标目录结构

```text
Anima/
  Train/
  Val/
  Qwen--Qwen3-VL-2B-Instruct/
  stablediffusion/
  models/
    bge-small-zh-v1.5/
  configs/
    default.yaml
    prompts/
      caption_basic.txt
      caption_structured.txt
      caption_verified.txt
      query_parser.txt
      reranker.txt
      content_writer.txt
      sd_prompt.txt
  src/anima_search/
    config.py
    schemas.py
    data/manifest.py
    annotation/qwen_client.py
    annotation/pipeline.py
    annotation/validation.py
    indexing/documents.py
    indexing/bm25_index.py
    indexing/vector_index.py
    retrieval/query_parser.py
    retrieval/fusion.py
    retrieval/search.py
    retrieval/reranker.py
    training/pairs.py
    training/train_embedder.py
    generation/prompt_builder.py
    generation/sd_generator.py
    runtime/model_manager.py
    evaluation/ground_truth.py
    evaluation/metrics.py
    evaluation/ablation.py
    app/service.py
    app/ui.py
  scripts/
    build_manifest.py
    annotate_images.py
    compare_prompts.py
    build_indexes.py
    build_training_pairs.py
    train_retriever.py
    evaluate_retrieval.py
    run_ablation.py
    launch_app.py
  tests/
    fixtures/
    unit/
    integration/
  artifacts/
    manifests/
    annotations/
    indexes/
    checkpoints/
    evaluation/
    generated/
  report/
    main.tex
    sections/
    figures/
  README.md
  DEPLOYMENT.md
  plan.md
```

`artifacts/` 中的大文件不提交到 Git；只提交小型示例、指标 JSON、图表和生成这些产物的脚本。

## 4. 核心数据契约

图片标注采用一行一个对象的 JSONL。核心模型应在 `src/anima_search/schemas.py` 中定义：

```python
from pydantic import BaseModel, Field


class ImageAnnotation(BaseModel):
    image_id: str
    split: str
    relative_path: str
    sha256: str
    summary: str
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    scene: str
    attributes: list[str] = Field(default_factory=list)
    spatial_relations: list[str] = Field(default_factory=list)
    style: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    ocr_text: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(min_length=3)
    generation_prompt: str
    uncertainty: list[str] = Field(default_factory=list)
    model_version: str
    prompt_version: str


class SearchQuery(BaseModel):
    raw_text: str
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    scene: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    style: list[str] = Field(default_factory=list)
    required_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    image_id: str
    relative_path: str
    fused_score: float
    rerank_score: float | None = None
    evidence: list[str] = Field(default_factory=list)
    mismatch: list[str] = Field(default_factory=list)
    source: str = "real"
```

接口字段一旦进入批量标注阶段即冻结；如必须修改，提升 `prompt_version` 并重新生成相应产物，禁止静默混用不同 Schema。

## 5. 实施任务

### Task 1：建立兼容环境和项目骨架

**Files:**
- Modify: `pixi.toml`
- Modify: `.gitignore`
- Create: `src/anima_search/__init__.py`
- Create: `configs/default.yaml`
- Create: `tests/unit/test_environment.py`

- [ ] 将 Python 约束改为 `>=3.11,<3.12`，加入 PyTorch、Transformers、Accelerate、Sentence Transformers、FAISS CPU、Pydantic、PyYAML、Pillow、rank-bm25、Diffusers、Gradio、pandas、scikit-learn、matplotlib、seaborn 和 pytest。
- [ ] 在 `configs/default.yaml` 写入两个现有模型的相对路径、Train/Val 路径、产物目录、随机种子 `20260802`、标注温度 `0.1`、召回数量 `50`、重排数量 `10`。
- [ ] 在 `.gitignore` 忽略 `.pixi/`、`artifacts/annotations/`、`artifacts/indexes/`、`artifacts/checkpoints/`、`artifacts/generated/` 和 Python 缓存。
- [ ] 编写环境测试，断言 CUDA 可用、Qwen 配置存在、SD `model_index.json` 存在、Train 为 2000 张且 Val 为 369 张。
- [ ] 运行：`pixi install`。预期：依赖解析完成并更新 `pixi.lock`。
- [ ] 运行：`pixi run pytest tests/unit/test_environment.py -v`。预期：全部通过。
- [ ] 建议提交：`git add pixi.toml pixi.lock .gitignore configs src tests && git commit -m "build: establish reproducible runtime"`。

### Task 2：生成数据清单并检查数据质量

**Files:**
- Create: `src/anima_search/data/manifest.py`
- Create: `scripts/build_manifest.py`
- Create: `tests/unit/test_manifest.py`
- Output: `artifacts/manifests/train.jsonl`
- Output: `artifacts/manifests/val.jsonl`

- [ ] 先编写测试：临时创建正常 JPG、损坏文件和重复文件，断言扫描器返回尺寸、模式、SHA-256、损坏状态和重复组。
- [ ] 实现 `scan_split(root: Path, split: str) -> list[ManifestItem]`，按数字文件名排序，不依赖操作系统枚举顺序。
- [ ] 实现跨 Train/Val 的哈希重复检测；发现跨集合重复时保留记录，但从最终评测查询集中排除对应 Val 图片。
- [ ] 运行：`pixi run python scripts/build_manifest.py --config configs/default.yaml`。
- [ ] 验证：Train 清单 2000 条、Val 清单 369 条、每条图片均可解码；质量报告保存到 `artifacts/manifests/quality_report.json`。
- [ ] 运行：`pixi run pytest tests/unit/test_manifest.py -v`。预期：全部通过。
- [ ] 建议提交：`git add src/anima_search/data scripts/build_manifest.py tests/unit/test_manifest.py && git commit -m "feat: add deterministic dataset manifest"`。

### Task 3：定义标注 Schema 和三套 Prompt

**Files:**
- Create: `src/anima_search/schemas.py`
- Create: `src/anima_search/annotation/validation.py`
- Create: `configs/prompts/caption_basic.txt`
- Create: `configs/prompts/caption_structured.txt`
- Create: `configs/prompts/caption_verified.txt`
- Create: `tests/unit/test_annotation_schema.py`

- [ ] 编写失败测试，覆盖合法标注、缺少 `summary`、少于三条 `search_queries`、错误 split 和额外解释文字包围 JSON 的情况。
- [ ] 实现本计划第 4 节的数据模型，以及从模型输出中抽取首个合法 JSON 对象的解析器。
- [ ] P1 只要求一句客观描述；P2 一次性输出完整结构化 JSON；P3 要求先观察、再核对对象数量、空间关系、OCR 和不确定项，最终仅输出 JSON。
- [ ] 三套 Prompt 均明确禁止猜测人物身份、地点名称、品牌和不可见事件，并要求将不确定内容写入 `uncertainty`。
- [ ] 运行：`pixi run pytest tests/unit/test_annotation_schema.py -v`。预期：全部通过。
- [ ] 建议提交：`git add src/anima_search/schemas.py src/anima_search/annotation configs/prompts tests/unit/test_annotation_schema.py && git commit -m "feat: define versioned annotation contract"`。

### Task 4：实现 Qwen3-VL 推理适配器和批量标注

**Files:**
- Create: `src/anima_search/annotation/qwen_client.py`
- Create: `src/anima_search/annotation/pipeline.py`
- Create: `scripts/annotate_images.py`
- Create: `tests/unit/test_annotation_pipeline.py`
- Output: `artifacts/annotations/{split}.{prompt_version}.jsonl`

- [ ] 用假模型编写测试，覆盖成功解析、第一次非法 JSON 后重试、图片读取失败、已存在记录跳过和中断后恢复。
- [ ] 实现 `QwenVLClient.generate(image, prompt) -> str`，以 BF16 或 FP16、`device_map="auto"` 加载本地模型，推理入口不包含业务 Schema 逻辑。
- [ ] 实现逐条追加 JSONL、图片哈希去重、最多两次纠错重试和失败记录文件。
- [ ] 每条结果保存模型路径摘要、Prompt SHA-256、生成参数、耗时和峰值显存。
- [ ] 先运行 10 张冒烟测试：`pixi run python scripts/annotate_images.py --split Train --limit 10 --prompt caption_structured`。
- [ ] 检查 10 条均可通过 Pydantic 校验，再运行完整 Train 和 Val 标注。
- [ ] 运行：`pixi run pytest tests/unit/test_annotation_pipeline.py -v`。预期：全部通过。
- [ ] 建议提交：`git add src/anima_search/annotation scripts/annotate_images.py tests/unit/test_annotation_pipeline.py && git commit -m "feat: add resumable Qwen annotation pipeline"`。

### Task 5：完成 Prompt Engineering 对比实验

**Files:**
- Create: `scripts/compare_prompts.py`
- Create: `src/anima_search/evaluation/annotation_quality.py`
- Create: `tests/unit/test_annotation_quality.py`
- Output: `artifacts/evaluation/prompt_comparison.csv`
- Output: `artifacts/evaluation/prompt_comparison_summary.json`

- [ ] 从 Train 固定抽取 60 张分层样本，覆盖人物、动物、食物、建筑、自然、室内、夜景、文字和复杂关系。
- [ ] 对同一批图片运行 P1/P2/P3，固定随机种子和解码参数。
- [ ] 由两名成员分别对事实正确性、完整度、幻觉、可检索性按 1–5 分评分。
- [ ] 计算 JSON 有效率、平均字段填充率、平均描述长度、人工评分均值和标注者一致性。
- [ ] 选择综合得分最高的 Prompt 作为主标注版本；实验结果保留三种版本，不删除较弱版本。
- [ ] 运行：`pixi run python scripts/compare_prompts.py --sample-size 60`。预期：生成 CSV、汇总 JSON 和对比图。
- [ ] 建议提交：`git add scripts/compare_prompts.py src/anima_search/evaluation tests/unit/test_annotation_quality.py artifacts/evaluation && git commit -m "experiment: compare annotation prompts"`。

### Task 6：建立 BM25 与基础稠密向量索引

**Files:**
- Create: `src/anima_search/indexing/documents.py`
- Create: `src/anima_search/indexing/bm25_index.py`
- Create: `src/anima_search/indexing/vector_index.py`
- Create: `scripts/build_indexes.py`
- Create: `tests/unit/test_indexes.py`

- [ ] 编写测试数据，断言 `objects`、`scene`、`mood`、`ocr_text` 和 `summary` 被带字段前缀地组合为检索文档。
- [ ] 为 BM25 实现中文字符/词语兼容的确定性分词；OCR 和主体字段权重高于风格字段。
- [ ] 将 `models/bge-small-zh-v1.5` 作为默认向量模型路径，向量执行 L2 归一化并用 FAISS Inner Product 索引。
- [ ] 保存索引时同时保存图片 ID 顺序、模型路径、模型哈希、标注版本和构建参数。
- [ ] 运行：`pixi run python scripts/build_indexes.py --split Train` 和 `--split Val`。
- [ ] 运行：`pixi run pytest tests/unit/test_indexes.py -v`。预期：相同输入两次构建得到相同 ID 顺序，已知查询返回预期样本。
- [ ] 建议提交：`git add src/anima_search/indexing scripts/build_indexes.py tests/unit/test_indexes.py && git commit -m "feat: add sparse and dense indexes"`。

### Task 7：实现查询解析、字段过滤和 RRF 融合

**Files:**
- Create: `configs/prompts/query_parser.txt`
- Create: `src/anima_search/retrieval/query_parser.py`
- Create: `src/anima_search/retrieval/fusion.py`
- Create: `src/anima_search/retrieval/search.py`
- Create: `tests/unit/test_retrieval.py`

- [ ] 编写测试，输入“不要人物，找冷色调的雨夜城市”，断言 `excluded_terms` 含人物，scene 含城市，colors 含冷色，mood/attributes 含雨夜信息。
- [ ] 查询解析失败时退化为原始文本检索，不让单次格式错误导致搜索失败。
- [ ] 实现 `rrf(rankings, k=60)`，同一图片在多路结果中的贡献相加，结果按分数和 image_id 稳定排序。
- [ ] 先分别取得 BM25 Top-50、向量 Top-50，再应用硬排除条件，最后融合为 Top-30。
- [ ] 返回结果保留各分支名次和分数，供消融实验与解释模块使用。
- [ ] 运行：`pixi run pytest tests/unit/test_retrieval.py -v`。预期：解析、排除、融合和稳定排序测试全部通过。
- [ ] 建议提交：`git add configs/prompts/query_parser.txt src/anima_search/retrieval tests/unit/test_retrieval.py && git commit -m "feat: implement hybrid candidate retrieval"`。

### Task 8：构造训练对并微调轻量检索模型

**Files:**
- Create: `src/anima_search/training/pairs.py`
- Create: `src/anima_search/training/train_embedder.py`
- Create: `scripts/build_training_pairs.py`
- Create: `scripts/train_retriever.py`
- Create: `tests/unit/test_training_pairs.py`
- Output: `artifacts/checkpoints/retriever/`

- [ ] 正样本使用 Train 图片的 `search_queries` 与结构化检索文档配对。
- [ ] 随机负样本从不同 scene 中抽取；困难负样本从同 scene 但 objects、actions 或 mood 不匹配的候选中抽取。
- [ ] 写测试断言训练对中不存在 Val ID、正负 ID 不相同、每个正样本至少包含一个困难负样本。
- [ ] 使用对比学习损失训练 BGE Small，保存最佳 nDCG@10 checkpoint；验证数据从 Train 内部固定划分，不使用课程 Val。
- [ ] 固定随机种子，保存训练配置、loss 曲线、checkpoint 哈希和基础模型信息。
- [ ] 运行：`pixi run python scripts/build_training_pairs.py --config configs/default.yaml`。
- [ ] 运行：`pixi run python scripts/train_retriever.py --epochs 3 --batch-size 16 --grad-accum 2`；若显存不足，将 batch size 降至 8 并将 grad accumulation 提至 4，保持有效 batch 不变。
- [ ] 对比基础 BGE 与微调 BGE 在 Train 内部验证集上的 Recall@K 和 nDCG@10，只有指标提升时才将微调模型设为应用默认值。
- [ ] 建议提交：`git add src/anima_search/training scripts/build_training_pairs.py scripts/train_retriever.py tests/unit/test_training_pairs.py && git commit -m "feat: train domain retrieval embedder"`。

### Task 9：实现 Qwen3-VL 视觉重排与匹配解释

**Files:**
- Create: `configs/prompts/reranker.txt`
- Create: `src/anima_search/retrieval/reranker.py`
- Create: `tests/unit/test_reranker.py`

- [ ] 定义输出格式：每张候选图片包含 0–100 相关度、可见证据、不匹配项和置信度。
- [ ] 用假模型测试候选顺序变化、非法分数钳制、缺失候选保留原融合顺序和模型失败降级。
- [ ] 仅对融合 Top-10 调用视觉重排，最终展示 Top-8，避免每次查询处理过多图片。
- [ ] 最终分数使用 `0.35 * normalized_rrf + 0.65 * vlm_score`，该权重只用 Train 内部验证集确定并写入配置。
- [ ] 缓存键包含查询文本、图片哈希、Prompt 哈希和模型版本。
- [ ] 运行：`pixi run pytest tests/unit/test_reranker.py -v`。预期：全部通过。
- [ ] 建议提交：`git add configs/prompts/reranker.txt src/anima_search/retrieval/reranker.py tests/unit/test_reranker.py && git commit -m "feat: add explainable visual reranking"`。

### Task 10：实现模型资源管理和 Stable Diffusion 补图

**Files:**
- Create: `src/anima_search/runtime/model_manager.py`
- Create: `configs/prompts/sd_prompt.txt`
- Create: `src/anima_search/generation/prompt_builder.py`
- Create: `src/anima_search/generation/sd_generator.py`
- Create: `tests/unit/test_model_manager.py`
- Create: `tests/unit/test_prompt_builder.py`

- [ ] 编写状态机测试，断言加载 SD 前释放 Qwen、生成完成后可释放 SD、重复请求复用当前模型、异常后状态恢复为 unloaded。
- [ ] Qwen 将中文查询和检索属性转换为英文正向 Prompt 与负向 Prompt，输出结构化 JSON。
- [ ] SD 以 FP16、attention slicing、VAE slicing 和 CPU offload 运行，默认 512×512、固定 seed 可复现。
- [ ] 输出 PNG 和同名 JSON；JSON 保存 query、prompt、negative prompt、seed、steps、guidance scale、模型路径和生成时间。
- [ ] 生成图保存到 `artifacts/generated/`，`source` 固定为 `generated`，默认不加入真实图片索引。
- [ ] 运行 1 张冒烟测试并用 `nvidia-smi` 记录峰值；预期不发生 CUDA OOM。
- [ ] 建议提交：`git add src/anima_search/runtime src/anima_search/generation configs/prompts/sd_prompt.txt tests/unit/test_model_manager.py tests/unit/test_prompt_builder.py && git commit -m "feat: add memory-aware image generation"`。

### Task 11：实现图片问答、文案生成和应用服务层

**Files:**
- Create: `configs/prompts/content_writer.txt`
- Create: `src/anima_search/app/service.py`
- Create: `tests/unit/test_service.py`

- [ ] 定义 `SearchService.search()`、`answer_about_image()`、`write_content()` 和 `generate_image()` 四个稳定入口。
- [ ] 图片问答必须基于选中图片回答；不能从图片确认的信息以“不确定”表达。
- [ ] 文案支持标题、朋友圈短文和微型故事三种模板，并允许选择正式、幽默、治愈三种语气。
- [ ] 服务层测试使用假检索器和假模型，验证参数传递、错误信息、缓存和 generated/real 来源标识。
- [ ] 运行：`pixi run pytest tests/unit/test_service.py -v`。预期：全部通过。
- [ ] 建议提交：`git add configs/prompts/content_writer.txt src/anima_search/app/service.py tests/unit/test_service.py && git commit -m "feat: expose search and generation services"`。

### Task 12：构建 Gradio Demo

**Files:**
- Create: `src/anima_search/app/ui.py`
- Create: `scripts/launch_app.py`
- Create: `tests/integration/test_app_smoke.py`

- [ ] 页面包含“智能搜索”“图片详情”“内容生成”“实验面板”四个标签页。
- [ ] 搜索页提供查询框、场景/情绪/颜色筛选、是否启用视觉重排开关和固定尺寸结果网格。
- [ ] 每个结果展示真实/生成标识、融合分数、重排分数、证据和不匹配项。
- [ ] 详情页支持点击图片后查看完整结构化标注、相似图片和图片问答。
- [ ] 内容生成页支持从当前图片生成文案，以及经确认后调用 SD 生成补图。
- [ ] 实验面板展示当前模型版本、Prompt 版本、索引版本和离线评测结果，不在页面内重新训练模型。
- [ ] 运行：`pixi run python scripts/launch_app.py --share false`。预期：本机地址可打开，搜索、详情、文案和补图流程均可完成。
- [ ] 运行：`pixi run pytest tests/integration/test_app_smoke.py -v`。预期：全部通过。
- [ ] 建议提交：`git add src/anima_search/app/ui.py scripts/launch_app.py tests/integration/test_app_smoke.py && git commit -m "feat: build interactive multimodal demo"`。

### Task 13：建立 Val 人工真值集和检索指标

**Files:**
- Create: `src/anima_search/evaluation/ground_truth.py`
- Create: `src/anima_search/evaluation/metrics.py`
- Create: `scripts/evaluate_retrieval.py`
- Create: `tests/unit/test_metrics.py`
- Output: `artifacts/evaluation/val_queries.jsonl`
- Output: `artifacts/evaluation/val_relevance.csv`

- [ ] 设计 100 条 Val 查询，覆盖主体、动作、场景、颜色、情绪、OCR、组合条件、否定条件和抽象描述。
- [ ] 每条查询对候选图片标注 0/1/2 三级相关度，由两名成员独立标注；分歧通过讨论形成最终标签。
- [ ] 测试 Recall@1/5/10、MRR、mAP 和 nDCG@10 的手算小样本，确保指标实现正确。
- [ ] 评测脚本只读取 Val 索引和冻结的 relevance 文件，并在结果中记录配置哈希。
- [ ] 同时记录平均查询耗时、P95 查询耗时和峰值显存。
- [ ] 运行：`pixi run python scripts/evaluate_retrieval.py --config configs/default.yaml`。
- [ ] 预期：生成 `retrieval_metrics.json`、逐查询结果 CSV 和失败案例列表。
- [ ] 建议提交：`git add src/anima_search/evaluation scripts/evaluate_retrieval.py tests/unit/test_metrics.py artifacts/evaluation/val_queries.jsonl artifacts/evaluation/val_relevance.csv && git commit -m "experiment: establish retrieval benchmark"`。

### Task 14：运行消融实验和生成质量评测

**Files:**
- Create: `src/anima_search/evaluation/ablation.py`
- Create: `scripts/run_ablation.py`
- Create: `tests/unit/test_ablation_matrix.py`
- Output: `artifacts/evaluation/ablation_results.csv`
- Output: `report/figures/*.pdf`

- [ ] 固定以下检索实验矩阵：P1/P2/P3；BM25/基础 BGE/混合检索；微调前/微调后；视觉重排关闭/开启。
- [ ] 测试矩阵生成器，断言每个配置组合只运行一次，断点恢复不会覆盖已有结果。
- [ ] 对 30–50 个生成请求人工评价图文一致性、视觉质量、风格符合度和整体偏好，每项 1–5 分。
- [ ] 保存代表性成功案例、失败案例和错误类型：对象遗漏、关系错误、OCR 错误、抽象情绪误判、重排反转错误、生成内容偏离。
- [ ] 使用固定配色生成检索指标柱状图、消融折线图、延迟/效果权衡图和人工评分图，导出矢量 PDF。
- [ ] 运行：`pixi run python scripts/run_ablation.py --resume`。预期：所有组合状态为 completed，表格与图可直接用于 LaTeX。
- [ ] 建议提交：`git add src/anima_search/evaluation/ablation.py scripts/run_ablation.py tests/unit/test_ablation_matrix.py artifacts/evaluation report/figures && git commit -m "experiment: complete ablation study"`。

### Task 15：完成系统级验证、文档和课程交付物

**Files:**
- Create: `README.md`
- Create: `DEPLOYMENT.md`
- Create: `tests/integration/test_end_to_end.py`
- Create: `report/main.tex`
- Create: `report/sections/abstract.tex`
- Create: `report/sections/introduction.tex`
- Create: `report/sections/related_work.tex`
- Create: `report/sections/method.tex`
- Create: `report/sections/experiments.tex`
- Create: `report/sections/results.tex`
- Create: `report/sections/conclusion.tex`

- [ ] 端到端测试在 10 张固定 fixture 图片上执行：建清单、读标注、建索引、检索、重排降级和文案生成，断言输出路径与 Schema。
- [ ] 运行完整测试：`pixi run pytest -v`。预期：全部通过。
- [ ] 使用全新产物目录执行 README 中的复现命令，确保不存在对开发机临时文件的隐式依赖。
- [ ] README 说明项目目标、目录、安装、模型放置、标注、建索引、训练、评测和启动 Demo 的完整命令。
- [ ] DEPLOYMENT 说明 Windows、CUDA、6GB 显存策略、CPU 降级、常见 OOM 与模型路径问题。
- [ ] LaTeX 报告包含 Abstract、Introduction、Related Work、Method、Experiments、Results、Conclusion、References 和 Appendix，正文控制在 20 页以内。
- [ ] 报告必须呈现 Prompt 对比、检索消融、训练前后、重排前后、性能成本、成功案例、失败案例和局限性。
- [ ] 每名成员单独提交个人贡献报告；贡献与 Git 提交、实验记录和负责模块对应。
- [ ] 准备演示材料：查询复杂条件、解释检索结果、图片问答、文案生成和 SD 补图，视频不超过 3 分钟。
- [ ] 建议提交：`git add README.md DEPLOYMENT.md tests/integration report && git commit -m "docs: complete reproducible course delivery"`。

## 6. 推荐团队分工

五人团队：

1. 数据与 Prompt：清单、标注 Schema、Prompt 实验和标注质量。
2. 检索与训练：BM25、BGE、困难负样本、微调和索引。
3. VLM 应用：查询解析、视觉重排、解释、图片问答和内容生成。
4. 生成与前端：模型资源管理、Stable Diffusion、Gradio 和演示。
5. 评测与论文：人工真值、指标、消融、绘图、LaTeX 和整合验证。

四人团队可将第 3、4 项合并；所有成员共同完成 Val 人工相关度标注，避免单人偏差。

## 7. 阶段门禁

- **Gate A：环境可用。** 两个本地模型各自完成一次推理，显存不溢出。
- **Gate B：标注可信。** 60 张 Prompt 对比结束，主 Prompt 冻结，JSON 有效率达到 98% 以上。
- **Gate C：基线成立。** BM25 与基础 BGE 能在固定查询集返回合理结果。
- **Gate D：训练有效。** 微调模型在 Train 内部验证集优于基础 BGE；若未提升，产品使用基础模型并将负结果如实写入报告。
- **Gate E：完整检索成立。** 混合召回、RRF、视觉重排和解释通过端到端测试。
- **Gate F：生成可用。** Qwen 与 SD 能顺序加载，补图不发生 OOM，生成图带完整元数据。
- **Gate G：评测冻结。** Val 查询和相关度标签完成后不再根据最终结果修改。
- **Gate H：课程交付完成。** 代码、README、部署文档、Demo、报告、个人报告和可选视频齐全。

## 8. 最终验收标准

- 2369 张图片均有图片清单，所有可解码图片均有版本化结构化标注。
- 标注流程可断点恢复，非法输出有日志且可重试。
- 用户可以使用包含主体、动作、场景、颜色、情绪和排除条件的中文查询。
- 检索结果同时提供图片、分数、可见证据和不匹配项。
- 至少报告 Recall@1/5/10、MRR、mAP、nDCG@10、平均延迟、P95 延迟和峰值显存。
- 至少完成 Prompt、召回策略、模型训练和视觉重排四类消融实验。
- 图片问答、三类文案和 Stable Diffusion 补图能够从同一 Demo 访问。
- 真实图与生成图在数据、界面和报告中均有明确来源标记。
- Train/Val 泄漏检查通过，最终评测能由冻结配置复现。
- README 中的命令能从环境安装一路运行到 Demo 和评测结果。

## 9. 风险与降级策略

| 风险 | 检测方式 | 降级方案 |
|---|---|---|
| Python 3.14 与深度学习依赖不兼容 | 环境测试无法导入 torch/transformers | 固定 Python 3.11 并重建 pixi 环境 |
| Qwen 输出非法 JSON | Schema 校验失败率 | 两次纠错重试，仍失败则进入失败队列 |
| 伪标注幻觉 | 60 张人工评分和失败案例统计 | 使用两阶段 Prompt，提高 uncertainty 使用率 |
| 中文向量模型效果不足 | 基础模型 Val 指标 | 保留 BM25，使用混合检索和困难负样本训练 |
| 微调后指标下降 | 冻结评测对比 | 产品回退基础 BGE，报告负实验结果 |
| Qwen 重排延迟过高 | P95 延迟 | Top-10 降至 Top-5，保留融合排序结果 |
| Qwen/SD 同驻显存导致 OOM | GPU 冒烟测试 | 模型互斥加载、CPU offload、512 分辨率 |
| SD 生成偏离查询 | 人工一致性评分 | 展示 Prompt 并允许用户修改后再次生成 |
| 人工评测主观性强 | 标注者一致性 | 两人独立标注，记录分歧与裁决规则 |
| 生成图混入真实检索结果 | 来源字段测试 | `source` 强类型校验，生成图默认不入索引 |

## 10. 开始实施前的检查

- [ ] 复核并修改本文件中的功能边界、模型路径和团队分工。
- [ ] 确认允许额外下载小型中文向量模型 `bge-small-zh-v1.5`。
- [ ] 确认是否初始化 Git；若初始化，先提交当前权重路径配置和计划文件，不提交权重本身。
- [ ] 确认报告语言和学校 LaTeX 模板。
- [ ] 从 Task 1 开始逐项执行，不跨过阶段门禁并行推进依赖任务。
