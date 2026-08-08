# Anima：智能图文搜索与内容生成平台

本项目使用 Qwen3-VL 为无标注真实场景图片生成结构化描述，通过 BM25、BGE 文本向量和 RRF 融合实现中文自然语言搜图，再使用 Qwen3-VL 查看候选图片并进行视觉重排与解释。当前 `deliver/` 任务使用 Qwen3-VL-8B-Instruct 在 A100 上完成 M1 本地结构化标注；原有 2B 标注、检索和 Demo 链路仍然保留。

## 1. 功能

- 为 Train/Val 图片建立带哈希、尺寸和重复信息的数据清单。
- 使用三种 Prompt 比较普通描述、结构化描述和两阶段复核描述。
- 使用 Qwen3-VL-2B 批量生成结构化 JSONL 伪标注，支持断点续跑。
- 使用 BM25 与 BGE 建立稀疏、稠密双索引。
- 使用 RRF 融合多路召回结果，并支持否定条件过滤。
- 使用 Train 伪标注构造正样本、随机负样本和困难负样本，微调 BGE。
- 使用 Qwen3-VL 对 Top-N 图片进行视觉重排并输出匹配证据。
- 支持图片问答、标题、朋友圈文案和微型故事。
- 使用 Stable Diffusion 1.5 生成补充图片，真实图和生成图严格区分。
- 提供命令行搜索、Gradio Demo、标准检索指标和消融实验矩阵。

## 当前 deliver 任务：M1 本地 Qwen 标注

本节是当前任务的运行入口。M1 暂时跳过 M0，不进行图片预分类，不调用外部 API，也不使用原有 `configs/prompts/caption_verified.txt`。Prompt 和输出契约只读取以下文件：

```text
deliver/prompt_v1.md
deliver/annotation_payload.schema.json
deliver/candidate_record.schema.json
```

程序会从 `prompt_v1.md` 原样提取 System Prompt 和 User Prompt。模型只生成 `annotation`，客户端再添加图片 hash、模型 ID、状态和原始响应路径，最终每张图片在 `candidates_local.jsonl` 中占一行。

推荐先完成 12 张 Train smoke 门禁，再进入完整 Train 和 Val。代码也支持显式使用 `--mode full` 跳过 smoke，但这样会失去小样本格式、显存和吞吐检查。

### M1.1 A100 服务器环境

建议使用 Linux、Python 3.11、CUDA 版 PyTorch 和单张 A100 80GB：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先根据服务器 CUDA 环境安装 PyTorch。例如使用 CUDA 12.8 wheel：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

再从 `pyproject.toml` 安装 M1 本地推理依赖：

```bash
python -m pip install -e ".[m1-local]"
```

可选安装 Flash Attention 2：

```bash
python -m pip install flash-attn --no-build-isolation
```

如果没有安装 `flash-attn`，代码会自动降级到 PyTorch SDPA，并把实际 attention implementation 写入 `run_summary.json`。

### M1.2 下载或放置 Qwen3-VL-8B-Instruct

默认配置读取：

```text
models/Qwen3-VL-8B-Instruct/
```

可以使用 Hugging Face CLI 下载：

```bash
python -m pip install huggingface_hub
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir models/Qwen3-VL-8B-Instruct
```

如果 checkpoint 位于其他目录，不需要修改代码，可在运行时覆盖：

```bash
python scripts/annotate_m1_local.py \
  --config configs/m1_local.yaml \
  --model-path /data/models/Qwen3-VL-8B-Instruct \
  --limit 1 \
  --batch-size 1
```

`--model-id` 用于指定写入候选记录的精确模型 ID；不传时固定记录 `Qwen/Qwen3-VL-8B-Instruct`。

### M1.3 建立原始 Train manifest

将课程图片放在：

```text
Train/*.jpg
Val/*.jpg
```

运行原有清单脚本：

```bash
python scripts/build_manifest.py --config configs/default.yaml
```

本阶段使用的来源清单是：

```text
artifacts/manifests/train.jsonl
```

### M1.4 人工确定 12 张 smoke 图片

编辑 `configs/m1_smoke_ids.txt`，填入恰好 12 个互不重复的 Train `image_id`，每行一个，例如：

```text
train-14
train-105
train-287
```

这里的 ID 只是格式示例，不是规定选择。最终 12 张应由人工查看后确定，覆盖普通照片、多人、文字、室内、食物、自然、弱光、展品、插画、文档或屏幕、低分辨率和极端宽高比。不能用模型先分类再自动选择。

### M1.5 生成统一处理图和 smoke manifest

```bash
python scripts/prepare_m1_smoke.py --config configs/m1_local.yaml
```

该命令会：

- 检查恰好选择了 12 张有效、非重复的 Train 图片；
- 核对原图 SHA-256；
- 应用 EXIF 方向、转换 RGB，并按固定像素预算等比例缩小；
- 使用固定 JPEG 参数写入处理图；
- 为最终处理文件计算 `processed_sha256`。

输出：

```text
artifacts/m1/shared/images_smoke.jsonl
artifacts/m1/shared/images/img-v1/*.jpg
```

### M1.6 一图能力探针

第一次加载模型时先只跑一张，确认模型、显存、Prompt 和 JSON 格式：

```bash
python scripts/annotate_m1_local.py \
  --config configs/m1_local.yaml \
  --mode smoke \
  --split train \
  --limit 1 \
  --batch-size 1
```

成功或失败都会在候选 JSONL 中写入明确记录。重新执行完整 12 图任务时，这一张会通过 `image_id + processed_sha256` 自动跳过，不会重复生成。如果一图探针失败，修复环境后增加 `--retry-failed` 重新运行。

### M1.7 运行 12 图批量标注

A100 80GB 上建议从 batch size 8 开始：

```bash
python scripts/annotate_m1_local.py \
  --config configs/m1_local.yaml \
  --mode smoke \
  --split train \
  --batch-size 8
```

也可以测试 `--batch-size 12` 或 `--batch-size 16`。如果一个批次推理失败或 CUDA OOM，程序会自动把该批次对半拆分，直到单张；实际尝试过的 batch size 会记录在 `run_summary.json`。

默认生成参数：

```text
model: Qwen/Qwen3-VL-8B-Instruct
dtype: bfloat16
do_sample: false
temperature: 0 的等效确定性生成
max_new_tokens: 4096
visual pixel budget: 256-1280 × 32 × 32
```

这些参数都集中在 `configs/m1_local.yaml`。不要修改 `deliver/prompt_v1.md` 或两个 Schema 来适配模型输出。

### M1.8 运行完整 Train 和 Val

smoke 通过后，运行完整 Train：

```bash
pixi run python scripts/annotate_m1_local.py \
  --mode full \
  --split train \
  --batch-size 8
```

然后单独运行完整 Val：

```bash
pixi run python scripts/annotate_m1_local.py \
  --mode full \
  --split val \
  --batch-size 8
```

如果对应的全量处理 manifest 不存在，标注脚本会先自动完成统一预处理。也可以提前显式运行预处理：

```bash
pixi run python scripts/prepare_m1_dataset.py --split train
pixi run python scripts/prepare_m1_dataset.py --split val
```

全量处理 manifest 分别位于：

```text
artifacts/m1/shared/train/images.jsonl
artifacts/m1/shared/val/images.jsonl
```

`--mode full` 也支持 `--limit N`，可在正式全量运行前额外进行较大的吞吐探针。例如：

```bash
pixi run python scripts/annotate_m1_local.py \
  --mode full \
  --split train \
  --limit 100 \
  --batch-size 8
```

### M1.9 输出文件

```text
artifacts/m1/local_run/train/
  candidates_local.jsonl
  run_summary.json
  raw/
    train-14.attempt-01.txt
    train-105.attempt-01.txt
    ...

artifacts/m1/local_run/val/
  candidates_local.jsonl
  run_summary.json
  raw/
    val-12.attempt-01.txt
    ...
```

`candidates_local.jsonl` 是汇总文件，不是每张图片一个 JSONL。每张图在文件中占一行，格式严格遵守 `deliver/candidate_record.schema.json`。模型原始文本才按图片分别保存在 `raw/` 中，包括无法解析或校验失败的响应。

Train 和 Val 始终写入不同目录，不会混合在同一 JSONL 中。smoke 与完整 Train 共用 Train 输出；由于处理规则相同，已经成功的 smoke 记录会在完整 Train 运行中自动跳过。

### M1.10 校验结果

```bash
python scripts/validate_m1_candidates.py --config configs/m1_local.yaml
```

上面的默认命令校验 Train。校验 Val 时覆盖输入路径：

```bash
python scripts/validate_m1_candidates.py \
  --config configs/m1_local.yaml \
  --input artifacts/m1/local_run/val/candidates_local.jsonl
```

校验内容包括：

- annotation 和 candidate 的 Draft 2020-12 JSON Schema；
- 连续的 `e1/e2/...` 与 `t1/t2/...`；
- relation 和 event 的实体引用；
- bbox 坐标顺序；
- `count=null` 时 `count_exact=false`；
- 非 person 的 `attire_zh` 必须为空；
- `secondary_types` 不得包含 `primary_type`。

程序不会猜测或自动修改事实。严格 JSON、Schema 或语义校验失败时，保留原始响应，并写入 `status="failed"`、`annotation=null` 和明确的 `error`。

### M1.11 断点续跑

直接重新执行相同命令即可：

```bash
python scripts/annotate_m1_local.py \
  --config configs/m1_local.yaml \
  --mode full \
  --split train \
  --batch-size 8
```

默认情况下，已经存在且 `processed_sha256` 一致的 succeeded/failed 记录都会跳过。如果要重新处理失败记录，同时保留成功记录，运行：

```bash
python scripts/annotate_m1_local.py \
  --config configs/m1_local.yaml \
  --mode full \
  --split train \
  --batch-size 8 \
  --retry-failed
```

如果 manifest 中同一 `image_id` 的处理图 hash 发生变化，程序会拒绝继续，防止旧结果与新图片混合。

## 2. 硬件与系统建议

- Windows 10/11 64 位。
- NVIDIA GPU，建议显存不低于 6 GB。
- 当前机器已检测到 RTX 4050 Laptop 6 GB。
- 建议使用 NVIDIA 驱动支持的 CUDA 版 PyTorch。
- Python 必须使用 3.11；不要使用原先配置中的 Python 3.14。
- 磁盘至少预留 15 GB，用于环境、BGE、伪标注、索引和生成结果。

Qwen3-VL 与 Stable Diffusion 由 `ModelManager` 互斥加载，避免同时占用 6 GB 显存。批量标注、训练和全量评测仍然是耗时操作，建议先使用 `--limit 10` 检查流程。

## 3. 必需模型及放置位置

项目默认从本地目录读取模型，不会在运行时自动联网下载。

### 3.1 Qwen3-VL（M1 使用 8B，原有链路使用 2B）

当前 `deliver` M1 默认路径：

```text
models/Qwen3-VL-8B-Instruct/
```

对应配置为 `configs/m1_local.yaml`。下面的 2B 路径只供 Anima 原有标注、检索和 Demo 链路使用。

默认路径：

```text
E:\programs\Anima\Qwen--Qwen3-VL-2B-Instruct\snapshots\master\
```

该目录至少需要：

```text
config.json
generation_config.json
preprocessor_config.json
chat_template.json
tokenizer.json
tokenizer_config.json
model.safetensors
```

当前工作区中的 Qwen 权重已经位于正确位置。

如果模型放在其他位置，修改 `configs/default.yaml`：

```yaml
models:
  qwen_vl: D:/models/Qwen3-VL-2B-Instruct
```

### 3.2 Stable Diffusion 1.5

默认路径：

```text
E:\programs\Anima\stablediffusion\
```

代码使用 Diffusers 目录格式，需要：

```text
stablediffusion/
  model_index.json
  feature_extractor/
  safety_checker/
  scheduler/
  text_encoder/
  tokenizer/
  unet/
  vae/
```

根目录中的 `v1-5-pruned-emaonly.ckpt` 和 `v1-5-pruned.ckpt` 不是本项目运行所必需的；代码优先读取上述 Diffusers 组件目录。当前工作区中的 Stable Diffusion 已位于默认位置。

如果模型放在其他位置，修改：

```yaml
models:
  stable_diffusion: D:/models/stable-diffusion-v1-5
```

### 3.3 BGE Small Chinese

默认路径：

```text
E:\programs\Anima\models\bge-small-zh-v1.5\
```

该模型当前需要额外准备。安装环境后，在项目根目录运行：

```powershell
pixi run python -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/bge-small-zh-v1.5', local_dir='models/bge-small-zh-v1.5')"
```

下载后目录中应包含 `config.json`、Tokenizer 文件和模型权重文件。如果你使用其他中文 Sentence Transformers 模型，将其放在任意本地目录，并修改：

```yaml
models:
  embedder: D:/models/your-chinese-embedding-model
```

## 4. 图片数据放置

默认目录必须是：

```text
E:\programs\Anima\Train\*.jpg
E:\programs\Anima\Val\*.jpg
```

当前数据规模：

- Train：2000 张 JPG，用于伪标注、训练和困难负样本。
- Val：369 张 JPG，只用于最终检索评测。

不要把 Val 图片加入训练对。代码在 `build_training_pairs()` 中会拒绝 `split != "Train"` 的标注。

## 5. 创建环境（Pixi 或 Mamba 二选一）

不要同时使用 Pixi 和 Mamba 安装同一套依赖。你可以选择下面任意一种环境管理方式；环境创建完成后，后续命令只使用对应的命令前缀。

### 5.1 使用 Mamba（`mamba install` 方案）

Windows 上建议先安装 [Miniforge](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)。Miniforge 会同时提供 `conda` 和 `mamba`，并默认使用 `conda-forge`。安装完成后，重新打开 **Miniforge Prompt** 或 PowerShell。

确认 Mamba 可用：

```powershell
mamba --version
```

创建独立的 Python 3.11 环境：

```powershell
mamba create -n anima python=3.11 pip -c conda-forge
mamba activate anima
```

使用 Mamba 安装能从 conda-forge 获取的基础依赖：

```powershell
mamba install -n anima -c conda-forge `
  numpy=1.26 pillow pyyaml pydantic pandas scikit-learn `
  matplotlib seaborn jieba faiss-cpu pytest
```

使用 PyTorch 官方 Conda channel 安装 CUDA 版 PyTorch：

```powershell
mamba install -n anima -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.4
```

激活环境后安装需要较新版本或在 Windows 上更容易通过 PyPI 获取的 Python 包：

```powershell
python -m pip install --upgrade `
  "transformers>=4.57,<5" "accelerate>=1,<2" "diffusers>=0.35,<1" `
  "sentence-transformers>=3,<6" "gradio>=5,<7" "rank-bm25>=0.2,<1" modelscope
```

确认当前命令使用的是 Mamba 环境：

```powershell
where.exe python
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

在 Mamba 环境中，后续所有命令都直接使用 `python`：

```powershell
python scripts\build_manifest.py --config configs\default.yaml
python scripts\annotate_images.py --config configs\default.yaml --split Train --limit 10
```

如果某个包在 conda-forge 中无法解析，保留 Mamba 安装的 PyTorch、CUDA、FAISS 和基础科学计算包，再用上面的 `python -m pip install` 补装该包。不要在 `base` 环境中安装项目依赖。

### 5.2 使用 Pixi

打开 PowerShell：

```powershell
cd E:\programs\Anima
```

确认 Pixi：

```powershell
pixi --version
```

根据 `pixi.toml` 创建 Python 3.11 环境：

```powershell
pixi install
```

`pixi.toml` 已经从 Python 3.14 改为 Python 3.11。仓库中原有的 Python 3.14 锁文件已移除；首次安装会根据新配置生成 `pixi.lock`。此后提交新的锁文件即可固定环境版本。

如果安装得到的是 CPU 版 PyTorch，可在 Pixi 环境中安装 CUDA 12.8 wheel：

```powershell
pixi run python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

检查 Python、PyTorch 和 CUDA：

```powershell
pixi run python --version
pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

本项目所有 Pixi 命令都采用以下形式：

```powershell
pixi run python scripts\脚本名.py 参数
```

如果你已经手动激活了兼容环境，也可以把前缀 `pixi run` 去掉，直接运行：

```powershell
python scripts\脚本名.py 参数
```

**Mamba 用户请跳过所有 `pixi install` 命令。** 激活 `anima` 环境后，将后文每条命令中的 `pixi run python` 替换为 `python`；例如：

```powershell
mamba activate anima
python scripts\build_manifest.py --config configs\default.yaml
python scripts\build_indexes.py --config configs\default.yaml --split Val
python scripts\launch_app.py --config configs\default.yaml --split val
```

## 6. 修改统一配置

所有模型、数据和运行参数位于：

```text
configs/default.yaml
```

首次运行前重点检查：

```yaml
data:
  train_dir: Train
  val_dir: Val
  artifacts_dir: artifacts
models:
  qwen_vl: Qwen--Qwen3-VL-2B-Instruct/snapshots/master
  stable_diffusion: stablediffusion
  embedder: models/bge-small-zh-v1.5
runtime:
  device: cuda
  dtype: float16
```

RTX 4050 6 GB 建议保持 `float16`、512×512 生成分辨率和 Top-10 视觉重排。

## 7. 全流程运行

以下命令均从 `E:\programs\Anima` 执行。

### Step 1：生成图片清单

```powershell
pixi run python scripts\build_manifest.py --config configs\default.yaml
```

输出：

```text
artifacts/manifests/train.jsonl
artifacts/manifests/val.jsonl
artifacts/manifests/quality_report.json
```

先查看 `quality_report.json` 中的 `invalid` 和 `duplicates`。损坏图片不会进入标注；跨 Train/Val 的重复图应从人工评测集排除。

### Step 2：比较三种 Prompt

先用 10 张图片检查输出：

```powershell
pixi run python scripts\compare_prompts.py --split Train --sample-size 10
```

正式比较 60 张：

```powershell
pixi run python scripts\compare_prompts.py --split Train --sample-size 60
```

输出：

```text
artifacts/evaluation/prompt_outputs.jsonl
```

每张图片分别产生 `caption_basic`、`caption_structured` 和 `caption_verified` 输出。建议导入表格，由两名成员分别评价事实正确性、完整度、幻觉和可检索性。默认主流程使用 `caption_verified_v1`。

### Step 3：小样本结构化标注

先处理 10 张 Train：

```powershell
pixi run python scripts\annotate_images.py --config configs\default.yaml --split Train --limit 10
```

输出：

```text
artifacts/annotations/train.caption_verified_v1.jsonl
```

如果模型输出无效 JSON，程序最多纠错重试两次。最终失败记录在：

```text
artifacts/annotations/train.caption_verified_v1.failures.jsonl
```

脚本支持断点续跑：重新执行时会跳过输出 JSONL 中已经存在的 `image_id`。

### Step 4：全量标注 Train 和 Val

```powershell
pixi run python scripts\annotate_images.py --config configs\default.yaml --split Train
pixi run python scripts\annotate_images.py --config configs\default.yaml --split Val
```

输出：

```text
artifacts/annotations/train.caption_verified_v1.jsonl
artifacts/annotations/val.caption_verified_v1.jsonl
```

两条命令会运行较长时间。可以随时正常终止，之后执行同一命令继续。

### Step 5：建立 Train 和 Val 检索索引

确认 BGE 模型已经放入 `models/bge-small-zh-v1.5`，然后运行：

```powershell
pixi run python scripts\build_indexes.py --config configs\default.yaml --split Train
pixi run python scripts\build_indexes.py --config configs\default.yaml --split Val
```

输出：

```text
artifacts/indexes/train/bm25.pkl
artifacts/indexes/train/vector/vectors.faiss
artifacts/indexes/train/vector/metadata.json
artifacts/indexes/train/annotations.json
artifacts/indexes/val/bm25.pkl
artifacts/indexes/val/vector/vectors.faiss
artifacts/indexes/val/vector/metadata.json
artifacts/indexes/val/annotations.json
```

### Step 6：在命令行中搜图

不使用视觉重排，只运行基础混合检索：

```powershell
pixi run python scripts\search_cli.py "不要人物，寻找冷色调的雨夜城市" --split val
```

启用 Qwen3-VL 视觉重排：

```powershell
pixi run python scripts\search_cli.py "不要人物，寻找冷色调的雨夜城市" --split val --rerank
```

返回 JSON 包含图片相对路径、RRF 分数、VLM 分数、匹配证据和不匹配项。

### Step 7：构造检索训练数据

```powershell
pixi run python scripts\build_training_pairs.py --config configs\default.yaml
```

输出：

```text
artifacts/training/pairs.jsonl
```

训练对只从 Train 标注构建。每条记录包含 query、positive、negative、正负图片 ID 和负样本类型。

### Step 8：微调 BGE 检索模型

RTX 4050 6 GB 的起始参数：

```powershell
pixi run python scripts\train_retriever.py --config configs\default.yaml --epochs 3 --batch-size 16
```

如果发生 CUDA OOM，将 batch size 改为 8：

```powershell
pixi run python scripts\train_retriever.py --config configs\default.yaml --epochs 3 --batch-size 8
```

输出：

```text
artifacts/checkpoints/retriever/
```

要使用微调后的模型，编辑 `configs/default.yaml`：

```yaml
models:
  embedder: artifacts/checkpoints/retriever
```

然后重新构建 Train/Val 向量索引：

```powershell
pixi run python scripts\build_indexes.py --config configs\default.yaml --split Train
pixi run python scripts\build_indexes.py --config configs\default.yaml --split Val
```

如果微调模型指标没有超过基础 BGE，把 `embedder` 改回 `models/bge-small-zh-v1.5` 并如实记录负实验结果。

### Step 9：建立 Val 评测集初稿

```powershell
pixi run python scripts\create_eval_set.py --config configs\default.yaml --count 100
```

输出：

```text
artifacts/evaluation/val_queries.jsonl
artifacts/evaluation/val_relevance.csv
```

脚本用每张 Val 图片的一条生成查询作为初始 query，并把来源图片标为相关度 2。正式报告前必须人工修改：

- 将查询分类为主体、动作、场景、颜色、情绪、OCR、组合条件或否定条件。
- 人工改写自动生成的查询，避免与伪标注中的 `search_queries` 原文相同。
- 为每条查询补充其他相关候选图片。
- relevance 使用 0、1、2：不相关、部分相关、高度相关。
- 至少两名成员独立标注并处理分歧。
- 完成人工改写后，把 `val_queries.jsonl` 中每条记录的 `reviewed` 改为 `true`，并将 `category` 从 `auto_seed` 改为真实类别；否则评测脚本会拒绝运行。

### Step 10：运行检索评测

不启用视觉重排：

```powershell
pixi run python scripts\evaluate_retrieval.py --config configs\default.yaml --queries artifacts\evaluation\val_queries.jsonl --relevance artifacts\evaluation\val_relevance.csv
```

启用视觉重排：

```powershell
pixi run python scripts\evaluate_retrieval.py --config configs\default.yaml --queries artifacts\evaluation\val_queries.jsonl --relevance artifacts\evaluation\val_relevance.csv --rerank
```

输出：

```text
artifacts/evaluation/retrieval_metrics.json
artifacts/evaluation/retrieval_details.csv
```

指标包括 Recall@1/5/10、MRR、mAP、nDCG@10 和平均查询耗时。

### Step 11：生成消融实验矩阵

```powershell
pixi run python scripts\run_ablation.py --dry-run
```

写入 CSV：

```powershell
pixi run python scripts\run_ablation.py
```

输出：

```text
artifacts/evaluation/ablation_plan.csv
```

矩阵覆盖三种 Prompt、BM25/稠密/混合检索、微调前后和视觉重排开关。每种 Prompt 需要使用不同 `prompt_version` 生成标注和索引，避免覆盖实验产物。

### Step 12：启动 Gradio Demo

默认搜索 Val：

```powershell
pixi run python scripts\launch_app.py --config configs\default.yaml --split val --host 127.0.0.1 --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

搜索 Train：

```powershell
pixi run python scripts\launch_app.py --config configs\default.yaml --split train --host 127.0.0.1 --port 7860
```

需要 Gradio 临时公网链接时显式增加 `--share`：

```powershell
pixi run python scripts\launch_app.py --config configs\default.yaml --split val --share
```

Demo 包含：

- 智能搜索和 Qwen 视觉重排开关；
- 结构化标注与图片问答；
- 标题、朋友圈文案和微型故事；
- Stable Diffusion 补图；
- 当前模型和 Prompt 版本展示。

## 8. Stable Diffusion 补图参数

默认配置：

```yaml
generation:
  width: 512
  height: 512
  steps: 30
  guidance_scale: 7.5
  seed: 20260802
```

生成结果位于：

```text
artifacts/generated/generated-<seed>.png
artifacts/generated/generated-<seed>.json
```

同名 JSON 保存正向 Prompt、负向 Prompt、seed、尺寸、步数、guidance scale 和模型路径。生成结果的来源固定为 `generated`，默认不进入真实图片索引。

## 9. 测试命令

运行全部测试：

```powershell
pixi run python -m pytest -q
```

只运行轻量单元测试：

```powershell
pixi run python -m pytest tests\unit -q
```

只检查服务接口：

```powershell
pixi run python -m pytest tests\integration\test_service_contract.py -q
```

测试文件不会自动运行 Qwen、Stable Diffusion、全量标注或训练任务。

## 10. 产物目录

```text
artifacts/
  manifests/       图片清单和质量报告
  annotations/     Qwen 结构化伪标注
  indexes/         BM25、FAISS 和标注快照
  training/        训练对
  checkpoints/     微调后的检索模型
  evaluation/      查询、相关度、指标和消融结果
  generated/       Stable Diffusion 图片及生成参数
```

模型权重、批量标注、索引和生成图片已加入 `.gitignore`。报告中需要的最终指标 JSON、CSV 和图表可按课程提交要求单独整理。

## 11. 常见问题

### `pixi run python` 无法启动

原 `.pixi` 环境可能来自 Python 3.14。重新运行：

```powershell
pixi install
```

如果仍然失败，确认没有其他进程占用 `.pixi`，再由 Pixi 重建环境。不要手动把系统 Python 3.14 填入该环境。

### `Qwen3VLForConditionalGeneration` 无法导入

Qwen3-VL 需要较新的 Transformers。确认：

```powershell
pixi run python -c "import transformers; print(transformers.__version__)"
```

项目要求 `transformers>=4.57,<5`。

### CUDA OOM

- 保持 Qwen 和 SD 的互斥模型管理，不要在另一个 Python 进程中同时加载它们。
- 将检索训练 batch size 从 16 降到 8 或 4。
- 保持 SD 分辨率 512×512。
- 将 `retrieval.rerank_count` 从 10 改为 5。
- 结束其他占用 GPU 的程序后再启动 Demo。

### 找不到 BGE 模型

检查目录：

```powershell
Get-ChildItem models\bge-small-zh-v1.5
```

然后确认 `configs/default.yaml` 的 `models.embedder` 与实际路径一致。

### 标注输出大量失败

查看：

```text
artifacts/annotations/train.caption_verified_v1.failures.jsonl
artifacts/annotations/val.caption_verified_v1.failures.jsonl
```

优先检查显存、图片是否损坏、Qwen 模型文件是否齐全，以及 Transformers 是否支持 Qwen3-VL。修正问题后重新执行相同标注命令，已成功图片会被跳过。

### 搜索结果为空

确认已经对对应 split 建立索引：

```text
artifacts/indexes/train/
artifacts/indexes/val/
```

如果查询包含过多否定词，结构化过滤可能排除全部候选；先用不带否定条件的查询检查基础索引。

## 12. 建议实验表格

论文至少报告：

| 实验 | Prompt | 检索器 | 训练 | VLM 重排 |
|---|---|---|---|---|
| Baseline 1 | 普通描述 | BM25 | 否 | 否 |
| Baseline 2 | 结构化描述 | BGE | 否 | 否 |
| Proposed A | 两阶段描述 | BM25+BGE+RRF | 否 | 否 |
| Proposed B | 两阶段描述 | BM25+微调 BGE+RRF | 是 | 否 |
| Full | 两阶段描述 | BM25+微调 BGE+RRF | 是 | 是 |

同时报告 Prompt 的 JSON 有效率、事实正确性、完整度和幻觉率；生成部分报告图文一致性、视觉质量、风格符合度和人工偏好。

## 13. 推荐首次运行顺序

首次不要直接运行全量任务，按以下顺序排错：

```powershell
cd E:\programs\Anima
pixi install
pixi run python scripts\build_manifest.py
pixi run python scripts\compare_prompts.py --sample-size 10
pixi run python scripts\annotate_images.py --split Train --limit 10
```

确认 10 张图片的 JSON 正常后，再执行：

```powershell
pixi run python scripts\annotate_images.py --split Train
pixi run python scripts\annotate_images.py --split Val
pixi run python scripts\build_indexes.py --split Train
pixi run python scripts\build_indexes.py --split Val
pixi run python scripts\build_training_pairs.py
pixi run python scripts\train_retriever.py --epochs 3 --batch-size 16
pixi run python scripts\create_eval_set.py --count 100
pixi run python scripts\evaluate_retrieval.py
pixi run python scripts\launch_app.py --split val
```

完整设计、阶段门禁、风险和课程交付建议见 [plan.md](plan.md)。
