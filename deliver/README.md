# AskAlbum M1 双模型标注共享包

版本：draft 0.1  
当前决定：**暂时跳过 M0，从 M1 直接开始**

## 1. 当前方案

现在的工作流只有下面五步：

```text
1. 两边读取同一张处理后的图片
                  ↓
2. 两边使用同一份 Prompt 和同一份 JSON 输出格式
                  ↓
3. 本地开源 VLM 与 API VLM 分别独立标注
                  ↓
4. 各自输出一行一个结果的 JSONL 文件
                  ↓
5. 按场景、实体、数量、OCR、关系等字段比较两份答案
```

不再先做 CLIP 分类，不给模型添加类别提示，也不要求 M0 输出。M0 以后只作为可选实验：如果需要，再比较“统一 Prompt”和“增加分类提醒”的效果。

## 2. 两个人共享什么

双方只需要保证以下四样东西完全一致：

1. 同一批处理后的图片及 SHA-256；
2. 同一份 [`prompt_v1.md`](./prompt_v1.md)；
3. 同一份 VLM 输出格式 [`annotation_payload.schema.json`](./annotation_payload.schema.json)；
4. 同一份落盘格式 [`candidate_record.schema.json`](./candidate_record.schema.json)。

## 3. 两个人分别做什么

```text
共同准备 12 张 train smoke 图片
              |
       +------+------+
       |             |
       v             v
 coworker           你
 本地 VLM           API VLM
       |             |
       v             v
candidates_local  candidates_api
       +------+------+
              |
              v
       共同逐字段比较
```

- coworker 负责本地模型加载、推理、原始响应和 `candidates_local.jsonl`；
- 你负责 API 调用、限流、重试、缓存、费用和 `candidates_api.jsonl`；
- 双方共同确定 12 张图片、Prompt、Schema 和图片处理方法。

完整任务见 [`M1_双模型并行启动说明.md`](./M1_双模型并行启动说明.md)。

## 4. 三个 JSON 文件分别是什么

### annotation_payload.schema.json

这是给 VLM 的“答题表格式”，规定模型必须填写：

```text
场景、拍摄外观、实体、OCR、关系、事件、主观感受、描述、不确定项
```

它是规则，不是标注结果。

### m1_annotation_payload.example.json

这是已经按上述格式填好的一张示范答卷：

[`examples/m1_annotation_payload.example.json`](./examples/m1_annotation_payload.example.json)

### candidate_record.schema.json

模型答完以后，程序在外面加上 `image_id`、图片 hash、模型名、成功/失败状态和原始响应路径。这个文件规定整条落盘记录的格式。

合法示例见：

[`examples/candidate_record.example.json`](./examples/candidate_record.example.json)

## 5. 两份答案不逐字比较

比较按字段进行：

- 场景 enum 直接比较；
- 实体通过名称同义词、位置和框判断是不是同一对象；
- 数量在实体对齐后比较；
- OCR 统一空格和全半角后比较字符差异；
- 关系在实体对齐后比较“谁、什么关系、对谁”；
- caption 不比较措辞，只检查事实是否一致、有没有新增幻觉；
- 氛围和美学是主观字段，不作为硬冲突。

第一轮 12 张先人工并排核对，不需要先实现复杂自动评分。

## 6. 当前已经有和还没有的东西

已经有：

- 共享 Prompt；
- VLM 输出 Schema；
- 每张图落盘记录 Schema；
- 两份合法示例；
- 双人分工、12 图流程和完成条件。

还没有：

- 图片 manifest 和统一预处理代码；
- 本地模型调用代码；
- API 调用代码；
- `candidates_local.jsonl` 和 `candidates_api.jsonl`；
- 自动比较和最终融合代码。

因此这个包可以直接交给两个 Codex 开始写代码，但不是一个已经能直接运行的程序。

## 7. 阅读顺序

1. 本 README；
2. [`M1_双模型并行启动说明.md`](./M1_双模型并行启动说明.md)；
3. [`prompt_v1.md`](./prompt_v1.md)；
4. [`annotation_payload.schema.json`](./annotation_payload.schema.json)；
5. [`candidate_record.schema.json`](./candidate_record.schema.json)；
6. 两个 [`examples/`](./examples) 示例。

全项目核心提案和课程 PDF 放在 [`reference/`](./reference) 中，只作背景材料。核心提案中的 M0 当前不执行。
