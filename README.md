# DigitalTwin

DigitalTwin 是一个面向两类场景的 RAG 交互系统：使用微信聊天记录构建人物风格对话，使用 PDF 教材构建支持文字、OCR 与图片检索的数字助教。项目采用分层架构，将模型调用、向量库、检索策略、业务服务和 Flask API 解耦，便于独立测试和替换组件。

## 核心能力

- 人物对话：清洗并增量导入聊天记录，按人物管理独立的 Chroma collection。
- 按需检索：ReAct 路由根据当前问题和最近会话选择直接回答或调用检索工具。
- 历史感知查询理解：结合最近会话，用一次结构化模型调用完成指代消解与查询改写。
- 混合召回：使用 Dense/MMR 与轻量中文 BM25 双路召回，通过加权 RRF 融合排名；单通道异常时自动降级。
- 候选重排：用一次结构化 LLM 调用重排首轮 Top-N 候选，记录重排前后的分数与名次，解析失败时保持原排名。
- 时间约束检索：把“去年”“某月之后”等显式时间条件规范化为 ISO 日期，并下推为 Chroma 与 BM25 共用的范围过滤。
- 上下文预算：对召回片段做规范化去重、近重复检测、单片段限长与句界截断，并保留教材引用编号和实际来源的一致性。
- 时间邻域扩展：语义命中聊天消息后，按会话 ID 和消息序号补齐前后消息，恢复完整对话语境。
- 教材问答：并行召回多模态文本、OCR 文本和教材图片；使用 RRF 融合两个文本排序。
- 图文回答：将命中图片交给视觉语言模型，并通过 `[图1]` 等标记嵌入前端回答。
- 可复现评测：对比基础相似度检索与“查询改写 + 混合召回 + 时间邻域扩展”，统计 Hit Rate、MRR、Recall 和延迟。
- 可观测性：使用 OpenTelemetry 采集模型和数据库调用，支持接入 Tempo、Prometheus、Loki 与 Grafana。

## 系统架构

```text
浏览器 / API 调用方
        │
        ▼
Flask API（人物对话 / 数字助教）
        │
        ▼
Service（RAGService / TextbookRAGService）
        │
        ├── 人物对话：路由 → 查询理解/过滤 → 混合召回 → 重排/邻域 → 去重与预算 → LLM
        │
        └── 教材问答：多模态文本 ─┐
                    OCR 文本 ─────┼→ RRF 文本融合 → 图文上下文 → VL 模型
                    教材图片 ─────┘
        │
        ▼
Infrastructure（DashScope / ChromaDB / OpenTelemetry）
```

## 技术栈

| 模块 | 实现 |
|---|---|
| Web API | Python 3.10+、Flask、Flask-CORS |
| 模型服务 | DashScope Qwen 文本模型与视觉语言模型 |
| 检索与排序 | ChromaDB、LangChain Chroma、MMR、BM25、加权 RRF、LLM Reranker |
| 向量模型 | `text-embedding-v4`、`multimodal-embedding-v1` |
| PDF 处理 | PyMuPDF，原生文本提取、OCR、图片导出 |
| 可观测性 | OpenTelemetry、Prometheus、Tempo、Loki、Grafana |
| 前端 | 原生 HTML、CSS、JavaScript |

## 项目结构

```text
DigitalTwin/
├── src/
│   ├── api/                  # Flask 应用、配置与路由
│   ├── cli/                  # 服务启动、数据导入、索引与评测命令
│   ├── evaluation/           # 检索评测模型、指标与报告逻辑
│   ├── infrastructure/       # LLM、Embedding、ChromaDB、Telemetry 客户端
│   ├── loaders/              # 微信 CSV 与 PDF 加载、清洗、切分
│   ├── rag/                  # 查询处理、ReAct 路由、RAG 检索引擎
│   └── services/             # 人物对话、教材问答与导入服务
├── frontend/                 # 人物对话与数字助教页面
├── tests/                    # 单元测试和组件测试
├── evaluation/               # 评测数据格式、示例与本地结果目录
├── scripts/                  # 启动、课程导入和监控脚本
├── monitoring/               # Prometheus、Tempo、Loki 配置
├── docs/                     # 架构、API 与迁移文档
├── data/                     # 本地原始数据，不提交版本库
├── output/                   # PDF 图片和结构化导出，不提交版本库
└── chroma_db*/               # 本地向量库，不提交版本库
```

`src/cli/` 只负责解析参数和组装组件，业务逻辑位于 `services/`、`rag/` 与 `infrastructure/`，因此 CLI 和 Web API 可以复用同一套实现。

## 快速开始

### 1. 安装

在项目根目录创建 Python 3.10 或更高版本的虚拟环境，然后安装项目：

```bash
python -m pip install -e ".[dev]"
```

### 2. 配置

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

Linux 或 macOS：

```bash
cp .env.example .env
```

至少需要填写：

```env
DASHSCOPE_API_KEY=your-api-key-here
```

模型、collection、召回数量与可观测性开关均可在 [.env.example](./.env.example) 中配置。不要提交包含真实密钥的 `.env`。

### 3. 准备本地数据

聊天记录放在 `data/csv/`。导入器支持 `msg` 或 `message` 作为消息列，并读取 `talker`、`is_sender`、`CreateTime` / `chat_time`、`room_name` / `room` 等元数据。涉及真实聊天内容时应先脱敏并获得数据使用授权。

默认课程导入脚本期望以下文件：

```text
data/pdf/notes1_2022.pdf
data/pdf/notes7_2022.pdf
data/pdf/textbook.pdf
```

其中 notes 文件进入多模态文本与图片索引，textbook 进入 OCR 文本索引。原始数据、导出图片和向量库均被 Git 忽略，仓库不附带这些运行数据。

### 4. 导入数据

交互式创建人物并导入微信 CSV：

```bash
python -m src.cli.import_wechat_csv
```

人物索引使用秒级数值 `chat_time` 执行时间过滤，并使用 `conversation_id` 和
`message_index` 完成时间邻域扩展。升级已有向量库后，请在交互式导入中选择
一次“全量导入”，为旧记录补齐这些元数据字段。

一键导入默认课程资料：

```bash
bash scripts/import_course_materials.sh
```

脚本默认执行增量导入，并将断点记录写入 `chroma_db_mm/import_tracking/`。需要清空目标 collections 后重建时，直接运行 CLI 并显式添加 `--reset`：

```bash
python -m src.cli.import_course_materials --reset
```

可用 `python -m src.cli.import_course_materials --help` 查看文件路径、并发数、批大小和 collection 参数。

### 5. 启动

课程资料使用默认脚本配置启动：

```bash
bash scripts/start_server.sh
```

或直接使用 Python 模块和 `.env` 中的配置：

```bash
python -m src.cli.run_server
```

启动后访问：

- 人物对话：<http://localhost:8080>
- 数字助教：<http://localhost:8080/tutor>

执行 `pip install -e .` 后，也可以使用 `digitaltwin-server`、`digitaltwin-import`、`digitaltwin-import-course-materials` 等命令行入口。

## 两条 RAG 链路

### 人物风格对话

1. ReAct 路由读取用户问题和最近会话，输出 `retrieve` 或 `respond`。
2. 需要检索时，QueryProcessor 结合最近会话，用一次结构化调用生成独立检索查询，并提取实体及 ISO 时间范围。
3. 对显式时间约束生成秒级时间戳条件，同一条件分别下推到 Chroma 向量检索和 BM25 缓存索引；无时间约束或解析失败时不施加过滤。
4. RAGEngine 获取 Dense/MMR 与 BM25 候选，用加权 RRF 融合不同量纲的排名；任一通道失败时使用另一通道继续回答。
5. BM25 索引在 collection 第一次被查询时从 Chroma 原文懒加载并缓存；导入数据后新启动的服务会重建索引，也可调用 `invalidate()` 主动失效。
6. LLM Reranker 在一次结构化调用中对 Top-N 候选评分；异常、空响应或 JSON 解析失败时保留 RRF 排名。
7. 围绕前几个重排后的命中点补齐同一会话的前后消息，按原对话顺序组织并对重叠邻域去重。
8. ContextBuilder 再执行规范化近重复检测、单片段限长和总字符预算；长片段优先在句界截断，避免一个超长结果挤占全部上下文。
9. 模型结合人物提示词、实际入选的对话片段和会话历史生成回复。

路由无法解析或模型调用失败时默认选择检索，以减少遗漏相关记忆的风险。系统不保存或返回模型的内部推理过程。

### PDF 图文知识问答

1. PDF 导入阶段提取文本块、页级 OCR 文本和页面图片，分别写入三个 collection。
2. 查询阶段使用教材领域改写，并召回多模态文本、OCR 文本和图片。
3. 多模态文本与 OCR 文本来自不同向量空间，使用 Reciprocal Rank Fusion 合并排名，而不是直接比较分数。
4. 命中图片作为多模态输入发送给视觉语言模型，回答中的图片标记由前端替换为实际图片。

OCR 通道不可用时会降级为多模态文本与图片检索，不让单个外部调用中断整条问答链路。

## 检索评测

复制示例并补充脱敏的人工标注查询：

```bash
Copy-Item evaluation/retrieval_queries.example.jsonl evaluation/retrieval_queries.jsonl
```

运行基础相似度检索与“查询改写 + 时间过滤 + 混合召回 + 重排 + 邻域扩展 + 上下文预算”对比：

```bash
python -m src.cli.evaluate_retrieval \
  --dataset evaluation/retrieval_queries.jsonl \
  --collection persona_xxxxxxxx \
  --k 5
```

报告默认写入 `evaluation/results/`，包含 Hit Rate@K、MRR@K、Recall@K、Context Precision@K、平均/P50/P95 延迟以及逐样本变化。优化组的指标基于字符预算内实际进入提示词的片段计算。`evaluation/retrieval_queries.example.jsonl` 只描述格式，不代表真实实验结果；项目指标应由完整标注集和保存的评测报告支撑。详见 [evaluation/README.md](./evaluation/README.md)。

## 测试与 CI

```bash
python -m pytest
```

测试通过 Mock 隔离外部模型和向量服务，不会产生 API 费用。PDF 集成用例在本地测试文件缺失时会跳过。GitHub Actions 会在 Python 3.10 的 Linux 和 Windows 环境执行同一测试套件。

覆盖率：

```bash
python -m pytest --cov=src --cov-report=term-missing
```

## 可观测性

将 `.env` 中的 `OTEL_ENABLED` 设为 `true` 后，可导出模型和数据库调用的 trace、日志和指标。启动本地监控容器：

```bash
bash scripts/start_monitoring.sh
```

默认地址：

| 服务 | 地址 |
|---|---|
| Grafana | <http://localhost:3000> |
| Prometheus | <http://localhost:9090> |
| Tempo | `http://localhost:4318`（OTLP HTTP） |
| Loki | `http://localhost:3100/otlp`（OTLP HTTP） |

停止并移除这些监控容器：

```bash
bash scripts/stop_monitoring.sh
```

监控脚本负责启动容器，但仓库没有预置 Grafana 数据源和仪表盘；首次使用需要在 Grafana 中手动添加 Prometheus、Tempo 和 Loki。

## HTTP 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/` | 人物对话页面 |
| `POST` | `/chat` | 人物对话 |
| `POST` | `/reset` | 清空人物对话会话 |
| `GET` | `/api/personas` | 查询人物列表 |
| `DELETE` | `/api/personas/<persona_id>` | 删除人物及其 collection |
| `GET` | `/tutor` | 数字助教页面 |
| `POST` | `/tutor/chat` | 教材问答，支持 SSE 流式输出 |
| `POST` | `/tutor/import` | 在后台触发 PDF 多模态导入 |
| `POST` | `/tutor/reset` | 清空助教会话 |
| `GET` | `/tutor/stats` | 查询教材 collections 统计 |

请求和响应示例见 [API 文档](./docs/api.md)。

## 仓库边界与已知限制

- 本仓库聚焦 RAG 推理与检索评测，不包含 PyTorch / PEFT LoRA 训练代码、训练数据或 SwanLab 实验记录。
- 仓库目前不包含 Locust 压测脚本与可复核的 TTFT 报告；性能数字应在补齐压测资产后再作为仓库结论。
- 会话保存在单进程内存中，服务重启后丢失，也不支持多实例共享。
- API 尚未实现身份认证、租户隔离、限流和生产级任务队列；不要直接暴露到公网。
- `/tutor/import` 只返回后台任务已启动，详细进度需通过应用日志查看。

## 进一步阅读

- [架构设计](./docs/architecture.md)
- [API 接口](./docs/api.md)
- [迁移指南](./docs/MIGRATION.md)
- [检索评测说明](./evaluation/README.md)
