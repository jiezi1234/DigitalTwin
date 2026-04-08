# DigitalTwin

基于 Agentic RAG 的数字分身与数字助教系统。通过微信聊天记录构建个人数字分身，通过 PDF 教材构建支持图文混合检索的 AI 助教。本项目已全面重构升级为 **ReAct 智能体架构**，剥离了传统的线性逻辑，实现了自主决策工作流。

## 功能

- **数字分身 (Agent 化)** — 基于 ReAct 思维模式的主动智能体驱动。能够根据上下文自主决策是否触发记忆库检索，生成高度模拟本人的智能对话。
- **动态技能库 (Skills)** — 将所有外挂能力（记忆检索、数据注入、PDF 解析导出等）模块化为即插即用的标准 Skill 格式，供大模型自行调用。
- **数字助教** — 导入 PDF 教材，基于多模态检索进行图文混合问答。
- **RAG 检索增强** — 查询改写、指代消解、MMR 多样性搜索。
- **可观测性** — OpenTelemetry 追踪全生命周期的 Agent Workflow + Prometheus + Grafana 监控栈。

## 架构与工作流

```text
主监控与用户交互 (Client / UI)
    ↓
API 路由层 (Flask)
    ↓
应用服务层 (RAGService 基于 ReActAgent 运作)  <--【核心工作流】
    │   1. 意图思考 (Thought): "用户在问昨天吃了什么，我需查记忆..."
    │   2. 决定动作 (Action): 调用 `chat_history_retrieval` 技能
    │   3. 获取观测 (Observation): 获得向量库匹配结果
    │   4. 最终回答 (Final Answer): "昨天去吃了烤肉..."
    ↓ 
技能组件装配库 (src/skills)
    ├── chat_history_retrieval (基于 RAGEngine 的聊天历史检索)
    ├── wechat_csv_import (向量知识库语料注入)
    ├── course_material_import (课程教材结构化拆库)
    ├── multimodal_pdf_rebuild (多模态图文重构)
    └── pdf_assets_export (提取导出)
    ↓
底层基础设施 (LLMClient + DBClient + RAGEngine + Telemetry)
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | Flask 3.0 + Flask-CORS |
| Agent 架构 | 原生手写 ReAct 机制智能体 |
| LLM | 通义千问 (`qwen-plus` / `qwen-vl-plus`) via DashScope |
| 向量数据库 | ChromaDB |
| 多模态感知 | `multimodal-embedding-v1` + `text-embedding-v4` |
| RAG 框架 | 自构建的 Query Processor & RAGEngine |
| 可观测性 | OpenTelemetry + Prometheus + Loki + Grafana |
| 前端 | 原生 HTML/CSS/JS |

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/jiezi1234/DigitalTwin.git
cd DigitalTwin

# 创建 conda 环境
conda create -n DT python=3.10
conda activate DT

# 安装依赖
pip install -e .
```

### 配置

复制 `.env.example` 为 `.env`，填入你的 DashScope API Key：

```bash
cp .env.example .env
```

```env
DASHSCOPE_API_KEY=your-api-key-here
```

### 导入数据 (作为技能调用)

项目所有核心脚手架已改造为标准 Skill 套件目录结构：

```bash
# 导入微信聊天记录 CSV (构建数字分身基石)
python -m src.skills.wechat_csv_import.scripts.import_wechat_csv_cli

# 课程资料导入（默认：增量导入）
# notes*.pdf -> textbook_mm_text_embeddings
# textbook.pdf -> textbook_ocr_text_embeddings
python -m src.skills.course_material_import.scripts.import_course_materials_cli \
  --persist-dir ./chroma_db_mm \
  --notes-files data/pdf/notes1_2022.pdf data/pdf/notes7_2022.pdf \
  --textbook-file data/pdf/textbook.pdf

# 全量重建（清空后重建）
python -m src.skills.course_material_import.scripts.import_course_materials_cli \
  --persist-dir ./chroma_db_mm \
  --notes-files data/pdf/notes1_2022.pdf data/pdf/notes7_2022.pdf \
  --textbook-file data/pdf/textbook.pdf \
  --reset

# 仅导入 notes（跳过 textbook OCR）
python -m src.skills.course_material_import.scripts.import_course_materials_cli \
  --persist-dir ./chroma_db_mm \
  --notes-files data/pdf/notes1_2022.pdf data/pdf/notes7_2022.pdf \
  --skip-textbook
```

也可以使用交互控制台选择要导入的文档：

```bash
python start_assistant.py
```

课程教材导入后的默认产物（以 `--persist-dir ./chroma_db_mm` 为例）：
- 向量库目录：`./chroma_db_mm`
- 导出目录：`./output/course_mm`
- tracking 目录：`./chroma_db_mm/import_tracking`

注意：服务运行时的 `CHROMA_PERSIST_DIR` 必须与导入时使用的 `--persist-dir` 一致，否则会出现“导入成功但检索不到”的现象。

### 启动服务

```bash
bash scripts/start_server.sh
```

或：

```bash
python -m src.run_server
```

- 数字分身：http://localhost:8080
- 数字助教：http://localhost:8080/tutor

此时你在服务端日志 `logs/app.log` 会看到清晰的 `[Agent Workflow]` 决策打点，监控整个问答流从思考到执行。

## 项目核心结构

```
src/
├── agent/                         # Agent 主架构
│   └── react_agent.py                 # ReAct 大脑控制器
├── api/                           # Flask 路由及端点暴露
├── infrastructure/                # ChromaDB / 大模型 Client 封装
├── loaders/                       # PDF/CSV 解析原生驱动
├── rag/                           # RAG 底层算法核心机制
├── services/                      # 服务层 (装载 Agent 和 Web 生命周期)
│   └── rag_service.py                 # 分身聊天业务中枢
├── skills/                        # ★ 即插即用式的标准化技能插件组
│   ├── base_skill/                    # 技能基础父类规范 (SKILL.md)
│   ├── chat_history_retrieval/        # 提供分身检索挂载
│   ├── course_material_import/        # 课程大文件注入处理
│   ├── multimodal_pdf_rebuild/        # RAG 索引重建
│   ├── pdf_assets_export/             # 游离资源提取
│   └── wechat_csv_import/             # 分身微调数据喂食
└── run_server.py                  # 服务主启入口

frontend/                          # 前端页面
scripts/                           # 服务级启停 Shell 脚本
tests/                             # 功能测试
logs/                              # 系统与 Agent Workflow 日志追踪
```

## 教材检索说明

数字助教依然采用双路混排检索：
- 文本块检索：`textbook_mm_text_embeddings`
- 图片检索：`textbook_mm_image_embeddings`

`textbook.pdf` 的 OCR 文本目前会导入到 `textbook_ocr_text_embeddings`。配合前端展示流渲染 `[图N]` 占位符。

## 监控可视化

全套微服务探针和系统级打桩 (基于 OpenTelemetry)

```bash
# 启动监控服务（Prometheus + Loki + Grafana）
bash scripts/start_monitoring.sh

# 停止
bash scripts/stop_monitoring.sh
```
