# DigitalTwin

基于 RAG 的数字分身与数字助教系统。通过微信聊天记录构建个人数字分身，通过 PDF 教材构建支持图文混合检索的 AI 助教。

## 功能

- **数字分身** — 导入微信聊天记录，生成模拟本人说话风格的 AI 对话
- **数字助教** — 导入 PDF 教材，基于教材内容进行图文混合问答
- **RAG 检索增强** — 查询改写、指代消解、MMR 多样性搜索
- **多模态教材检索** — 文本块与图片分库索引，支持图片检索与嵌入式引用
- **断点续跑导入** — 教材导入支持 tracking、增量更新与中断恢复
- **可观测性** — OpenTelemetry 追踪 + Prometheus + Loki + Grafana 监控栈

## 架构

```
API 路由层 (Flask)
    ↓
服务层 (RAGService / TextbookRAGService)
    ↓
RAG 引擎层 (RAGEngine + QueryProcessor)
    ↓
基础设施层 (LLMClient + DBClient + Telemetry)
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | Flask 3.0 + Flask-CORS |
| LLM | 通义千问 (`qwen-plus` / `qwen-vl-plus`) via DashScope |
| 向量数据库 | ChromaDB |
| 嵌入模型 | `multimodal-embedding-v1` + `text-embedding-v4` |
| RAG 框架 | LangChain |
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

### 导入数据

```bash
# 导入微信聊天记录 CSV（数字分身）
python -m src.import_wechat_csv

# 导入课程教材
# notes1_2022.pdf / notes7_2022.pdf -> 多模态文本块 + 图片向量
# textbook.pdf -> OCR 文本提取后入文本向量库
bash scripts/import_course_materials.sh
```

课程教材导入后的默认产物：
- 向量库目录：`./chroma_db_mm`
- 导出目录：`./output/course_mm`
- tracking 目录：`./chroma_db_mm/import_tracking`

默认 collection：
- `textbook_mm_text_embeddings`
- `textbook_mm_image_embeddings`
- `textbook_ocr_text_embeddings`

说明：
- `scripts/import_course_materials.sh` 当前默认执行全量重建，因为脚本里带了 `--reset`
- 如果希望启用真正的增量/断点续跑，请移除脚本中的 `--reset`
- `notes1_2022.pdf` 与 `notes7_2022.pdf` 走多模态导入
- `textbook.pdf` 走 OCR 页级处理、批次级文本向量化和断点续跑

### 启动服务

```bash
bash scripts/start_server.sh
```

- 数字分身：http://localhost:8080
- 数字助教：http://localhost:8080/tutor

`scripts/start_server.sh` 默认连接：
- `./chroma_db_mm`
- `./output/course_mm`
- `textbook_mm_text_embeddings`
- `textbook_mm_image_embeddings`
- `textbook_ocr_text_embeddings`

## 项目结构

```
src/
├── api/                           # Flask 路由层
│   ├── app.py                         # 应用工厂
│   ├── config.py                      # 配置管理
│   └── routes/                        # 路由端点
│       ├── chatbot.py                     # 聊天接口
│       ├── persona.py                     # 分身管理接口
│       └── tutor.py                       # 助教接口
├── infrastructure/                # 基础设施层
│   ├── db_client.py                   # ChromaDB 客户端
│   ├── llm_client.py                  # DashScope LLM 客户端
│   ├── multimodal_embedding_client.py # 多模态向量客户端
│   ├── text_embedding_client.py       # 文本向量客户端
│   ├── telemetry.py                   # OpenTelemetry 配置
│   ├── persona_manager.py            # 分身元数据管理
│   └── document.py                    # 统一文档模型
├── loaders/                       # 数据加载层
│   ├── base.py                        # DataLoader 基类 + 工厂
│   ├── csv_loader.py                  # 微信 CSV 加载器
│   └── pdf_loader.py                  # PDF 文档加载器
├── rag/                           # RAG 引擎层
│   ├── rag_engine.py                  # 向量检索引擎
│   ├── query_processor.py            # 查询优化（改写 / 指代消解）
│   └── self_rag.py                    # Self-RAG
├── services/                      # 业务服务层
│   ├── rag_service.py                 # 数字分身服务
│   ├── textbook_rag_service.py        # 数字助教服务
│   ├── import_service.py              # 通用导入服务
│   └── multimodal_pdf_service.py      # 多模态 PDF 导入服务
├── run_server.py                  # 服务入口
├── import_wechat_csv.py           # CSV 导入脚本
├── import_pdf.py                  # 旧版 PDF 导入脚本
├── export_pdf_assets.py           # PDF 文本块 / 图片导出脚本
├── rebuild_multimodal_pdf_index.py # 多模态 PDF 重建脚本
└── import_course_materials.py     # 课程材料导入脚本

frontend/                          # 前端页面
scripts/                           # 启动 / 导入脚本
tests/                             # 测试
docs/                              # 文档
monitoring/                        # 监控栈配置
```

## 教材检索说明

当前数字助教采用双路教材检索：
- 文本块检索：`textbook_mm_text_embeddings`
- 图片检索：`textbook_mm_image_embeddings`

回答阶段会：
- 将命中的图片作为多模态输入发送给视觉模型
- 在回答中使用 `[图1]`、`[图2]` 等占位符
- 由前端将图片嵌入到回答正文中

`textbook.pdf` 的 OCR 文本目前会导入到 `textbook_ocr_text_embeddings`，用于保留扫描教材的文本索引。

## 监控

项目集成了完整的可观测性栈（需要 Docker）：

```bash
# 启动监控服务（Prometheus + Loki + Grafana）
bash scripts/start_monitoring.sh

# 停止
bash scripts/stop_monitoring.sh
```

## 测试

```bash
pytest tests/ -v
```

## 文档

- [架构设计](./docs/architecture.md)
- [API 接口](./docs/api.md)
- [迁移指南](./docs/MIGRATION.md)（从旧版 DigitalTwin 迁移）

## 环境变量

详见 [.env.example](./.env.example)，主要配置项：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_API_KEY` | DashScope API 密钥 | — |
| `CHAT_MODEL` | 对话模型 | `qwen-plus` |
| `TUTOR_VL_MODEL` | 助教视觉模型 | `qwen-vl-plus` |
| `EMBED_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `MM_EMBED_MODEL` | 多模态嵌入模型 | `multimodal-embedding-v1` |
| `CHROMA_PERSIST_DIR` | ChromaDB 存储路径 | `./chroma_db` |
| `PDF_EXPORT_ROOT` | 教材图片导出根目录 | `./output` |
| `TUTOR_MM_TEXT_COLLECTION` | 助教多模态文本 collection | `textbook_mm_text_embeddings` |
| `TUTOR_MM_IMAGE_COLLECTION` | 助教多模态图片 collection | `textbook_mm_image_embeddings` |
| `TUTOR_OCR_TEXT_COLLECTION` | OCR 文本 collection | `textbook_ocr_text_embeddings` |
| `DASHSCOPE_MM_MAX_CONCURRENCY` | 多模态 embedding 全局并发门限 | `2` |
| `DASHSCOPE_TEXT_MAX_CONCURRENCY` | 文本 embedding 全局并发门限 | `2` |
| `RAG_QUERY_REWRITING_ENABLED` | 查询改写 | `true` |
| `RAG_COREFERENCE_RESOLUTION_ENABLED` | 指代消解 | `true` |
| `OTEL_ENABLED` | 启用 OpenTelemetry | `false` |
