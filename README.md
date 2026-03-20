# DigitalTwin

基于 RAG 的数字分身 & 数字助教系统。通过微信聊天记录构建个人数字分身，通过 PDF 教材构建 AI 助教。

## 功能

- **数字分身** — 导入微信聊天记录，生成模拟本人说话风格的 AI 对话
- **数字助教** — 导入 PDF 教材，基于教材内容回答学生提问
- **RAG 检索增强** — 查询改写、指代消解、MMR 多样性搜索
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
| LLM | 通义千问 (qwen-plus) via DashScope |
| 向量数据库 | ChromaDB |
| 嵌入模型 | DashScope text-embedding-v4 |
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

# 导入 PDF 教材（数字助教）
python -m src.import_pdf
```

### 启动服务

```bash
python -m src.run_server
```

- 数字分身：http://localhost:8080
- 数字助教：http://localhost:8080/tutor

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
│   └── import_service.py             # 数据导入服务
├── run_server.py                  # 服务入口
├── import_wechat_csv.py           # CSV 导入脚本
└── import_pdf.py                  # PDF 导入脚本

frontend/                          # 前端页面
tests/                             # 测试
docs/                              # 文档
monitoring/                        # 监控栈配置
```

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
| `EMBED_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `CHROMA_PERSIST_DIR` | ChromaDB 存储路径 | `./chroma_db` |
| `RAG_QUERY_REWRITING_ENABLED` | 查询改写 | `true` |
| `RAG_COREFERENCE_RESOLUTION_ENABLED` | 指代消解 | `true` |
| `OTEL_ENABLED` | 启用 OpenTelemetry | `false` |
