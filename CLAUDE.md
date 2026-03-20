# 项目速查

## 行为规范

- 在完成任务后，同步更新项目目录下的 CLAUDE.md ，更新时只在 CLAUDE.md 中已有的内容模块中做增减 ，如非必要不添加新内容 ，例如对某部分的详细说明，即使添加也先询问
- 关于模型的选择以及其他项目配置，先在 @.env 和 @.env.example 中添加或修改相关配置，在实际代码中使用 @.env 中的配置
- 优先组合而非继承：通过注入独立组件（DBClient、LLMClient 等）来组装服务，避免深层继承链

## 项目概述

- 对 DigitalTwin 项目的重构，统一 RAG 架构，支持数字分身与数字助教
- 分层架构：Infrastructure → Loaders → RAG Engine → Services → API

## 常用命令

```bash
# 激活环境
mamba activate DT

# 安装依赖（开发模式）
pip install -e .

# 启动主服务（在项目根目录运行）
python -m src.run_server
# 默认监听 0.0.0.0:8080
# 数字分身: http://localhost:8080
# 数字助教: http://localhost:8080/tutor

# 导入微信聊天 CSV
python -m src.import_wechat_csv

# 导入 PDF 教材
python -m src.import_pdf

# 运行测试
pytest tests/ -v

# 监控栈（Prometheus + Loki + Grafana）
bash scripts/start_monitoring.sh
bash scripts/stop_monitoring.sh
```

## 项目结构

```
DigitalTwin-Refactor/
├── src/
│   ├── api/                       # Flask 路由层
│   │   ├── app.py                     # 应用工厂 & 蓝图注册
│   │   ├── config.py                  # 配置管理
│   │   └── routes/                    # 路由端点（chatbot / persona / tutor）
│   ├── infrastructure/            # 基础设施层
│   │   ├── db_client.py               # ChromaDB 客户端
│   │   ├── llm_client.py             # DashScope LLM 客户端
│   │   ├── telemetry.py              # OpenTelemetry 配置
│   │   ├── persona_manager.py        # 分身元数据管理
│   │   └── document.py               # 统一文档数据模型
│   ├── loaders/                   # 数据加载层
│   │   ├── base.py                    # DataLoader 基类 + 工厂
│   │   ├── csv_loader.py             # 微信 CSV 加载器
│   │   └── pdf_loader.py             # PDF 文档加载器
│   ├── rag/                       # RAG 引擎层
│   │   ├── rag_engine.py             # 向量检索引擎
│   │   ├── query_processor.py        # 查询优化（改写 / 指代消解）
│   │   └── self_rag.py               # Self-RAG 实现
│   ├── services/                  # 业务服务层
│   │   ├── rag_service.py            # 数字分身 RAG 服务
│   │   ├── textbook_rag_service.py   # 数字助教 RAG 服务
│   │   └── import_service.py         # 数据导入服务
│   ├── run_server.py              # 服务入口
│   ├── import_wechat_csv.py       # CSV 导入脚本
│   └── import_pdf.py              # PDF 导入脚本
├── frontend/                  # 前端静态文件
├── tests/                     # 测试用例
├── data/                      # 数据目录（csv/ pdf/）
├── chroma_db/                 # ChromaDB 持久化
├── docs/                      # 文档（architecture.md / api.md / MIGRATION.md）
├── scripts/                   # 监控栈启停脚本
├── monitoring/                # 监控栈（Prometheus + Loki + Grafana）
│   ├── prometheus.yml             # Prometheus 配置
│   ├── prometheus_data/           # Prometheus 持久化数据
│   ├── loki-config.yml            # Loki 配置
│   ├── loki_data/                 # Loki 持久化数据
│   └── grafana_data/              # Grafana 持久化数据
├── docker-compose.yaml        # 监控服务编排
├── pyproject.toml
├── requirements.txt
├── .env / .env.example
└── README.md
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Flask 3.0 + Flask-CORS |
| LLM | 通义千问（qwen-plus）via DashScope API |
| 向量数据库 | ChromaDB（本地持久化） |
| 嵌入模型 | DashScope text-embedding-v4 |
| RAG 框架 | LangChain（langchain-chroma、langchain-community） |
| 可观测性 | OpenTelemetry + Prometheus + Loki + Grafana |
| 前端 | 原生 HTML/CSS/JS |
| 环境管理 | Conda（使用 miniforge 环境名：DT） |

## 文档索引

- `README.md` — 项目概述
- `docs/architecture.md` — 系统架构与设计
- `docs/api.md` — API 参考与示例
- `docs/MIGRATION.md` — 从 DigitalTwin 迁移指南
