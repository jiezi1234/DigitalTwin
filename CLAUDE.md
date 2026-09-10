# 项目速查

## 行为规范

- 在完成任务后，同步更新项目目录下的 CLAUDE.md ，更新时只在 CLAUDE.md 中已有的内容模块中做增减 ，如非必要不添加新内容 ，例如对某部分的详细说明，即使添加也先询问
- 关于模型的选择以及其他项目配置，先在 @.env 和 @.env.example 中添加或修改相关配置，在实际代码中使用 @.env 中的配置
- 优先组合而非继承：通过注入独立组件（DBClient、LLMClient 等）来组装服务，避免深层继承链

## 项目概述

- 对 DigitalTwin 项目的重构，统一 RAG 架构，支持数字分身与数字助教
- 教材问答使用多模态文本、OCR 文本和图片三通道召回，文本结果通过 RRF 融合
- 人物对话使用 ReAct 路由在直接回答和调用检索工具之间选择
- 查询理解结合有限长度的最近会话，用一次结构化调用完成指代消解与查询改写
- 人物检索组合 Dense/MMR 与轻量中文 BM25，通过加权 RRF 融合并支持单通道降级
- 混合召回后使用一次结构化 LLM 调用重排 Top-N 候选，失败时保留原排名
- 查询理解产生的 ISO 时间范围通过统一过滤器下推到 Dense/MMR 与 BM25
- 生成前统一执行上下文近重复去除、单片段限长与整体字符预算，教材引用只映射实际入选片段
- 人物检索围绕语义命中消息按 conversation_id/message_index 扩展时间邻域并去重
- 分层架构：Infrastructure → Loaders → RAG Engine → Services → API

## 常用命令

```bash
# 激活环境
mamba activate DT

# 安装依赖（开发模式）
pip install -e ".[dev]"

# 启动主服务（在项目根目录运行）
python -m src.cli.run_server
# 默认监听 0.0.0.0:8080
# 数字分身: http://localhost:8080
# 数字助教: http://localhost:8080/tutor

# 导入微信聊天 CSV
python -m src.cli.import_wechat_csv

# 导入 PDF 教材
python -m src.cli.import_pdf

# 运行测试
python -m pytest
# 默认同时发现 test_*.py 与 *_tests.py

# 运行人物对话检索对比评测
python -m src.cli.evaluate_retrieval --dataset evaluation/retrieval_queries.jsonl --collection <collection> --k 5

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
│   ├── cli/                       # 服务启动、导入、索引与评测入口
│   ├── evaluation/                # 检索评测模型、指标与报告逻辑
│   ├── infrastructure/            # 基础设施层
│   │   ├── db_client.py               # ChromaDB 客户端
│   │   ├── llm_client.py             # DashScope LLM 客户端
│   │   ├── multimodal_embedding_client.py # 多模态向量客户端
│   │   ├── text_embedding_client.py   # 文本向量客户端
│   │   ├── telemetry.py              # OpenTelemetry 配置
│   │   ├── persona_manager.py        # 分身元数据管理
│   │   └── document.py               # 统一文档数据模型
│   ├── loaders/                   # 数据加载层
│   │   ├── base.py                    # DataLoader 基类 + 工厂
│   │   ├── csv_loader.py             # 微信 CSV 加载器
│   │   └── pdf_loader.py             # PDF 文档加载器
│   ├── rag/                       # RAG 引擎层
│   │   ├── rag_engine.py             # 混合召回、RRF 融合与邻域扩展
│   │   ├── bm25_retriever.py         # 中文友好的懒加载 BM25 检索器
│   │   ├── llm_reranker.py            # 可降级的结构化 LLM 候选重排器
│   │   ├── metadata_filter.py          # 查询时间范围到 Chroma 条件的转换
│   │   ├── context_builder.py          # 上下文去重、预算、截断与入选统计
│   │   ├── query_processor.py        # 历史感知查询理解（改写 / 指代消解）
│   │   └── react_router.py           # ReAct 检索工具路由
│   ├── services/                  # 业务服务层
│   │   ├── rag_service.py            # 数字分身 RAG 服务
│   │   ├── textbook_rag_service.py   # 数字助教 RAG 服务
│   │   ├── import_service.py          # 数据导入服务
│   │   └── multimodal_pdf_service.py  # 多模态 PDF 导入服务
├── frontend/                  # 前端静态文件
├── tests/                     # 测试用例
├── evaluation/                # 检索标注格式、评测说明与本地结果
├── data/                      # 数据目录（csv/ pdf/）
├── chroma_db/                 # ChromaDB 持久化
├── docs/                      # 文档（architecture.md / api.md / MIGRATION.md）
├── scripts/                   # 服务、课程导入与监控脚本
├── monitoring/                # 监控栈（Prometheus + Loki + Grafana）
│   ├── prometheus.yml             # Prometheus 配置
│   ├── loki-config.yml            # Loki 配置
│   └── tempo-config.yml          # Tempo 配置
├── pyproject.toml
├── requirements.txt
├── .env / .env.example
└── README.md
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Flask 3.0 + Flask-CORS |
| LLM | 通义千问（qwen-plus / qwen-vl-plus）via DashScope API |
| 向量数据库 | ChromaDB（本地持久化） |
| 嵌入模型 | DashScope text-embedding-v4 / multimodal-embedding-v1 |
| RAG 框架 | LangChain（langchain-chroma、langchain-community） |
| 可观测性 | OpenTelemetry + Prometheus + Loki + Grafana |
| 前端 | 原生 HTML/CSS/JS |
| 环境管理 | Conda（使用 miniforge 环境名：DT） |

## 文档索引

- `README.md` — 项目概述
- `docs/architecture.md` — 系统架构与设计
- `docs/api.md` — API 参考与示例
- `docs/MIGRATION.md` — 从 DigitalTwin 迁移指南
