# 项目速查

## 行为规范

- 在完成任务后，同步更新项目目录下的 CLAUDE.md ，更新时只在 CLAUDE.md 中已有的内容模块中做增减 ，如非必要不添加新内容 ，例如对某部分的详细说明，即使添加也先询问
- 关于模型的选择以及其他项目配置，先在 @.env 和 @.env.example 中添加或修改相关配置，在实际代码中使用 @.env 中的配置

## 项目概述

- 将微信聊天记录转化为向量数据库

## 常用命令

```bash
# 激活环境
mamba activate DT

# 安装依赖
pip install -r requirements.txt

# 导入聊天数据并生成向量嵌入（在项目根目录运行）
python -m src.test_csv_final

# 导入课本PDF到向量数据库（数字助教）
python -m src.import_textbook

# 启动主服务（在项目根目录运行）
python -m src.app
# 默认监听 0.0.0.0:8080
# 数字分身（统一页面，含助教切换）: http://localhost:8080
# 数字助教（独立页面）: http://localhost:8080/tutor
```

## 项目结构

```
DigitalTwin/
├── src/                # 所有代码文件
│   ├── app.py              # Flask 主服务，对话接口（路由层）
│   ├── test_csv_final.py   # CSV 数据导入 & 嵌入生成脚本（ThreadPoolExecutor 并行嵌入，绕过 LangChain 直写 ChromaDB）
│   ├── preprocess_csv      # csv文件预处理，包含去重
│   ├── models/             # 数据模型
│   │   ├── __init__.py
│   │   └── chat_record_model.py  # 标准化聊天记录模型（支持多种CSV格式）
│   ├── core/               # 核心业务逻辑
│   │   ├── rag_service.py      # RAG 向量检索服务
│   │   ├── self_rag.py         # Self-RAG 反思增强检索（支持 HF / Qwen 双后端）
│   │   ├── textbook_rag_service.py  # 课本RAG检索服务（数字助教）
│   │   └── persona_manager.py  # 分身管理（personas.json CRUD）
│   ├── utils/              # 通用工具函数
│   │   ├── csv_loader.py       # 微信聊天记录 CSV 加载器（自动格式检测）
│   │   ├── doc_loader.py       # PDF文档加载器（数字助教）
│   │   └── tracking.py         # 增量导入跟踪（哈希去重）
│   └── front/              # 前端静态文件（AURA 风格，数字分身+助教统一页面）
├── csv/                # 微信聊天记录 CSV 数据
├── csv_clean/          # 预处理后的 CSV 数据
├── chroma_db/          # ChromaDB 本地持久化目录（含 personas.json）
├── logs/app.log        # 日志文件
├── requirements.txt
├── .env                # 环境变量（含 API Key，不入库）
└── .env.example        # 环境变量模板
```


## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Flask 3.0 + Flask-CORS |
| LLM | 通义千问（qwen-plus）via DashScope API |
| 向量数据库 | ChromaDB（本地持久化） |
| 嵌入模型 | DashScope text-embedding-v4（每次 API 请求上限 10 条）|
| RAG 框架 | LangChain（langchain-chroma、langchain-community） |
| 前端 | 原生 HTML/CSS/JS（AURA 设计风格，Three.js WebGL 背景，Phosphor Icons） |
| 环境管理 | Conda（使用miniforge  环境名：DT） |

## 文档索引

- `README.md` — 完整部署指南、API 说明、架构图
- `doc/数据模型说明` — 对系统用于导入向量数据库的CSV文件字段的说明
- `doc/设计文档.docx` — 系统设计文档
- `doc/数字分身_ppt展示.pptx` — 项目演示 PPT
