# DigitalTwin - 数字分身聊天机器人

基于 RAG（检索增强生成）技术的个性化 AI 聊天机器人系统，通过分析微信聊天记录，创建能够模拟你交流风格的数字分身。

## 项目简介

DigitalTwin 是一个智能对话系统，它可以：

- 导入并分析你的微信聊天历史记录
- 使用向量数据库存储对话语义
- 基于历史对话风格生成个性化回复
- 支持创建和管理多个数字分身
- 提供友好的 Web 聊天界面

通过 RAG 检索增强和大语言模型技术，系统能够学习你的说话方式、常用词汇和思维模式，创建一个"数字版的你"。

## 功能特性

- **智能对话**：基于 Qwen 大语言模型，生成自然流畅的对话
- **语义检索**：使用 MMR 算法兼顾相关性与多样性，检索历史对话
- **时间上下文**：自动扩展时间窗口内的相邻消息，还原对话场景
- **多分身管理**：支持创建多个数字分身，各分身独立参数和向量集合
- **个性化回复**：模拟你的交流风格和语言习惯
- **会话管理**：支持多会话管理，可随时重置对话
- **增量更新**：智能识别新数据，避免重复导入
- **CSV 预处理**：导入前自动去重、过滤无效消息

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Flask 3.0 + Flask-CORS |
| LLM | 通义千问（qwen-plus）via DashScope API |
| 向量数据库 | ChromaDB（本地持久化，无需独立数据库服务） |
| 嵌入模型 | DashScope text-embedding-v4 |
| RAG 框架 | LangChain（langchain-chroma、langchain-community） |
| 前端 | 原生 HTML/CSS/JS，无框架 |
| 环境管理 | Conda / Miniforge（环境名：DT） |

## 系统架构

```
┌─────────────┐
│   用户输入   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│        Flask Web Server         │
│  ┌───────────────────────────┐  │
│  │   Chat / Persona API     │  │
│  └────────────┬──────────────┘  │
│               │                  │
│               ▼                  │
│  ┌───────────────────────────┐  │
│  │     PersonaManager        │  │
│  │  （多分身选择 & 参数加载） │  │
│  └────────────┬──────────────┘  │
│               │                  │
│               ▼                  │
│  ┌───────────────────────────┐  │
│  │      RAG Service          │  │
│  │  ┌─────────────────────┐  │  │
│  │  │ MMR 向量检索        │  │  │
│  │  │ + 时间窗口扩展      │  │  │
│  │  │ ChromaDB            │  │  │
│  │  └─────────────────────┘  │  │
│  │  ┌─────────────────────┐  │  │
│  │  │ DashScope Embedding │  │  │
│  │  └─────────────────────┘  │  │
│  └────────────┬──────────────┘  │
│               │                  │
│               ▼                  │
│  ┌───────────────────────────┐  │
│  │    Qwen LLM Generator     │  │
│  └────────────┬──────────────┘  │
└───────────────┼──────────────────┘
                │
                ▼
          ┌──────────┐
          │   响应    │
          └──────────┘
```

## 安装与启动步骤

### 1. 环境要求

- Python 3.8 或更高版本
- Conda / Miniforge
- 阿里云 DashScope API 密钥

### 2. 创建并激活环境

```bash
conda create -n DT python=3.11
conda activate DT
# 或使用 mamba
mamba activate DT
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 文件为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的配置（主要确保 `DASHSCOPE_API_KEY` 正确填写）：

```env
# API配置
DASHSCOPE_API_KEY=your_dashscope_api_key_here
CHAT_MODEL=qwen-plus
EMBED_MODEL=text-embedding-v4

# ChromaDB 本地向量数据库配置
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=wechat_embeddings

# Flask服务配置
FLASK_HOST=0.0.0.0
FLASK_PORT=8080
```

完整配置项参见 `.env.example`。

### 5. 导入聊天数据

1. 在项目根目录下创建 `csv/` 文件夹，放入导出的微信聊天记录 CSV 文件。
2. （可选）运行预处理脚本，去重和过滤无效消息：

```bash
python -m src.preprocess_csv
```

预处理后的文件输出到 `csv_clean/` 目录。

3. 运行数据导入脚本（在项目根目录运行）：

```bash
python -m src.test_csv_final
```

脚本会引导你选择或创建分身，并选择导入模式：

- **全量导入**：清空该分身的向量集合，重新导入所有数据
- **增量更新**：只导入新增的聊天记录，智能去重

脚本会自动读取 CSV 文件、过滤无关消息、并行生成向量嵌入、存储到 ChromaDB。

### 6. 启动服务器

```bash
python -m src.app
```

服务器将在 `http://localhost:8080` 启动。

### 7. 访问 Web 界面

打开浏览器访问 `http://localhost:8080`，选择分身后即可开始对话。

## API 接口

### 发送消息

```http
POST /chat
Content-Type: application/json

{
  "message": "你好",
  "session_id": "session-123",
  "persona_id": "可选，指定分身ID"
}
```

响应：

```json
{
  "status": "success",
  "reply": "你好！有什么可以帮你的吗？",
  "session_id": "session-123"
}
```

### 重置会话

```http
POST /reset
Content-Type: application/json

{
  "session_id": "session-123"
}
```

### 分身管理

```http
GET  /api/personas           # 列出所有分身
DELETE /api/personas/:id     # 删除分身
```

### 健康检查和运行统计

```http
GET /health
GET /stats
```

## 项目结构

```
DigitalTwin/
├── src/                    # 所有代码文件
│   ├── app.py                  # Flask 主服务，对话接口（路由层）
│   ├── test_csv_final.py       # CSV 数据导入 & 嵌入生成脚本
│   ├── preprocess_csv.py       # CSV 预处理（去重、过滤无效消息）
│   ├── core/                   # 核心业务逻辑
│   │   ├── rag_service.py          # RAG 向量检索服务（MMR + 时间窗口）
│   │   └── persona_manager.py      # 分身管理（personas.json CRUD）
│   ├── utils/                  # 通用工具函数
│   │   ├── csv_loader.py           # 微信聊天记录 CSV 加载器
│   │   └── tracking.py             # 增量导入跟踪（哈希去重）
│   └── front/                  # 前端静态文件
│       ├── index.html              # 主聊天界面
│       ├── script.js               # 客户端逻辑
│       ├── styles.css              # 样式文件
│       └── test.html               # 测试页面
├── csv/                    # 微信聊天记录 CSV 数据
├── csv_clean/              # 预处理后的 CSV 数据
├── chroma_db/              # ChromaDB 本地持久化目录（含 personas.json）
├── logs/                   # 日志目录
├── requirements.txt
├── .env                    # 环境变量（含 API Key，不入库）
└── .env.example            # 环境变量模板
```

## 常见问题

### Q: 如何获取 DashScope API 密钥？

A: 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)，注册并创建 API 密钥。

### Q: 运行脚本时长时间没有输出？

A: `langchain` 相关模块首次导入需要较长时间（可能数十秒），请耐心等待控制台出现提示。

### Q: 找不到 `csv` 文件夹直接退出？

A: 请在项目根目录手动创建 `csv/` 文件夹并放入聊天记录 CSV 文件。

### Q: 如何提高回复的准确性？

A: 可以通过 `personas.json` 中各分身的 `rag_params` 和 `model_params` 调整检索和生成参数，也可以充实聊天数据源。
