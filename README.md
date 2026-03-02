# DigitalTwin - 数字孪生聊天机器人

基于 RAG（检索增强生成）技术的个性化 AI 聊天机器人系统，通过分析微信聊天记录，创建能够模拟你交流风格的数字孪生体。

## 项目简介

DigitalTwin 是一个智能对话系统，它可以：

- 导入并分析你的微信聊天历史记录
- 使用向量数据库存储对话语义
- 基于历史对话风格生成个性化回复
- 提供友好的 Web 聊天界面

通过机器学习和自然语言处理技术，系统能够学习你的说话方式、常用词汇和思维模式，创建一个"数字版的你"。

## 功能特性

- ✨ **智能对话**：基于 Qwen 大语言模型，生成自然流畅的对话
- 🔍 **语义检索**：使用向量数据库快速检索相关历史对话
- 🎯 **个性化回复**：模拟你的交流风格和语言习惯
- 💾 **会话管理**：支持多会话管理，可随时重置对话
- 🌙 **界面友好**：现代化的聊天界面，支持深色模式
- 🎤 **语音输入**：支持浏览器语音输入功能（部分浏览器）
- 📈 **增量更新**：智能识别新数据，避免重复导入

## 技术栈

### 后端

- **框架**：Flask 3.0.0
- **AI 引擎**：
  - Qwen（通义千问）大语言模型
  - DashScope Embeddings (text-embedding-v3)
  - LangChain RAG 框架
- **数据库**：ChromaDB 本地向量数据库（无需独立安装服务）
- **语言**：Python 3.8+

### 前端

- HTML5 + CSS3 + JavaScript
- 原生实现，无需额外框架
- 响应式设计

## 系统架构

```
┌─────────────┐
│   用户输入   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│      Flask Web Server       │
│  ┌───────────────────────┐  │
│  │   Chat API Endpoint   │  │
│  └──────────┬────────────┘  │
│             │                │
│             ▼                │
│  ┌───────────────────────┐  │
│  │    RAG Service        │  │
│  │  ┌─────────────────┐  │  │
│  │  │ 向量检索         │  │  │
│  │  │ ChromaDB        │  │  │
│  │  └─────────────────┘  │  │
│  │  ┌─────────────────┐  │  │
│  │  │ 语义匹配         │  │  │
│  │  │ DashScope       │  │  │
│  │  └─────────────────┘  │  │
│  └───────────┬───────────┘  │
│              │               │
│              ▼               │
│  ┌───────────────────────┐  │
│  │   Qwen LLM Generator  │  │
│  └───────────┬───────────┘  │
└──────────────┼───────────────┘
               │
               ▼
         ┌──────────┐
         │   响应    │
         └──────────┘
```

## 安装与启动步骤

### 1. 环境要求

- Python 3.8 或更高版本
- 阿里云 DashScope API 密钥

### 2. 安装依赖

在项目根目录下，运行以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

> **注意**：部分 `langchain` 相关依赖包首次导入时会比较慢，请耐心等待。

### 3. 配置环境变量

复制 `.env.example` 文件为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的配置（主要确保 `QWEN_API_KEY` 正确填写）：

```env
# Qwen API配置 (同时用于DashScope Embedding)
QWEN_API_KEY=your_dashscope_api_key_here
QWEN_MODEL=qwen-plus

# ChromaDB 本地向量数据库配置
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=wechat_embeddings

# Flask服务配置
FLASK_HOST=0.0.0.0
FLASK_PORT=8080
```

### 4. 导入聊天数据

1. 确保在项目根目录下有一个 `csv/` 文件夹。
2. 将导出的微信聊天记录 CSV 文件（例如 `zhenhuan_chat.csv`）放置在该文件夹中。
3. 运行数据导入脚本：

```bash
python test_csv_final.py
```

脚本启动阶段会加载 `langchain` 相关依赖，可能需要 30~60 秒，请耐心等待。加载完成后，会出现如下提示：

```
请选择导入模式:
1. 全量导入（清空数据库重新导入所有数据）
2. 增量更新（只导入新增的聊天记录）
请输入选项 (1/2，默认为2):
```

- **初次使用**：输入 `1` 进行全量导入。
- **后续更新**：输入 `2` 进行增量更新，可以智能去重。

脚本会自动：

- 读取 `csv/` 文件夹下的 CSV 文件
- 过滤无关消息，生成向量嵌入
- 存储到本地的 `chroma_db/` 文件夹

### 5. 启动服务器

数据导入完成后，启动 Flask 后端服务器：

```bash
python app.py
```

服务器将在 `http://localhost:8080` 启动，提示：

```
🚀 Flask聊天机器人服务器启动中...
✅ RAG服务初始化成功
🏠 本地访问: http://localhost:8080
```

### 6. 访问 Web 界面

打开浏览器访问：

```
http://localhost:8080
```

你将看到聊天界面，可以开始与你的数字孪生体对话。后续启动只需运行 `python app.py` 即可，无需再次录入数据。

## API 接口

#### 1. 发送消息

```http
POST /chat
Content-Type: application/json

{
  "message": "你好",
  "session_id": "session-123"
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

#### 2. 重置会话

```http
POST /reset
Content-Type: application/json

{
  "session_id": "session-123"
}
```

#### 3. 健康检查和运行统计

```http
GET /health
GET /stats
```

## 项目结构

```
DigitalTwin/
├── app.py                 # Flask 主应用入口
├── rag_service.py         # RAG 向量数据库和服务管理
├── rag_client.py          # RAG 相关调用
├── test_csv_final.py      # 将聊天记录生成向量数据到 ChromaDB 
├── requirements.txt       # Python 依赖列表
├── .env.example           # 环境变量配置模板
├── README.md              # 项目文档（本文件）
├── front/                 # 前端界面文件
│   ├── index.html         # 主聊天界面
│   ├── script.js          # 客户端逻辑
│   ├── styles.css         # 样式文件
│   └── test.html          # 测试页面
├── csv/                   # 微信聊天数据存放目录
└── chroma_db/             # 本地向量数据库存储目录
```

## 常见问题

### Q: 如何获取 DashScope API 密钥？

A: 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)，注册并创建 API 密钥。

### Q: 运行 `test_csv_final.py` 或 `app.py` 没反应？

A: `langchain` 库中诸多模块首次引入（导包）需要较长时间（可能会等待数十秒）。控制台虽然没有任何输出，但实际上程序正在运行加载中。请耐心等待出现控制台提示输出。

### Q: 找不到 `csv` 文件夹直接退出？

A: 请确保在项目根目录手动建立 `csv` 文件夹并放入含有你需要导入分析的对话 CSV 源数据。

### Q: 如何提高回复的准确性？

A: 尝试调整 `.env` 下的参数，或者充实个人的聊天数据源。
