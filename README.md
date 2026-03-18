# DigitalTwin RAG 重构版本

重构后的 DigitalTwin 项目，统一了数字分身和数字助教的 RAG 架构。

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置环境变量
```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key 等配置
```

### 运行测试
```bash
pytest tests/ -v
```

## 架构

- `src/infrastructure/` - 基础设施层（LLMClient、DBClient、Telemetry）
- `src/loaders/` - 数据加载层（CSV、PDF 等）
- `src/rag/` - RAG 引擎层
- `src/services/` - 对外服务层
- `tests/` - 单元和集成测试

详见 `docs/architecture.md`
