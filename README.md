# DigitalTwin RAG 重构项目

一个统一架构的数字分身和数字助教 RAG 系统，包含完整的 OpenTelemetry 可观测性支持。

## 🎯 项目目标

重构原 DigitalTwin 项目，解决以下问题：

- **代码重复**: 分身和助教两个 RAG 实现几乎完全相同
- **可维护性**: 架构不清晰，组件耦合
- **可扩展性**: 添加新的数据源或查询策略困难
- **可观测性**: 缺乏完整的追踪和监控

## ✨ 核心改进

### 分层架构

```
┌─────────────────────────────────────┐
│       Services Layer                │
│  (RAGService, TextbookRAGService)   │
├─────────────────────────────────────┤
│       RAG Engine Layer              │
│  (RAGEngine, QueryProcessor)        │
├─────────────────────────────────────┤
│       Data Loaders Layer            │
│  (DataLoaderFactory, CSV/PDF)       │
├─────────────────────────────────────┤
│     Infrastructure Layer            │
│  (LLMClient, DBClient, Telemetry)   │
└─────────────────────────────────────┘
```

### 统一接口

- **DataLoader**: 统一的数据加载接口，支持工厂模式
- **RAGEngine**: 统一的搜索引擎，支持多种格式化策略
- **QueryProcessor**: 统一的查询处理，支持可配置的策略
- **LLMClient**: 统一的 LLM API 调用，带自动追踪

## 📊 项目统计

| 指标 | 数值 |
|------|------|
| 总文件数 | 28 |
| 测试数 | 37 |
| 文档行数 | 1,400+ |
| 测试全部通过 | ✅ |

## 🚀 快速开始

### 安装

```bash
cd /home/moka/projects/DigitalTwin-Refactor
pip install -e .
```

### 配置

```bash
export DASHSCOPE_API_KEY=sk-xxx
export OTEL_ENABLED=true
```

### 运行测试

```bash
pytest tests/ -v
```

## 📚 文档

- **[架构设计](./docs/architecture.md)** - 系统架构和设计
- **[API 接口](./docs/api.md)** - 接口文档和示例
- **[迁移指南](./docs/MIGRATION.md)** - 从原项目迁移

## 🏗️ 项目结构

```
src/
├── infrastructure/    # 基础设施（LLM、DB、追踪）
├── loaders/          # 数据加载（CSV、PDF、工厂）
├── rag/              # RAG 引擎（查询处理、搜索）
├── services/         # 服务层（分身服务、助教服务）
├── models/           # 数据模型
├── core/             # 核心模块
└── utils/            # 工具函数

tests/                # 37 个测试
docs/                 # 完整文档
```

## ✅ 完成清单

- ✅ 基础设施层（LLMClient、DBClient、Telemetry）
- ✅ 数据加载层（DataLoader 工厂、CSV/PDF 加载器）
- ✅ RAG 引擎层（QueryProcessor、RAGEngine）
- ✅ 服务层（RAGService、TextbookRAGService）
- ✅ 单元测试（26 个）
- ✅ 集成测试（5 个）
- ✅ 系统测试（6 个）
- ✅ 完整文档（1,400+ 行）

**总计: 37 个测试全部通过 ✅**
