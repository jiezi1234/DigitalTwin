---
name: chat_history_retrieval
description: RAG 搜索技能，用于根据关键词调取数字分身的过往记录。
---

# Chat History Retrieval Skill
本技能负责挂载 `RAGEngine`，通过向量相近度匹配用户聊天历史。

## 用法
专为基于 ReAct 模式调度的 AI Agent 提供数据挂载服务。
它会将诸如 `"吃了什么"` 这样的用户模糊/间接意图词，传入重构引擎或向量检索引擎，最终组装返回一串标准格式的、时间相近的用户聊天记录作为大语言模型下一步的系统观测值 (Observation)。
