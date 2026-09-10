# 检索评测

评测集采用 JSONL，每行包含一个查询和至少一个相关结果标签：

```json
{"id":"chat-001","query":"我以前说过最喜欢什么运动？","persona":{"name":"示例人物"},"relevant":[{"source_file":"example.csv","content_contains":"羽毛球"}]}
```

`relevant` 中除 `content_contains` 外的字段均与检索结果 metadata 精确匹配。
标注集应使用脱敏数据，正式实验建议至少包含近千条人工核验查询。

运行基础相似度检索与“查询改写 + 时间过滤 + 混合召回 + 重排 + 邻域扩展 + 上下文预算”对比：

```bash
python -m src.cli.evaluate_retrieval \
  --dataset evaluation/retrieval_queries.jsonl \
  --collection persona_xxxxxxxx \
  --k 5
```

首次运行可以通过 `--max-cases 20` 控制模型调用成本。报告默认保存到
`evaluation/results/`，包含 Hit Rate@K、MRR@K、Recall@K、Context Precision@K、平均/P50/P95 延迟，
以及提升和退化的样本编号。

使用 `--no-hybrid-search`、`--no-reranking`、`--no-metadata-filtering`、
`--no-context-optimization`、`--no-query-rewriting` 或 `--no-neighbor-expansion`
可分别关闭混合召回、候选重排、时间过滤、上下文预算、查询改写和邻域扩展
进行消融实验。查询改写和重排都会产生模型调用，试跑时建议配合
`--max-cases` 控制成本。旧索引没有
`conversation_id` 和 `message_index` 时会自动保留原语义检索结果，不会报错。

时间范围过滤要求人物索引中的 `chat_time` 为秒级数值。旧索引包含字符串日期或
毫秒时间戳时，应先全量重新导入聊天数据。

示例文件只说明数据格式，不代表真实实验结果。简历中的指标必须以完整标注集生成的报告为准。

## 回答级评测

复制回答评测示例，并填入系统实际生成的答案及其实际上下文：

```powershell
Copy-Item evaluation/answer_cases.example.jsonl evaluation/answer_cases.jsonl
python -m src.cli.evaluate_answers --dataset evaluation/answer_cases.jsonl
```

每条记录包含 `query`、`contexts`、`answer`、`answerable`，可选
`expected_contains`。默认报告包含：

- `abstention_accuracy`：可回答问题正常回答、不可回答问题正确拒答的比例。
- `citation_precision`：回答中的文本引用有多少落在有效上下文编号内。
- `citation_coverage`：可回答且未拒答的样本中，有多少至少包含一个有效引用。
- `answer_keyword_recall`：人工标注关键词在答案中的覆盖率。

需要语义级忠实度时显式添加 `--llm-judge`。该模式每个样本增加一次模型调用，
输出 `groundedness`，适合在小规模人工复核集上运行；不要把未经人工抽检的模型评分直接写进简历。

## 证据阈值校准

证据门禁的目标不是判断“有没有返回结果”，而是判断最高相关结果是否足以支撑回答。每条 JSONL 记录标注一个文本或图片检索结果：

```json
{"id":"text-hard-negative-1","modality":"text","score":0.63,"sufficient":false,"category":"same_topic_wrong_fact"}
```

`score` 使用线上门禁所见的 0 到 1 相似度，`sufficient` 由人工判断该证据能否支撑查询。负例应包含同主题错误事实、相邻但不回答问题的段落和视觉相似错误页面，而不应只有随机无关内容。

```powershell
Copy-Item evaluation/evidence_calibration.example.jsonl evaluation/evidence_calibration.jsonl
python -m src.cli.calibrate_evidence --dataset evaluation/evidence_calibration.jsonl
```

校准器按文本与图片分别选择 F1 最高的阈值；F1 并列时优先降低误放行率。报告中的建议值需经过独立验证集和人工错误分析，再写入 `.env`。示例数据只展示格式，不能作为简历指标。
