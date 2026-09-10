# 检索评测

评测集采用 JSONL，每行包含一个查询和至少一个相关结果标签：

```json
{"id":"chat-001","query":"我以前说过最喜欢什么运动？","persona":{"name":"示例人物"},"relevant":[{"source_file":"example.csv","content_contains":"羽毛球"}]}
```

`relevant` 中除 `content_contains` 外的字段均与检索结果 metadata 精确匹配。
标注集应使用脱敏数据，正式实验建议至少包含近千条人工核验查询。

运行基础相似度检索与“查询改写 + 混合召回 + LLM 重排 + 时间邻域扩展”对比：

```bash
python -m src.cli.evaluate_retrieval \
  --dataset evaluation/retrieval_queries.jsonl \
  --collection persona_xxxxxxxx \
  --k 5
```

首次运行可以通过 `--max-cases 20` 控制模型调用成本。报告默认保存到
`evaluation/results/`，包含 Hit Rate@K、MRR@K、Recall@K、平均/P50/P95 延迟，
以及提升和退化的样本编号。

使用 `--no-hybrid-search`、`--no-reranking`、`--no-query-rewriting` 或
`--no-neighbor-expansion` 可分别关闭混合召回、候选重排、查询改写和邻域扩展
进行消融实验。查询改写和重排都会产生模型调用，试跑时建议配合
`--max-cases` 控制成本。旧索引没有
`conversation_id` 和 `message_index` 时会自动保留原语义检索结果，不会报错。

示例文件只说明数据格式，不代表真实实验结果。简历中的指标必须以完整标注集生成的报告为准。
