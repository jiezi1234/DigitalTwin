---
name: wechat_csv_import
description: 用于将导出的微信聊天记录 CSV 文件注入数字分身的向量库。
---

# WeChat CSV Import Skill
本技能对接底层的 CSV 解析 Loader，并将数据注入到特定的数字人设 (Persona) 库中。

## 用法
支持交互式运行 (CLI)：
`python -m src.skills.wechat_csv_import.scripts.import_wechat_csv_cli`

支持 Agent 原生接口调用。
