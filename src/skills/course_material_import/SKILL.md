---
name: course_material_import
description: 用于重载导入核心的课程材料PDF和教学教材，构建多模态索引或OCR索引。
---

# Course Material Import Skill
批量解析复杂包含表格、图片的 PDF 并执行对应的文本块划分。

## 用法
CLI 调用入口：
`python -m src.skills.course_material_import.scripts.import_course_materials_cli`

同时提供了标准 Agent 的调用基类接口支持。
