---
name: multimodal_pdf_rebuild
description: 批量重建 PDF 多模态向量索引。
---

# Multimodal PDF Rebuild Skill
清空及重建文本区与图片区的两路独立集合索引。应对模型效果发生漂移或库重置时的救场机制。

## 用法
CLI 调用入口：
`python -m src.skills.multimodal_pdf_rebuild.scripts.rebuild_multimodal_pdf_index_cli "data/pdf/*.pdf"`
