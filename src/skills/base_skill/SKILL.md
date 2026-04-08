---
name: base_skill
description: 智能体底层能够挂载的统一技能规范（接口）。
---

# Base Skill
此组件定义了 Digital Twin Agent 能够调用的所有标准衍生技能必须继承的基础类 `BaseSkill`。

## 规格要求
所有实现类的子技能，必须覆写：
- `name` (String): Agent 调用工具时传递的精确代号名。
- `description` (String): 描述工具的功能及何时被触发，这是放入系统提示词用于告诉大模型何时使用的核心文档。
- `run(self, action_input: str) -> str`: 具体执行操作的方法，接收模型的单个字符串输入，并必须返回字符串格式的观测结果(Observation)。
