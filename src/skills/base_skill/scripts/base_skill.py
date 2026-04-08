from abc import ABC, abstractmethod

class BaseSkill(ABC):
    """
    Agent 可以调用的基础技能接口
    """
    # 技能的名称，Agent 将使用此名称调用技能
    name: str = ""
    # 技能的描述，告诉 Agent 该技能的用途和参数预期
    description: str = ""

    @abstractmethod
    def run(self, action_input: str) -> str:
        """
        执行技能的逻辑
        
        Args:
            action_input: Agent 传入的参数字符串
            
        Returns:
            执行结果的字符串表示（Observation）
        """
        pass
