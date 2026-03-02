"""
RAG API 客户端封装
用于连接和查询RAG向量数据库服务
"""

import requests
from typing import Optional, Dict, Any


class RAGClient:
    """RAG API客户端类"""

    def __init__(self, server_ip: str = "127.0.0.1", port: int = 12345):
        """
        初始化RAG客户端
        :param server_ip: RAG服务器IP地址
        :param port: RAG服务端口
        """
        self.base_url = f"http://{server_ip}:{port}"
        self.server_ip = server_ip
        self.port = port

        # 测试连接
        if not self._test_connection():
            raise ConnectionError(f"无法连接到RAG服务器: {self.base_url}")

    def _test_connection(self) -> bool:
        """测试与RAG服务器的连接"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def query(
        self,
        question: str,
        max_results: int = 50,
        similarity_threshold: float = 0.0,
        include_nearby: bool = True,
        time_window_minutes: int = 30,
        nearby_per_result: int = 8
    ) -> Optional[Dict[str, Any]]:
        """
        查询RAG服务获取相关上下文
        :param question: 用户问题
        :param max_results: 最大返回结果数(默认50,上限50)
        :param similarity_threshold: 相似度阈值
        :param include_nearby: 是否包含时间相近的记录(默认True)
        :param time_window_minutes: 时间窗口(分钟,默认30分钟)
        :param nearby_per_result: 每个相似结果附近获取的记录数(默认8条)
        :return: 查询结果字典，失败返回None
        """
        try:
            # 使用简化的查询接口
            response = requests.post(
                f"{self.base_url}/query_simple",
                params={
                    "question": question,
                    "max_results": max_results,
                    "include_nearby": include_nearby,
                    "time_window_minutes": time_window_minutes,
                    "nearby_per_result": nearby_per_result
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"RAG查询失败，状态码: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            print("RAG查询超时")
            return None
        except requests.exceptions.ConnectionError:
            print(f"无法连接到RAG服务器: {self.base_url}")
            return None
        except Exception as e:
            print(f"RAG查询异常: {e}")
            return None

    def get_health_status(self) -> Optional[Dict[str, Any]]:
        """获取RAG服务健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"获取健康状态失败: {e}")
            return None

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """获取RAG服务统计信息"""
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return None
