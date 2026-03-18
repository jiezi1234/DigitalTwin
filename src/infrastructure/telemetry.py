"""
OpenTelemetry 配置和初始化模块
支持可配置的追踪级别：light、full、custom
"""

import os
import logging
import threading
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

# Thread-safe singleton lock
_singleton_lock = threading.Lock()

# 默认的高层操作追踪
_LIGHT_SPANS = {
    "llm.api_call",
    "db.vector_search",
    "loader.load",
}

# 完整追踪包括中间步骤
_FULL_SPANS = _LIGHT_SPANS | {
    "rag.search",
    "query.process",
    "query.coreference_resolution",
    "query.rewriting",
    "db.connect",
    "format.context",
}


class TelemetryManager:
    """管理 OpenTelemetry 生命周期和配置"""

    def __init__(self):
        self.enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"
        self.trace_level = os.getenv("OTEL_TRACE_LEVEL", "light")
        self._tracer_provider: Optional[TracerProvider] = None

        if self.enabled:
            self._init_tracer_provider()
            logger.info(f"OpenTelemetry 已启用，追踪级别: {self.trace_level}")
        else:
            logger.debug("OpenTelemetry 已禁用")

    def _init_tracer_provider(self):
        """初始化 TracerProvider 和 Exporter"""
        resource = Resource.create({
            "service.name": "digitaltwin-rag",
            "service.version": "0.1.0",
        })

        self._tracer_provider = TracerProvider(resource=resource)

        # 仅在启用导出时才配置 OTLP 导出器
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            try:
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                self._tracer_provider.add_span_processor(
                    BatchSpanProcessor(exporter)
                )
                logger.info(f"OTLP 导出器已配置: {otlp_endpoint}")
            except (ConnectionError, ValueError, RuntimeError) as e:
                logger.warning(f"OTLP 导出器配置失败: {e}")
            except Exception as e:
                logger.warning(f"OTLP 导出器配置失败 (unexpected error): {e}")

        # 设置全局 tracer provider
        trace.set_tracer_provider(self._tracer_provider)

    def should_trace(self, span_name: str) -> bool:
        """根据配置判断是否应该追踪某个 span"""
        if not self.enabled:
            return False

        if self.trace_level == "full":
            return span_name in _FULL_SPANS
        elif self.trace_level == "light":
            return span_name in _LIGHT_SPANS
        elif self.trace_level == "custom":
            custom_patterns = os.getenv("OTEL_CUSTOM_SPAN_PATTERNS", "").split(",")
            # Exact match: check if span_name exactly matches any pattern
            return any(span_name == pattern.strip() for pattern in custom_patterns if pattern.strip())
        return False

    def get_tracer(self, name: str) -> trace.Tracer:
        """获取命名的 tracer"""
        if self._tracer_provider:
            return self._tracer_provider.get_tracer(name)
        return trace.get_tracer(name)

    def shutdown(self):
        """关闭 tracer provider 并清理资源"""
        if self._tracer_provider:
            try:
                self._tracer_provider.force_flush(timeout_millis=5000)
                self._tracer_provider.shutdown()
                logger.info("Telemetry manager 已关闭")
            except Exception as e:
                logger.warning(f"关闭 telemetry 时出错: {e}")


# 全局单例
_telemetry_manager: Optional[TelemetryManager] = None


def init_telemetry() -> TelemetryManager:
    """初始化全局 telemetry manager（线程安全）"""
    global _telemetry_manager
    if _telemetry_manager is None:
        with _singleton_lock:
            if _telemetry_manager is None:
                _telemetry_manager = TelemetryManager()
    return _telemetry_manager


def get_telemetry() -> TelemetryManager:
    """获取全局 telemetry manager"""
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = init_telemetry()
    return _telemetry_manager


def get_tracer(name: str) -> trace.Tracer:
    """获取命名的 tracer（便捷函数）"""
    return get_telemetry().get_tracer(name)
