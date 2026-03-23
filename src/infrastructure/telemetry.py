"""
OpenTelemetry 配置和初始化模块
支持可配置的追踪级别：light、full、custom
"""

import os
import logging
import threading
from typing import Optional
# 基础 API 导入
try:
    from opentelemetry import trace, metrics
    HAS_OTEL_API = True
except ImportError:
    HAS_OTEL_API = False

# SDK 和导出器导入组件（延迟加载或受保护加载）
_OTLP_HTTP_AVAILABLE = False
try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    _OTLP_HTTP_AVAILABLE = True
except ImportError:
    OTLPSpanExporter = None
    OTLPLogExporter = None

_PROMETHEUS_AVAILABLE = False
try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    PrometheusMetricReader = None

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
        self._tracer_provider = None
        self._meter_provider = None
        self._logger_provider = None

        if self.enabled and not HAS_OTEL_API:
            logger.warning("OpenTelemetry API 未安装，无法启用可观测性功能。")
            self.enabled = False

        if self.enabled:
            try:
                self._init_telemetry()
                logger.info(f"OpenTelemetry 已启用，追踪级别: {self.trace_level}")
            except Exception as e:
                logger.warning(f"OpenTelemetry 初始化失败，已回退到禁用状态: {e}")
                self.enabled = False
        else:
            logger.debug("OpenTelemetry 已禁用")

    def _init_telemetry(self):
        """初始化 TracerProvider 和 MeterProvider（使用受保护的导入）"""
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
            from opentelemetry._logs import set_logger_provider
        except ImportError as e:
            raise ImportError(f"缺少 OpenTelemetry SDK 组件: {e}")

        resource = Resource.create({
            "service.name": "digitaltwin-rag",
            "service.version": "0.1.0",
        })

        # 1. 追踪初始化
        self._tracer_provider = TracerProvider(resource=resource)
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            if _OTLP_HTTP_AVAILABLE:
                try:
                    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                    self._tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                    logger.info(f"OTLP 追踪导出器已配置: {otlp_endpoint}")
                except Exception as e:
                    logger.warning(f"OTLP 追踪导出器配置失败: {e}")
            else:
                logger.warning("OTLP HTTP 导出器未安装，跳过追踪导出配置。")
        
        trace.set_tracer_provider(self._tracer_provider)

        # 2. 指标初始化
        try:
            if _PROMETHEUS_AVAILABLE:
                prometheus_port = int(os.getenv("OTEL_PROMETHEUS_PORT", "9464"))
                from opentelemetry.exporter.prometheus import start_http_server
                
                try:
                    start_http_server(port=prometheus_port, addr="0.0.0.0")
                    reader = PrometheusMetricReader()
                    
                    self._meter_provider = MeterProvider(
                        resource=resource,
                        metric_readers=[reader]
                    )
                    metrics.set_meter_provider(self._meter_provider)
                    logger.info(f"Prometheus 指标端点已启动，端口: {prometheus_port}")
                except Exception as e:
                    logger.warning(f"Prometheus 服务器启动失败: {e}")
            else:
                logger.warning("Prometheus 导出器未安装，跳过指标配置。")
        except Exception as e:
            logger.warning(f"指标初始化失败: {e}")

        # 3. 日志初始化
        try:
            self._logger_provider = LoggerProvider(resource=resource)
            log_endpoint = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://localhost:3100/otlp/v1/logs")
            
            if _OTLP_HTTP_AVAILABLE:
                try:
                    log_exporter = OTLPLogExporter(endpoint=log_endpoint)
                    self._logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
                    logger.info(f"OTLP 日志导出器已配置: {log_endpoint}")
                except Exception as e:
                    logger.warning(f"OTLP 日志导出器配置失败: {e}")
            
            set_logger_provider(self._logger_provider)
            
            log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
            handler = LoggingHandler(level=log_level, logger_provider=self._logger_provider)
            logging.getLogger().addHandler(handler)
        except Exception as e:
            logger.warning(f"日志初始化失败: {e}")

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
            return any(span_name == pattern.strip() for pattern in custom_patterns if pattern.strip())
        return False

    def get_tracer(self, name: str):
        """获取命名的 tracer"""
        if self.enabled and self._tracer_provider:
            return self._tracer_provider.get_tracer(name)
        if HAS_OTEL_API:
            return trace.get_tracer(name)
        return None

    def get_meter(self, name: str):
        """获取命名的 meter"""
        if self.enabled and self._meter_provider:
            return self._meter_provider.get_meter(name)
        if HAS_OTEL_API:
            return metrics.get_meter(name)
        return None

    def shutdown(self):
        """关闭 telemetry manager 并清理资源"""
        if self._tracer_provider:
            try:
                self._tracer_provider.force_flush(timeout_millis=5000)
                self._tracer_provider.shutdown()
            except Exception:
                pass
        if self._meter_provider:
            try:
                self._meter_provider.shutdown()
            except Exception:
                pass
        if self._logger_provider:
            try:
                self._logger_provider.shutdown()
            except Exception:
                pass
        logger.info("Telemetry manager 已关闭")


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


def get_tracer(name: str):
    """获取命名的 tracer（便捷函数）"""
    return get_telemetry().get_tracer(name)

def get_meter(name: str):
    """获取命名的 meter（便捷函数）"""
    return get_telemetry().get_meter(name)
