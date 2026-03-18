import os
from unittest.mock import patch
from src.infrastructure.telemetry import TelemetryManager, get_tracer


def test_telemetry_disabled_by_default():
    """禁用状态下不应该创建导出器"""
    with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
        manager = TelemetryManager()
        assert manager.enabled is False


def test_telemetry_enabled():
    """启用时应该正确初始化"""
    with patch.dict(os.environ, {"OTEL_ENABLED": "true", "OTEL_TRACE_LEVEL": "light"}):
        manager = TelemetryManager()
        assert manager.enabled is True
        assert manager.trace_level == "light"


def test_get_tracer_returns_valid_tracer():
    """获取的 tracer 应该能创建 span"""
    tracer = get_tracer(__name__)
    assert tracer is not None
    # 应该支持创建 span
    with tracer.start_as_current_span("test_span") as span:
        assert span is not None


def test_tracer_filtering_by_level():
    """不同的追踪级别应该过滤不同的 span"""
    # Test light level
    with patch.dict(os.environ, {"OTEL_ENABLED": "true", "OTEL_TRACE_LEVEL": "light"}):
        manager = TelemetryManager()
        assert manager.should_trace("llm.api_call") is True
        assert manager.should_trace("db.vector_search") is True
        assert manager.should_trace("rag.search") is False  # Not in light level

    # Test full level
    with patch.dict(os.environ, {"OTEL_ENABLED": "true", "OTEL_TRACE_LEVEL": "full"}):
        manager = TelemetryManager()
        assert manager.should_trace("llm.api_call") is True
        assert manager.should_trace("rag.search") is True  # In full level
        assert manager.should_trace("unknown.span") is False


def test_tracer_filtering_custom_patterns():
    """自定义追踪模式"""
    with patch.dict(os.environ, {
        "OTEL_ENABLED": "true",
        "OTEL_TRACE_LEVEL": "custom",
        "OTEL_CUSTOM_SPAN_PATTERNS": "llm.api_call,db.vector_search"
    }):
        manager = TelemetryManager()
        assert manager.should_trace("llm.api_call") is True
        assert manager.should_trace("db.vector_search") is True
        assert manager.should_trace("rag.search") is False


def test_telemetry_shutdown():
    """测试关闭功能"""
    with patch.dict(os.environ, {"OTEL_ENABLED": "true"}):
        manager = TelemetryManager()
        # Should not raise any exceptions
        manager.shutdown()
