from unittest.mock import MagicMock

from src.infrastructure.llm_client import LLMClient
from src.rag.react_router import ReActDecision, ReActRetrievalRouter


def make_router(response, fallback_action="retrieve"):
    llm_client = MagicMock(spec=LLMClient)
    llm_client.call.return_value = response
    return (
        ReActRetrievalRouter(
            llm_client=llm_client,
            model="router-test",
            history_messages=2,
            fallback_action=fallback_action,
        ),
        llm_client,
    )


def test_react_router_selects_retrieval_tool():
    router, llm_client = make_router('{"action":"retrieve"}')

    decision = router.decide(
        query="我以前说过最喜欢什么？",
        conversation=[{"role": "user", "content": "聊聊兴趣"}],
        persona={"name": "张三"},
    )

    assert decision == ReActDecision(action="retrieve")
    assert llm_client.call.call_args.kwargs["model"] == "router-test"


def test_react_router_selects_direct_response_from_fenced_json():
    router, _ = make_router('```json\n{"action":"respond"}\n```')

    assert router.decide(query="你好").action == "respond"


def test_react_router_falls_back_to_retrieval_on_invalid_output():
    router, _ = make_router("无法判断")

    assert router.decide(query="昨天发生了什么？").action == "retrieve"


def test_react_router_limits_conversation_history():
    router, llm_client = make_router('{"action":"respond"}')
    conversation = [
        {"role": "user", "content": "第一条"},
        {"role": "assistant", "content": "第二条"},
        {"role": "user", "content": "第三条"},
    ]

    router.decide(query="你好", conversation=conversation)

    prompt = llm_client.call.call_args.kwargs["messages"][0]["content"]
    assert "第一条" not in prompt
    assert "第二条" in prompt
    assert "第三条" in prompt
