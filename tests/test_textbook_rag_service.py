from unittest.mock import MagicMock

from src.infrastructure.llm_client import LLMClient
from src.infrastructure.multimodal_embedding_client import MultiModalEmbeddingClient
from src.infrastructure.text_embedding_client import TextEmbeddingClient
from src.services.textbook_rag_service import TextbookRAGService


def make_service():
    llm_client = MagicMock(spec=LLMClient)
    llm_client.call.return_value = "数据库事务 ACID"
    db_client = MagicMock()
    mm_client = MagicMock(spec=MultiModalEmbeddingClient)
    mm_client.embed_query.return_value = [0.1, 0.2]
    text_client = MagicMock(spec=TextEmbeddingClient)
    text_client.embed_query.return_value = [0.3, 0.4]
    service = TextbookRAGService(
        llm_client=llm_client,
        db_client=db_client,
        text_collection_name="mm_text",
        image_collection_name="mm_image",
        ocr_collection_name="ocr_text",
        mm_client=mm_client,
        text_embedding_client=text_client,
    )
    return service, db_client, mm_client, text_client


def test_textbook_retrieve_queries_and_fuses_all_channels():
    service, db_client, mm_client, text_client = make_service()
    shared_metadata = {"source_file": "book.pdf", "page": 1, "chunk_index": 0}
    db_client.search_by_embedding.side_effect = [
        [("事务具有ACID特性", shared_metadata, 0.91)],
        [("[image] ACID图", {"source_file": "book.pdf", "page": 2}, 0.82)],
        [
            ("事务具有ACID特性", shared_metadata, 0.88),
            (
                "原子性表示全部成功或全部失败",
                {"source_file": "scan.pdf", "page": 8},
                0.80,
            ),
        ],
    ]

    result = service.retrieve("ACID是什么？", text_k=4, image_k=2, ocr_k=3)

    assert mm_client.embed_query.called
    text_client.embed_query.assert_called_once_with("数据库事务 ACID")
    assert [
        call.kwargs["collection_name"]
        for call in db_client.search_by_embedding.call_args_list
    ] == [
        "mm_text",
        "mm_image",
        "ocr_text",
    ]
    assert result["text_results"][0][0] == "事务具有ACID特性"
    assert (
        result["text_results"][0][1]["retrieval_channels"] == "multimodal_text,ocr_text"
    )
    assert len(result["ocr_text_results"]) == 2
    assert len(result["image_results"]) == 1


def test_textbook_retrieve_degrades_when_ocr_channel_fails():
    service, db_client, _, text_client = make_service()
    db_client.search_by_embedding.side_effect = [
        [("多模态文本", {"source_file": "book.pdf", "page": 1}, 0.9)],
        [],
    ]
    text_client.embed_query.side_effect = RuntimeError("temporary failure")

    result = service.retrieve("测试", text_k=4, image_k=2, ocr_k=3)

    assert [item[0] for item in result["text_results"]] == ["多模态文本"]
    assert result["ocr_text_results"] == []


def test_textbook_retrieve_uses_recent_conversation_for_query_understanding():
    service, db_client, mm_client, _ = make_service()
    service.llm_client.call.return_value = (
        '{"standalone_query":"数据库事务的ACID特性是什么",'
        '"entities":["数据库事务","ACID"],"time_range":null}'
    )
    db_client.search_by_embedding.side_effect = [[], [], []]

    service.retrieve(
        "它有哪些特性？",
        conversation=[
            {"role": "user", "content": "什么是数据库事务？"},
            {"role": "assistant", "content": "事务是一组操作。"},
        ],
    )

    mm_client.embed_query.assert_called_once_with("数据库事务的ACID特性是什么")
    prompt = service.llm_client.call.call_args.kwargs["messages"][0]["content"]
    assert "什么是数据库事务" in prompt
