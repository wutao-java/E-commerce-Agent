import os

import pytest

from domain import ChatCommand
from rag.service import CourseRagService


@pytest.mark.skipif(
    os.getenv("COURSE_RAG_VERIFY_MILVUS") != "1",
    reason="需要可用的 embedding 服务、Milvus 和已发布的课程索引",
)
def test_real_rag_retrieves_seven_day_return_first() -> None:
    service = CourseRagService()
    command = ChatCommand(
        session_id="rag-test",
        runtime_user_id="test-user",
        user_message="签收八天了，包装没拆，可以无理由退货吗？",
    )

    plan, hits, debug = service.retrieve(
        command,
        "refund_request",  # 固定意图，避免意图识别影响检索测试
    )

    print("改写后的查询：", plan.rewritten_query)
    print("调试信息：", debug)
    for index, hit in enumerate(hits, start=1):
        print(
            index,
            hit.chunk.chunk_id,
            hit.score,
            hit.vector_score,
            hit.keyword_score,
            hit.sources,
            hit.chunk.text,
        )

    assert hits
    assert hits[0].chunk.chunk_id == "after-sale-seven-day-return"
