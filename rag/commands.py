"""手动发布课程索引与离线质量检查。"""

from __future__ import annotations

import argparse
import json
import logging
import time

from config import configure_logging, get_settings
from rag.quality import run_rag_quality_check
from rag.service import CourseRagService


logger = logging.getLogger(__name__)


def main() -> None:
    """解析管理命令，并执行索引重建或离线质量评估。"""
    parser = argparse.ArgumentParser(description="课程 RAG 的 Milvus 管理命令")
    parser.add_argument("action", choices=["rebuild", "evaluate"])
    action = parser.parse_args().action
    settings = get_settings()
    configure_logging(settings.server.log_level, settings.server.log_file)
    started_at = time.perf_counter()
    logger.info("RAG command started action=%s", action)
    service = CourseRagService()
    if action == "rebuild":
        collection, count = service.publish()
        logger.info(
            "RAG command completed action=%s collection=%s chunk_count=%d duration_ms=%.2f",
            action,
            collection,
            count,
            (time.perf_counter() - started_at) * 1000,
        )
        print(json.dumps({"collection": collection, "chunk_count": count}, ensure_ascii=False))
    else:
        result = run_rag_quality_check(service)
        logger.info(
            "RAG command completed action=%s total_cases=%d passed_cases=%d duration_ms=%.2f",
            action,
            result["total_cases"],
            result["passed_cases"],
            (time.perf_counter() - started_at) * 1000,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
