"""手动发布课程索引与离线质量检查。"""

from __future__ import annotations

import argparse
import json

from rag.quality import run_rag_quality_check
from rag.service import CourseRagService


def main() -> None:
    """解析管理命令，并执行索引重建或离线质量评估。"""
    parser = argparse.ArgumentParser(description="课程 RAG 的 Milvus 管理命令")
    parser.add_argument("action", choices=["rebuild", "evaluate"])
    action = parser.parse_args().action
    service = CourseRagService()
    if action == "rebuild":
        collection, count = service.publish()
        print(json.dumps({"collection": collection, "chunk_count": count}, ensure_ascii=False))
    else:
        print(json.dumps(run_rag_quality_check(service), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
