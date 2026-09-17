"""离线检查课程问题集；不在每次聊天时执行。"""

from __future__ import annotations

import json

from agent.intent_rules import plan_intent_by_rules
from config.settings import PROJECT_ROOT
from domain import ChatCommand
from rag.service import CourseRagService


CASES_PATH = PROJECT_ROOT / "knowledge" / "course" / "quality_cases.json"


def run_rag_quality_check(service: CourseRagService) -> dict:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        command = ChatCommand(session_id="course-quality", runtime_user_id="course-user", user_message=case["question"])
        intent = plan_intent_by_rules(command.user_message)
        _, hits, _ = service.retrieve(command, intent.intent if intent else "unknown")
        retrieved = {hit.chunk.chunk_id for hit in hits}
        expected = set(case["expected_chunk_ids"])
        must_fallback = bool(case["must_fallback"])
        results.append({
            "case_id": case["case_id"],
            "retrieved_chunk_ids": sorted(retrieved),
            "recall_at_k": len(retrieved & expected) / len(expected) if expected else float(not retrieved),
            "precision_at_k": len(retrieved & expected) / len(retrieved) if retrieved else float(must_fallback),
            "passed": (not retrieved) if must_fallback else bool(retrieved & expected),
        })
    return {"total_cases": len(results), "passed_cases": sum(item["passed"] for item in results), "results": results}
