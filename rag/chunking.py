"""沿原文段落边界建立父子块，并保留引用所需的来源信息。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from rag.document_processing import DocumentBlock, ParsedDocument


@dataclass(frozen=True)
class KnowledgeChunk:
    """Milvus 中的一条子块记录；父块全文随子块冗余存储。"""

    # 文档 ID 稳定，版本 ID 标识一次不可变内容，父子 ID 在该版本内确定。
    chunk_id: str
    kb_id: str
    doc_id: str
    doc_version_id: str
    parent_id: str
    child_index: int
    # 只对子块向量化；命中时直接返回父块补足规则的条件和例外。
    child_text: str
    parent_text: str
    source_name: str
    source_format: str
    sha256: str
    parser_version: str
    title: str
    # 无可靠页码的 Markdown / Word 保持 None，不能伪造引用位置。
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    topic: str
    audience: str
    # 有效期统一为 UTC 毫秒，供 Milvus 两路检索共用过滤条件。
    valid_from_ms: int
    valid_to_ms: int


def _milliseconds(value: datetime | None, default: int) -> int:
    """将带时区时间转为 UTC 毫秒；None 使用调用方指定的开放边界。"""

    if value is None:
        return default
    if value.tzinfo is None:
        raise ValueError("Validity timestamps must have a timezone")
    return int(value.astimezone(timezone.utc).timestamp() * 1000)


def chunk_document(
    document: ParsedDocument, *, kb_id: str, doc_id: str, doc_version_id: str,
    topic: str, audience: str = "all", valid_from: datetime | None = None,
    valid_to: datetime | None = None, max_parent_chars: int = 4000,
    max_child_chars: int = 900, overlap_chars: int = 180,
) -> list[KnowledgeChunk]:
    """按章节和语义块分组；单个段落或表格过长时要求人工复核。"""
    if not all((kb_id, doc_id, doc_version_id, topic, audience)):
        raise ValueError("Knowledge identifiers, topic and audience are required")
    if not (0 <= overlap_chars < max_child_chars <= max_parent_chars):
        raise ValueError("Require 0 <= overlap < child budget <= parent budget")
    start = _milliseconds(valid_from, 0)
    end = _milliseconds(valid_to, 253402300799000)
    if start >= end:
        raise ValueError("valid_from must precede valid_to")

    # 先按标题路径形成父块；超出父块预算时只在完整块之间断开。
    parents: list[tuple[list[str], list[DocumentBlock]]] = []
    current: list[DocumentBlock] = []
    path: list[str] = []
    for block in document.blocks:
        if block.kind == "heading":
            if current:
                parents.append((path, current))
                current = []
            path = block.heading_path
            continue
        block_path = block.heading_path
        prefix = " / ".join(block_path)
        if len(prefix) + len(block.text) + 1 > max_child_chars:
            # 原子规则不能从中间截断，否则适用条件可能与结论分离。
            raise ValueError("Oversized atomic block requires manual review")
        if current and (block_path != path or len(prefix) + sum(len(b.text) + 1 for b in current) + len(block.text) + 1 > max_parent_chars):
            parents.append((path, current))
            current = []
        path = block_path
        current.append(block)
    if current:
        parents.append((path, current))

    chunks: list[KnowledgeChunk] = []
    for parent_number, (heading_path, blocks) in enumerate(parents, start=1):
        prefix = " / ".join(heading_path)
        parent_text = "\n".join(filter(None, [prefix, *(block.text for block in blocks)]))
        parent_id = f"{doc_version_id}:p{parent_number:04d}"
        # 子块也只沿完整块切分，最多携带前一子块的末尾语义块。
        groups: list[list[DocumentBlock]] = []
        group: list[DocumentBlock] = []
        for block in blocks:
            if group and len(prefix) + sum(len(b.text) + 1 for b in group) + len(block.text) + 1 > max_child_chars:
                groups.append(group)
                overlap = group[-1] if len(group[-1].text) <= overlap_chars else None
                # 重叠不得超过子块预算，也不会跨越当前父块边界。
                group = [overlap] if overlap and len(prefix) + len(overlap.text) + len(block.text) + 2 <= max_child_chars else []
            group.append(block)
        if group:
            groups.append(group)
        for child_number, child_blocks in enumerate(groups, start=1):
            # 引用范围包含实际进入子块的内容，也包含可能重叠的块。
            pages = [page for b in child_blocks for page in (b.page_start, b.page_end) if page is not None]
            chunks.append(KnowledgeChunk(
                chunk_id=f"{parent_id}:c{child_number:04d}", kb_id=kb_id,
                doc_id=doc_id, doc_version_id=doc_version_id, parent_id=parent_id,
                child_index=child_number, child_text="\n".join(filter(None, [prefix, *(b.text for b in child_blocks)])),
                parent_text=parent_text, source_name=document.source_name,
                source_format=document.source_format, sha256=document.sha256,
                parser_version=document.parser_version, title=heading_path[-1] if heading_path else document.source_name,
                heading_path=heading_path, page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None, topic=topic, audience=audience,
                valid_from_ms=start, valid_to_ms=end,
            ))
    return chunks
