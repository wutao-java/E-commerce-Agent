"""将课程 Markdown 解析为保留章节边界的知识片段。"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from config.settings import PROJECT_ROOT
from domain.rag import KnowledgeChunk, SourceDocument


KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge" / "course"
# 章节级元数据写在独立的 HTML 注释中，例如 ``<!-- chunk_id:x; keywords:a,b -->``。
SECTION_METADATA = re.compile(r"^<!--\s*(.*?)\s*-->$")
ACTIVITY_DATES = re.compile(r"活动时间为\s*(\d{4}-\d{2}-\d{2}).*?至\s*(\d{4}-\d{2}-\d{2})", re.S)


def load_source_documents(directory: Path = KNOWLEDGE_DIR) -> list[SourceDocument]:
    """读取目录中的课程 Markdown 文档及其 YAML front matter。

    Args:
        directory: 课程知识文件所在目录。

    Returns:
        按文件名排序的源文档列表。

    Raises:
        ValueError: 文档缺少完整 front matter，或 metadata 不是映射结构。
    """
    documents: list[SourceDocument] = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        if not raw.startswith("---\n"):
            raise ValueError(f"课程文档缺少 front matter: {path.name}")
        metadata_text, separator, body = raw[4:].partition("\n---\n")
        if not separator:
            raise ValueError(f"课程文档 front matter 未闭合: {path.name}")
        metadata = yaml.safe_load(metadata_text)
        if not isinstance(metadata, dict):
            raise ValueError(f"课程文档 metadata 无效: {path.name}")
        documents.append(SourceDocument(
            source_path=path.name,
            title=str(metadata.get("title") or path.stem),
            metadata=metadata,
            body=body,
        ))
    return documents


def _section_metadata(line: str) -> dict[str, str]:
    """解析单行章节元数据注释；普通正文行返回空字典。"""
    match = SECTION_METADATA.fullmatch(line.strip())
    if not match:
        return {}
    metadata: dict[str, str] = {}
    for part in match.group(1).split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def _split_section(text: str, size: int, overlap: int) -> list[str]:
    """优先按中文标点切分章节，并为相邻片段保留上下文重叠。

    Args:
        text: 完整章节正文。
        size: 单个片段允许的最大字符数。
        overlap: 相邻片段重复保留的字符数。

    Returns:
        已去除空行且长度受限的知识片段。

    Raises:
        ValueError: 重叠长度不小于片段长度。
    """
    if overlap >= size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(normalized) <= size:
        return [normalized]
    # 标点保留在前一分句中，使切片后的文本仍保持自然语义。
    parts = re.split(r"(?<=[。！？；])|\n", normalized)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if current and len(current) + len(part) > size:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""
        # 单个分句超过上限时按字符硬切，确保所有片段都不超过 size。
        while len(part) > size:
            room = size - len(current)
            current += part[:room]
            chunks.append(current.strip())
            part = part[room:]
            current = current[-overlap:] if overlap else ""
        current += part
    if current.strip() and (not chunks or current.strip() != chunks[-1]):
        chunks.append(current.strip())
    return chunks


def build_knowledge_chunks(documents: list[SourceDocument], size: int = 420, overlap: int = 80) -> list[KnowledgeChunk]:
    """按 Markdown 二级章节构建可检索的课程知识片段。

    Args:
        documents: 已解析 front matter 的源文档。
        size: 单个知识片段的最大字符数。
        overlap: 相邻片段的重叠字符数。

    Returns:
        保留来源、章节、有效期和父级正文的知识片段列表。

    Raises:
        ValueError: 生成了重复的知识片段 ID，或切片参数无效。
    """
    chunks: list[KnowledgeChunk] = []
    identifiers: set[str] = set()
    for document in documents:
        section = document.title
        lines: list[str] = []
        metadata: dict[str, str] = {}

        def flush() -> None:
            """将当前章节缓冲区转换为一个或多个知识片段。"""
            nonlocal lines, metadata
            text = "\n".join(lines).strip()
            if not text:
                lines, metadata = [], {}
                return
            base_id = metadata.get("chunk_id") or f"{Path(document.source_path).stem}-s{len(chunks) + 1}"
            keywords = [term.strip() for term in metadata.get("keywords", "").split(",") if term.strip()]
            dates = ACTIVITY_DATES.search(text)
            valid_from = date.fromisoformat(dates.group(1)) if dates else None
            valid_until = date.fromisoformat(dates.group(2)) if dates else None
            pieces = _split_section(text, size, overlap)
            for index, piece in enumerate(pieces, start=1):
                chunk_id = base_id if len(pieces) == 1 else f"{base_id}-c{index}"
                if chunk_id in identifiers:
                    raise ValueError(f"重复的课程知识 ID: {chunk_id}")
                identifiers.add(chunk_id)
                chunks.append(KnowledgeChunk(
                    chunk_id=chunk_id,
                    title=document.title,
                    source_path=document.source_path,
                    section=section,
                    topic=str(document.metadata.get("domain") or "faq"),
                    status=metadata.get("effective_status") or str(document.metadata.get("effective_status") or "active"),
                    keywords=keywords,
                    text=piece,
                    parent_text=text,
                    valid_from=valid_from,
                    valid_until=valid_until,
                ))
            lines, metadata = [], {}

        for raw_line in document.body.splitlines():
            if raw_line.startswith("## "):
                flush()
                section = raw_line[3:].strip()
            elif raw_line.startswith("# "):
                continue
            elif parsed := _section_metadata(raw_line):
                metadata.update(parsed)
            else:
                lines.append(raw_line)
        flush()
    return chunks


def is_available(chunk: KnowledgeChunk, reference_date: date, historical: bool = False) -> bool:
    """判断知识片段在参考日期和历史查询语境下是否可检索。

    Args:
        chunk: 待判断的知识片段。
        reference_date: 课程检索使用的业务参考日期。
        historical: 是否明确允许召回历史或已过期资料。

    Returns:
        片段满足状态及有效期约束时返回 ``True``。
    """
    if chunk.status in {"expired", "historical"}:
        return historical
    if chunk.status != "active":
        return False
    if chunk.valid_from and reference_date < chunk.valid_from:
        return False
    if chunk.valid_until and reference_date > chunk.valid_until:
        return historical
    return True
