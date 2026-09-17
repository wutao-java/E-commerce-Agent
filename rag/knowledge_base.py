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
SECTION_METADATA = re.compile(r"^<!--\s*(.*?)\s*-->$")
ACTIVITY_DATES = re.compile(r"活动时间为\s*(\d{4}-\d{2}-\d{2}).*?至\s*(\d{4}-\d{2}-\d{2})", re.S)


def load_source_documents(directory: Path = KNOWLEDGE_DIR) -> list[SourceDocument]:
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
    if overlap >= size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(normalized) <= size:
        return [normalized]
    parts = re.split(r"(?<=[。！？；])|\n", normalized)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if current and len(current) + len(part) > size:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""
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
    chunks: list[KnowledgeChunk] = []
    identifiers: set[str] = set()
    for document in documents:
        section = document.title
        lines: list[str] = []
        metadata: dict[str, str] = {}

        def flush() -> None:
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
    if chunk.status in {"expired", "historical"}:
        return historical
    if chunk.status != "active":
        return False
    if chunk.valid_from and reference_date < chunk.valid_from:
        return False
    if chunk.valid_until and reference_date > chunk.valid_until:
        return historical
    return True
