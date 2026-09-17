"""发布已审核的文档版本组合，保留旧版本以支持回滚。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag.chunking import chunk_document
from rag.document_processing import ParsedDocument
from rag.embeddings import BailianEmbeddings
from rag.milvus_store import MilvusKnowledgeStore
from rag.models import DocumentVersion, KnowledgeBase, KnowledgeDocument, KnowledgeRelease, ReleaseDocument


@dataclass
class KnowledgeSource:
    """一份入库文档的稳定身份、内容版本及适用范围。"""

    doc_id: str
    doc_version_id: str
    parsed: ParsedDocument
    source_uri: str
    topic: str
    # 上游完成解析质量和业务规则审核后才可设置为 True。
    approved: bool
    audience: str = "all"
    valid_from: datetime | None = None
    valid_to: datetime | None = None


async def _fail_release(sessions: async_sessionmaker[AsyncSession], release_id: str) -> None:
    """记录未发布成功的批次；不清理已入库版本，便于事后核对。"""

    async with sessions.begin() as session:
        release = await session.get(KnowledgeRelease, release_id)
        if release is not None:
            release.status = "failed"


async def _build_snapshot(
    sessions: async_sessionmaker[AsyncSession], store: MilvusKnowledgeStore,
    embeddings: BailianEmbeddings, *, kb_id: str, release_id: str,
    sources: list[KnowledgeSource], fingerprint: str | None, ready_status: str,
) -> str:
    """持久化完整版本清单，只向共用集合写入尚未入库的版本。"""

    collection = store.collection_name(kb_id, release_id)
    if not sources:
        raise ValueError("A complete source snapshot is required")
    if len({source.doc_id for source in sources}) != len(sources) or len({source.doc_version_id for source in sources}) != len(sources):
        raise ValueError("Each document and version must occur once per snapshot")
    # 所有文档先生成父子块；分块出错时尚未创建数据库发布记录。
    chunks = [chunk for source in sources for chunk in chunk_document(
        source.parsed, kb_id=kb_id, doc_id=source.doc_id, doc_version_id=source.doc_version_id,
        topic=source.topic, audience=source.audience, valid_from=source.valid_from,
        valid_to=source.valid_to,
    )]
    if not chunks:
        raise ValueError("Approved snapshot contains no searchable text")

    # 先持久化不可变文档版本与 building 批次，保留失败发布的审计线索。
    async with sessions.begin() as session:
        if await session.get(KnowledgeRelease, release_id) is not None:
            raise ValueError("Release ID already exists and cannot be reused")
        base = await session.get(KnowledgeBase, kb_id)
        if base is None:
            session.add(KnowledgeBase(kb_id=kb_id))
            await session.flush()
        for source in sources:
            version = await session.get(DocumentVersion, source.doc_version_id)
            # 复用版本 ID 时，文件内容、来源及解析器必须与原记录完全一致。
            identity = (kb_id, source.doc_id, source.parsed.sha256, source.parsed.source_name,
                        source.source_uri, source.parsed.source_format, source.parsed.parser_version)
            if version:
                if (version.kb_id, version.doc_id, version.sha256, version.source_name,
                    version.source_uri, version.source_format, version.parser_version) != identity:
                    raise ValueError("Document version ID is immutable")
            else:
                if await session.get(KnowledgeDocument, (kb_id, source.doc_id)) is None:
                    session.add(KnowledgeDocument(kb_id=kb_id, doc_id=source.doc_id))
                    await session.flush()
                session.add(DocumentVersion(
                    doc_version_id=source.doc_version_id, kb_id=kb_id, doc_id=source.doc_id,
                    sha256=source.parsed.sha256, source_name=source.parsed.source_name,
                    source_uri=source.source_uri, source_format=source.parsed.source_format,
                    parser_version=source.parsed.parser_version,
                ))
        session.add(KnowledgeRelease(
            release_id=release_id, kb_id=kb_id, collection_name=collection,
            embedding_model=embeddings.model, dimension=embeddings.dimension,
            fingerprint=fingerprint, status="building",
        ))
        await session.flush()
        session.add_all(ReleaseDocument(release_id=release_id, doc_version_id=s.doc_version_id) for s in sources)

    try:
        staged = await asyncio.to_thread(store.stage, kb_id, release_id, chunks, embeddings)
        if staged != collection:
            raise RuntimeError("Staged collection differs from release audit")
        if ready_status != "building":
            async with sessions.begin() as session:
                (await session.get(KnowledgeRelease, release_id)).status = ready_status
    except Exception:
        await _fail_release(sessions, release_id)
        raise
    return collection


async def _activate_snapshot(
    sessions: async_sessionmaker[AsyncSession], store: MilvusKnowledgeStore,
    *, kb_id: str, release_id: str, expected_status: str,
    reviewer: str | None = None,
) -> None:
    """锁定知识库，核验向量后原子切换 MySQL 中的生效版本组合。"""
    async with sessions.begin() as session:
        base = (await session.execute(select(KnowledgeBase).where(
            KnowledgeBase.kb_id == kb_id).with_for_update())).scalar_one()
        release = await session.get(KnowledgeRelease, release_id)
        if release is None or release.kb_id != kb_id or release.status != expected_status:
            raise ValueError("Release is not ready for this publication step")
        if release.collection_name != store.collection_name(kb_id, release_id):
            raise ValueError("Release points to a different knowledge base collection")
        versions = (await session.scalars(select(ReleaseDocument.doc_version_id).where(
            ReleaseDocument.release_id == release_id))).all()
        await asyncio.to_thread(store.verify_versions, kb_id, versions)
        previous = await session.get(KnowledgeRelease, base.active_release_id) if base.active_release_id else None
        release.status = "published"
        release.published_at = datetime.now(timezone.utc)
        if reviewer is not None:
            release.reviewed_by = reviewer
            release.reviewed_at = release.published_at
        if previous:
            previous.status = "superseded"
        base.active_release_id = release_id


async def active_version_ids(sessions: async_sessionmaker[AsyncSession], kb_id: str) -> list[str]:
    """读取当前发布的完整文档版本集合；未发布时不允许普通检索。"""
    async with sessions() as session:
        base = await session.get(KnowledgeBase, kb_id)
        if base is None or base.active_release_id is None:
            return []
        return list((await session.scalars(select(ReleaseDocument.doc_version_id).where(
            ReleaseDocument.release_id == base.active_release_id).order_by(ReleaseDocument.doc_version_id))).all())


async def prepare_snapshot(
    sessions: async_sessionmaker[AsyncSession], store: MilvusKnowledgeStore,
    embeddings: BailianEmbeddings, *, kb_id: str, release_id: str,
    sources: list[KnowledgeSource], fingerprint: str,
) -> str:
    """准备并向量化完整快照，等待人工审核，不改变线上版本。"""
    if len(fingerprint) != 64:
        raise ValueError("A SHA-256 snapshot fingerprint is required")
    return await _build_snapshot(sessions, store, embeddings, kb_id=kb_id,
                                 release_id=release_id, sources=sources,
                                 fingerprint=fingerprint, ready_status="pending_review")


async def approve_prepared_snapshot(
    sessions: async_sessionmaker[AsyncSession], store: MilvusKnowledgeStore,
    *, kb_id: str, release_id: str, reviewer: str,
) -> None:
    """审核者确认待审核版本后才使其对普通检索生效。"""
    if not reviewer.strip() or len(reviewer) > 128:
        raise ValueError("A non-empty reviewer (max 128 chars) is required")
    await _activate_snapshot(sessions, store, kb_id=kb_id, release_id=release_id,
                             expected_status="pending_review", reviewer=reviewer.strip())


async def publish_snapshot(
    sessions: async_sessionmaker[AsyncSession], store: MilvusKnowledgeStore,
    embeddings: BailianEmbeddings, *, kb_id: str, release_id: str,
    sources: list[KnowledgeSource],
) -> str:
    """兼容已有的显式审核调用：增量写入并立即发布版本组合。"""
    if not sources or not all(source.approved for source in sources):
        raise ValueError("A complete, explicitly approved source snapshot is required")
    collection = await _build_snapshot(sessions, store, embeddings, kb_id=kb_id,
                                       release_id=release_id, sources=sources,
                                       fingerprint=None, ready_status="building")
    try:
        await _activate_snapshot(sessions, store, kb_id=kb_id, release_id=release_id,
                                 expected_status="building")
    except Exception:
        await _fail_release(sessions, release_id)
        raise
    return collection


async def rollback_snapshot(
    sessions: async_sessionmaker[AsyncSession], store: MilvusKnowledgeStore,
    *, kb_id: str, release_id: str,
) -> None:
    """旧向量保持不变，仅把 MySQL 生效版本组合切回已发布批次。"""
    async with sessions.begin() as session:
        base = (await session.execute(select(KnowledgeBase).where(
            KnowledgeBase.kb_id == kb_id).with_for_update())).scalar_one()
        target = await session.get(KnowledgeRelease, release_id)
        if target is None or target.kb_id != kb_id or target.status not in ("published", "superseded"):
            raise ValueError("Rollback target must be a previously published KB release")
        if target.collection_name != store.collection_name(kb_id, release_id):
            raise ValueError("Rollback target points to another collection")
        versions = (await session.scalars(select(ReleaseDocument.doc_version_id).where(
            ReleaseDocument.release_id == release_id))).all()
        await asyncio.to_thread(store.verify_versions, kb_id, versions)
        if base.active_release_id == release_id:
            return
        current = await session.get(KnowledgeRelease, base.active_release_id)
        target.status = "published"
        current.status = "superseded"
        base.active_release_id = release_id
