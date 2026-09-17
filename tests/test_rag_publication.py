"""离线验证增量版本审计、审核隔离与历史批次回滚。"""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.base import Base
from rag.publication import KnowledgeSource, active_version_ids, publish_snapshot, rollback_snapshot
from rag.models import KnowledgeBase, KnowledgeRelease, DocumentVersion

from test_rag_pipeline import sample_document


class FakeStore:
    """模拟同一集合中的版本写入和发布前核验。"""

    def __init__(self) -> None:
        self.dimension = 1024
        self.versions = set()
        self.fail_stage = False
        self.fail_activate = False

    @staticmethod
    def collection_name(kb_id, release_id):
        return f"rag_{kb_id}"

    def stage(self, kb_id, release_id, chunks, embeddings):
        assert chunks and all(c.doc_version_id for c in chunks)
        if self.fail_stage:
            raise RuntimeError("embedding unavailable")
        self.versions.update(chunk.doc_version_id for chunk in chunks)
        return f"rag_{kb_id}"

    def verify_versions(self, kb_id, version_ids):
        if self.fail_activate:
            raise RuntimeError("version verification failed")
        if not set(version_ids) <= self.versions:
            raise RuntimeError("document version is missing")


class FakeEmbeddings:
    """发布审计使用的固定模型名称与维度。"""

    model = "qwen3.7-text-embedding"
    dimension = 1024


def make_source(document=None):
    """构造一份已审核的文档版本作为完整快照输入。"""

    return KnowledgeSource(doc_id="refund", doc_version_id="v1", parsed=document or sample_document(),
                           source_uri="object://refund.pdf", topic="after_sale", approved=True)


@pytest.mark.parametrize("failure", ["stage", "activate"])
def test_failed_publication_keeps_active_versions_and_audit(tmp_path, failure):
    """入库或审核校验失败时，不将新版本暴露给查询。"""

    async def scenario():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        store = FakeStore()
        store.fail_stage = failure == "stage"
        store.fail_activate = failure == "activate"
        with pytest.raises(RuntimeError):
            await publish_snapshot(sessions, store, FakeEmbeddings(), kb_id="shop", release_id="r1",
                                   sources=[make_source()])
        async with sessions() as session:
            assert (await session.get(KnowledgeRelease, "r1")).status == "failed"
            assert (await session.get(DocumentVersion, "v1")).sha256 == "a" * 64
        assert await active_version_ids(sessions, "shop") == []
        await engine.dispose()
    asyncio.run(scenario())


def test_immutable_versions_snapshot_and_rollback(tmp_path):
    """旧向量保留；仅 MySQL 当前批次决定普通检索和回滚版本。"""

    async def scenario():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        store = FakeStore()
        await publish_snapshot(sessions, store, FakeEmbeddings(), kb_id="shop", release_id="r1",
                               sources=[make_source()])
        assert await active_version_ids(sessions, "shop") == ["v1"]
        async with sessions() as session:
            assert (await session.get(KnowledgeBase, "shop")).active_release_id == "r1"
        changed = sample_document()
        changed.sha256 = "b" * 64
        with pytest.raises(ValueError, match="immutable"):
            await publish_snapshot(sessions, store, FakeEmbeddings(), kb_id="shop", release_id="r2",
                                   sources=[make_source(changed)])
        assert await active_version_ids(sessions, "shop") == ["v1"]
        new_source = KnowledgeSource(doc_id="refund", doc_version_id="v2", parsed=changed,
                                     source_uri="object://refund-v2.pdf", topic="after_sale", approved=True)
        store.fail_activate = True
        with pytest.raises(RuntimeError, match="version verification failed"):
            await publish_snapshot(sessions, store, FakeEmbeddings(), kb_id="shop",
                                   release_id="r_failed", sources=[new_source])
        assert store.versions == {"v1", "v2"}
        assert await active_version_ids(sessions, "shop") == ["v1"]
        async with sessions() as session:
            assert (await session.get(KnowledgeRelease, "r_failed")).status == "failed"
            assert (await session.get(KnowledgeBase, "shop")).active_release_id == "r1"
        store.fail_activate = False
        await publish_snapshot(sessions, store, FakeEmbeddings(), kb_id="shop", release_id="r2",
                               sources=[new_source])
        assert await active_version_ids(sessions, "shop") == ["v2"]
        await rollback_snapshot(sessions, store, kb_id="shop", release_id="r1")
        assert await active_version_ids(sessions, "shop") == ["v1"]
        async with sessions() as session:
            assert (await session.get(KnowledgeBase, "shop")).active_release_id == "r1"
            assert (await session.get(KnowledgeRelease, "r2")).status == "superseded"
        await engine.dispose()
    asyncio.run(scenario())


def test_only_explicitly_approved_sources_are_published(tmp_path):
    """未审核的文档无法进入对外发布快照。"""

    async def scenario():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine)
        source = make_source()
        source.approved = False
        with pytest.raises(ValueError, match="approved"):
            await publish_snapshot(sessions, FakeStore(), FakeEmbeddings(), kb_id="shop",
                                   release_id="r1", sources=[source])
        await engine.dispose()
    asyncio.run(scenario())
