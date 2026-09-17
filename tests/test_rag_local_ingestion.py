"""启动准备本地文档版本，但必须显式审核才能切换生效批次。"""

import asyncio
import hashlib
import importlib
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.base import Base
from rag.local_ingestion import approve_local_release, prepare_local_snapshots
from rag.publication import active_version_ids
from rag.models import KnowledgeBase, KnowledgeRelease
from test_rag_pipeline import sample_document
from test_rag_publication import FakeEmbeddings, FakeStore


def local_fixture(tmp_path):
    root = tmp_path / "data"
    source = root / "shop" / "refund" / "v1" / "refund.md"
    source.parent.mkdir(parents=True)
    source.write_text("# 退货\n七天内退货。", encoding="utf-8")
    manifest = root / "shop" / "manifest.json"
    manifest.write_text(json.dumps({
        "kb_id": "shop", "release_id": "r1", "documents": [
            {"doc_id": "refund", "doc_version_id": "v1", "path": "refund/v1/refund.md",
             "topic": "after_sale", "audience": "all"}
        ],
    }, ensure_ascii=False), encoding="utf-8")
    return root, source, manifest


async def sessions_for(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def test_startup_stages_once_without_switching_alias(tmp_path, monkeypatch):
    root, source, manifest_path = local_fixture(tmp_path)
    parsed_paths = []

    def parser(path):
        parsed_paths.append(path)
        parsed = sample_document()
        parsed.source_name = path.name
        parsed.source_format = "md"
        parsed.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return parsed

    monkeypatch.setattr("rag.local_ingestion.parse_document", parser)

    class CountingStore(FakeStore):
        calls = 0

        def stage(self, kb_id, release_id, chunks, embeddings):
            self.calls += 1
            return super().stage(kb_id, release_id, chunks, embeddings)

    async def scenario():
        engine, sessions = await sessions_for(tmp_path)
        store = CountingStore()
        assert await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root) == ["r1"]
        assert await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root) == []
        assert store.calls == 1 and parsed_paths == [source]
        assert await active_version_ids(sessions, "shop") == []
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["release_id"] = "r2"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="identical|already prepared"):
            await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        manifest["release_id"] = "r1"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        async with sessions() as session:
            release = await session.get(KnowledgeRelease, "r1")
            assert release.status == "pending_review"
            assert release.fingerprint and release.reviewed_by is None
            assert (await session.get(KnowledgeBase, "shop")).active_release_id is None
        with pytest.raises(ValueError, match="reviewer"):
            await approve_local_release(sessions, store, data_root=root,
                                        kb_id="shop", release_id="r1", reviewer="")
        assert await active_version_ids(sessions, "shop") == []
        await approve_local_release(sessions, store, data_root=root,
                                    kb_id="shop", release_id="r1", reviewer="operator")
        assert await active_version_ids(sessions, "shop") == ["v1"]
        async with sessions() as session:
            release = await session.get(KnowledgeRelease, "r1")
            assert release.status == "published" and release.reviewed_by == "operator"
            assert release.reviewed_at is not None
        await engine.dispose()
    asyncio.run(scenario())


def test_changed_file_cannot_be_approved_without_new_release(tmp_path, monkeypatch):
    root, source, _ = local_fixture(tmp_path)

    def parser(path):
        parsed = sample_document()
        parsed.source_name = path.name
        parsed.source_format = "md"
        parsed.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return parsed

    monkeypatch.setattr("rag.local_ingestion.parse_document", parser)

    async def scenario():
        engine, sessions = await sessions_for(tmp_path)
        store = FakeStore()
        await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        source.write_text("# 退货\n规则已更新。", encoding="utf-8")
        with pytest.raises(ValueError, match="changed|fingerprint"):
            await approve_local_release(sessions, store, data_root=root,
                                        kb_id="shop", release_id="r1", reviewer="operator")
        assert await active_version_ids(sessions, "shop") == []
        with pytest.raises(ValueError, match="release ID|changed|fingerprint"):
            await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        await engine.dispose()
    asyncio.run(scenario())


def test_existing_release_does_not_hide_deleted_milvus_versions(tmp_path, monkeypatch):
    root, source, _ = local_fixture(tmp_path)

    def parser(path):
        parsed = sample_document()
        parsed.source_name = path.name
        parsed.source_format = "md"
        parsed.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return parsed

    monkeypatch.setattr("rag.local_ingestion.parse_document", parser)

    async def scenario():
        engine, sessions = await sessions_for(tmp_path)
        store = FakeStore()
        await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        store.versions.clear()
        with pytest.raises(RuntimeError, match="missing"):
            await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        await engine.dispose()

    asyncio.run(scenario())


def test_new_document_and_new_version_keep_historical_source(tmp_path, monkeypatch):
    root, source, manifest_path = local_fixture(tmp_path)

    def parser(path):
        parsed = sample_document()
        parsed.source_name = path.name
        parsed.source_format = "md"
        parsed.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return parsed

    monkeypatch.setattr("rag.local_ingestion.parse_document", parser)

    async def scenario():
        engine, sessions = await sessions_for(tmp_path)
        store = FakeStore()
        await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        await approve_local_release(sessions, store, data_root=root,
                                    kb_id="shop", release_id="r1", reviewer="operator")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shipping = root / "shop" / "shipping" / "shipping_v1" / "shipping.md"
        shipping.parent.mkdir(parents=True)
        shipping.write_text("# 配送\n正常配送。", encoding="utf-8")
        manifest["release_id"] = "r2"
        manifest["documents"].append({"doc_id": "shipping", "doc_version_id": "shipping_v1",
                                       "path": "shipping/shipping_v1/shipping.md", "topic": "delivery"})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        assert await active_version_ids(sessions, "shop") == ["v1"]
        await approve_local_release(sessions, store, data_root=root,
                                    kb_id="shop", release_id="r2", reviewer="operator")
        assert await active_version_ids(sessions, "shop") == ["shipping_v1", "v1"]

        updated = source.parent.parent / "v2" / "refund.md"
        updated.parent.mkdir()
        updated.write_text("# 退货\n更新后的规则。", encoding="utf-8")
        manifest["release_id"] = "r3"
        manifest["documents"][0].update(doc_version_id="v2", path="refund/v2/refund.md")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        assert source.exists() and await active_version_ids(sessions, "shop") == ["shipping_v1", "v1"]
        await approve_local_release(sessions, store, data_root=root,
                                    kb_id="shop", release_id="r3", reviewer="operator")
        assert await active_version_ids(sessions, "shop") == ["shipping_v1", "v2"]
        source.write_text("# 退货\n历史原件被修改。", encoding="utf-8")
        with pytest.raises(ValueError, match="Unlisted document"):
            await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        await engine.dispose()

    asyncio.run(scenario())


def test_missing_unlisted_or_unsafe_documents_block_preparation(tmp_path, monkeypatch):
    root, source, manifest = local_fixture(tmp_path)
    unlisted = source.parent / "surprise.pdf"
    unlisted.write_bytes(b"new document")

    async def scenario():
        engine, sessions = await sessions_for(tmp_path)
        store = FakeStore()
        with pytest.raises(ValueError, match="unlisted"):
            await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        unlisted.unlink()
        source.unlink()
        with pytest.raises(ValueError, match="missing"):
            await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        source.write_text("# 退货\n七天内退货。", encoding="utf-8")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["documents"][0]["path"] = "../../../secrets.md"
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="path|directory"):
            await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        assert await active_version_ids(sessions, "shop") == []
        await engine.dispose()
    asyncio.run(scenario())


def test_blank_data_directory_does_not_require_external_services(tmp_path):
    async def scenario():
        engine, sessions = await sessions_for(tmp_path)
        assert await prepare_local_snapshots(sessions, FakeStore(), FakeEmbeddings(),
                                             data_root=tmp_path / "data") == []
        await engine.dispose()
    asyncio.run(scenario())


def test_real_markdown_parser_builds_pending_snapshot(tmp_path):
    root, _, _ = local_fixture(tmp_path)

    async def scenario():
        engine, sessions = await sessions_for(tmp_path)
        store = FakeStore()
        assert await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root) == ["r1"]
        assert await active_version_ids(sessions, "shop") == []
        async with sessions() as session:
            assert (await session.get(KnowledgeRelease, "r1")).status == "pending_review"
        await engine.dispose()
    asyncio.run(scenario())


def test_failed_approval_retains_pending_release(tmp_path, monkeypatch):
    root, _, _ = local_fixture(tmp_path)

    def parser(path):
        parsed = sample_document()
        parsed.source_name = path.name
        parsed.source_format = "md"
        parsed.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return parsed

    monkeypatch.setattr("rag.local_ingestion.parse_document", parser)

    async def scenario():
        engine, sessions = await sessions_for(tmp_path)
        store = FakeStore()
        await prepare_local_snapshots(sessions, store, FakeEmbeddings(), data_root=root)
        store.fail_activate = True
        with pytest.raises(RuntimeError, match="version verification failed"):
            await approve_local_release(sessions, store, data_root=root,
                                        kb_id="shop", release_id="r1", reviewer="operator")
        assert await active_version_ids(sessions, "shop") == []
        async with sessions() as session:
            assert (await session.get(KnowledgeRelease, "r1")).status == "pending_review"
        await engine.dispose()
    asyncio.run(scenario())


def test_app_startup_prepares_without_exposing_publication(monkeypatch):
    app_module = importlib.import_module("web.app")
    called = []

    async def prepare():
        called.append("prepared")
        return ["r1"]

    monkeypatch.setattr("rag.local_ingestion.prepare_startup", prepare)
    with TestClient(app_module.create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.app.state.rag_prepared_releases == ["r1"]
    assert called == ["prepared"]


def test_preparation_failure_keeps_http_service_available(monkeypatch):
    app_module = importlib.import_module("web.app")

    async def prepare():
        raise RuntimeError("invalid manifest")

    monkeypatch.setattr("rag.local_ingestion.prepare_startup", prepare)
    with TestClient(app_module.create_app()) as client:
        assert client.get("/health").status_code == 200
        assert "invalid manifest" in client.app.state.rag_preparation_error
