"""从项目 data 目录增量准备待审核版本，审核前不改变当前生效批次。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field
from pymilvus import MilvusClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config.database import get_session_factory
from config.rag import get_rag_settings
from config.settings import PROJECT_ROOT
from rag.document_processing import parse_document
from rag.embeddings import BailianEmbeddings
from rag.milvus_store import MilvusKnowledgeStore
from rag.models import DocumentVersion, KnowledgeRelease, ReleaseDocument
from rag.publication import KnowledgeSource, approve_prepared_snapshot, prepare_snapshot


_ID = re.compile(r"[A-Za-z0-9_]{1,40}\Z")
_FORMATS = {".md", ".docx", ".pdf"}


class LocalDocument(BaseModel):
    """清单中一份原件的业务身份和适用范围。"""

    model_config = ConfigDict(extra="forbid")
    doc_id: str
    doc_version_id: str
    path: str
    topic: str = Field(min_length=1)
    audience: str = "all"
    valid_from: str | None = None
    valid_to: str | None = None


class LocalManifest(BaseModel):
    """一个知识库某次完整快照的文件清单；文件顺序不影响指纹。"""

    model_config = ConfigDict(extra="forbid")
    kb_id: str
    release_id: str
    documents: list[LocalDocument] = Field(min_length=1)


@dataclass(frozen=True)
class LocalSnapshot:
    manifest: LocalManifest
    paths: list[Path]
    hashes: list[str]
    historical_paths: list[Path]


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _is_ignored(path: Path, directory: Path) -> bool:
    parts = path.relative_to(directory).parts
    return any(part.startswith(".") or part.startswith("~$") for part in parts) or path.suffix.lower() in {".tmp", ".part"}


def _read_snapshot(data_root: Path, kb_dir: Path) -> LocalSnapshot:
    manifest_path = kb_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"Knowledge directory requires manifest.json: {kb_dir.name}")
    manifest = LocalManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.kb_id != kb_dir.name or not _ID.fullmatch(manifest.kb_id) or not _ID.fullmatch(manifest.release_id):
        raise ValueError("Manifest knowledge base or release ID is invalid")

    paths: list[Path] = []
    hashes: list[str] = []
    listed: set[Path] = set()
    historical_paths: list[Path] = []
    doc_ids: set[str] = set()
    version_ids: set[str] = set()
    for document in manifest.documents:
        parts = PurePosixPath(document.path).parts
        if (len(parts) != 3 or "\\" in document.path or document.path.startswith("/")
            or any(part in (".", "..") for part in parts)
            or parts[:2] != (document.doc_id, document.doc_version_id)
            or not _ID.fullmatch(document.doc_id) or not _ID.fullmatch(document.doc_version_id)):
            raise ValueError(f"Document path must be doc_id/doc_version_id/filename: {document.path}")
        path = kb_dir.joinpath(*parts)
        if path.suffix.lower() not in _FORMATS or _is_ignored(path, kb_dir):
            raise ValueError(f"Unsupported or temporary document path: {document.path}")
        if (not path.is_file() or any(part.is_symlink() for part in (path, path.parent, path.parent.parent))):
            raise ValueError(f"Document missing or symbolic link: {document.path}")
        if not path.resolve().is_relative_to(data_root.resolve()):
            raise ValueError(f"Document path escapes data directory: {document.path}")
        if path in listed or document.doc_id in doc_ids or document.doc_version_id in version_ids:
            raise ValueError("Duplicate document, version or path in manifest")
        for value in (document.valid_from, document.valid_to):
            if value and datetime.fromisoformat(value).tzinfo is None:
                raise ValueError("Validity timestamps must include a timezone")
        listed.add(path)
        doc_ids.add(document.doc_id)
        version_ids.add(document.doc_version_id)
        paths.append(path)
        hashes.append(_digest(path))

    # 缺少清单条目时不把文件从完整快照中静默遗漏，也不自动发布删除。
    for path in kb_dir.rglob("*"):
        if path.is_symlink() and not _is_ignored(path, kb_dir):
            raise ValueError(f"Symbolic link in data directory: {path}")
        if not path.is_file() or path == manifest_path or _is_ignored(path, kb_dir):
            continue
        if path.suffix.lower() not in _FORMATS:
            raise ValueError(f"Unsupported document in data directory: {path}")
        if path not in listed:
            historical_paths.append(path)
    return LocalSnapshot(manifest, paths, hashes, historical_paths)


async def _validate_historical_files(sessions: async_sessionmaker[AsyncSession],
                                     snapshot: LocalSnapshot, kb_dir: Path) -> None:
    async with sessions() as session:
        for path in snapshot.historical_paths:
            parts = path.relative_to(kb_dir).parts
            version = await session.get(DocumentVersion, parts[1]) if len(parts) == 3 else None
            if (version is None or version.kb_id != snapshot.manifest.kb_id or version.doc_id != parts[0]
                or version.source_uri != f"data/{snapshot.manifest.kb_id}/{path.relative_to(kb_dir).as_posix()}"
                or version.sha256 != await asyncio.to_thread(_digest, path)):
                raise ValueError(f"Unlisted document in data directory: {path}")


def _fingerprint(snapshot: LocalSnapshot, *, model: str, dimension: int) -> str:
    records = [
        {"document": doc.model_dump(mode="json"), "sha256": sha256}
        for doc, sha256 in zip(snapshot.manifest.documents, snapshot.hashes, strict=True)
    ]
    payload = {
        "kb_id": snapshot.manifest.kb_id,
        "documents": sorted(records, key=lambda record: record["document"]["doc_id"]),
        "parser_version": version("docling"),
        "embedding_model": model,
        "dimension": dimension,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _knowledge_directories(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    directories = []
    for entry in sorted(data_root.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink() or not entry.is_dir():
            raise ValueError(f"Unexpected entry in data directory: {entry}")
        directories.append(entry)
    return directories


async def prepare_local_snapshots(
    sessions: async_sessionmaker[AsyncSession], store: MilvusKnowledgeStore,
    embeddings: BailianEmbeddings, *, data_root: Path,
) -> list[str]:
    """检查完整清单，只准备尚未入库的版本，不切换生效批次。"""
    prepared: list[str] = []
    for kb_dir in _knowledge_directories(data_root):
        snapshot = await asyncio.to_thread(_read_snapshot, data_root, kb_dir)
        await _validate_historical_files(sessions, snapshot, kb_dir)
        manifest = snapshot.manifest
        fingerprint = _fingerprint(snapshot, model=embeddings.model, dimension=embeddings.dimension)
        async with sessions() as session:
            existing = await session.get(KnowledgeRelease, manifest.release_id)
            same_content = (await session.execute(select(KnowledgeRelease).where(
                KnowledgeRelease.kb_id == manifest.kb_id,
                KnowledgeRelease.fingerprint == fingerprint,
            ))).scalar_one_or_none()
        if existing:
            if existing.kb_id != manifest.kb_id or existing.fingerprint != fingerprint:
                raise ValueError("Manifest changed; use a new release ID for the changed snapshot")
            if existing.status not in ("pending_review", "published", "superseded"):
                raise ValueError("Existing release ID requires manual reconciliation")
            async with sessions() as session:
                versions = (await session.scalars(select(ReleaseDocument.doc_version_id).where(
                    ReleaseDocument.release_id == manifest.release_id))).all()
            await asyncio.to_thread(store.verify_versions, manifest.kb_id, versions)
            continue
        if same_content:
            raise ValueError(f"Identical snapshot is already prepared as {same_content.release_id}")

        sources: list[KnowledgeSource] = []
        for document, path, sha256 in zip(manifest.documents, snapshot.paths, snapshot.hashes, strict=True):
            parsed = await asyncio.to_thread(parse_document, path)
            if parsed.sha256 != sha256 or _digest(path) != sha256:
                raise ValueError(f"Document changed during preparation: {path}")
            if parsed.warnings:
                raise ValueError(f"Document needs parsing review before staging: {path}: {parsed.warnings}")
            sources.append(KnowledgeSource(
                doc_id=document.doc_id, doc_version_id=document.doc_version_id,
                parsed=parsed, source_uri=f"data/{manifest.kb_id}/{document.path}",
                topic=document.topic, audience=document.audience, approved=False,
                valid_from=datetime.fromisoformat(document.valid_from) if document.valid_from else None,
                valid_to=datetime.fromisoformat(document.valid_to) if document.valid_to else None,
            ))
        await prepare_snapshot(sessions, store, embeddings, kb_id=manifest.kb_id,
                               release_id=manifest.release_id, sources=sources, fingerprint=fingerprint)
        prepared.append(manifest.release_id)
    return prepared


async def approve_local_release(
    sessions: async_sessionmaker[AsyncSession], store: MilvusKnowledgeStore,
    *, data_root: Path, kb_id: str, release_id: str, reviewer: str,
) -> None:
    """审核者确认原件与准备时完全一致后，才发布该集合。"""
    async with sessions() as session:
        release = await session.get(KnowledgeRelease, release_id)
    if not _ID.fullmatch(kb_id) or release is None or release.kb_id != kb_id or release.status != "pending_review":
        raise ValueError("Release is not waiting for review")
    if store.dimension != release.dimension:
        raise ValueError("Configured Milvus dimension differs from prepared release")
    snapshot = await asyncio.to_thread(_read_snapshot, data_root, data_root / kb_id)
    await _validate_historical_files(sessions, snapshot, data_root / kb_id)
    if snapshot.manifest.release_id != release_id or _fingerprint(
        snapshot, model=release.embedding_model, dimension=release.dimension
    ) != release.fingerprint:
        raise ValueError("Source or manifest changed since preparation; use a new release ID")
    await approve_prepared_snapshot(sessions, store, kb_id=kb_id,
                                    release_id=release_id, reviewer=reviewer)


async def prepare_startup() -> list[str]:
    """启动钩子；没有知识目录时不创建数据库或模型连接。"""
    data_root = PROJECT_ROOT / "data"
    if not _knowledge_directories(data_root):
        return []
    settings = get_rag_settings()
    if not settings.milvus_uri or not settings.api_key.get_secret_value() or not settings.embedding_endpoint:
        raise RuntimeError("Local RAG preparation requires Milvus, Bailian and database configuration")
    embeddings = BailianEmbeddings(api_key=settings.api_key.get_secret_value(),
                                  endpoint=settings.embedding_endpoint, model=settings.embedding_model,
                                  dimension=settings.dimension)
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token.get_secret_value())
    try:
        store = MilvusKnowledgeStore(client=client, dimension=settings.dimension)
        return await prepare_local_snapshots(get_session_factory(), store, embeddings, data_root=data_root)
    finally:
        client.close()
