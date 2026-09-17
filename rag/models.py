"""用 MySQL 保存文档版本、发布批次及当前服务快照的审计信息。"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import ForeignKey, ForeignKeyConstraint, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


def _utcnow() -> datetime:
    """新版本与发布记录统一使用 UTC 时间。"""

    return datetime.now(timezone.utc)


class KnowledgeBase(Base):
    """每个知识库只记录一个当前对外服务的发布批次。"""

    __tablename__ = "rag_knowledge_base"

    kb_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    active_release_id: Mapped[str | None] = mapped_column(String(40), nullable=True)


class KnowledgeDocument(Base):
    """同一知识库中稳定的文档身份，不随文件内容更新而变化。"""

    __tablename__ = "rag_document"

    kb_id: Mapped[str] = mapped_column(String(40), ForeignKey("rag_knowledge_base.kb_id"), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)


class DocumentVersion(Base):
    """一次不可变的文档内容及原文件定位，用哈希校验版本身份。"""

    __tablename__ = "rag_document_version"
    __table_args__ = (ForeignKeyConstraint(["kb_id", "doc_id"], ["rag_document.kb_id", "rag_document.doc_id"]),)

    doc_version_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    kb_id: Mapped[str] = mapped_column(String(40))
    doc_id: Mapped[str] = mapped_column(String(128))
    sha256: Mapped[str] = mapped_column(String(64))
    source_name: Mapped[str] = mapped_column(String(512))
    source_uri: Mapped[str] = mapped_column(String(1024))
    source_format: Mapped[str] = mapped_column(String(12))
    parser_version: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


class KnowledgeRelease(Base):
    """一个完整快照对应的 Milvus 集合、模型配置与发布状态。"""

    __tablename__ = "rag_release"

    release_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    kb_id: Mapped[str] = mapped_column(String(40), ForeignKey("rag_knowledge_base.kb_id"), index=True)
    collection_name: Mapped[str] = mapped_column(String(128))
    embedding_model: Mapped[str] = mapped_column(String(128))
    dimension: Mapped[int] = mapped_column(Integer)
    # 本地启动准备记录清单与原件指纹；旧的直接发布入口可不填写。
    fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    # building / pending_review / published / superseded / failed。
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    published_at: Mapped[datetime | None] = mapped_column(nullable=True)


class ReleaseDocument(Base):
    """固定某次完整发布所采用的各文档版本，便于审计和回滚。"""

    __tablename__ = "rag_release_document"

    release_id: Mapped[str] = mapped_column(String(40), ForeignKey("rag_release.release_id"), primary_key=True)
    doc_version_id: Mapped[str] = mapped_column(String(128), ForeignKey("rag_document_version.doc_version_id"), primary_key=True)
