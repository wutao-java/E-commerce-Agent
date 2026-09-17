"""受信任运维使用的待审核快照查看和人工发布命令。"""

from __future__ import annotations

import argparse
import asyncio
import json

from pymilvus import MilvusClient
from sqlalchemy import select

from config.database import dispose_engine, get_session_factory
from config.rag import get_rag_settings
from config.settings import PROJECT_ROOT
from rag.local_ingestion import approve_local_release
from rag.milvus_store import MilvusKnowledgeStore
from rag.models import DocumentVersion, KnowledgeRelease, ReleaseDocument


async def _run(args: argparse.Namespace) -> None:
    try:
        sessions = get_session_factory()
        if args.action == "inspect":
            async with sessions() as session:
                release = await session.get(KnowledgeRelease, args.release_id)
                if release is None or release.kb_id != args.kb_id:
                    raise ValueError("Release not found in the specified knowledge base")
                versions = (await session.execute(
                    select(DocumentVersion).join(ReleaseDocument,
                        ReleaseDocument.doc_version_id == DocumentVersion.doc_version_id)
                    .where(ReleaseDocument.release_id == args.release_id)
                    .order_by(DocumentVersion.doc_id)
                )).scalars().all()
                print(json.dumps({
                    "kb_id": release.kb_id, "release_id": release.release_id,
                    "status": release.status, "embedding_model": release.embedding_model,
                    "documents": [{"doc_id": v.doc_id, "doc_version_id": v.doc_version_id,
                                   "source_uri": v.source_uri, "sha256": v.sha256} for v in versions],
                }, ensure_ascii=False, indent=2))
        else:
            if not args.confirm:
                raise ValueError("Approval requires --confirm after inspecting the source documents")
            settings = get_rag_settings()
            if not settings.milvus_uri:
                raise RuntimeError("MILVUS_URI must be configured before publication")
            async with sessions() as session:
                release = await session.get(KnowledgeRelease, args.release_id)
                if release is None or release.kb_id != args.kb_id or (
                    release.embedding_model != settings.embedding_model or release.dimension != settings.dimension
                ):
                    raise ValueError("Configured embedding model differs from prepared release")
            client = MilvusClient(uri=settings.milvus_uri,
                                  token=settings.milvus_token.get_secret_value())
            try:
                store = MilvusKnowledgeStore(client=client, dimension=settings.dimension)
                await approve_local_release(sessions, store, data_root=PROJECT_ROOT / "data",
                                            kb_id=args.kb_id, release_id=args.release_id,
                                            reviewer=args.reviewer)
                print(f"Published {args.kb_id}/{args.release_id}")
            finally:
                client.close()
    finally:
        await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or approve a prepared RAG release")
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("inspect", "approve"):
        command = actions.add_parser(action)
        command.add_argument("--kb-id", required=True)
        command.add_argument("--release-id", required=True)
        if action == "approve":
            command.add_argument("--reviewer", required=True)
            command.add_argument("--confirm", action="store_true")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
