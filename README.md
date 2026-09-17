# E-commerce Agent

E-commerce Agent 是一个面向电子商务客服场景的 FastAPI 服务。当前版本在通用配置、日志、异步 MySQL 和 JWT 基础上，提供聊天接口、规则优先的结构化意图识别和 OpenAI-compatible 模型回答能力。

## 当前能力

- 基于 FastAPI 提供 HTTP 服务和 OpenAPI 文档
- 提供 `/chat` 聊天接口和 `/capabilities` 能力声明
- 支持规则优先、分类模型兜底的结构化意图识别
- 支持回答模型失败时回退到确定性安全话术
- 支持 INI 配置、`.env` 文件和系统环境变量
- 支持控制台日志和按大小轮转的文件日志
- 基于 SQLAlchemy 2.x 和 asyncmy 提供异步 MySQL 引擎与会话
- 基于 PyJWT 提供 RS256 JWT 签发与校验
- 提供无需鉴权的健康检查接口
- 在应用关闭时释放已创建的数据库连接池
- 使用 Pytest 覆盖模型客户端、聊天契约、意图识别和依赖边界

> 当前 `/chat` 尚未接入商品、订单、RAG 检索、工具调用和售后工作流，不能执行退款、赔偿或其他业务动作；RAG 入库与检索可作为独立业务模块调用。

## 技术栈

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic 2.x
- SQLAlchemy 2.x
- asyncmy
- PyJWT
- Pytest

## 项目结构

```text
E-commerce-agent/
├── agent/
│   ├── customer_service_agent.py # 客服流程编排
│   ├── intent_service.py         # 规则与分类模型的意图识别编排
│   └── intent_rules.py           # 高置信意图规则
├── config/
│   ├── database.py         # 异步数据库配置、引擎和会话管理
│   ├── logging_config.py   # 控制台与轮转文件日志配置
│   └── settings.py         # INI、.env 和环境变量加载
├── db/
│   └── base.py             # SQLAlchemy 声明式模型基类
├── domain/
│   ├── chat.py             # Agent 内部聊天命令与结果
│   └── intent.py           # 意图类型与识别结果
├── llm/
│   ├── client.py           # OpenAI-compatible 通用客户端
│   ├── intent_classifier.py # 意图分类模型适配器
│   └── answer_generator.py # 客服回答模型适配器
├── security/
│   └── jwt.py              # JWT 配置、签发与校验
├── tests/                  # 自动化测试
├── web/
│   ├── routers/
│   │   ├── chat.py         # HTTP DTO 与内部契约转换
│   │   └── health.py       # 健康检查路由
│   └── app.py              # FastAPI 应用工厂与生命周期管理
├── .env.example            # 本地环境变量示例
├── application.ini         # 默认应用配置
├── main.py                 # 本地启动入口
└── pyproject.toml          # 项目元数据与依赖
```

## 环境要求

- Python 3.11 或更高版本
- 使用数据库功能时需要可访问的 MySQL 实例
- 使用 JWT 功能时需要匹配的 RSA 私钥和公钥

数据库和 JWT 资源均按需加载。未配置 MySQL 或 RSA 密钥时，健康检查等不依赖这些资源的功能仍可正常启动。

## 快速开始

### Windows PowerShell

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe main.py
```

复制 `.env.example` 后，服务默认监听 `127.0.0.1:8000`。

### Linux 或 macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env
.venv/bin/python -m pytest
.venv/bin/python main.py
```

服务启动后可访问：

- 健康检查：<http://127.0.0.1:8000/health>
- Swagger UI：<http://127.0.0.1:8000/docs>
- ReDoc：<http://127.0.0.1:8000/redoc>
- OpenAPI 描述：<http://127.0.0.1:8000/openapi.json>

## 健康检查

PowerShell 请求示例：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

响应内容：

```json
{
  "status": "ok",
  "service": "e-commerce-agent"
}
```

该接口仅表示 HTTP 服务可以正常响应，不会主动探测 MySQL 等外部依赖。

## 配置说明

默认配置文件为项目根目录下的 `application.ini`。应用启动时还会自动加载同目录下的 `.env` 文件，配置优先级从高到低为：

1. 当前进程的系统环境变量
2. `.env` 文件中的变量
3. `application.ini` 中占位符声明的默认值

如需使用其他 INI 文件，可设置 `AGENT_CENTER_CONFIG`：

```powershell
$env:AGENT_CENTER_CONFIG = "D:\config\e-commerce-agent.ini"
.\.venv\Scripts\python.exe main.py
```

配置会在进程内缓存。修改配置后应重新启动服务，使新值生效。

### 服务配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SERVICE_NAME` | `e-commerce-agent` | 服务名称，同时用于 FastAPI 标题和健康检查响应 |
| `SERVER_HOST` | `0.0.0.0` | 服务监听地址；`.env.example` 为本地开发设置成 `127.0.0.1` |
| `SERVER_PORT` | `8000` | 服务监听端口，范围为 1～65535 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `LOG_FILE` | 空 | 日志文件路径；为空时仅输出到控制台 |
| `START_INTEGRATIONS` | `false` | 外部集成开关预留项，当前未绑定具体集成 |

配置 `LOG_FILE` 后，应用会自动创建日志目录。单个日志文件最大为 10 MiB，并保留 5 个历史文件。

### MySQL 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | 空 | SQLAlchemy 异步 MySQL 连接地址，使用数据库功能时必填 |
| `DATABASE_POOL_SIZE` | `10` | 连接池常驻连接数，最小值为 1 |
| `DATABASE_MAX_OVERFLOW` | `20` | 连接池允许临时扩展的连接数，最小值为 0 |
| `DATABASE_POOL_TIMEOUT` | `30` | 等待可用连接的最长秒数 |
| `DATABASE_POOL_RECYCLE` | `1800` | 连接回收周期，单位为秒 |
| `DATABASE_ECHO` | `false` | 是否输出 SQL 日志，生产环境建议保持关闭 |

连接地址示例：

```dotenv
DATABASE_URL=mysql+asyncmy://app_user:app_password@127.0.0.1:3306/ecommerce?charset=utf8mb4
```

用户名或密码包含特殊字符时，需要先进行 URL 编码。通用数据库能力按需加载；RAG 审计表由 `db/migrations/001_rag.sql`、`002_rag_review.sql` 顺序创建或升级。

业务路由中可通过 `config.database.get_session` 获取按请求关闭的异步会话：

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from config.database import get_session


async def example(session: AsyncSession = Depends(get_session)) -> None:
    ...
```

### JWT 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `JWT_PRIVATE_KEY` | 空 | 用于签发令牌的 PEM 私钥 |
| `JWT_PUBLIC_KEY` | 空 | 用于校验令牌的 PEM 公钥 |
| `JWT_ALGORITHM` | `RS256` | JWT 签名算法 |
| `JWT_EXPIRE_MINUTES` | `60` | 默认有效期，单位为分钟，最小值为 1 |

在 `.env` 或部署平台中配置 PEM 密钥时，可用 `\n` 表示换行，程序会在使用前还原。例如：

```dotenv
JWT_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----
JWT_PUBLIC_KEY=-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----
```

签发和校验示例：

```python
from security.jwt import create_token, decode_token

token = create_token({"sub": "user-1"})
claims = decode_token(token)
```

签发时会自动写入 UTC 时间的 `iat` 和 `exp` 声明，并覆盖调用方传入的同名字段。私钥或公钥为空时，实际执行签发或校验操作会抛出明确的配置异常。

## 运行测试

执行全部测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

仅执行单个测试文件：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_lesson_04_intent.py
```

当前测试覆盖：

- OpenAI-compatible 消息和模型客户端
- `/chat` 请求校验、响应契约和异常映射
- 规则优先、分类模型兜底的结构化意图识别
- 回答模型失败时的确定性安全兜底
- 核心模块不反向依赖 Web 层的架构边界

## 文档处理（RAG 入库准备）

`rag.document_processing.parse_document(path)` 接收 `.md`、`.docx`、`.pdf` 文件，返回有序的正文块、标题路径、PDF 实际页码、文件 SHA-256 和解析警告。表格保留为 Markdown 文本，便于后续以整张表为单位切片；Word 和 Markdown 没有可靠的原文件页码，结果中保持为空。旧版 `.doc` 请先转换为 `.docx`。

本地原文统一放在项目根目录的 `data/`，推荐按 `data/<kb_id>/<doc_id>/<doc_version_id>/<原文件名>` 归档，例如 `data/shop/refund_policy/refund_v1/refund.pdf`。同一文档的新版本另建目录，不覆盖旧版。`data/` 下的原件默认不纳入 Git；实际部署需将这个目录挂载到持久化存储，避免容器重建后丢失文件。从项目根目录运行以下示例：

```python
from rag.document_processing import parse_document

document = parse_document("data/shop/refund_policy/refund_v1/refund.pdf")
for block in document.blocks:
    print(block.kind, block.heading_path, block.page_start, block.text)
```

PDF 使用 Docling 进行版面与表格识别，并通过 RapidOCR 识别扫描页。首次解析 PDF 会下载版面和表格模型；部署时应在构建或预热阶段缓存模型，运行时保留 Hugging Face 缓存目录，离线部署可在模型备齐后设置 `HF_HUB_OFFLINE=1`。转换不完整或没有可提取文字会报错；缺少页码定位、图片没有文字说明等情况记录在 `warnings` 中，入库调用方须先复核。当前模块只解析文件，不负责上传接口、语义切片或向量入库，也不会改变 `/chat` 的行为。

## RAG 快照发布与混合检索

### 本地 Milvus（Docker Desktop）

先启动 Docker Desktop，在本项目目录运行以下命令。独立 Compose 项目使用 Milvus 2.6.23 Standalone、etcd 和 MinIO；仅 Milvus 端口绑定本机，三者的数据分别保存在 Docker 命名卷中，不依赖 Spring Boot 的 Compose 项目。

```powershell
docker compose -f docker-compose.milvus.yml up -d --wait
docker compose -f docker-compose.milvus.yml ps
.\.venv\Scripts\python.exe -c "from pymilvus import MilvusClient; c = MilvusClient(uri='http://127.0.0.1:19530'); print(c.list_collections()); c.close()"
```

在 `.env` 设置 `MILVUS_URI=http://127.0.0.1:19530`；本地端口只供本机开发使用。使用 `docker compose -f docker-compose.milvus.yml down` 停止服务，**不要添加 `-v`**，否则会删除持久化的知识库数据。该 Compose 使用仅限本机开发的默认 MinIO 凭据和未开启认证的 Milvus，不要直接对公网开放。

新增的 `rag.chunking`、`rag.embeddings`、`rag.milvus_store`、`rag.publication` 和 `rag.reranker` 是可调用的业务模块，**没有接入 `/chat` 或提供文件上传接口**。父块按标题和完整语义块组织，子块只在同一父块内按段落边界重叠；过长的原子段落/表格直接要求人工复核。每条 Milvus 子块包含父块原文、章节路径、文件名、实际 PDF 页码（其他格式为空）、内容哈希、版本、范围和生效时间；向量只对 `child_text` 生成。

新数据库依次执行 [建表迁移](db/migrations/001_rag.sql) 和 [待审核字段迁移](db/migrations/002_rag_review.sql)；已使用旧版建表 SQL 的数据库须额外执行 [单集合迁移](db/migrations/003_rag_single_collection.sql)，并将原有 `rag_<kb_id>_<release_id>` 集合改名为 `rag_<kb_id>`、同步更新对应的 `rag_release.collection_name`，确认无线上别名后再启用新代码。确保 Milvus 2.6.x 可访问。在 `.env` 配置 `DATABASE_URL`、`MILVUS_URI`、可选的 `MILVUS_TOKEN`、`BAILIAN_API_KEY`、`BAILIAN_EMBEDDING_ENDPOINT` 和 `BAILIAN_RERANK_ENDPOINT`。北京地域百炼业务空间接口分别为 `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding` 和 `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks`；用实际业务空间 ID 替换占位符。默认 embedding 模型是支持稠密和模型稀疏双路输出的 `qwen3.7-text-embedding`，1024 维；`qwen3.7-text-embedding-flash` 当前文档未列入双路支持范围，不能直接替换。更换模型或维度仍需单独迁移整个索引，不能把两种模型的向量混在同一集合。

已有多个历史发布集合或线上别名的环境必须先制定合并向量及回滚迁移方案，不能直接按单批次重命名。

### 本地准备与审核发布

在 `data/shop/manifest.json` 写入**当前待发布的完整文档版本列表**，文档路径相对于 `data/shop/`。新增文件时保留原清单条目、加入新文档并更新 `release_id`；同一文档换版时保留 `doc_id`，修改 `doc_version_id` 和路径，旧版原件留在原目录：

```json
{
  "kb_id": "shop",
  "release_id": "release_001",
  "documents": [
    {
      "doc_id": "refund_policy",
      "doc_version_id": "refund_v1",
      "path": "refund_policy/refund_v1/refund.pdf",
      "topic": "after_sale",
      "audience": "all",
      "valid_from": "2026-09-16T00:00:00+08:00"
    }
  ]
}
```

`RAG_PREPARE_ON_START=true` 为默认值。服务启动时若 `data/` 为空，不连接 RAG 外部服务；存在知识清单时核对清单、文件、哈希和解析告警，仅向 `rag_<kb_id>` 写入尚未入库的文档版本，MySQL 批次标为 `pending_review`，**新版本尚不参与普通检索**。已入库的版本会核对切片，不会重复调用 embedding；完全相同的清单直接跳过，但若集合或版本被手动删除会报错，须人工核查。旧版原件可留在原目录，未列入当前清单的文件仅在 MySQL 中已有匹配路径和哈希的历史版本时才允许存在。原文、解析器或业务范围变化需新的文档版本 ID 与发布 ID；模型变化需单独迁移整个集合。缺文件、多出未登记文件或解析告警会阻止准备，现有 HTTP 服务和已发布版本不受影响。多进程同时启动时同一批次 ID 由 MySQL 约束避免重复准备；若留下 `building` 或 `failed` 批次，须先人工核查。启动准备可能延迟首次就绪，也会产生新版本的 embedding 费用；无需自动准备可设置 `RAG_PREPARE_ON_START=false`。

运维人员先检查原件、解析结果及批次哈希，再从受信任的服务器终端发布；没有开放无鉴权的审核 HTTP 接口：

```powershell
.\.venv\Scripts\python.exe -m rag.commands inspect --kb-id shop --release-id release_001
.\.venv\Scripts\python.exe -m rag.commands approve --kb-id shop --release-id release_001 --reviewer operator_name --confirm
```

审核发布前会再次核对原件与清单指纹；若准备后文件被改动，须使用新版本和新发布 ID 重新准备。审核只在 MySQL 中切换当前生效的版本组合，不会再次生成向量；审核失败会保留 `pending_review` 和原生效版本供核查或重试。原文件所在目录需要持久挂载，并限制有权修改清单、原件及执行审核命令的人员。

### 检索与回滚

```python
from datetime import datetime, timezone
import asyncio

from pymilvus import MilvusClient
from config.database import dispose_engine, get_session_factory
from config.rag import get_rag_settings
from rag.embeddings import BailianEmbeddings
from rag.milvus_store import MilvusKnowledgeStore
from rag.publication import active_version_ids
from rag.reranker import BailianReranker, retrieve

settings = get_rag_settings()
key = settings.api_key.get_secret_value()
embeddings = BailianEmbeddings(api_key=key, endpoint=settings.embedding_endpoint,
                              model=settings.embedding_model, dimension=settings.dimension)
store = MilvusKnowledgeStore(client=MilvusClient(uri=settings.milvus_uri,
                              token=settings.milvus_token.get_secret_value()), dimension=settings.dimension)
reranker = BailianReranker(api_key=key, endpoint=settings.rerank_endpoint)
async def load_versions():
    try:
        return await active_version_ids(get_session_factory(), "shop")
    finally:
        await dispose_engine()

versions = asyncio.run(load_versions())
evidence = retrieve(query="拆封后可以退款吗？", kb_id="shop", store=store,
                    embeddings=embeddings, reranker=reranker, topic="after_sale",
                    version_ids=versions, at=datetime.now(timezone.utc))
for hit in evidence:
    print(hit.parent_text, hit.source_name, hit.heading_path, hit.page_start)
```

`doc_id` 是稳定文档身份，`doc_version_id` 不可复用修改内容，`release_id` 唯一且不重复；每个知识库只有一个集合，旧文档版本的向量保留以便回滚。发布期间用数据库行锁串行化同一知识库的生效版本切换；未审核版本虽已入库，但检索必须带上 `active_version_ids` 返回的版本列表。需要回滚时由受信任的管理流程调用 `rag.publication.rollback_snapshot`，仅改变 MySQL 的当前发布指针。查询默认双路各召回 30 个候选、RRF 融合、重排，再取最多 4 个不同父块；这些数值需用真实标注集调优。证据不足时由后续回答层兜底；实时订单、库存、物流等仍应访问业务接口。

## 开发约定

- 新增 ORM 模型时继承 `db.base.Base`
- 新增路由后在 `web.app.create_app` 中注册
- 敏感信息只通过环境变量或安全的密钥管理服务提供，不提交真实 `.env`、数据库密码或 RSA 密钥
- 新增功能时同步补充测试和本 README 中对应的配置、接口及启动说明
