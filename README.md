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
- 基于 PyJWT 校验 Spring Boot 登录后签发的 RS256 Access Token
- 提供无需鉴权的健康检查接口
- 在应用关闭时释放已创建的数据库连接池
- 使用 Pytest 覆盖模型客户端、聊天契约、意图识别和依赖边界

> 默认 `/chat` 尚未启用 RAG，也未接入商品、订单、工具调用和售后工作流，不能执行退款、赔偿或其他业务动作。课程 RAG 仅供隔离练习。

## 课程 RAG 沙箱（第 08–16 课）

默认关闭，不影响当前 `/chat` 或 Spring Boot 商城。课程原文位于 `knowledge/course/`，向量仅写入独立的 `course_rag_<内容指纹>` Milvus collection。文档中的历史活动不代表今天有效的商城政策；正式业务知识与实时订单、库存、物流等信息均未接入。

启用前需要可访问的 Milvus Standalone（默认 `http://127.0.0.1:19530`）和支持 OpenAI-compatible `/embeddings` 的独立 embedding Key。没有 Milvus 时可运行 `docker compose -f docker-compose.course-milvus.yml up -d`；若本机 19530 端口已有 Milvus，则直接复用，避免端口冲突。安装依赖后，在本机 `.env` 中设置：

```dotenv
COURSE_RAG_EMBEDDING_API_KEY=课程embedding服务的Key
COURSE_RAG_ENABLED=true
SERVER_HOST=127.0.0.1
SERVER_PORT=8001
```

如需复现课程中的 2026 春季案例，可以**只在隔离环境**设置 `COURSE_RAG_AS_OF_DATE=2026-03-15`；留空按当天日期过滤过期活动。回答模型仍由原有 `AGENT_OPENAI_*` 配置控制。不同服务商时，分别设置 `COURSE_RAG_EMBEDDING_BASE_URL` 与 `AGENT_OPENAI_BASE_URL`，不可混用密钥。

先执行核心构建，再启动隔离端口的服务：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m rag.commands rebuild
.\.venv\Scripts\python.exe main.py
```

索引按文档内容、切片参数和 embedding 模型生成版本；修改这些项目后再次执行 `rebuild`，并**重启 Agent 进程**以读取新版本。不会删除旧 collection；可在确认不再使用之后另行清理。首次聊天会校验索引完整性，缺少索引或 Milvus/embedding 服务不可用时只返回保守话术，不会用关键词命中冒充向量检索。

```powershell
$body = @{ session_id = 'course-08-16'; runtime_user_id = 'course-user'; user_message = '金卡会员买降噪耳机，会员价还能叠加优惠券吗？' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/chat -ContentType 'application/json' -Body $body
.\.venv\Scripts\python.exe -m rag.commands evaluate
```

`citations` 来自实际知识命中；`session_state.rag` 展示改写、两路召回、索引版本、置信度与检索缓存状态。仅缓存稳定知识的命中 ID/片段，最多 256 项、有效 5 分钟，不缓存最终回答或实时业务数据。商业 reranker 可选配置 `COURSE_RAG_RERANK_BASE_URL`、`COURSE_RAG_RERANK_MODEL`、`COURSE_RAG_RERANK_API_KEY`；不配置时沿用课程轻量重排。质量检查仅通过 `evaluate` 离线运行。

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
│   └── jwt.py              # Spring Boot 委托 JWT 验签
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
- 调用 `/chat` 时需要配置 Spring Boot 签发密钥对应的 RSA 公钥

数据库和 JWT 资源均按需加载。未配置 MySQL 或 RSA 公钥时，健康检查等不依赖这些资源的功能仍可正常启动，但 `/chat` 会返回 `503`。

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

用户名或密码包含特殊字符时，需要先进行 URL 编码。通用数据库能力按需加载。

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
| `AUTH_JWT_PUBLIC_KEY` | 空 | Spring Boot 签发密钥对应的 PEM 公钥 |
| `AUTH_JWT_ALGORITHM` | `RS256` | 允许的签名算法，目前固定为 RS256 |
| `AUTH_JWT_ISSUER` | `commerce-backend` | 必须匹配的签发方 |
| `AUTH_JWT_AGENT_AUDIENCE` | `ecommerce-agent` | 必须包含的 Agent audience |
| `AUTH_JWT_REQUIRED_SCOPE` | `agent:chat` | 调用 `/chat` 必须具有的 scope |
| `AUTH_JWT_LEEWAY_SECONDS` | `5` | 时间声明允许的时钟偏差秒数 |

Agent 只验签，不生成令牌，也不得保存 Spring Boot 的私钥。在 `.env` 或部署平台中配置 PEM 公钥时，可用 `\n` 表示换行，程序会在使用前还原：

```dotenv
AUTH_JWT_PUBLIC_KEY=-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----
```

`/chat` 只接受 Spring Boot 登录或注册后签发的 Access Token，并校验 `iss`、`aud`、`exp`、`iat`、`nbf`、`sub`、`jti`、`token_use` 和 `scope`。请求体中的旧身份字段仅用于向后兼容，可信身份始终来自 JWT。Spring Boot 与 Agent 必须使用相同的 issuer、Agent audience 和公钥。

调用链约定为：浏览器或 Apifox 登录 Spring Boot 获得默认 30 分钟的 RS256 Access Token，之后使用 Bearer Token 调用 Spring Boot 或 Agent；Spring Boot 转调 Agent、Agent 访问 `/api/agent/facts/**` 时都原样转发该 Token。

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

## 开发约定

- 新增 ORM 模型时继承 `db.base.Base`
- 新增路由后在 `web.app.create_app` 中注册
- 敏感信息只通过环境变量或安全的密钥管理服务提供，不提交真实 `.env`、数据库密码或 RSA 密钥
- 新增功能时同步补充测试和本 README 中对应的配置、接口及启动说明
