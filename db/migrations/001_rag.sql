-- 首次发布前在 Agent 项目的 MySQL 数据库执行一次；时间字段统一写入 UTC。
-- 此脚本只建立审计表；同一知识库的版本共用 Milvus 集合。
-- 知识库只记录当前对外服务的发布 ID；具体向量在 Milvus 集合中。
CREATE TABLE rag_knowledge_base (
    kb_id VARCHAR(40) NOT NULL PRIMARY KEY COMMENT '知识库唯一标识',
    active_release_id VARCHAR(40) NULL COMMENT '当前对外服务的发布批次ID；未发布时为空'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- doc_id 表示长期稳定的文档身份，不等于某次上传的内容版本。
CREATE TABLE rag_document (
    kb_id VARCHAR(40) NOT NULL COMMENT '文档所属知识库ID',
    doc_id VARCHAR(128) NOT NULL COMMENT '文档的稳定标识，不随内容版本变化',
    PRIMARY KEY (kb_id, doc_id),
    CONSTRAINT fk_rag_document_kb FOREIGN KEY (kb_id) REFERENCES rag_knowledge_base(kb_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- doc_version_id、原文件哈希和来源形成不可变的文档内容记录。
CREATE TABLE rag_document_version (
    doc_version_id VARCHAR(128) NOT NULL PRIMARY KEY COMMENT '不可复用的文档内容版本ID',
    kb_id VARCHAR(40) NOT NULL COMMENT '文档所属知识库ID',
    doc_id VARCHAR(128) NOT NULL COMMENT '此版本对应的稳定文档ID',
    sha256 VARCHAR(64) NOT NULL COMMENT '原文件内容的SHA-256哈希值',
    source_name VARCHAR(512) NOT NULL COMMENT '原文件名',
    source_uri VARCHAR(1024) NOT NULL COMMENT '原文件的存储位置',
    source_format VARCHAR(12) NOT NULL COMMENT '原文件格式，如md、docx或pdf',
    parser_version VARCHAR(80) NOT NULL COMMENT '解析原文件所用解析器的版本',
    created_at DATETIME(6) NOT NULL COMMENT '文档版本记录创建时间（UTC）',
    KEY ix_rag_version_document (kb_id, doc_id),
    CONSTRAINT fk_rag_version_document FOREIGN KEY (kb_id, doc_id)
        REFERENCES rag_document(kb_id, doc_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 多个 release 可共用同一个集合；待审核版本不参与普通检索。
CREATE TABLE rag_release (
    release_id VARCHAR(40) NOT NULL PRIMARY KEY COMMENT '知识库发布批次的唯一标识',
    kb_id VARCHAR(40) NOT NULL COMMENT '发布批次所属知识库ID',
    collection_name VARCHAR(128) NOT NULL COMMENT '知识库共用的Milvus集合名称',
    embedding_model VARCHAR(128) NOT NULL COMMENT '构建向量索引所用的嵌入模型',
    dimension INT NOT NULL COMMENT '嵌入向量的维度',
    status VARCHAR(20) NOT NULL COMMENT '发布状态：building构建中、pending_review待审核、published已发布、superseded已取代、failed失败',
    created_at DATETIME(6) NOT NULL COMMENT '发布批次创建时间（UTC）',
    published_at DATETIME(6) NULL COMMENT '首次发布成功的时间（UTC）；未发布时为空',
    KEY ix_rag_release_kb (kb_id),
    CONSTRAINT fk_rag_release_kb FOREIGN KEY (kb_id) REFERENCES rag_knowledge_base(kb_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 固定每次完整发布引用的文档版本，不因后续上传新版本而改变。
CREATE TABLE rag_release_document (
    release_id VARCHAR(40) NOT NULL COMMENT '引用文档版本的发布批次ID',
    doc_version_id VARCHAR(128) NOT NULL COMMENT '发布批次包含的文档内容版本ID',
    PRIMARY KEY (release_id, doc_version_id),
    KEY ix_rag_release_doc_version (doc_version_id),
    CONSTRAINT fk_rag_release_document_release FOREIGN KEY (release_id)
        REFERENCES rag_release(release_id),
    CONSTRAINT fk_rag_release_document_version FOREIGN KEY (doc_version_id)
        REFERENCES rag_document_version(doc_version_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
