-- 为已准备但未发布的快照记录输入指纹和人工审核信息。
-- 在 001_rag.sql 之后执行一次；已有发布批次允许指纹为空。
ALTER TABLE rag_release
    ADD COLUMN fingerprint VARCHAR(64) NULL COMMENT '快照输入及构建配置的SHA-256指纹；旧批次可为空',
    ADD COLUMN reviewed_by VARCHAR(128) NULL COMMENT '审核发布的操作人；未经过人工审核时为空',
    ADD COLUMN reviewed_at DATETIME(6) NULL COMMENT '审核发布的时间（UTC）；未经过人工审核时为空';
