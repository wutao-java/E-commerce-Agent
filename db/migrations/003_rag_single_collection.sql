-- 已执行 001_rag.sql 的数据库先运行此迁移，再准备新的发布批次。
-- 当前每个 release 唯一的约束不允许多个批次共用知识库集合。
ALTER TABLE rag_release DROP INDEX collection_name;
