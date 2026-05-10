-- 01-create-rag-db.sql
-- 创建数据库
CREATE DATABASE rag_db;

-- 连接到新数据库（某些权限设置需要）
\c rag_db;

-- 授予 postgres 用户所有权限（已经是所有者，这步可选）
GRANT ALL PRIVILEGES ON DATABASE rag_db TO postgres;

-- 设置 rag_db 为 postgres 用户的默认数据库（可选）
-- 注意：这不会影响环境变量中的默认数据库设置