-- 正式入库独立前向迁移；正文/版本不可变，状态与撤权字段只由受控仓库方法更新。
CREATE TABLE harness_rag.knowledge_bases (
    tenant_id text NOT NULL,
    application_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    created_by text NOT NULL,
    revoked_at timestamptz,
    PRIMARY KEY(tenant_id, application_id, knowledge_base_id)
);
CREATE TABLE harness_rag.knowledge_grants (
    tenant_id text NOT NULL,
    application_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    subject_id text NOT NULL,
    PRIMARY KEY(tenant_id, application_id, knowledge_base_id, subject_id),
    FOREIGN KEY(tenant_id, application_id, knowledge_base_id)
      REFERENCES harness_rag.knowledge_bases
);
CREATE TABLE harness_rag.documents (
    tenant_id text NOT NULL,
    application_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    document_id text NOT NULL,
    revoked_at timestamptz,
    PRIMARY KEY(tenant_id, application_id, knowledge_base_id, document_id),
    FOREIGN KEY(tenant_id, application_id, knowledge_base_id)
      REFERENCES harness_rag.knowledge_bases
);
CREATE TABLE harness_rag.ingestion_jobs (
    job_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    application_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    document_id text NOT NULL,
    subject_id text NOT NULL,
    request_id text NOT NULL,
    fingerprint text NOT NULL,
    version_id text NOT NULL UNIQUE,
    source jsonb NOT NULL,
    upload jsonb NOT NULL,
    profile jsonb NOT NULL,
    state text NOT NULL CHECK(state IN ('queued','storing','parsing','persisting','succeeded','failed')),
    chunks_written integer NOT NULL DEFAULT 0 CHECK(chunks_written >= 0),
    error_code text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(tenant_id, application_id, knowledge_base_id, request_id),
    FOREIGN KEY(tenant_id, application_id, knowledge_base_id, document_id)
      REFERENCES harness_rag.documents,
    CHECK((state='failed') = (error_code IS NOT NULL))
);
CREATE TABLE harness_rag.document_versions (
    version_id text PRIMARY KEY REFERENCES harness_rag.ingestion_jobs(version_id),
    job_id text NOT NULL UNIQUE REFERENCES harness_rag.ingestion_jobs(job_id),
    source jsonb NOT NULL,
    parser_profile jsonb NOT NULL,
    content_hash text NOT NULL,
    chunk_count integer NOT NULL CHECK(chunk_count BETWEEN 1 AND 1000),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE harness_rag.chunks (
    chunk_id text PRIMARY KEY,
    version_id text NOT NULL REFERENCES harness_rag.document_versions,
    ordinal integer NOT NULL CHECK(ordinal BETWEEN 0 AND 999),
    start_offset integer NOT NULL CHECK(start_offset >= 0),
    end_offset integer NOT NULL CHECK(end_offset > start_offset),
    body text NOT NULL CHECK(length(body) BETWEEN 1 AND 8000),
    body_hash text NOT NULL,
    UNIQUE(version_id, ordinal),
    CHECK(length(body) = end_offset - start_offset)
);
CREATE FUNCTION harness_rag.reject_content_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'immutable ingestion content';
END;
$$;
CREATE TRIGGER immutable_document_version BEFORE UPDATE ON harness_rag.document_versions
FOR EACH ROW EXECUTE FUNCTION harness_rag.reject_content_update();
CREATE TRIGGER immutable_chunk BEFORE UPDATE ON harness_rag.chunks
FOR EACH ROW EXECUTE FUNCTION harness_rag.reject_content_update();
INSERT INTO harness_rag.schema_versions(version) VALUES(2);
