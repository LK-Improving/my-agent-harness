-- 运行检索扩展；需先安装core业务迁移及rag0001～0003，不自动执行。
CREATE TABLE harness_rag.knowledge_read_grants (
    tenant_id text NOT NULL, application_id text NOT NULL, knowledge_base_id text NOT NULL,
    subject_id text NOT NULL,
    PRIMARY KEY(tenant_id,application_id,knowledge_base_id,subject_id),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id) REFERENCES harness_rag.knowledge_bases
);
CREATE TABLE harness_rag.document_access (
    tenant_id text NOT NULL, application_id text NOT NULL, knowledge_base_id text NOT NULL,
    document_id text NOT NULL, restricted boolean NOT NULL DEFAULT true,
    PRIMARY KEY(tenant_id,application_id,knowledge_base_id,document_id),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id,document_id) REFERENCES harness_rag.documents
);
CREATE TABLE harness_rag.document_read_grants (
    tenant_id text NOT NULL, application_id text NOT NULL, knowledge_base_id text NOT NULL,
    document_id text NOT NULL, subject_id text NOT NULL,
    PRIMARY KEY(tenant_id,application_id,knowledge_base_id,document_id,subject_id),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id,document_id) REFERENCES harness_rag.documents
);
CREATE TABLE harness_rag.retrieval_requests (
    retrieval_id text PRIMARY KEY,
    run_id text NOT NULL REFERENCES harness.runs,
    knowledge_base_id text NOT NULL,
    tenant_id text NOT NULL, application_id text NOT NULL, subject_id text NOT NULL,
    request_id text NOT NULL, fingerprint text NOT NULL,
    ticket jsonb NOT NULL,
    parent_reserved_tokens bigint NOT NULL CHECK(parent_reserved_tokens > 0),
    state text NOT NULL CHECK(state IN ('queued','running','succeeded','failed')),
    result jsonb, result_hash text, error_code text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(run_id,knowledge_base_id),
    CHECK((state='succeeded') = (result IS NOT NULL)),
    CHECK((state='succeeded') = (result_hash IS NOT NULL)),
    CHECK((state='failed') = (error_code IS NOT NULL)),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id) REFERENCES harness_rag.knowledge_bases
);
CREATE TABLE harness_rag.retrieval_stages (
    retrieval_id text NOT NULL REFERENCES harness_rag.retrieval_requests,
    stage text NOT NULL CHECK(stage IN ('created','claimed','recall','evidence','completed','failed')),
    details jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(retrieval_id,stage)
);
CREATE FUNCTION harness_rag.protect_retrieval_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(OLD) - ARRAY['state','result','result_hash','error_code','updated_at']) IS DISTINCT FROM
       (to_jsonb(NEW) - ARRAY['state','result','result_hash','error_code','updated_at']) THEN
        RAISE EXCEPTION 'immutable retrieval identity';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER immutable_retrieval_identity BEFORE UPDATE ON harness_rag.retrieval_requests
FOR EACH ROW EXECUTE FUNCTION harness_rag.protect_retrieval_identity();
INSERT INTO harness_rag.schema_versions(version) VALUES(4);
