-- 可选生命周期任务；不级联删除运行、审批、回执、模型计量或官方checkpoint。
CREATE FUNCTION harness_rag.protect_document_tombstone() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(OLD) - 'revoked_at') IS DISTINCT FROM (to_jsonb(NEW) - 'revoked_at')
       OR (OLD.revoked_at IS NOT NULL AND NEW.revoked_at IS DISTINCT FROM OLD.revoked_at) THEN
        RAISE EXCEPTION 'immutable document revocation';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER immutable_document_revocation BEFORE UPDATE ON harness_rag.documents
FOR EACH ROW EXECUTE FUNCTION harness_rag.protect_document_tombstone();
-- 保留版本/切片身份和hash，只有已撤权文档允许单向清空正文。
ALTER TABLE harness_rag.chunks ALTER COLUMN body DROP NOT NULL;
ALTER TABLE harness_rag.chunks ADD COLUMN purged_at timestamptz;
ALTER TABLE harness_rag.chunks ADD CONSTRAINT chunk_redaction_state
CHECK((body IS NULL) = (purged_at IS NOT NULL));
CREATE FUNCTION harness_rag.protect_chunk_redaction() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.body IS NOT NULL AND NEW.body IS NULL AND NEW.purged_at IS NOT NULL
       AND (to_jsonb(OLD) - ARRAY['body','purged_at']) = (to_jsonb(NEW) - ARRAY['body','purged_at'])
       AND EXISTS(SELECT 1 FROM harness_rag.ingestion_jobs j JOIN harness_rag.documents d
           USING(tenant_id,application_id,knowledge_base_id,document_id)
           WHERE j.version_id=OLD.version_id AND d.revoked_at IS NOT NULL) THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'immutable chunk except revoked body redaction';
END;
$$;
DROP TRIGGER immutable_chunk ON harness_rag.chunks;
CREATE TRIGGER immutable_chunk BEFORE UPDATE ON harness_rag.chunks
FOR EACH ROW EXECUTE FUNCTION harness_rag.protect_chunk_redaction();
CREATE TABLE harness_rag.cleanup_jobs (
    job_id text PRIMARY KEY,
    tenant_id text NOT NULL, application_id text NOT NULL, knowledge_base_id text NOT NULL,
    document_id text NOT NULL, subject_id text NOT NULL, request_id text NOT NULL,
    fingerprint text NOT NULL, retain_until timestamptz NOT NULL,
    state text NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','blocked','manual','succeeded')),
    reason text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(tenant_id,application_id,knowledge_base_id,request_id),
    UNIQUE(tenant_id,application_id,knowledge_base_id,document_id),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id,document_id) REFERENCES harness_rag.documents
);
CREATE TABLE harness_rag.cleanup_steps (
    step_id text PRIMARY KEY, job_id text NOT NULL REFERENCES harness_rag.cleanup_jobs,
    position integer NOT NULL, kind text NOT NULL CHECK(kind IN ('index','source','postgres')),
    target jsonb NOT NULL,
    state text NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','running','failed','succeeded','retained')),
    attempts integer NOT NULL DEFAULT 0 CHECK(attempts BETWEEN 0 AND 3),
    error_code text,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(job_id,position)
);
-- 明确保留原因由可信管理面登记；普通TTL不得删除活动证据。
CREATE TABLE harness_rag.retention_holds (
    tenant_id text NOT NULL, application_id text NOT NULL, knowledge_base_id text NOT NULL,
    document_id text NOT NULL, hold_id text NOT NULL, reason_code text NOT NULL,
    until_at timestamptz NOT NULL,
    PRIMARY KEY(tenant_id,application_id,knowledge_base_id,document_id,hold_id),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id,document_id) REFERENCES harness_rag.documents
);
CREATE TABLE harness_rag.source_releases (
    tenant_id text NOT NULL, application_id text NOT NULL, knowledge_base_id text NOT NULL,
    document_id text NOT NULL, object_id text NOT NULL,
    released_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(tenant_id,application_id,knowledge_base_id,document_id,object_id),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id,document_id) REFERENCES harness_rag.documents
);
CREATE FUNCTION harness_rag.protect_cleanup_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(OLD) - ARRAY['state','reason','updated_at']) IS DISTINCT FROM
       (to_jsonb(NEW) - ARRAY['state','reason','updated_at']) THEN
        RAISE EXCEPTION 'immutable cleanup identity';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER immutable_cleanup_job BEFORE UPDATE ON harness_rag.cleanup_jobs
FOR EACH ROW EXECUTE FUNCTION harness_rag.protect_cleanup_identity();
CREATE FUNCTION harness_rag.protect_cleanup_step() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(OLD) - ARRAY['state','attempts','error_code','updated_at']) IS DISTINCT FROM
       (to_jsonb(NEW) - ARRAY['state','attempts','error_code','updated_at']) THEN
        RAISE EXCEPTION 'immutable cleanup target';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER immutable_cleanup_target BEFORE UPDATE ON harness_rag.cleanup_steps
FOR EACH ROW EXECUTE FUNCTION harness_rag.protect_cleanup_step();
INSERT INTO harness_rag.schema_versions(version) VALUES(5);
