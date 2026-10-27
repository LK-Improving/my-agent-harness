-- 可选正式双路索引；不修改既有正文/切片与官方saver。
ALTER TABLE harness_rag.chunks ADD CONSTRAINT chunk_version_hash_identity
UNIQUE(version_id,chunk_id,body_hash);
CREATE TABLE harness_rag.index_jobs (
    job_id text PRIMARY KEY,
    generation_id text NOT NULL UNIQUE,
    tenant_id text NOT NULL,
    application_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    subject_id text NOT NULL,
    request_id text NOT NULL,
    fingerprint text NOT NULL,
    version_ids jsonb NOT NULL,
    binding jsonb NOT NULL,
    scope jsonb NOT NULL,
    members_hash text NOT NULL,
    chunk_count integer NOT NULL CHECK(chunk_count BETWEEN 1 AND 1000),
    state text NOT NULL CHECK(state IN ('queued','building','ready','failed','offline')),
    batches_written integer NOT NULL DEFAULT 0 CHECK(batches_written >= 0),
    batches_expected integer NOT NULL DEFAULT 0 CHECK(batches_expected BETWEEN 0 AND 128),
    error_code text,
    readiness jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(tenant_id,application_id,knowledge_base_id,request_id),
    UNIQUE(tenant_id,application_id,knowledge_base_id,generation_id),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id) REFERENCES harness_rag.knowledge_bases,
    CHECK((state='failed') = (error_code IS NOT NULL)),
    CHECK(state NOT IN ('ready','offline') OR readiness IS NOT NULL)
);
CREATE TABLE harness_rag.generation_members (
    generation_id text NOT NULL REFERENCES harness_rag.index_jobs(generation_id),
    chunk_id text NOT NULL REFERENCES harness_rag.chunks,
    version_id text NOT NULL REFERENCES harness_rag.document_versions,
    body_hash text NOT NULL,
    PRIMARY KEY(generation_id,chunk_id),
    FOREIGN KEY(version_id,chunk_id,body_hash) REFERENCES harness_rag.chunks(version_id,chunk_id,body_hash)
);
CREATE TABLE harness_rag.knowledge_snapshots (
    snapshot_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    application_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    generation_id text NOT NULL REFERENCES harness_rag.index_jobs(generation_id),
    content_hash text NOT NULL,
    state text NOT NULL CHECK(state IN ('published','offline')),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(generation_id),
    UNIQUE(tenant_id,application_id,knowledge_base_id,snapshot_id),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id,generation_id)
      REFERENCES harness_rag.index_jobs(tenant_id,application_id,knowledge_base_id,generation_id),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id) REFERENCES harness_rag.knowledge_bases
);
CREATE TABLE harness_rag.snapshot_members (
    snapshot_id text NOT NULL REFERENCES harness_rag.knowledge_snapshots,
    chunk_id text NOT NULL REFERENCES harness_rag.chunks,
    version_id text NOT NULL REFERENCES harness_rag.document_versions,
    body_hash text NOT NULL,
    PRIMARY KEY(snapshot_id,chunk_id),
    FOREIGN KEY(version_id,chunk_id,body_hash) REFERENCES harness_rag.chunks(version_id,chunk_id,body_hash)
);
CREATE TABLE harness_rag.active_snapshots (
    tenant_id text NOT NULL,
    application_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    snapshot_id text REFERENCES harness_rag.knowledge_snapshots,
    revision bigint NOT NULL DEFAULT 0 CHECK(revision >= 0),
    PRIMARY KEY(tenant_id,application_id,knowledge_base_id),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id,snapshot_id)
      REFERENCES harness_rag.knowledge_snapshots(tenant_id,application_id,knowledge_base_id,snapshot_id),
    FOREIGN KEY(tenant_id,application_id,knowledge_base_id) REFERENCES harness_rag.knowledge_bases
);
CREATE TABLE harness_rag.index_audit (
    audit_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id text NOT NULL,
    application_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    subject_id text NOT NULL,
    action text NOT NULL,
    resource_id text NOT NULL,
    details jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TRIGGER immutable_generation_member BEFORE UPDATE ON harness_rag.generation_members
FOR EACH ROW EXECUTE FUNCTION harness_rag.reject_content_update();
CREATE TRIGGER immutable_snapshot_member BEFORE UPDATE ON harness_rag.snapshot_members
FOR EACH ROW EXECUTE FUNCTION harness_rag.reject_content_update();
CREATE FUNCTION harness_rag.protect_index_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(OLD) - ARRAY['state','batches_written','batches_expected','error_code','readiness','updated_at'])
       IS DISTINCT FROM
       (to_jsonb(NEW) - ARRAY['state','batches_written','batches_expected','error_code','readiness','updated_at']) THEN
        RAISE EXCEPTION 'immutable index identity';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER immutable_index_identity BEFORE UPDATE ON harness_rag.index_jobs
FOR EACH ROW EXECUTE FUNCTION harness_rag.protect_index_identity();
CREATE FUNCTION harness_rag.protect_snapshot_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(OLD) - 'state') IS DISTINCT FROM (to_jsonb(NEW) - 'state') THEN
        RAISE EXCEPTION 'immutable snapshot identity';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER immutable_snapshot_identity BEFORE UPDATE ON harness_rag.knowledge_snapshots
FOR EACH ROW EXECUTE FUNCTION harness_rag.protect_snapshot_identity();
INSERT INTO harness_rag.schema_versions(version) VALUES(3);
