-- 可选检索调用账本，与core schema4和官方checkpoint表独立。
CREATE SCHEMA IF NOT EXISTS harness_rag;
CREATE TABLE IF NOT EXISTS harness_rag.schema_versions (
    version integer PRIMARY KEY,
    installed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE harness_rag.model_contexts (
    context_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    application_id text NOT NULL,
    owner_kind text NOT NULL CHECK (owner_kind IN ('run','ingestion_job')),
    owner_id text NOT NULL,
    knowledge_base_id text NOT NULL,
    scope jsonb NOT NULL,
    limits jsonb NOT NULL,
    calls_used integer NOT NULL DEFAULT 0 CHECK(calls_used >= 0),
    input_bytes_used bigint NOT NULL DEFAULT 0 CHECK(input_bytes_used >= 0),
    deadline timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(tenant_id,application_id,owner_kind,owner_id,knowledge_base_id)
);
CREATE TABLE harness_rag.model_calls (
    context_id text NOT NULL REFERENCES harness_rag.model_contexts(context_id),
    call_id text NOT NULL,
    attempt_id text NOT NULL UNIQUE,
    request_fingerprint text NOT NULL,
    binding_hash text NOT NULL,
    task_type text NOT NULL CHECK(task_type IN ('embedding','rerank')),
    input_bytes bigint NOT NULL CHECK(input_bytes > 0),
    state text NOT NULL CHECK(state IN ('reserved','succeeded','failed')),
    result jsonb,
    failure jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz,
    PRIMARY KEY(context_id,call_id),
    CHECK((state='succeeded') = (result IS NOT NULL)),
    CHECK((state='failed') = (failure IS NOT NULL))
);
INSERT INTO harness_rag.schema_versions(version) VALUES(1);
