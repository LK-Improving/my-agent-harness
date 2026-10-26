-- 开发执行装配：持久模型/只读工具尝试、预算及本机终止证明。
ALTER TABLE harness.runs ADD COLUMN model_steps integer NOT NULL DEFAULT 0;
ALTER TABLE harness.runs ADD COLUMN model_reserved_tokens bigint NOT NULL DEFAULT 0;
ALTER TABLE harness.worker_registration ADD COLUMN process jsonb;
ALTER TABLE harness.tool_attempts ADD COLUMN requires_reconciliation boolean NOT NULL DEFAULT false;
CREATE TABLE harness.model_calls (
    model_call_id text PRIMARY KEY, run_id text NOT NULL REFERENCES harness.runs,
    input_hash text NOT NULL, request_fingerprint text NOT NULL, request jsonb NOT NULL,
    state text NOT NULL CHECK(state IN ('dispatched','succeeded','failed','unknown')),
    result jsonb, failure jsonb, retry_intent text NOT NULL DEFAULT 'never',
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(run_id,input_hash)
);
CREATE TABLE harness.read_operations (
    operation_id text PRIMARY KEY, run_id text NOT NULL REFERENCES harness.runs,
    logical_call_id text NOT NULL, tool_name text NOT NULL, spec_hash text NOT NULL,
    args_hash text NOT NULL, arguments jsonb NOT NULL, attempt_id text NOT NULL,
    state text NOT NULL CHECK(state IN ('dispatched','succeeded','failed','unknown')),
    receipt jsonb, UNIQUE(run_id,logical_call_id)
);
CREATE TABLE harness.terminated_workers (
    identity text PRIMARY KEY, process jsonb NOT NULL, proof jsonb NOT NULL,
    actor jsonb NOT NULL, verified_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE harness.answer_drafts (
    run_id text PRIMARY KEY REFERENCES harness.runs, draft_id text NOT NULL UNIQUE,
    model_attempt_id text NOT NULL, candidate_hash text NOT NULL, candidate jsonb NOT NULL,
    published_offset integer NOT NULL DEFAULT 0,
    state text NOT NULL CHECK(state IN ('published','committed','invalid'))
);
CREATE TABLE harness.read_windows (
    tenant_id text NOT NULL, subject_id text NOT NULL, application_id text NOT NULL,
    window_start timestamptz NOT NULL, used integer NOT NULL,
    PRIMARY KEY(tenant_id,subject_id,application_id)
);
CREATE TABLE harness.sse_connections (
    token text PRIMARY KEY, tenant_id text NOT NULL, subject_id text NOT NULL,
    application_id text NOT NULL, expires_at timestamptz NOT NULL
);
INSERT INTO harness.schema_versions(version) VALUES(3);
