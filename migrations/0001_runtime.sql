-- M1 业务表。显式运行迁移；不修改官方 checkpoint 表。
CREATE SCHEMA IF NOT EXISTS harness;
CREATE TABLE harness.schema_versions (
    version integer PRIMARY KEY, installed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE harness.tenant_admission (
    tenant_id text PRIMARY KEY
);
CREATE TABLE harness.request_windows (
    tenant_id text NOT NULL, subject_id text NOT NULL, application_id text NOT NULL,
    window_start timestamptz NOT NULL, used integer NOT NULL CHECK (used > 0),
    PRIMARY KEY (tenant_id, subject_id, application_id)
);
CREATE TABLE harness.releases (
    tenant_id text NOT NULL, application_id text NOT NULL, release_id text NOT NULL,
    agent_id text NOT NULL, manifest_hash text NOT NULL, manifest jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, application_id, release_id)
);
CREATE TABLE harness.active_releases (
    tenant_id text NOT NULL, application_id text NOT NULL, agent_id text NOT NULL,
    release_id text NOT NULL,
    PRIMARY KEY (tenant_id, application_id, agent_id),
    FOREIGN KEY (tenant_id, application_id, release_id)
        REFERENCES harness.releases (tenant_id, application_id, release_id)
);
CREATE TABLE harness.evaluation_contexts (
    context_id text PRIMARY KEY, tenant_id text NOT NULL, subject_id text NOT NULL,
    application_id text NOT NULL, release_id text NOT NULL, context jsonb NOT NULL,
    expires_at timestamptz NOT NULL, revoked_at timestamptz,
    FOREIGN KEY (tenant_id, application_id, release_id)
        REFERENCES harness.releases (tenant_id, application_id, release_id)
);
CREATE TABLE harness.sessions (
    session_id text PRIMARY KEY, tenant_id text NOT NULL, subject_id text NOT NULL,
    application_id text NOT NULL, agent_id text NOT NULL, release_id text NOT NULL,
    evaluation_context_id text REFERENCES harness.evaluation_contexts (context_id),
    revision bigint NOT NULL DEFAULT 0 CHECK (revision >= 0), active_run_id text,
    FOREIGN KEY (tenant_id, application_id, release_id)
        REFERENCES harness.releases (tenant_id, application_id, release_id)
);
CREATE TABLE harness.runs (
    run_id text PRIMARY KEY, session_id text NOT NULL REFERENCES harness.sessions,
    request_id text NOT NULL, request_hash text NOT NULL, request jsonb NOT NULL,
    manifest_hash text NOT NULL, environment_profile_hash text NOT NULL,
    principal jsonb NOT NULL, status text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued','running','succeeded','partial','failed','cancelled','expired')),
    revision bigint NOT NULL DEFAULT 0, event_seq bigint NOT NULL DEFAULT 0,
    event_floor bigint NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    queue_deadline timestamptz NOT NULL, expires_at timestamptz NOT NULL,
    execution_deadline timestamptz, execution_started_at timestamptz,
    execution_used_ms bigint NOT NULL DEFAULT 0,
    owner_id text, cancel_requested boolean NOT NULL DEFAULT false,
    recovery_required boolean NOT NULL DEFAULT false, result jsonb, error_code text,
    UNIQUE (session_id, request_id)
);
CREATE INDEX runs_queue ON harness.runs (created_at,run_id) WHERE status = 'queued';
ALTER TABLE harness.sessions ADD CONSTRAINT sessions_active_run_fk
    FOREIGN KEY (active_run_id) REFERENCES harness.runs (run_id);
CREATE TABLE harness.events (
    run_id text NOT NULL REFERENCES harness.runs, seq bigint NOT NULL,
    type text NOT NULL, payload jsonb NOT NULL,
    timestamp timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (run_id,seq)
);
CREATE UNIQUE INDEX events_terminal ON harness.events (run_id)
    WHERE type IN ('run.completed','run.failed','run.cancelled','run.expired');
CREATE TABLE harness.commits (
    commit_id text PRIMARY KEY, run_id text NOT NULL UNIQUE REFERENCES harness.runs,
    candidate_hash text NOT NULL, validation jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE harness.turns (
    session_id text NOT NULL REFERENCES harness.sessions, revision bigint NOT NULL,
    run_id text NOT NULL UNIQUE REFERENCES harness.runs, messages jsonb NOT NULL,
    PRIMARY KEY (session_id,revision)
);
CREATE TABLE harness.cancel_requests (
    run_id text NOT NULL REFERENCES harness.runs, request_id text NOT NULL,
    PRIMARY KEY (run_id,request_id)
);
CREATE TABLE harness.worker_registration (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    identity text NOT NULL, heartbeat_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    active_run_id text REFERENCES harness.runs,
    state text NOT NULL CHECK (state IN ('idle','running','draining'))
);
CREATE TABLE harness.audit (
    audit_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id text, subject_id text, action text NOT NULL, resource_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
INSERT INTO harness.schema_versions(version) VALUES (1);
