-- M5-02：累计评测准入与单向撤销；只追加，不修改旧迁移或官方checkpoint。
ALTER TABLE harness.evaluation_contexts
    ADD COLUMN session_count bigint NOT NULL DEFAULT 0 CHECK (session_count >= 0),
    ADD COLUMN run_count bigint NOT NULL DEFAULT 0 CHECK (run_count >= 0);

UPDATE harness.evaluation_contexts e SET
    session_count=(SELECT count(*) FROM harness.sessions s WHERE s.evaluation_context_id=e.context_id),
    run_count=(SELECT count(*) FROM harness.runs r JOIN harness.sessions s USING(session_id)
               WHERE s.evaluation_context_id=e.context_id);

CREATE TABLE harness.evaluation_revocations (
    context_id text NOT NULL REFERENCES harness.evaluation_contexts(context_id),
    request_id text NOT NULL,
    request_hash text NOT NULL CHECK (request_hash ~ '^[a-f0-9]{64}$'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(context_id,request_id)
);
CREATE INDEX evaluation_contexts_owner
    ON harness.evaluation_contexts(tenant_id,application_id,subject_id,expires_at);
INSERT INTO harness.schema_versions(version) VALUES (10);
