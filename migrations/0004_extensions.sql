-- 关键扩展决定为私有运行审计；同输入重放只能复用相同决定。
CREATE TABLE harness.extension_decisions (
    run_id text NOT NULL REFERENCES harness.runs(run_id),
    extension_id text NOT NULL,
    extension_version text NOT NULL,
    invocation_fingerprint text NOT NULL,
    stage text NOT NULL,
    spec jsonb NOT NULL,
    invocation jsonb NOT NULL,
    decision jsonb NOT NULL,
    decision_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(run_id, extension_id, extension_version, invocation_fingerprint)
);
INSERT INTO harness.schema_versions(version) VALUES(4);
