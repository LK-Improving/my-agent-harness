-- 只登记删除意图及撤权；完整正文副本传播/清理器留待M4-06。
CREATE TABLE harness.artifact_deletion_requests (
    session_id text NOT NULL REFERENCES harness.sessions(session_id),
    request_id text NOT NULL,
    artifact_ref text NOT NULL REFERENCES harness.artifacts(artifact_ref),
    request_hash text NOT NULL CHECK (request_hash ~ '^[a-f0-9]{64}$'),
    expected_content_hash text NOT NULL CHECK (expected_content_hash ~ '^[a-f0-9]{64}$'),
    actor jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(session_id,request_id)
);
CREATE INDEX artifact_deletion_ref ON harness.artifact_deletion_requests(artifact_ref);
CREATE INDEX read_operations_artifact_ref ON harness.read_operations((arguments->>'artifact_ref'))
    WHERE tool_name='read_artifact';
CREATE FUNCTION harness.protect_artifact_deletion() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'artifact deletion intent is immutable';
END;
$$;
CREATE TRIGGER artifact_deletion_immutable BEFORE UPDATE OR DELETE ON harness.artifact_deletion_requests
    FOR EACH ROW EXECUTE FUNCTION harness.protect_artifact_deletion();
CREATE FUNCTION harness.require_artifact_cleanup() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'artifact cleanup executor is not installed';
END;
$$;
CREATE TRIGGER artifacts_cleanup_required BEFORE DELETE ON harness.artifacts
    FOR EACH ROW EXECUTE FUNCTION harness.require_artifact_cleanup();
INSERT INTO harness.schema_versions(version) VALUES(6);
