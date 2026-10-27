-- 产物正文只追加；撤权字段允许单向更新，清理在M4-06另行实现。
CREATE TABLE harness.artifacts (
    artifact_ref text PRIMARY KEY,
    run_id text NOT NULL REFERENCES harness.runs(run_id),
    session_id text NOT NULL REFERENCES harness.sessions(session_id),
    operation_id text NOT NULL,
    receipt_hash text NOT NULL,
    media_type text NOT NULL CHECK (media_type IN ('text/plain','application/json')),
    body text NOT NULL,
    content_hash text NOT NULL CHECK (content_hash ~ '^[a-f0-9]{64}$'),
    total_chars integer NOT NULL CHECK (total_chars=char_length(body)),
    total_bytes integer NOT NULL CHECK (total_bytes=octet_length(body) AND total_bytes<=1048576),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    expires_at timestamptz NOT NULL,
    revoked_at timestamptz,
    UNIQUE(run_id,operation_id)
);
CREATE INDEX artifacts_session ON harness.artifacts(session_id);
CREATE FUNCTION harness.protect_artifact_content() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(NEW)-'revoked_at') IS DISTINCT FROM (to_jsonb(OLD)-'revoked_at')
       OR (OLD.revoked_at IS NOT NULL AND NEW.revoked_at IS DISTINCT FROM OLD.revoked_at) THEN
        RAISE EXCEPTION 'artifact content is immutable';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER artifacts_immutable BEFORE UPDATE ON harness.artifacts
    FOR EACH ROW EXECUTE FUNCTION harness.protect_artifact_content();
INSERT INTO harness.schema_versions(version) VALUES(5);
