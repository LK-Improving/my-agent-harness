-- 显式保留管理；不改官方saver，不级联审批、回执、guard或知识表。
ALTER TABLE harness.sessions ADD COLUMN created_at timestamptz NOT NULL DEFAULT clock_timestamp();
ALTER TABLE harness.sessions ADD COLUMN revoked_at timestamptz;
ALTER TABLE harness.sessions ADD COLUMN purged_at timestamptz;
ALTER TABLE harness.sessions ADD COLUMN retain_until timestamptz;
ALTER TABLE harness.sessions ADD COLUMN legal_hold boolean NOT NULL DEFAULT false;
ALTER TABLE harness.sessions ADD COLUMN retention_policy jsonb NOT NULL DEFAULT
    '{"version":"core-retention@1","session_seconds":2592000,"event_seconds":604800,"trace_seconds":604800,"private_seconds":7776000,"max_runs":256,"max_jobs":256,"max_attempts":3,"recheck_seconds":300,"external_trace_copies":false}';
CREATE TABLE harness.cleanup_jobs (
    job_id text PRIMARY KEY,
    session_id text NOT NULL REFERENCES harness.sessions,
    request_id text NOT NULL, request_hash text NOT NULL CHECK(request_hash ~ '^[a-f0-9]{64}$'),
    request jsonb NOT NULL, policy jsonb NOT NULL,
    kind text NOT NULL CHECK(kind IN ('session','events','artifact','trace','extensions')),
    phase text NOT NULL DEFAULT 'copies' CHECK(phase IN ('copies','payloads','done')),
    state text NOT NULL DEFAULT 'pending'
        CHECK(state IN ('pending','blocked','running','retry','manual','completed')),
    attempts integer NOT NULL DEFAULT 0 CHECK(attempts BETWEEN 0 AND 3),
    reasons jsonb NOT NULL DEFAULT '[]', target jsonb,
    token text, next_check_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(session_id,request_id)
);
CREATE INDEX cleanup_session ON harness.cleanup_jobs(session_id,state);
ALTER TABLE harness.artifacts ADD COLUMN purged_at timestamptz;
ALTER TABLE harness.artifacts ALTER COLUMN body DROP NOT NULL;
ALTER TABLE harness.artifacts ADD CONSTRAINT artifact_body_lifecycle CHECK
    ((body IS NOT NULL AND purged_at IS NULL) OR
     (body IS NULL AND purged_at IS NOT NULL AND revoked_at IS NOT NULL));
-- 原身份/hash/大小保留。清空只在已锁定会话、运行及running任务的业务事务中允许。
CREATE OR REPLACE FUNCTION harness.protect_artifact_content() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(NEW)-ARRAY['revoked_at','purged_at','body']) IS DISTINCT FROM
       (to_jsonb(OLD)-ARRAY['revoked_at','purged_at','body'])
       OR (OLD.revoked_at IS NOT NULL AND NEW.revoked_at IS DISTINCT FROM OLD.revoked_at)
       OR (OLD.purged_at IS NOT NULL AND NEW IS DISTINCT FROM OLD) THEN
        RAISE EXCEPTION 'artifact content is immutable';
    END IF;
    IF NEW.body IS DISTINCT FROM OLD.body OR NEW.purged_at IS DISTINCT FROM OLD.purged_at THEN
        IF NEW.body IS NOT NULL OR NEW.purged_at IS NULL OR NOT EXISTS (
            SELECT 1 FROM harness.cleanup_jobs j WHERE j.job_id=current_setting('harness.cleanup_job',true)
              AND j.session_id=OLD.session_id AND j.state='running' AND j.phase='payloads'
              AND (j.kind='session' OR (j.kind='artifact' AND j.request->>'artifact_ref'=OLD.artifact_ref))
        ) THEN
            RAISE EXCEPTION 'authorized cleanup required';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
CREATE FUNCTION harness.protect_cleanup_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(NEW)-ARRAY['phase','state','attempts','reasons','target','token','next_check_at','updated_at'])
        IS DISTINCT FROM
       (to_jsonb(OLD)-ARRAY['phase','state','attempts','reasons','target','token','next_check_at','updated_at'])
       OR NEW.attempts<OLD.attempts OR (OLD.state='completed' AND NEW IS DISTINCT FROM OLD)
       OR (OLD.target IS NOT NULL AND NEW.target IS DISTINCT FROM OLD.target) THEN
        RAISE EXCEPTION 'immutable cleanup identity';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER cleanup_identity BEFORE UPDATE ON harness.cleanup_jobs
FOR EACH ROW EXECUTE FUNCTION harness.protect_cleanup_identity();
CREATE FUNCTION harness.protect_session_revocation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (OLD.revoked_at IS NOT NULL AND NEW.revoked_at IS DISTINCT FROM OLD.revoked_at)
       OR (OLD.purged_at IS NOT NULL AND NEW.purged_at IS DISTINCT FROM OLD.purged_at)
       OR NEW.created_at IS DISTINCT FROM OLD.created_at OR NEW.retention_policy IS DISTINCT FROM OLD.retention_policy THEN
        RAISE EXCEPTION 'session lifecycle cannot be reset';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER session_revocation BEFORE UPDATE ON harness.sessions
FOR EACH ROW EXECUTE FUNCTION harness.protect_session_revocation();
INSERT INTO harness.schema_versions(version) VALUES(9);
