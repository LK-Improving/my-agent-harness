-- 候选为运行审计；会话摘要和游标只随成功业务事务发布。
ALTER TABLE harness.sessions ADD COLUMN summary_cursor_hash text;
CREATE TABLE harness.summary_candidates (
    run_id text PRIMARY KEY REFERENCES harness.runs(run_id),
    session_id text NOT NULL REFERENCES harness.sessions(session_id),
    source_revision bigint NOT NULL CHECK (source_revision>=0),
    expected_cursor_hash text,
    candidate jsonb NOT NULL,
    candidate_hash text NOT NULL CHECK (candidate_hash ~ '^[a-f0-9]{64}$'),
    state text NOT NULL DEFAULT 'staged' CHECK (state IN ('staged','committed','invalid')),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    expires_at timestamptz NOT NULL,
    invalidation_code text,
    commit_id text REFERENCES harness.commits(commit_id),
    CHECK ((state='staged' AND commit_id IS NULL AND invalidation_code IS NULL)
        OR (state='committed' AND commit_id IS NOT NULL AND invalidation_code IS NULL)
        OR (state='invalid' AND commit_id IS NULL AND invalidation_code IS NOT NULL))
);
CREATE TABLE harness.session_summaries (
    session_id text PRIMARY KEY REFERENCES harness.sessions(session_id),
    candidate jsonb NOT NULL,
    covered_revision bigint NOT NULL CHECK (covered_revision>=1),
    committed_revision bigint NOT NULL,
    cursor_hash text NOT NULL CHECK (cursor_hash ~ '^[a-f0-9]{64}$'),
    commit_run_id text NOT NULL REFERENCES harness.runs(run_id),
    commit_id text NOT NULL REFERENCES harness.commits(commit_id),
    CHECK (committed_revision>covered_revision)
);
CREATE FUNCTION harness.protect_summary_candidate() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(NEW)-ARRAY['state','invalidation_code','commit_id']) IS DISTINCT FROM
       (to_jsonb(OLD)-ARRAY['state','invalidation_code','commit_id'])
       OR (OLD.state<>'staged' AND NEW IS DISTINCT FROM OLD)
       OR (OLD.state='staged' AND NEW.state NOT IN ('staged','committed','invalid')) THEN
        RAISE EXCEPTION 'summary candidate is immutable';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER summary_candidate_immutable BEFORE UPDATE ON harness.summary_candidates
    FOR EACH ROW EXECUTE FUNCTION harness.protect_summary_candidate();
INSERT INTO harness.schema_versions(version) VALUES(8);
