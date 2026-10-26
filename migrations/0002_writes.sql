-- M2 追加业务表；不改写历史0001或官方checkpoint内部表。
ALTER TABLE harness.runs DROP CONSTRAINT runs_status_check;
ALTER TABLE harness.runs ADD CONSTRAINT runs_status_check CHECK (
    status IN ('queued','running','waiting_approval','reconciling','succeeded','partial','failed','cancelled','expired')
);
ALTER TABLE harness.runs ADD COLUMN tool_dispatches integer NOT NULL DEFAULT 0;
ALTER TABLE harness.runs ADD COLUMN suspension jsonb;
CREATE TABLE harness.operations (
    operation_id text PRIMARY KEY, run_id text NOT NULL REFERENCES harness.runs,
    logical_call_id text NOT NULL, principal jsonb NOT NULL, tool jsonb NOT NULL,
    tool_hash text NOT NULL, arguments jsonb NOT NULL, args_hash text NOT NULL,
    action_key text NOT NULL, plan_hash text NOT NULL, binding_hash text NOT NULL,
    approval_id text NOT NULL UNIQUE,
    state text NOT NULL CHECK (state IN ('prepared','dispatched','unknown','succeeded','not_executed','rejected')),
    attempts integer NOT NULL DEFAULT 0, first_dispatch_at timestamptz,
    receipt jsonb, output_error text, reconcile_count integer NOT NULL DEFAULT 0,
    reconcile_token text, UNIQUE(run_id,logical_call_id)
);
CREATE INDEX operations_run_state ON harness.operations(run_id,state);
CREATE TABLE harness.approvals (
    approval_id text PRIMARY KEY, operation_id text NOT NULL UNIQUE REFERENCES harness.operations,
    binding_hash text NOT NULL, revision bigint NOT NULL DEFAULT 0,
    state text NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','approved','rejected')),
    expires_at timestamptz NOT NULL, decided_by text,
    decision_request_id text, decision_hash text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE harness.business_action_guards (
    tenant_id text NOT NULL, application_id text NOT NULL, action_key text NOT NULL,
    operation_id text NOT NULL REFERENCES harness.operations,
    PRIMARY KEY(tenant_id,application_id,action_key)
);
CREATE TABLE harness.tool_attempts (
    attempt_id text PRIMARY KEY, operation_id text NOT NULL REFERENCES harness.operations,
    ordinal integer NOT NULL, state text NOT NULL CHECK(state IN ('dispatched','unknown','succeeded','not_executed')),
    dispatched_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz, UNIQUE(operation_id,ordinal)
);
CREATE TABLE harness.reconciliation_records (
    token text PRIMARY KEY, operation_id text NOT NULL REFERENCES harness.operations,
    run_id text NOT NULL REFERENCES harness.runs,
    request_id text NOT NULL, actor text NOT NULL,
    result jsonb, created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(run_id,request_id)
);
INSERT INTO harness.schema_versions(version) VALUES(2);
