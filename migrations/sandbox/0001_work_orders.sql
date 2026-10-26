-- 独立运营沙箱库；不得安装到Harness业务库。
CREATE SCHEMA IF NOT EXISTS work_order_sandbox;
CREATE TABLE work_order_sandbox.requests (
    operation_id text PRIMARY KEY, tenant_id text NOT NULL, application_id text NOT NULL,
    args_hash text NOT NULL, state text NOT NULL CHECK(state IN ('succeeded','not_executed')),
    result jsonb, visible_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE work_order_sandbox.work_orders (
    work_order_id text PRIMARY KEY, operation_id text NOT NULL UNIQUE,
    tenant_id text NOT NULL, application_id text NOT NULL, order_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(tenant_id,application_id,order_id)
);
