-- 不改变v1供应商请求/指纹；摘要身份在父run的同一调用账本中独立持久绑定。
ALTER TABLE harness.model_calls ADD COLUMN purpose text NOT NULL DEFAULT 'answer'
    CHECK (purpose IN ('answer','context_summary'));
ALTER TABLE harness.model_calls ADD COLUMN purpose_ref text;
ALTER TABLE harness.model_calls ADD COLUMN purpose_binding jsonb;
ALTER TABLE harness.model_calls ADD CONSTRAINT model_call_purpose_binding CHECK (
    (purpose='answer' AND purpose_ref IS NULL AND purpose_binding IS NULL)
    OR (purpose='context_summary' AND purpose_ref IS NOT NULL AND purpose_binding IS NOT NULL)
);
CREATE UNIQUE INDEX model_calls_summary_identity ON harness.model_calls(run_id,purpose_ref)
    WHERE purpose='context_summary';
INSERT INTO harness.schema_versions(version) VALUES(7);
