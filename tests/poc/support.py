"""仅供 M0 隔离探针使用的 PG 沙箱；不作为正式 runtime 发布。"""

import hashlib
import json
from contextlib import contextmanager
from typing import Any

import psycopg
from harness.contracts.hashing import canonical_hash
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row


@contextmanager
def saver(dsn: str):
    with PostgresSaver.from_conn_string(dsn) as checkpointer:
        checkpointer.setup()
        yield checkpointer


class UnknownOutcome(Exception):
    """外部写入结果尚不能确认，不允许当作普通工具错误重试。"""


class ProbeStore:
    """供应商写入与基座回执分表，模拟独立事务之间的失败窗口。"""

    def __init__(self, dsn: str):
        self.dsn = dsn
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sandbox_writes (
                    operation_id text PRIMARY KEY, args_hash text NOT NULL,
                    result jsonb NOT NULL, visible boolean NOT NULL DEFAULT true
                );
                CREATE TABLE IF NOT EXISTS probe_receipts (
                    operation_id text PRIMARY KEY, args_hash text NOT NULL,
                    status text NOT NULL, dispatches integer NOT NULL DEFAULT 0,
                    result jsonb
                );
                CREATE TABLE IF NOT EXISTS probe_commits (
                    run_id text PRIMARY KEY, result_hash text NOT NULL, result jsonb NOT NULL
                );
                CREATE TABLE IF NOT EXISTS probe_turns (run_id text PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS probe_events (
                    run_id text NOT NULL, kind text NOT NULL,
                    PRIMARY KEY(run_id, kind)
                );
                CREATE TABLE IF NOT EXISTS probe_artifacts (
                    ref text PRIMARY KEY, tenant_id text NOT NULL, body text NOT NULL,
                    content_hash text NOT NULL, revoked boolean NOT NULL DEFAULT false
                );
                CREATE TABLE IF NOT EXISTS probe_workers (
                    singleton boolean PRIMARY KEY CHECK(singleton), owner text NOT NULL
                )
            """)

    def connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def operation(self, operation_id: str):
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM probe_receipts WHERE operation_id=%s", (operation_id,)
            ).fetchone()

    def dispatch(self, operation_id: str, args: dict[str, Any], *, lose_response=False):
        args_hash = canonical_hash(args)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO probe_receipts(operation_id,args_hash,status) "
                "VALUES(%s,%s,'prepared') "
                "ON CONFLICT DO NOTHING",
                (operation_id, args_hash),
            )
            row = conn.execute(
                "SELECT * FROM probe_receipts WHERE operation_id=%s FOR UPDATE", (operation_id,)
            ).fetchone()
            if row["args_hash"] != args_hash:
                raise ValueError("同 operation 异参")
            if row["status"] == "succeeded":
                return row["result"]
            if row["status"] in {"unknown", "dispatched"}:
                raise UnknownOutcome(operation_id)
            conn.execute(
                "UPDATE probe_receipts SET status='dispatched',dispatches=dispatches+1 "
                "WHERE operation_id=%s",
                (operation_id,),
            )
        # 外部事务不能放在基座回执行锁里，写入与回执无共同事务。
        result = {"work_order_id": "wo-" + operation_id, "order_id": args["order_id"]}
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sandbox_writes(operation_id,args_hash,result,visible) "
                "VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (operation_id, args_hash, json.dumps(result), not lose_response),
            )
        if lose_response:
            with self.connect() as conn:
                conn.execute(
                    "UPDATE probe_receipts SET status='unknown' WHERE operation_id=%s",
                    (operation_id,),
                )
            raise UnknownOutcome(operation_id)
        self._record_success(operation_id, result)
        return result

    def _record_success(self, operation_id, result):
        with self.connect() as conn:
            conn.execute(
                "UPDATE probe_receipts SET status='succeeded',result=%s WHERE operation_id=%s",
                (json.dumps(result), operation_id),
            )

    def reconcile(self, operation_id: str):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sandbox_writes WHERE operation_id=%s AND visible", (operation_id,)
            ).fetchone()
        if not row:
            return {"conclusion": "still_unknown", "evidence": None}
        self._record_success(operation_id, row["result"])
        return {"conclusion": "confirmed_succeeded", "evidence": row["result"]}

    def make_visible(self, operation_id: str):
        with self.connect() as conn:
            conn.execute(
                "UPDATE sandbox_writes SET visible=true WHERE operation_id=%s", (operation_id,)
            )

    def commit(self, run_id: str, result: dict[str, Any]):
        result_hash = canonical_hash(result)
        with self.connect() as conn:
            old = conn.execute("SELECT * FROM probe_commits WHERE run_id=%s", (run_id,)).fetchone()
            if old:
                if old["result_hash"] != result_hash:
                    raise ValueError("提交候选发生变化")
                return old["result"]
            conn.execute(
                "INSERT INTO probe_commits VALUES(%s,%s,%s)",
                (run_id, result_hash, json.dumps(result)),
            )
            conn.execute("INSERT INTO probe_turns VALUES(%s)", (run_id,))
            conn.execute("INSERT INTO probe_events VALUES(%s,'completed')", (run_id,))
        return result

    def counts(self):
        with self.connect() as conn:
            return {
                name: conn.execute(f"SELECT count(*) AS n FROM {name}").fetchone()["n"]
                for name in ("sandbox_writes", "probe_commits", "probe_turns", "probe_events")
            }

    def claim_worker(self, owner: str) -> bool:
        with self.connect() as conn:
            return (
                conn.execute(
                    "INSERT INTO probe_workers VALUES(true,%s) "
                    "ON CONFLICT DO NOTHING RETURNING owner",
                    (owner,),
                ).fetchone()
                is not None
            )

    def save_artifact(self, ref: str, tenant: str, body: str):
        digest = hashlib.sha256(body.encode()).hexdigest()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO probe_artifacts(ref,tenant_id,body,content_hash) VALUES(%s,%s,%s,%s)",
                (ref, tenant, body, digest),
            )
        return digest

    def read_artifact(self, ref: str, tenant: str, offset: int, limit: int):
        if offset < 0 or not 0 < limit <= 4000:
            raise ValueError("非法分页")
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM probe_artifacts WHERE ref=%s AND tenant_id=%s AND NOT revoked",
                (ref, tenant),
            ).fetchone()
        if not row:
            raise PermissionError("产物不存在、已撤权或无读取权限")
        page = row["body"][offset : offset + limit]
        return {
            "content": page,
            "next_offset": offset + len(page),
            "content_hash": row["content_hash"],
            "has_more": offset + len(page) < len(row["body"]),
        }
