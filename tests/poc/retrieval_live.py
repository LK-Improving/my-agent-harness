"""M0-05 显式真实检索 PoC；隔离资源、先冻结门限、无自动重试，不由 pytest 收集。"""

import argparse
import json
import math
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from harness.adapters.models.qwen import QwenClient
from harness.bootstrap.event_loop import run
from harness.bootstrap.settings import load_qwen_settings
from harness.contracts.hashing import canonical_hash
from psycopg import sql
from pymilvus import DataType, Function, FunctionType, MilvusClient

from tests.poc.milvus_probe import load_jsonl, rrf

ROOT = Path(__file__).resolve().parents[2]
GUARD = 'tenant == "tenant-a" and kb == "kb-main" and snapshot == "s1"'


def metrics(cases, mode, k):
    """文档级宏平均；只有有答案标注参与质量分母，负例单列。"""
    positive = [case for case in cases if case["relevant_ids"]]
    values = {name: [] for name in ("recall", "mrr", "ndcg", "precision")}
    for case in positive:
        relevant = set(case["relevant_ids"])
        hits = [item in relevant for item in case[mode][:k]]
        values["recall"].append(sum(hits) / len(relevant))
        values["precision"].append(sum(hits) / k)
        values["mrr"].append(next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
        values["ndcg"].append(
            sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1)) / ideal
        )
    return {name + f"_at_{k}": sum(nums) / len(nums) for name, nums in values.items()}


class Probe:
    """只用于显式 PoC 的有界调用审计；不冒充 M3 正式检索网关。"""

    def __init__(self, dsn, uri):
        if (
            urlparse(dsn).hostname not in {"127.0.0.1", "localhost"}
            or urlparse(dsn).path != "/harness_m0"
        ):
            raise ValueError("仅允许本项目本机 harness_m0")
        endpoint = urlparse(uri)
        if endpoint.hostname not in {"127.0.0.1", "localhost"} or endpoint.port != 19531:
            raise ValueError("仅允许本项目本机 Milvus 19531")
        self.dsn = dsn
        self.namespace = "rag_poc_" + uuid4().hex
        self.collection = "harness_m0_live_" + uuid4().hex
        self.scoped = psycopg.conninfo.make_conninfo(
            dsn, options=f"-c search_path={self.namespace}"
        )
        self.client = MilvusClient(uri=uri, timeout=20)
        self.model = QwenClient(load_qwen_settings(ROOT / ".env"))
        self.calls = []

    def execute(self, statement, parameters=None):
        with psycopg.connect(self.scoped) as conn:
            cursor = conn.execute(statement, parameters)
            return cursor.fetchall() if cursor.description else []

    def call(self, kind, payload, invoke):
        if len(self.calls) >= 24:
            raise RuntimeError("PoC 真实模型调用超过24次上限")
        call = {
            "id": uuid4().hex,
            "kind": kind,
            "input_hash": canonical_hash(payload),
            "state": "sent",
        }
        self.calls.append(call)
        # 派发前落库；不跨外部请求持有业务事务或锁，不自动重试。
        self.execute("INSERT INTO probe_calls VALUES(%s,%s::jsonb)", (call["id"], json.dumps(call)))
        try:
            result = run(invoke())
        except Exception:
            call["state"] = "failed_billing_unknown"
            self.execute(
                "UPDATE probe_calls SET record=%s::jsonb WHERE id=%s",
                (json.dumps(call), call["id"]),
            )
            raise
        call.update(
            state="succeeded",
            request_id=result.request_id,
            usage=result.usage.model_dump(exclude_none=True) if result.usage else None,
        )
        self.execute(
            "UPDATE probe_calls SET record=%s::jsonb WHERE id=%s", (json.dumps(call), call["id"])
        )
        return result

    def setup(self, corpus, vectors):
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        for name in ("id", "tenant", "kb", "snapshot"):
            schema.add_field(name, DataType.VARCHAR, is_primary=name == "id", max_length=128)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=4096,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "jieba"},
        )
        schema.add_field(
            "dense", DataType.FLOAT_VECTOR, dim=self.model.settings.embedding_dimensions
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )
        indexes = self.client.prepare_index_params()
        indexes.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
        indexes.add_index(
            field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
        )
        self.client.create_collection(
            self.collection, schema=schema, index_params=indexes, consistency_level="Strong"
        )
        rows = [
            dict(
                id=doc["id"],
                tenant=doc["tenant"],
                kb=doc["kb"],
                snapshot=doc["snapshot"],
                text=doc["text"],
                dense=vector,
            )
            for doc, vector in zip(corpus, vectors, strict=True)
        ]
        self.client.insert(self.collection, rows)
        self.client.flush(self.collection)
        self.execute(
            "CREATE TABLE authority(id text PRIMARY KEY,tenant text,kb text,"
            "snapshot text,revoked boolean,body text)"
        )
        self.execute("CREATE TABLE active(snapshot text PRIMARY KEY)")
        self.execute("INSERT INTO active VALUES('s1')")
        with psycopg.connect(self.scoped) as conn, conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO authority VALUES(%s,%s,%s,%s,%s,%s)",
                [
                    (d["id"], d["tenant"], d["kb"], d["snapshot"], d["revoked"], d["text"])
                    for d in corpus
                ],
            )

    def search(self, mode, query, vector, guard=GUARD, broken=False):
        result = self.client.search(
            self.collection,
            data=[vector if mode == "dense" else query],
            anns_field="missing_vector" if broken else ("dense" if mode == "dense" else "sparse"),
            filter=guard,
            limit=12,
            output_fields=["tenant", "kb", "snapshot"],
        )
        for hit in result[0]:
            assert hit["entity"]["tenant"] == "tenant-a"
            assert hit["entity"]["kb"] == "kb-main"
            assert hit["entity"]["snapshot"] == ("s2" if '"s2"' in guard else "s1")
        return [str(hit["id"]) for hit in result[0]]

    def authorized(self, ids, broken=False):
        # 召回仅带ID与过滤元数据；PG权威身份/活动代际/撤权校验后才取正文。
        if broken:
            self.execute("SELECT id FROM deliberately_missing_authority")
        rows = self.execute(
            "SELECT id,body FROM authority WHERE id=ANY(%s) AND tenant='tenant-a' "
            "AND kb='kb-main' AND snapshot=(SELECT snapshot FROM active) AND NOT revoked",
            (ids,),
        )
        found = dict(rows)
        return [(item, found[item]) for item in ids if item in found]

    def pipeline(
        self,
        query,
        vector,
        *,
        dense_broken=False,
        bm25_broken=False,
        rerank_broken=False,
        pg_broken=False,
    ):
        rankings = []
        diagnostics = []
        for mode, broken in (("dense", dense_broken), ("bm25", bm25_broken)):
            try:
                rankings.append(self.search(mode, query, vector, broken=broken))
            except Exception:
                if not broken:
                    raise
                diagnostics.append(mode + "_unavailable_actual_rpc")
        if not any(rankings):
            raise RuntimeError("双路无有效结果，不能伪装成功")
        authorized = self.authorized(rrf(*rankings)[:12], broken=pg_broken)
        ids = [item[0] for item in authorized]
        assert not (set(ids) & {"c10", "c11", "c12", "c13"})
        if rerank_broken:
            # 受控端口故障，明确区别于真实供应商故障；不得声称远端发生过503。
            diagnostics.append("rerank_unavailable_controlled_boundary")
            return ids[:5], diagnostics
        texts = [item[1] for item in authorized]
        result = self.call(
            "rerank",
            {"query": query, "ids": ids},
            lambda: self.model.rerank(query, texts, top_n=min(5, len(texts))),
        )
        return [ids[item.index] for item in result.output.results], diagnostics

    def evaluate(self, corpus, queries, frozen):
        texts = [item["text"] for item in corpus] + [item["query"] for item in queries]
        vectors = []
        for offset in range(0, len(texts), 10):
            batch = texts[offset : offset + 10]
            embedded = self.call(
                "embedding", {"texts": batch}, lambda batch=batch: self.model.embed(batch)
            )
            assert embedded.model == self.model.settings.embedding_model
            vectors.extend(
                item.embedding for item in sorted(embedded.data, key=lambda item: item.index)
            )
        assert all(any(value != 0 for value in vector) for vector in vectors)
        self.setup(corpus, vectors[: len(corpus)])
        analyzer = self.client.run_analyzer(
            "西湖预约与良渚团体预约", analyzer_params={"tokenizer": "jieba"}
        ).tokens
        assert analyzer and "西湖" in analyzer and "良渚" in analyzer
        cases = []
        for query, vector in zip(queries, vectors[len(corpus) :], strict=True):
            dense = [
                item[0] for item in self.authorized(self.search("dense", query["query"], vector))
            ]
            sparse = [
                item[0] for item in self.authorized(self.search("bm25", query["query"], vector))
            ]
            ranked, diagnostics = self.pipeline(query["query"], vector)
            cases.append(
                dict(
                    case_id=query["id"],
                    split=query["split"],
                    relevant_ids=query["relevant_ids"],
                    dense=dense[:5],
                    bm25=sparse[:5],
                    hybrid=rrf(dense, sparse)[:5],
                    rerank=ranked,
                    diagnostics=diagnostics,
                    negative_case_kind=query.get("kind", "unanswerable")
                    if not query["relevant_ids"]
                    else None,
                )
            )
        first = queries[0]["query"]
        vector = vectors[len(corpus)]
        failures = {}
        for mode in ("dense", "bm25", "rerank"):
            before = len(self.calls)
            ranking, diagnostics = self.pipeline(first, vector, **{mode + "_broken": True})
            assert "c01" in ranking and diagnostics
            failures[mode] = dict(
                passed=True, diagnostics=diagnostics, provider_calls=len(self.calls) - before
            )
        before = len(self.calls)
        try:
            self.pipeline(first, vector, pg_broken=True)
        except psycopg.errors.UndefinedTable:
            assert len(self.calls) == before
            failures["pg"] = dict(passed=True, mode="blocked_before_rerank", provider_calls=0)
        else:
            raise AssertionError("PG失败必须阻断")
        lifecycle = self.lifecycle(corpus[0], vectors[0], first, vector)
        scores = {
            split: {
                mode: metrics([case for case in cases if case["split"] == split], mode, 5)
                for mode in ("dense", "bm25", "hybrid", "rerank")
            }
            for split in ("development", "holdout")
        }
        requirements = frozen["policy"]["requirements"]
        gates = {}
        for split, modes in scores.items():
            for mode in ("dense", "bm25", "hybrid"):
                gates[split + "_" + mode + "_recall"] = (
                    modes[mode]["recall_at_5"] >= requirements[mode + "_recall_at_5_min"]
                )
            gates[split + "_rerank_mrr"] = (
                modes["rerank"]["mrr_at_5"] >= requirements["rerank_mrr_at_5_min"]
            )
        leaks = sum(
            bool(set(case[mode]) & {"c10", "c11", "c12", "c13"})
            for case in cases
            for mode in ("dense", "bm25", "hybrid", "rerank")
        )
        gates.update(
            no_leaks=leaks == 0,
            all_cases=len(cases) == len(queries),
            fault_probes=len(failures) == 4,
            lifecycle=all(lifecycle.values()),
        )
        return dict(
            status="passed" if all(gates.values()) else "failed",
            gates=gates,
            server_version=self.client.get_server_version(),
            sdk_version=version("pymilvus"),
            analyzer_output=analyzer,
            embedding_model=self.model.settings.embedding_model,
            embedding_dimensions=self.model.settings.embedding_dimensions,
            rerank_model=self.model.settings.rerank_model,
            metrics=scores,
            cases=cases,
            unauthorized_results=leaks,
            failure_probes=failures,
            lifecycle=lifecycle,
            provider_calls=self.calls,
            semantic_vector_mode="real-qwen-embedding",
            negative_cases_scope="只验证身份隔离，不验证无答案决策或拒答质量",
            not_verified=[
                "生产入库/删除传播",
                "正式M3检索网关及providers独立端口",
                "生产数据质量和容量",
                "真实远端Rerank宕机",
                "服务关闭时运营API端到端回归",
            ],
        )

    def lifecycle(self, original, vector, query, query_vector):
        # 新代际必须在两路都能召回后才激活；旧索引故意保留以验证权威回表。
        new_id = "c01-generation2"
        self.client.insert(
            self.collection,
            [
                dict(
                    id=new_id,
                    tenant="tenant-a",
                    kb="kb-main",
                    snapshot="s2",
                    text=original["text"],
                    dense=vector,
                )
            ],
        )
        self.client.flush(self.collection)
        self.execute(
            "INSERT INTO authority VALUES(%s,'tenant-a','kb-main','s2',false,%s)",
            (new_id, original["text"]),
        )
        assert self.authorized([new_id]) == []
        guard = GUARD.replace('"s1"', '"s2"')
        for mode in ("dense", "bm25"):
            assert new_id in self.search(mode, query, query_vector, guard=guard)
        self.execute("UPDATE active SET snapshot='s2'")
        assert self.authorized(["c01"]) == [] and self.authorized([new_id])
        self.execute("UPDATE authority SET revoked=true WHERE id=%s", (new_id,))
        assert self.authorized([new_id]) == []
        try:
            self.client.delete(self.collection, filter='missing_cleanup_field == "x"')
        except Exception:
            pass
        else:
            raise AssertionError("清理故障未发生")
        assert new_id in self.search("dense", query, query_vector, guard=guard)
        assert self.authorized([new_id]) == []
        self.client.delete(self.collection, ids=[new_id])
        for mode in ("dense", "bm25"):
            assert not self.search(mode, query, query_vector, guard=guard)
        return dict(
            new_generation_ready_before_activation=True,
            old_snapshot_denied=True,
            revocation_immediate=True,
            failed_cleanup_does_not_restore_access=True,
            both_indexes_deleted=True,
        )

    def close(self):
        try:
            if self.client.has_collection(self.collection):
                self.client.drop_collection(self.collection)
            assert not self.client.has_collection(self.collection)
        finally:
            self.client.close()
            with psycopg.connect(self.dsn, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(self.namespace)
                    )
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-live", action="store_true", required=True)
    args = parser.parse_args()
    assert args.confirm_live
    load_dotenv(ROOT / ".env", override=False)
    folder = ROOT / ".runtime"
    folder.mkdir(exist_ok=True)
    corpus = [dict(item, kb="kb-main") for item in load_jsonl("corpus.jsonl")]
    corpus.append(dict(corpus[0], id="c13", kb="kb-other"))
    queries = [dict(item, split="development") for item in load_jsonl("queries.jsonl")]
    queries += [dict(item, split="holdout") for item in load_jsonl("holdout.jsonl")]
    policy = json.loads((ROOT / "evals/m0/retrieval-live-policy.json").read_text("utf-8"))
    frozen = dict(
        corpus_hash=canonical_hash(corpus),
        queries_hash=canonical_hash(queries),
        policy_hash=canonical_hash(policy),
        policy=policy,
        frozen_at=datetime.now(UTC).isoformat(),
    )
    (folder / "m0-retrieval-live-frozen.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), "utf-8"
    )
    probe = Probe(os.environ["HARNESS_TEST_DSN"], os.environ["HARNESS_MILVUS_URI"])
    report = dict(status="failed", frozen=frozen, started_at=datetime.now(UTC).isoformat())
    try:
        with psycopg.connect(probe.dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(probe.namespace)))
        probe.execute("CREATE TABLE probe_calls(id text PRIMARY KEY, record jsonb NOT NULL)")
        report.update(probe.evaluate(corpus, queries, frozen))
    except Exception as exc:
        # 不保存第三方异常正文，其中可能有连接参数、请求正文或其他敏感值。
        report.update(error_type=type(exc).__name__, provider_calls=probe.calls)
        raise
    finally:
        try:
            probe.close()
            report["isolated_resources_removed"] = True
        finally:
            report["finished_at"] = datetime.now(UTC).isoformat()
            (folder / "m0-retrieval-live-report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
            )
    print(
        json.dumps(dict(status=report["status"], gates=report["gates"], metrics=report["metrics"]))
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
