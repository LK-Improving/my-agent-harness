"""M0 中文哈希稠密/BM25兼容探针；真实语义验收另见 retrieval_live。"""

import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from psycopg import sql
from pymilvus import DataType, Function, FunctionType, MilvusClient

ROOT = Path(__file__).resolve().parents[2]


def deterministic_vector(text: str, dimension: int = 128) -> list[float]:
    """字符二元组哈希向量仅验证稠密检索通路，不代表语义 Embedding。"""
    vector = [0.0] * dimension
    for offset in range(max(1, len(text) - 1)):
        token = text[offset : offset + 2]
        slot = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4]) % dimension
        vector[slot] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def rrf(*rankings: list[str], k0: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k0 + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def load_jsonl(name: str):
    return [
        json.loads(line)
        for line in (ROOT / "evals/m0" / name).read_text("utf-8").splitlines()
        if line.strip()
    ]


def run_probe(dsn: str, uri: str):
    """只操作本项目隔离 schema/collection；异常仍保留明确未通过状态。"""
    db = urlparse(dsn)
    endpoint = urlparse(uri)
    if db.hostname not in {"127.0.0.1", "localhost"} or db.path != "/harness_m0":
        raise ValueError("只允许显式本机 harness_m0 测试数据库")
    if endpoint.hostname not in {"127.0.0.1", "localhost"} or endpoint.port != 19531:
        raise ValueError("只允许本项目127.0.0.1:19531隔离Milvus")
    corpus = load_jsonl("corpus.jsonl")
    queries = load_jsonl("queries.jsonl")
    namespace = "poc_" + uuid4().hex
    collection = "harness_m0_" + uuid4().hex
    client = MilvusClient(uri=uri, timeout=15)
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
    scoped = psycopg.conninfo.make_conninfo(dsn, options=f"-c search_path={namespace}")
    try:
        with psycopg.connect(scoped) as conn:
            conn.execute(
                "CREATE TABLE kb_authority(id text PRIMARY KEY,tenant text,"
                "snapshot text,revoked boolean,body text)"
            )
            with conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO kb_authority VALUES(%s,%s,%s,%s,%s)",
                    [
                        (doc["id"], doc["tenant"], doc["snapshot"], doc["revoked"], doc["text"])
                        for doc in corpus
                    ],
                )
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("tenant", DataType.VARCHAR, max_length=128)
        schema.add_field("snapshot", DataType.VARCHAR, max_length=128)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=4096,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "jieba"},
        )
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=128)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )
        indexes = client.prepare_index_params()
        indexes.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
        indexes.add_index(
            field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
        )
        client.create_collection(
            collection, schema=schema, index_params=indexes, consistency_level="Strong"
        )
        client.insert(
            collection,
            [
                {
                    "id": doc["id"],
                    "tenant": doc["tenant"],
                    "snapshot": doc["snapshot"],
                    "text": doc["text"],
                    "dense": deterministic_vector(doc["text"]),
                }
                for doc in corpus
            ],
        )
        client.flush(collection)
        analyzer = client.run_analyzer(
            "西湖预约与良渚团体预约", analyzer_params={"tokenizer": "jieba"}
        ).tokens
        results = []

        def authorized(ids: list[str], tenant: str, snapshot: str):
            # 这里只在PG权威校验后取正文；召回阶段仅交付ID而非未授权文本。
            with psycopg.connect(scoped) as conn:
                rows = conn.execute(
                    "SELECT id FROM kb_authority WHERE id=ANY(%s) AND "
                    "tenant=%s AND snapshot=%s AND NOT revoked",
                    (ids, tenant, snapshot),
                ).fetchall()
            allowed = {row[0] for row in rows}
            return [item for item in ids if item in allowed]

        for query in queries:
            # 过滤范围取自可信测试身份；不把客户端/模型字符串直接拼进表达式。
            if query["tenant"] != "tenant-a" or query["snapshot"] != "s1":
                raise ValueError("样本身份超出本轮已授权测试范围")
            guard = 'tenant == "tenant-a" and snapshot == "s1"'
            dense = client.search(
                collection,
                data=[deterministic_vector(query["query"])],
                anns_field="dense",
                filter=guard,
                limit=12,
            )
            sparse = client.search(
                collection, data=[query["query"]], anns_field="sparse", filter=guard, limit=12
            )
            dense_ids = authorized([str(hit["id"]) for hit in dense[0]], "tenant-a", "s1")
            sparse_ids = authorized([str(hit["id"]) for hit in sparse[0]], "tenant-a", "s1")
            hybrid_ids = rrf(dense_ids, sparse_ids)[:5]
            results.append(
                {
                    "case_id": query["id"],
                    "dense": dense_ids[:5],
                    "bm25": sparse_ids[:5],
                    "hybrid": hybrid_ids,
                    "relevant_ids": query["relevant_ids"],
                    "kind": query["kind"],
                }
            )
        answers = [case for case in results if case["relevant_ids"]]
        recall = {
            mode: sum(
                len(set(case[mode]) & set(case["relevant_ids"])) / len(case["relevant_ids"])
                for case in answers
            )
            / len(answers)
            for mode in ("dense", "bm25", "hybrid")
        }
        forbidden = {"c10", "c11", "c12"}
        leaks = sum(
            len(set(case[mode]) & forbidden)
            for case in results
            for mode in ("dense", "bm25", "hybrid")
        )
        return {
            "server_version": client.get_server_version(),
            "analyzer_output": analyzer,
            "dense_vector_mode": "deterministic-character-hash",
            "semantic_quality_verified": False,
            "recall_at_5": recall,
            "leaks": leaks,
            "cases": results,
            "not_verified": ["真实通义Embedding/Rerank", "单路故障/PG故障注入", "删除传播"],
        }
    finally:
        try:
            if client.has_collection(collection):
                client.drop_collection(collection)
        finally:
            client.close()
            with psycopg.connect(dsn, autocommit=True) as conn:
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(namespace)))


if __name__ == "__main__":
    load_dotenv(ROOT / ".env", override=False)
    # 必须由开发者显式运行；不在导入、安装、健康检查或应用启动时执行。
    report = run_probe(os.environ["HARNESS_TEST_DSN"], os.environ["HARNESS_MILVUS_URI"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
