"""M3-01纯边界、原文与编排用例；仅定义，本轮不执行。"""

import asyncio
import hashlib

import pytest

from harness.adapters.ingestion_sources import LocalSourceStore
from harness.adapters.ingestion_text import IsolatedTextParser
from harness.bootstrap.ingestion import read_staged
from harness.contracts.hashing import canonical_hash
from harness.contracts.ingestion import IngestionJob, IngestionProfile, SourceRef, Upload
from harness.contracts.runtime import Principal, RuntimeFault
from harness.ingestion.pipeline import IngestionService, validate_upload
from harness.ingestion.text import parse_text


def upload(**changes):
    return Upload(request_id="request-1", document_id="doc-1", filename="资料.txt",
                  media_type="text/plain").model_copy(update=changes)


@pytest.mark.parametrize("filename", ["../x.txt", "..\\x.txt", "C:x.txt", "x.pdf", "x\x00.txt"])
def test_filename_is_not_a_path(filename):
    with pytest.raises(RuntimeFault, match="INGESTION_UNSUPPORTED_FILENAME"):
        validate_upload(upload(filename=filename), b"x", IngestionProfile())


@pytest.mark.parametrize("body", [b"", b"\xff", b"\x00", b" \n "])
def test_empty_binary_and_invalid_utf8_are_rejected(body):
    with pytest.raises(RuntimeFault):
        parse_text(body, IngestionProfile())


def test_normalized_chinese_source_offsets_and_hashes_are_stable():
    chunks = parse_text(("\ufeff" + "中文\r\n" * 100).encode(),
                        IngestionProfile(chunk_characters=100))
    assert "".join(c.body for c in chunks) == "中文\n" * 100
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.end - c.start == len(c.body) and c.body_hash == canonical_hash(c.body)
               for c in chunks)
    assert chunks[0].start == 0 and chunks[-1].end == 300


def test_upload_and_chunk_count_limits():
    with pytest.raises(RuntimeFault, match="UPLOAD_SIZE"):
        validate_upload(upload(), b"xx", IngestionProfile(max_upload_bytes=1))
    with pytest.raises(RuntimeFault, match="PARSE_LIMIT"):
        parse_text(b"x" * 101, IngestionProfile(chunk_characters=100, max_chunks=1))


async def test_source_store_is_idempotent_and_never_overwrites(tmp_path):
    body = "原文".encode()
    ref = SourceRef(object_id="a" * 64, sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body))
    store = LocalSourceStore(tmp_path)
    await store.put(ref, body)
    await store.put(ref, body)
    assert (tmp_path / ref.object_id).read_bytes() == body
    (tmp_path / ref.object_id).write_bytes(b"tampered")
    with pytest.raises(RuntimeFault, match="SOURCE_CONFLICT"):
        await store.put(ref, body)
    with pytest.raises(RuntimeFault, match="SOURCE_HASH_MISMATCH"):
        await store.put(ref, b"other")


def test_staging_path_and_size_are_bounded(tmp_path):
    (tmp_path / "x.txt").write_bytes(b"123")
    assert read_staged(tmp_path, "x.txt", 3) == b"123"
    with pytest.raises(RuntimeFault, match="SIZE"):
        read_staged(tmp_path, "x.txt", 2)
    with pytest.raises(RuntimeFault, match="PATH"):
        read_staged(tmp_path, "../x.txt", 3)


async def test_fixed_parser_runs_on_selector_compatible_thread():
    chunks = await IsolatedTextParser().parse("中文资料".encode(), IngestionProfile())
    assert chunks[0].body == "中文资料"
    with pytest.raises(RuntimeFault, match="INVALID_UTF8"):
        await IsolatedTextParser().parse(b"\xff", IngestionProfile())


class Repository:
    def __init__(self):
        self.job = IngestionJob(job_id="job-1", knowledge_base_id="kb-1", document_id="doc-1",
                                version_id="version-1", state="queued")
        self.transitions = []

    async def begin(self, *args):
        return self.job

    async def advance(self, actor, job_id, expected, target):
        assert self.job.state == expected
        self.transitions.append(target)
        self.job = self.job.model_copy(update={"state": target})

    async def finish(self, actor, job_id, chunks):
        self.job = self.job.model_copy(update={"state": "succeeded", "chunks_written": len(chunks)})
        return self.job

    async def fail(self, actor, job_id, code):
        self.job = self.job.model_copy(update={"state": "failed", "error_code": code})


class Parser:
    async def parse(self, body, profile):
        return parse_text(body, profile)


class FailingSource:
    async def put(self, *args):
        raise RuntimeFault("SOURCE_UNAVAILABLE", 503)


def actor():
    return Principal(tenant_id="tenant-1", application_id="app-1", subject_id="manager-1",
                     roles=frozenset({"knowledge_manager"}), resource_scopes=frozenset({"kb:kb-1"}))


async def test_pipeline_commits_after_source_and_duplicate_does_not_parse(tmp_path):
    repository = Repository()
    service = IngestionService(repository, LocalSourceStore(tmp_path), Parser(), IngestionProfile())
    job = await service.upload(actor(), "kb-1", upload(), b"text")
    assert job.state == "succeeded" and job.chunks_written == 1
    assert repository.transitions == ["storing", "parsing", "persisting"]
    assert await service.upload(actor(), "kb-1", upload(), b"text") == job
    assert len(repository.transitions) == 3


async def test_storage_failure_is_persisted_without_parsing():
    repository = Repository()
    service = IngestionService(repository, FailingSource(), Parser(), IngestionProfile())
    with pytest.raises(RuntimeFault, match="SOURCE_UNAVAILABLE"):
        await service.upload(actor(), "kb-1", upload(), b"text")
    assert repository.job.state == "failed" and repository.job.error_code == "SOURCE_UNAVAILABLE"
    assert repository.transitions == ["storing"]


async def test_cancellation_is_not_converted_to_success(tmp_path):
    class CancelledParser:
        async def parse(self, *args):
            raise asyncio.CancelledError()

    repository = Repository()
    service = IngestionService(repository, LocalSourceStore(tmp_path), CancelledParser(), IngestionProfile())
    with pytest.raises(asyncio.CancelledError):
        await service.upload(actor(), "kb-1", upload(), b"text")
    assert repository.job.state == "failed" and repository.job.error_code == "INGESTION_CANCELLED"


async def test_disabled_cli_does_not_load_identity_or_connect(tmp_path, monkeypatch):
    import argparse

    from harness.bootstrap.ingestion import execute

    monkeypatch.setattr("harness.bootstrap.providers.model_values", lambda root: {})
    with pytest.raises(RuntimeFault, match="RAG_DISABLED"):
        await execute(tmp_path, argparse.Namespace(command="create-kb", kb_id="kb-1"))


def test_parser_timeout_kills_and_reaps_fixed_process(monkeypatch):
    import subprocess
    import threading

    class HangingProcess:
        returncode = None

        def __init__(self):
            self.killed = False
            self.reaped = False

        def communicate(self, input=None, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("fixed-parser", timeout)
            self.reaped = True
            return b"", None

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -1

    process = HangingProcess()
    monkeypatch.setattr("harness.adapters.ingestion_text.subprocess.Popen", lambda *a, **k: process)
    ticks = iter((0, 0, 2))
    monkeypatch.setattr("harness.adapters.ingestion_text.time.monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeFault, match="PARSE_TIMEOUT"):
        IsolatedTextParser._parse(b"text", IngestionProfile(parse_timeout_seconds=1), threading.Event())
    assert process.killed and process.reaped
