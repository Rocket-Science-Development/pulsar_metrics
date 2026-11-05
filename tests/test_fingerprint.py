"""Model fingerprint: digests of saved artifacts, evidence records carrying them, ids of records without them unchanged"""

import hashlib
import json
import os
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.evidence import SCHEMA_PATH, Evidence, dumps, evidence_from_analyzer, evidence_from_result, loads
from pulsar_metrics.fingerprint import (
    ModelFingerprint,
    declared_fingerprint,
    directory_manifest,
    fingerprint_path,
    verify_fingerprint,
)
from pulsar_metrics.metrics.base import MetricResults
from pulsar_metrics.quality import DataQualityAnalyzer

from .test_evidence import run, window

PRODUCED = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)
HEX = "ab" * 32
# digest of make_model_dir(): pinned, and the same in pulsar_data_collection.fingerprint (0.14)
MODEL_DIR_DIGEST = "18d881ea17494c6bf6b94335a9d49633ea008e0465f9c4c5ba57dff0890ceb7f"
# evidence_id of RECORD_ARGS computed by pulsar-metrics 0.15.0, before model_fingerprint existed
ID_015 = "73163dd21432a284b2583f28d6aebf6ae0660acfdc9912643667c0e092271bd5"
RECORD_ARGS = dict(
    model_id="credit_model",
    model_version="1.0",
    window={"id": "2026-W44", "start": "2026-10-26T00:00:00Z", "end": "2026-11-02T00:00:00Z"},
    reference={"id": "ref-2026-01"},
    produced_at=PRODUCED,
)


def ks_result():
    return MetricResults(
        metric_name="ks_2samp", metric_type="drift", feature_name="income", metric_value=0.01, threshold=0.05, drift_status=True
    )


def record(**kwargs):
    return evidence_from_result(ks_result(), "REG-0001", **{**RECORD_ARGS, **kwargs})


# ------------------------------------------------------------------ the value
def test_fingerprint_normalizes_and_validates_the_digest():
    fp = ModelFingerprint(digest="SHA256:" + HEX.upper(), source="declared")
    assert fp.digest == HEX and fp.algorithm == "sha256" and str(fp) == f"sha256:{HEX}"
    assert ModelFingerprint(algorithm="sha512", digest="0f" * 64, source="artifact").algorithm == "sha512"
    for bad in (
        {"digest": "ab" * 31, "source": "declared"},  # too short
        {"digest": "zz" * 32, "source": "declared"},  # not hexadecimal
        {"digest": HEX, "source": "declared", "algorithm": "md5"},
        {"digest": "ab" * 32, "source": "declared", "algorithm": "sha512"},  # sha256 length for sha512
        {"digest": "sha512:" + HEX, "source": "declared"},  # prefix does not match the algorithm
        {"digest": HEX, "source": "weights"},  # unknown source
        {"digest": HEX, "source": "declared", "ref": "x" * 600},
    ):
        with pytest.raises(ValidationError):
            ModelFingerprint(**bad)


def test_matches_compares_algorithm_and_digest_only():
    a = declared_fingerprint(HEX, ref="registry:credit/7")
    assert a.matches({"digest": HEX, "source": "endpoint", "ref": "other"})
    assert not a.matches(declared_fingerprint("cd" * 32)) and not a.matches(None)


# ------------------------------------------------------------------ artifacts
def test_file_digest_is_the_sha256_of_its_bytes(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(os.urandom(3 * 1024 * 1024 + 17))  # several chunks
    fp = fingerprint_path(path)
    assert fp.digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert fp.source == "artifact" and fp.ref == "model.joblib"  # the name, not the full path
    assert fingerprint_path(path, algorithm="sha512").digest == hashlib.sha512(path.read_bytes()).hexdigest()
    assert fingerprint_path(path, ref="credit/7").ref == "credit/7"
    with pytest.raises(FileNotFoundError):
        fingerprint_path(tmp_path / "missing.pkl")


def make_model_dir(root, order):
    files = {"config.json": b'{"a": 1}', "weights/part-1.bin": b"\x00" * 1000, ".hidden": b"h", "Z.txt": b"z"}
    for name in order:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[name])
    return root


def test_directory_digest_is_deterministic_and_covers_every_file(tmp_path):
    names = ["config.json", "weights/part-1.bin", ".hidden", "Z.txt"]
    a = make_model_dir(tmp_path / "a", names)
    b = make_model_dir(tmp_path / "b", list(reversed(names)))  # other creation order
    assert fingerprint_path(a).digest == fingerprint_path(b).digest == MODEL_DIR_DIGEST
    manifest = directory_manifest(a)
    lines = manifest.splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == [".hidden", "Z.txt", "config.json", "weights/part-1.bin"]
    assert lines[0].split("  ")[0] == hashlib.sha256(b"h").hexdigest()
    assert fingerprint_path(a).digest == hashlib.sha256(manifest.encode()).hexdigest()

    (b / ".hidden").write_bytes(b"H")  # a hidden file is not ignored
    assert fingerprint_path(a).digest != fingerprint_path(b).digest
    (b / ".hidden").write_bytes(b"h")
    (b / "Z.txt").rename(b / "Y.txt")  # a renamed file changes the digest
    assert fingerprint_path(a).digest != fingerprint_path(b).digest


def test_directory_without_files_is_refused(tmp_path):
    (tmp_path / "empty" / "sub").mkdir(parents=True)
    with pytest.raises(ValueError, match="no file"):
        fingerprint_path(tmp_path / "empty")


def test_links_to_directories_and_broken_links_are_refused(tmp_path):
    root = make_model_dir(tmp_path / "m", ["config.json"])
    (tmp_path / "outside").mkdir()
    try:
        os.symlink(tmp_path / "outside", root / "linked", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links not available")
    with pytest.raises(ValueError, match="directory"):
        fingerprint_path(root)
    os.remove(root / "linked") if os.path.islink(root / "linked") else os.rmdir(root / "linked")
    os.symlink(tmp_path / "nowhere.bin", root / "broken.bin")
    with pytest.raises(ValueError, match="broken"):
        fingerprint_path(root)


def test_verify_fingerprint(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model v1")
    rec = record(model_fingerprint=fingerprint_path(path))
    assert verify_fingerprint(rec, path) and verify_fingerprint(rec.to_dict(), path)
    path.write_bytes(b"model v2")
    assert not verify_fingerprint(rec, path)
    with pytest.raises(ValueError, match="no model_fingerprint"):
        verify_fingerprint(record(), path)


# ------------------------------------------------------------------ evidence records
def test_ids_of_records_without_fingerprint_are_unchanged():
    """The fingerprint enters the hash only when present: records of 0.15 keep their evidence_id"""
    rec = record()
    assert rec.model_fingerprint is None and rec.evidence_id == ID_015
    data = rec.to_dict()
    del data["model_fingerprint"]  # a record written by 0.15 has no such key
    assert Evidence.from_dict(data).evidence_id == ID_015


def test_fingerprint_is_part_of_the_identity():
    fp = declared_fingerprint(HEX, ref="registry:credit/7")
    rec = record(model_fingerprint=fp)
    assert rec.evidence_id != ID_015
    assert record(model_fingerprint={"digest": "cd" * 32, "source": "declared"}).evidence_id != rec.evidence_id
    # source and ref are labels: same digest, same id
    assert record(model_fingerprint={"digest": HEX, "source": "endpoint", "ref": "x"}).evidence_id == rec.evidence_id

    data = json.loads(dumps([rec]))[0]
    assert data["model_fingerprint"] == {"algorithm": "sha256", "digest": HEX, "source": "declared", "ref": "registry:credit/7"}
    assert loads(json.dumps(data))[0] == rec
    data["model_fingerprint"]["digest"] = "cd" * 32  # altered fingerprint: the id no longer matches
    with pytest.raises(ValidationError, match="evidence_id"):
        Evidence.from_dict(data)
    del data["model_fingerprint"]  # removed fingerprint: refused as well
    with pytest.raises(ValidationError, match="evidence_id"):
        Evidence.from_dict(data)


def test_record_with_fingerprint_matches_the_committed_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(record(model_fingerprint=declared_fingerprint(HEX)).to_dict(), schema)
    jsonschema.validate(record().to_dict(), schema)


def test_analyzers_carry_their_fingerprint(tmp_path):
    path = tmp_path / "credit.joblib"
    path.write_bytes(b"weights")
    fp = fingerprint_path(path)
    current, reference = window(300, 0.5, "2026-01-08", 1), window(300, 0, "2026-01-01", 2)

    analyzer = run(current, reference)
    plain = evidence_from_analyzer(analyzer, "REG-0001")
    assert all(r.model_fingerprint is None for r in plain)
    explicit = evidence_from_analyzer(analyzer, "REG-0001", model_fingerprint=fp)
    assert explicit and all(r.model_fingerprint == fp for r in explicit)

    analyzer = Analyzer(name="weekly", model_id="credit_model", model_version="1.0", model_fingerprint=fp.model_dump())
    analyzer.add_drift_metrics(metrics_list=["ks_2samp"], features_list=["income"])
    analyzer.run(current=current, reference=reference, options={})
    records = evidence_from_analyzer(analyzer, "REG-0001")
    assert records and all(r.model_fingerprint == fp and verify_fingerprint(r, path) for r in records)
    with pytest.raises(ValidationError):
        Analyzer(name="w", model_id="m", model_version="1", model_fingerprint={"digest": "nope", "source": "declared"})

    dq = DataQualityAnalyzer(name="dq", model_fingerprint=fp)
    meta = ["model_id", "model_version"]
    dq.run(current=current.drop(columns=meta), reference=reference.drop(columns=meta))
    assert all(r.model_fingerprint == fp for r in evidence_from_analyzer(dq, "REG-0001"))
