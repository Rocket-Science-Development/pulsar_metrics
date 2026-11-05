"""Model fingerprint: which exact model was measured.

`model_version` of an evidence record is a free label chosen by the user: two different models may carry the same
label. A `ModelFingerprint` is a digest of the model itself, so that a record can prove which bytes were measured:

- `source="artifact"`: digest of the saved model file or directory, computed by Pulsar (`fingerprint_path`);
- `source="endpoint"`: digest reported by the serving endpoint (e.g. the Ollama model digest), not recomputed here;
- `source="declared"`: digest given by the user (e.g. from a model registry), not checked by Pulsar.

Digest of a file: the hash of its bytes, read in chunks (the same as `sha256sum <file>`). Digest of a directory: the
hash of its manifest (`directory_manifest`), one line `<hex digest of the file>  <relative path>` per file, sorted by
relative path (POSIX separators, UTF-8), every file included (hidden files too). Symbolic links to files are followed
(the content is hashed under the link's name); links to directories, broken links and special files raise instead of
being skipped silently. Empty directories, permissions and modification times are not part of the digest.

An in-memory object (a fitted scikit-learn estimator, a torch module) is not fingerprinted: its pickle bytes depend on
the pickle protocol, the library versions and the order of its attributes, so the same model gives several digests.
Fingerprint the saved artifact the server loads (the .joblib, .pkl, .onnx or .safetensors file, or the model
directory), and check it later with `verify_fingerprint`.
"""

import hashlib
import os
import re
import stat
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

Algorithm = Literal["sha256", "sha512"]
Source = Literal["artifact", "endpoint", "declared"]
HEX_LENGTHS = {"sha256": 64, "sha512": 128}
CHUNK_BYTES = 1024 * 1024
MAX_REF_LENGTH = 512


class ModelFingerprint(BaseModel):
    """Digest of the measured model (field `model_fingerprint` of an evidence record)"""

    model_config = ConfigDict(extra="ignore", json_schema_serialization_defaults_required=True, frozen=True)

    algorithm: Algorithm = Field("sha256", description="Hash algorithm of the digest: sha256 or sha512")
    digest: str = Field(description="Lowercase hexadecimal digest (64 characters for sha256, 128 for sha512)")
    source: Source = Field(
        description="artifact: saved model file or directory hashed by Pulsar; endpoint: digest reported by the serving "
        "endpoint; declared: digest given by the user, not checked by Pulsar"
    )
    ref: Optional[str] = Field(
        None,
        max_length=MAX_REF_LENGTH,
        description="What was hashed: file or directory name, served model tag (e.g. 'ollama:llama3.2:3b'), registry "
        "entry. A label: not part of evidence_id",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        """Accept "sha256:<hex>" as digest (the prefix must match the algorithm) and uppercase hexadecimal"""
        if isinstance(data, dict) and isinstance(data.get("digest"), str):
            data = dict(data)
            digest = data["digest"].strip()
            prefix, sep, rest = digest.partition(":")
            if sep:
                algorithm = data.get("algorithm", "sha256")
                if prefix.lower() != algorithm:
                    raise ValueError(f"digest prefix {prefix!r} does not match algorithm {algorithm!r}")
                digest = rest
            data["digest"] = digest.lower()
        return data

    @model_validator(mode="after")
    def _check_digest(self):
        length = HEX_LENGTHS[self.algorithm]
        if not re.fullmatch(f"[0-9a-f]{{{length}}}", self.digest):
            raise ValueError(f"digest must be {length} hexadecimal characters for {self.algorithm}")
        return self

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.digest}"

    def matches(self, other: Union["ModelFingerprint", dict, None]) -> bool:
        """Same algorithm and digest (source and ref are labels and are not compared)"""
        if other is None:
            return False
        other = ModelFingerprint.model_validate(other)
        return (self.algorithm, self.digest) == (other.algorithm, other.digest)


def _hash_file(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root: Path) -> list[tuple[str, Path]]:
    """(relative POSIX path, path) of every file under root; raises on what cannot be hashed instead of skipping it"""
    found = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            if os.path.islink(os.path.join(directory, name)):
                raise ValueError(f"symbolic link to a directory in the model directory: {os.path.join(directory, name)}")
        for name in filenames:
            path = Path(directory) / name
            try:
                mode = os.stat(path).st_mode  # follows a link to a file
            except FileNotFoundError:
                raise ValueError(f"broken symbolic link in the model directory: {path}") from None
            if not stat.S_ISREG(mode):
                raise ValueError(f"not a regular file in the model directory: {path}")
            found.append((path.relative_to(root).as_posix(), path))
    return sorted(found, key=lambda item: item[0].encode("utf-8"))


def directory_manifest(path: Union[str, Path], algorithm: Algorithm = "sha256") -> str:
    """Manifest hashed by `fingerprint_path` for a directory: `<hex>  <relative path>` per file, sorted (the format of
    `sha256sum`, so that `sha256sum -c` can check it from the directory)"""
    if algorithm not in HEX_LENGTHS:
        raise ValueError(f"algorithm must be one of {sorted(HEX_LENGTHS)}")
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    files = _files(root)
    if not files:
        raise ValueError(f"no file in the model directory {root}: nothing to fingerprint")
    return "".join(f"{_hash_file(p, algorithm)}  {rel}\n" for rel, p in files)


def fingerprint_path(path: Union[str, Path], algorithm: Algorithm = "sha256", ref: Optional[str] = None) -> ModelFingerprint:
    """Fingerprint (source "artifact") of a saved model file or directory

    ref defaults to the file or directory name (not its full path, which may name a user or a host)
    """
    if algorithm not in HEX_LENGTHS:
        raise ValueError(f"algorithm must be one of {sorted(HEX_LENGTHS)}")
    path = Path(path)
    if path.is_dir():
        digest = hashlib.new(algorithm, directory_manifest(path, algorithm).encode("utf-8")).hexdigest()
    elif path.is_file():
        digest = _hash_file(path, algorithm)
    else:
        raise FileNotFoundError(f"no model file or directory at {path}")
    return ModelFingerprint(algorithm=algorithm, digest=digest, source="artifact", ref=ref if ref is not None else path.name)


def declared_fingerprint(digest: str, algorithm: Algorithm = "sha256", ref: Optional[str] = None) -> ModelFingerprint:
    """Fingerprint (source "declared") of a digest known from elsewhere, e.g. a model registry"""
    return ModelFingerprint(algorithm=algorithm, digest=digest, source="declared", ref=ref)


def _fingerprint_of(record) -> Optional[ModelFingerprint]:
    if isinstance(record, ModelFingerprint):
        return record
    value = record.get("model_fingerprint") if isinstance(record, dict) else getattr(record, "model_fingerprint", None)
    return None if value is None else ModelFingerprint.model_validate(value)


def verify_fingerprint(record, path: Union[str, Path]) -> bool:
    """True when the model file or directory at `path` has the digest of the record's model_fingerprint

    record : Evidence, dict of an evidence record, or ModelFingerprint. ValueError when it carries no fingerprint.
    The digest is recomputed with the record's algorithm, whatever its source (an "endpoint" digest matches only if
    the endpoint reports the hash of the same bytes).
    """
    expected = _fingerprint_of(record)
    if expected is None:
        raise ValueError("the record carries no model_fingerprint")
    return fingerprint_path(path, algorithm=expected.algorithm).matches(expected)
