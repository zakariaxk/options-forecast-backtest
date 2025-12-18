from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None
    BotoCoreError = ClientError = Exception


S3_SCHEME = "s3://"


@dataclass(frozen=True)
class StorageURI:
    """Represents a storage location that may live on S3 or the local filesystem."""

    uri: str

    def is_s3(self) -> bool:
        return self.uri.startswith(S3_SCHEME)

    def bucket(self) -> str:
        if not self.is_s3():
            raise ValueError("Not an S3 URI")
        without_scheme = self.uri[len(S3_SCHEME) :]
        return without_scheme.split("/", 1)[0]

    def key(self) -> str:
        if not self.is_s3():
            raise ValueError("Not an S3 URI")
        without_scheme = self.uri[len(S3_SCHEME) :]
        parts = without_scheme.split("/", 1)
        if len(parts) == 1:
            return ""
        return parts[1]

    def local_path(self, base_dir: Path | None = None) -> Path:
        if self.is_s3():
            bucket = self.bucket()
            key = self.key()
            root = base_dir or Path("data") / bucket
            path = root / key
        else:
            path = Path(self.uri)
        return path.resolve()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _get_s3_resource():
    if boto3 is None:
        raise RuntimeError("boto3 is required for S3 operations but is not installed")
    session_kwargs = {}
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if endpoint_url:
        session_kwargs["endpoint_url"] = endpoint_url
    return boto3.resource("s3", **session_kwargs)


def compute_content_hash(data: bytes) -> str:
    """Return deterministic md5 checksum."""
    return md5(data).hexdigest()


def write_json(obj: Any, uri: str) -> str:
    payload = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    checksum = compute_content_hash(payload)
    storage = StorageURI(uri)
    if storage.is_s3():
        resource = _get_s3_resource()
        try:
            resource.Object(storage.bucket(), storage.key()).put(Body=payload)
        except (BotoCoreError, ClientError) as exc:  # pragma: no cover - network operation
            raise RuntimeError(f"Failed to write JSON to {uri}: {exc}") from exc
    else:
        path = storage.local_path()
        _ensure_parent(path)
        path.write_bytes(payload)
    return checksum


def read_json(uri: str) -> Any:
    storage = StorageURI(uri)
    if storage.is_s3():  # pragma: no cover - network operation
        resource = _get_s3_resource()
        try:
            body = resource.Object(storage.bucket(), storage.key()).get()["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Failed to read JSON from {uri}: {exc}") from exc
        return json.loads(body)
    path = storage.local_path()
    with path.open("rb") as fh:
        return json.load(fh)


def write_parquet(df: pd.DataFrame, uri: str, *, index: bool = False) -> str:
    storage = StorageURI(uri)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=index)
    payload = buffer.getvalue()
    if storage.is_s3():  # pragma: no cover - network operation
        resource = _get_s3_resource()
        try:
            resource.Object(storage.bucket(), storage.key()).put(Body=payload)
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Failed to write parquet to {uri}: {exc}") from exc
        return compute_content_hash(payload)
    path = storage.local_path()
    _ensure_parent(path)
    with path.open("wb") as fh:
        fh.write(payload)
    return compute_content_hash(path.read_bytes())


def read_parquet(uri: str, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    storage = StorageURI(uri)
    if storage.is_s3():  # pragma: no cover - network operation
        resource = _get_s3_resource()
        try:
            body = resource.Object(storage.bucket(), storage.key()).get()["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Failed to read parquet from {uri}: {exc}") from exc
        return pd.read_parquet(io.BytesIO(body), columns=list(columns) if columns else None)
    path = storage.local_path()
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def list_local(uri: str) -> list[Path]:
    storage = StorageURI(uri)
    if storage.is_s3():  # pragma: no cover - network operation
        resource = _get_s3_resource()
        objects = resource.Bucket(storage.bucket()).objects.filter(Prefix=storage.key())
        return [Path(obj.key) for obj in objects]
    path = storage.local_path()
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return sorted(p for p in path.glob("**/*") if p.is_file())
