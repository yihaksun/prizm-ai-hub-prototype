"""AI Hub / Object Storage 접근 공통 클라이언트.

dataset.py / model.py / code.py 는 전부 이 모듈을 통해서만
AI Hub(메타데이터 Resolve)와 MinIO(S3 호환 오브젝트 스토리지)에 접근한다.
사용자(ML 개발자)는 이 파일을 직접 쓸 일이 없다 - Asset Key + Version만 알면 된다.

Asset은 두 가지 형태를 지원한다.
  - "file": 파일 하나 (예: iris.csv, yolov8n.pt) - 압축 불필요.
  - "dir" : 폴더 통째로 (예: images/ + labels/ + data.yaml) - 압축(zip) 없이
            S3에 개별 오브젝트로 그대로 올라간다. Key+Version prefix 아래
            원래 폴더 구조 그대로 저장/복원된다.
"""
import os
import time

import boto3
import requests
from botocore.client import Config

AI_HUB_URL = os.environ.get("MLOPS_AI_HUB_URL", "http://ai-hub:8000")
S3_ENDPOINT = os.environ.get("MLOPS_S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.environ.get("MLOPS_S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("MLOPS_S3_SECRET_KEY", "minioadmin")
BUCKET = os.environ.get("MLOPS_BUCKET", "mlops-assets")


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ping(timeout=5):
    """AI Hub 연결/응답 확인. Preflight Check에서 사용."""
    r = requests.get(f"{AI_HUB_URL}/health", timeout=timeout)
    r.raise_for_status()
    return r.json()


def resolve(asset_type, key, version="latest", retries=3):
    """Logical Asset(type/key/version) -> Physical Asset(bucket/object_key) 변환."""
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{AI_HUB_URL}/assets/resolve",
                params={"type": asset_type, "key": key, "version": version},
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1)
    raise last_err


def register(asset_type, key, version, object_key, bucket=None, kind="file"):
    payload = {
        "type": asset_type,
        "key": key,
        "version": version,
        "bucket": bucket or BUCKET,
        "object_key": object_key,
        "kind": kind,
    }
    r = requests.post(f"{AI_HUB_URL}/assets/register", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def download_object(bucket, object_key, dest_path):
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    _s3().download_file(bucket, object_key, dest_path)
    return dest_path


def upload_object(local_path, object_key, bucket=None):
    bucket = bucket or BUCKET
    _s3().upload_file(local_path, bucket, object_key)
    return {"bucket": bucket, "object_key": object_key}


def list_objects(bucket, prefix):
    s3 = _s3()
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def upload_asset(asset_type, key, path, version=None):
    """path가 폴더면 압축 없이 파일 그대로 개별 업로드 (구조 보존),
    파일이면 단일 오브젝트로 업로드한다.
    """
    version = version or time.strftime("v%Y%m%d%H%M%S")
    base_prefix = f"{asset_type}/{key}/{version}"

    if os.path.isdir(path):
        n = 0
        for root, _dirs, files in os.walk(path):
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, path).replace(os.sep, "/")
                upload_object(full, f"{base_prefix}/{rel}")
                n += 1
        register(asset_type, key, version, base_prefix, kind="dir")
        print(f"[prizm.{asset_type}] uploaded {key}:{version} (dir, {n} files, no compression)")
    else:
        filename = os.path.basename(path)
        object_key = f"{base_prefix}/{filename}"
        upload_object(path, object_key)
        register(asset_type, key, version, object_key, kind="file")
        print(f"[prizm.{asset_type}] uploaded {key}:{version}")

    return {"key": key, "version": version}


def download_asset(asset_type, key, version="latest", dest_dir="./data"):
    """dest_dir: 원하는 로컬 다운로드 경로를 직접 지정할 수 있다.

    - kind="dir" 인 자산은 폴더 구조 그대로 dest_dir 밑에 복원되고, dest_dir 자체를 반환한다
      (그 안의 data.yaml 등을 바로 이어서 쓰면 된다. 압축 해제 단계가 필요 없다).
    - kind="file" 인 자산은 dest_dir 안에 원래 파일명으로 받아지고, 그 파일 경로를 반환한다.
    """
    info = resolve(asset_type, key, version)
    kind = info.get("kind", "file")
    bucket = info["bucket"]

    if kind == "dir":
        prefix = info["object_key"].rstrip("/") + "/"
        object_keys = list_objects(bucket, prefix)
        if not object_keys:
            raise FileNotFoundError(f"no objects found under s3://{bucket}/{prefix}")
        for object_key in object_keys:
            rel = object_key[len(prefix):]
            download_object(bucket, object_key, os.path.join(dest_dir, rel))
        print(
            f"[prizm.{asset_type}] {key}:{info['version']} "
            f"(dir, {len(object_keys)} files, 압축 없음) -> {dest_dir}"
        )
        return dest_dir

    filename = os.path.basename(info["object_key"])
    dest_path = os.path.join(dest_dir, filename)
    download_object(bucket, info["object_key"], dest_path)
    print(f"[prizm.{asset_type}] {key}:{info['version']} -> {dest_path}")
    return dest_path
