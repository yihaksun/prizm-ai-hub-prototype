"""Prizm AI Hub - 프로토타입.

Logical Asset (type/key/version) <-> Physical Asset (S3 bucket/object_key)
를 매핑하는 아주 얇은 레지스트리.

실제 제품에서는 여기에 Lineage/Permission/Owner 관리가 들어가지만,
이 프로토타입은 "경로 문제를 어떻게 제거하는가"를 보여주는 데
필요한 최소 기능(등록 / 조회)만 구현한다.
"""
import json
import os
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Prizm AI Hub (Prototype)")

CATALOG_PATH = os.environ.get("CATALOG_PATH", "/data/catalog.json")
_lock = threading.Lock()


def _load() -> dict:
    if not os.path.exists(CATALOG_PATH):
        return {}
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _save(catalog: dict) -> None:
    os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


class RegisterRequest(BaseModel):
    type: str
    key: str
    version: str
    bucket: str
    object_key: str
    kind: str = "file"  # "file" | "dir" - dir는 object_key를 prefix로 취급 (압축 없이 다건 저장)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assets/register")
def register(req: RegisterRequest):
    with _lock:
        catalog = _load()
        catalog.setdefault(req.type, {}).setdefault(req.key, {})[req.version] = {
            "bucket": req.bucket,
            "object_key": req.object_key,
            "kind": req.kind,
        }
        _save(catalog)
    return {"status": "registered", **req.model_dump()}


@app.get("/assets/resolve")
def resolve(type: str, key: str, version: str = "latest"):
    catalog = _load()
    versions = catalog.get(type, {}).get(key)
    if not versions:
        raise HTTPException(status_code=404, detail=f"asset not found: {type}/{key}")
    if version == "latest":
        version = sorted(versions.keys())[-1]
    if version not in versions:
        raise HTTPException(
            status_code=404, detail=f"version not found: {type}/{key}:{version}"
        )
    info = versions[version]
    return {"type": type, "key": key, "version": version, **info}


@app.get("/assets")
def list_assets():
    return _load()
