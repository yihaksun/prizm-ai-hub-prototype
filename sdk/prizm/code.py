"""Notebook(Code) Asset API.

ML 개발자가 직접 호출하기보다는, MLOps Core(Generic Airflow Executor)가
등록된 표준 Notebook을 실행하기 위해 내부적으로 사용한다.

    notebook_path = prizm.code.download("demo.door_defect_yolov8_training", "v1")
"""
from . import _client


def download(key, version="latest", dest_dir="./code"):
    return _client.download_asset("code", key, version, dest_dir)


def upload(key, path, version=None):
    return _client.upload_asset("code", key, path, version)
