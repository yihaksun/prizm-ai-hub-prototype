"""ML 개발자가 Notebook에서 쓰는 Model Asset API.

- 초기 가중치(pretrained weight)도 로컬 파일이 아니라 AI Hub에서 받는다.
- 학습 결과물도 로컬 폴더에 방치하지 않고 AI Hub(-> MLflow에서 조회 가능)에 등록한다.
- 폴더 단위 모델(예: 여러 파일로 구성된 체크포인트)도 압축 없이 그대로 오간다.

    base_weight = prizm.model.download("demo.yolov8n_base", "v1", dest_dir="./models")
    ...
    prizm.model.upload("demo.door_defect_yolov8", "./output/train/weights/best.pt")
"""
from . import _client


def download(key, version="latest", dest_dir="./models"):
    """dest_dir: 원하는 로컬 다운로드 경로를 직접 지정할 수 있다."""
    return _client.download_asset("model", key, version, dest_dir)


def upload(key, path, version=None):
    return _client.upload_asset("model", key, path, version)
