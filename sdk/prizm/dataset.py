"""ML 개발자가 Notebook에서 쓰는 Dataset Asset API.

물리 경로(S3 URI, 로컬 폴더, 사내 공유 드라이브)를 몰라도 된다.
Key + Version만 알면 된다. 압축(zip) 없이 폴더 그대로 저장/복원된다 -
images/, labels/, data.yaml 같은 구조가 있는 데이터셋도 그대로 오간다.

    dataset_dir = prizm.dataset.download("demo.door_defect", "v1", dest_dir="./datasets/door_defect")
    data_yaml = os.path.join(dataset_dir, "data.yaml")
"""
from . import _client


def download(key, version="latest", dest_dir="./data"):
    """dest_dir: 원하는 로컬 다운로드 경로를 직접 지정할 수 있다."""
    return _client.download_asset("dataset", key, version, dest_dir)


def upload(key, path, version=None):
    """데이터 파이프라인 / 관리자가 Dataset Asset을 등록할 때 사용.
    (일반 ML 개발자는 보통 이 함수를 직접 호출하지 않는다 - 데이터는
    이미 AI Hub에 등록되어 있고, download()만 쓰는 것이 정상 흐름이다.)

    path가 폴더면 압축 없이 그 구조 그대로 업로드된다.
    """
    return _client.upload_asset("dataset", key, path, version)
