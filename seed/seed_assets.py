"""AI Hub 초기 자산 시딩 스크립트.

실제 제품에서는 데이터 파이프라인/관리자가 하는 일을,
프로토타입 데모에서는 이 스크립트가 한 번 대신 수행한다.

등록하는 자산:
  - dataset: demo.iris, demo.wine        (표준 Notebook 1, 2용 - sklearn 내장 데이터)
  - dataset: demo.door_defect            (표준 Notebook 3 - YOLOv8용, coco8 예제로 대체)
  - model  : demo.yolov8n_base           (YOLOv8 사전학습 가중치)
  - code   : demo.iris_training / demo.wine_training / demo.door_defect_yolov8_training
"""
import os
import shutil
import sys
import time
from pathlib import Path

import requests
import yaml

sys.path.insert(0, "/opt/prizm-sdk")
import prizm  # noqa: E402
from prizm import _client  # noqa: E402

NOTEBOOKS_DIR = Path("/opt/seed/notebooks")


def wait_for_ai_hub(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{_client.AI_HUB_URL}/health", timeout=3)
            if r.status_code == 200:
                print("[seed] AI Hub ready")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("AI Hub not reachable within timeout")


def seed_tabular_dataset(name):
    from sklearn.datasets import load_iris, load_wine

    loader = {"iris": load_iris, "wine": load_wine}[name]
    data = loader(as_frame=True)
    df = data.frame
    path = f"/tmp/{name}.csv"
    df.to_csv(path, index=False)
    prizm.dataset.upload(f"demo.{name}", path, version="v1")


def seed_door_defect_dataset():
    """coco8(초소형 예제 데이터셋)을 door_defect 데모 자산으로 재포장한다.

    실제 공정 이미지가 아니라 대체 데이터이지만, 구조(images/labels/data.yaml)는
    표준 Notebook이 기대하는 형태 그대로다.
    """
    # ultralytics는 최초 import 시점의 cwd를 기준으로 기본 datasets_dir 을
    # 계산해서 내부 상수(ultralytics.utils.DATASETS_DIR)로 고정해버린다
    # (이후 settings.update()로 값을 바꿔도 이 상수엔 반영되지 않는다).
    # ./seed 가 읽기 전용 마운트이므로, ultralytics를 import하기 전에
    # 먼저 쓰기 가능한 /tmp 로 cwd를 옮겨서 그 기본값 자체를 쓰기 가능하게 만든다.
    cwd = os.getcwd()
    os.chdir("/tmp")
    try:
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.utils import DATASETS_DIR

        print("[seed] downloading coco8 sample dataset (one-time, seed-side only)...")
        check_det_dataset("coco8.yaml")
        datasets_root = Path(DATASETS_DIR)
    finally:
        os.chdir(cwd)

    src_dir = datasets_root / "coco8"
    if not src_dir.exists():
        raise RuntimeError(f"coco8 dataset not found at expected path: {src_dir}")

    bundle_root = Path("/tmp/door_defect_bundle")
    bundle_dir = bundle_root / "door_defect"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    shutil.copytree(src_dir / "images", bundle_dir / "images")
    shutil.copytree(src_dir / "labels", bundle_dir / "labels")

    import ultralytics

    cfg_path = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco8.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["path"] = "."
    cfg.pop("download", None)
    (bundle_dir / "data.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )

    # 압축(zip) 없이 폴더 그대로 업로드한다 - S3에는 images/labels/data.yaml이
    # 개별 오브젝트로 그대로 올라가고, prizm.dataset.download()가 같은 구조로 복원한다.
    prizm.dataset.upload("demo.door_defect", str(bundle_dir), version="v1")


def seed_yolov8_base_model():
    from ultralytics import YOLO

    cwd = os.getcwd()
    os.chdir("/tmp")
    try:
        YOLO("yolov8n.pt")  # 없으면 자동 다운로드 (seed 단계에서만 인터넷 사용)
    finally:
        os.chdir(cwd)
    weight_path = "/tmp/yolov8n.pt"
    if not os.path.exists(weight_path):
        raise RuntimeError("yolov8n.pt was not downloaded to /tmp")
    prizm.model.upload("demo.yolov8n_base", weight_path, version="v1")


def seed_notebook(code_key, filename):
    prizm.code.upload(code_key, str(NOTEBOOKS_DIR / filename), version="v1")


if __name__ == "__main__":
    wait_for_ai_hub()

    print("== dataset assets ==")
    seed_tabular_dataset("iris")
    seed_tabular_dataset("wine")
    seed_door_defect_dataset()

    print("== model assets ==")
    seed_yolov8_base_model()

    print("== code (notebook) assets ==")
    seed_notebook("demo.iris_training", "demo_iris_training.ipynb")
    seed_notebook("demo.wine_training", "demo_wine_training.ipynb")
    seed_notebook("demo.door_defect_yolov8_training", "demo_yolov8_training.ipynb")

    # "라이브 비교" 데모용 - Rule Validator를 거치지 않고 강제로 등록한다.
    # (Windows 로컬 코드를 수정 없이 그대로 실행하면 실제로 어떻게 실패하는지 보여주는 용도)
    seed_notebook("demo._live_before_asis", "before_asis_door_defect.ipynb")

    print("[seed] all assets registered.")
