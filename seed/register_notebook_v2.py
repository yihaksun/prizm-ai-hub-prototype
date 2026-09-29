"""노트북 v2(PRIZM 실행 연동)를 AI Hub 코드 자산으로 등록한다 (1회성).

    docker compose --profile seed run --rm --entrypoint python seed register_notebook_v2.py

demo.door_defect 데이터셋과 demo.yolov8n_base 가중치는 seed_assets.py가 먼저 등록해 두어야 한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/opt/prizm-sdk")
import prizm  # noqa: E402
from seed_assets import wait_for_ai_hub  # noqa: E402

CODE_KEY = "demo.door_defect_yolov8_training"
NOTEBOOK = Path("/opt/seed/notebooks/demo_yolov8_training_v2.ipynb")

if __name__ == "__main__":
    wait_for_ai_hub()
    result = prizm.code.upload(CODE_KEY, str(NOTEBOOK), version="v2")
    print("[seed] registered notebook v2:", result)
