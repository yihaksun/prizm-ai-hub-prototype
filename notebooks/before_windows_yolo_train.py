"""
door_defect_train.py
문 표면 결함(Door Defect) 탐지 모델 학습 스크립트

작성 환경: Windows 11 / 로컬 Anaconda / RTX 3060
상태     : 본인 PC에서는 정상 동작, 실행/평가 결과 확인 완료

──────────────────────────────────────────────────────────────
[이 파일이 실제로 ML 개발자가 MLOps 팀에 "이거 재학습되게 해주세요"
 라며 전달하는 코드의 전형적인 모습이다.]

로컬에서는 문제없이 돌아가지만, 이 코드를 그대로 MLOps 실행환경
(Linux 컨테이너 / Airflow / A100 GPU 클러스터)에 올리면 아래 지점들에서
전부 깨진다. 지금까지는 이걸 MLOps 엔지니어가 한 줄씩 수작업으로
고쳐주고 있었고, 그게 바로 "사람이 Translation Layer가 되어 있다"는
문제의 실체다.
──────────────────────────────────────────────────────────────
"""

import json
import os
import time

from ultralytics import YOLO

# ────────────────────────────────────────────────────────────
# [문제 1] 데이터 경로 - 로컬 PC 전용 절대경로 (Windows 드라이브 문자)
#          → 이 폴더는 작성자 PC에만 존재. Linux 컨테이너엔 당연히 없음.
DATA_ROOT = r"C:\Users\jwlee\Desktop\door_project\dataset"
DATA_YAML = os.path.join(DATA_ROOT, "data.yaml")

# [문제 2] 사전학습 가중치 - 개인 PC에 미리 받아둔 로컬 파일
#          → 서버/다른 PC엔 이 파일이 없어서 매번 수동으로 복사해줘야 함.
BASE_WEIGHT = r"C:\Users\jwlee\Downloads\yolov8n.pt"

# [문제 3] 결과 저장 경로 - 그냥 로컬 바탕화면
#          → MLOps가 학습 결과물을 어디서 가져와야 하는지 알 방법이 없음.
OUTPUT_DIR = r"C:\Users\jwlee\Desktop\door_project\runs"
RUN_NAME = "door_defect_v1"
LOG_PATH = os.path.join(OUTPUT_DIR, RUN_NAME, "train_log.txt")

# [문제 4] GPU 하드코딩 - 본인 PC(RTX 3060, GPU 1개) 기준 설정
#          → 학습 서버는 A100 멀티 GPU + 다른 CUDA/Driver 버전.
#            이 값 그대로 두면 자원 배정과 충돌하거나 무시됨.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 0

# [문제 5] 사내 공유 드라이브 의존 - 로컬 PC에서만 매핑된 네트워크 드라이브
#          → 서버 컨테이너에는 Z: 드라이브 자체가 존재하지 않음.
LABEL_MAP_PATH = r"Z:\공유\비전팀\door_defect\label_map.json"
with open(LABEL_MAP_PATH, encoding="utf-8") as f:
    LABEL_MAP = json.load(f)  # {"0": "scratch", "1": "dent", "2": "paint_peel", ...}

# ────────────────────────────────────────────────────────────
# 학습/증강 하이퍼파라미터 - 실험하면서 여기저기 값 바꿔가며 하드코딩됨.
# 재학습 때마다 이 블록 전체를 다시 손으로 맞춰야 함.
EPOCHS = 50
IMGSZ = 640
BATCH = 16
WORKERS = 8
SEED = 0

OPTIMIZER = "SGD"
LR0 = 0.01
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
COS_LR = False
PATIENCE = 50
CLOSE_MOSAIC = 10
LABEL_SMOOTHING = 0.0
SAVE_PERIOD = -1

# 증강(augmentation) 파라미터 - 데이터셋 바뀔 때마다 재조정
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 0.0
TRANSLATE = 0.1
SCALE = 0.5
SHEAR = 0.0
PERSPECTIVE = 0.0
FLIPUD = 0.0
FLIPLR = 0.5
MOSAIC = 1.0
MIXUP = 0.0

# 평가 임계값
CONF_THRES = 0.25
IOU_THRES = 0.7


def main():
    model = YOLO(BASE_WEIGHT)  # 로컬 파일을 직접 로드

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        seed=SEED,
        device=DEVICE,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        cos_lr=COS_LR,
        patience=PATIENCE,
        close_mosaic=CLOSE_MOSAIC,
        label_smoothing=LABEL_SMOOTHING,
        save_period=SAVE_PERIOD,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        project=OUTPUT_DIR,
        name=RUN_NAME,
    )

    metrics = model.val(conf=CONF_THRES, iou=IOU_THRES)

    # 결과를 그냥 로컬 텍스트 파일에 손으로 남김 (버전 관리도, 조회 UI도 없음)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} mAP50={metrics.box.map50:.4f}\n")

    # 학습 결과는 로컬 바탕화면 runs 폴더에 그대로 남는다.
    # 이후 "이 모델 파일 좀 서버에 올려주세요" 라고 메신저로
    # best.pt를 직접 전달하는 방식으로 등록이 이루어짐 (버전 관리 없음).
    print(rf"학습 완료. 결과: {OUTPUT_DIR}\{RUN_NAME}\weights\best.pt")


if __name__ == "__main__":
    main()
