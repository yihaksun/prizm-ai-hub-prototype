# MLOps Core 실행 환경 (Runtime Catalog 프로토타입 - CPU 단일 이미지)
FROM apache/airflow:2.10.2-python3.11

USER airflow

# torch/torchvision을 CPU 전용 wheel로 먼저, 같은 인덱스에서 함께 설치한다.
# - 그냥 두면 ultralytics가 끌어오는 기본 torch가 CUDA 런타임(수백 MB~1GB대
#   nvidia_* 패키지들)을 통째로 받는다 - CPU Runtime 이미지엔 다 낭비.
# - torch와 torchvision을 서로 다른 소스(CPU/CUDA)에서 따로 설치하면 ABI가
#   어긋나 "operator torchvision::nms does not exist" 같은 에러가 난다.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt \
    && python -m ipykernel install --user --name python3 --display-name python3

# MLOps SDK 설치 (사용자 Notebook과 Airflow Generic Executor가 동일하게 사용)
COPY --chown=airflow:root sdk /opt/prizm-sdk
RUN pip install --no-cache-dir -e /opt/prizm-sdk

# ultralytics -> opencv-python(GUI 빌드) 가 요구하는 X11 공유 라이브러리
# (slim 베이스 이미지엔 기본으로 없음)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow
