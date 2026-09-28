# PRIZM 베이스 런타임 이미지 v1 (CPU)
#
# Airflow를 포함하지 않는다. 노트북은 Airflow 워커 프로세스가 아니라
# 이 이미지로 만든 파생 이미지의 컨테이너 안에서 실행된다.
#
# 파이썬 환경을 둘로 나눈다:
#   - 사용자 환경(site-packages): ML 스택 + ipykernel.
#     사용자가 코드 자산에 추가한 requirements가 여기 설치된다.
#   - 도구 venv(/opt/prizm/tools): papermill, jupyterlab, jupyter-server.
#     사용자가 무엇을 설치하든 편집·실행 도구와 충돌하지 않는다.
FROM python:3.11-slim

# ultralytics -> opencv-python(GUI 빌드)이 요구하는 X11 공유 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 파생 이미지 빌드가 uv로 사용자 requirements를 설치한다
RUN pip install --no-cache-dir uv==0.4.20

# ── 사용자 환경 ─────────────────────────────────────────────
# torch/torchvision은 CPU 전용 wheel로 같은 인덱스에서 함께 설치한다.
# 서로 다른 소스(CPU/CUDA)에서 따로 설치하면 ABI가 어긋나
# "operator torchvision::nms does not exist" 같은 에러가 난다.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY runtime/base/requirements-user.txt /tmp/requirements-user.txt
RUN pip install --no-cache-dir -r /tmp/requirements-user.txt \
    && python -m ipykernel install --name python3 --display-name python3

# MLOps SDK (사용자 Notebook이 mlops.data/model 로 자산을 주고받는다)
COPY sdk /opt/prizm-sdk
RUN pip install --no-cache-dir /opt/prizm-sdk

# ── 도구 venv ───────────────────────────────────────────────
RUN python -m venv /opt/prizm/tools \
    && /opt/prizm/tools/bin/pip install --no-cache-dir \
         papermill==2.6.0 \
         jupyterlab==4.2.5 \
         jupyter-server==2.14.2

# 도구 venv에서 실행된 papermill이 사용자 환경의 커널스펙을 찾을 수 있어야 한다
ENV JUPYTER_PATH=/usr/local/share/jupyter
ENV PATH=$PATH:/opt/prizm/tools/bin
