"""Generic Notebook Executor DAG.

핵심 주장: "과제별로 DAG를 새로 만들지 않는다."

이 DAG 하나가 AI Hub에 등록된 *어떤* 표준 Notebook이든 실행한다.
어떤 Notebook을 실행할지는 오직 DAG 실행 시점의 파라미터로 결정된다.

    airflow dags trigger mlops_notebook_executor --conf \
        '{"code_key": "demo.iris_training", "code_version": "v1"}'

    airflow dags trigger mlops_notebook_executor --conf \
        '{"code_key": "demo.wine_training", "code_version": "v1"}'

    airflow dags trigger mlops_notebook_executor --conf \
        '{"code_key": "demo.door_defect_yolov8_training", "code_version": "v1"}'

세 개의 완전히 다른 학습 과제(로지스틱 회귀 / 랜덤포레스트 / YOLOv8 탐지)가
전부 같은 DAG, 같은 Task 3개로 실행된다 - Airflow DAG를 새로 개발할 필요가 없다.

Preflight 범위는 의도적으로 가볍다: 실제 학습을 미리 돌리지 않는다.
"정식 Training Job을 시작할 최소 조건이 갖춰졌는가"만 확인한다.
"""
from __future__ import annotations

import datetime
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

AI_HUB_URL = os.environ.get("MLOPS_AI_HUB_URL", "http://ai-hub:8000")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
RUN_ROOT = "/opt/airflow/mlops_run"

# DockerOperator가 자식 컨테이너를 띄울 때 쓰는 값. 호스트 경로여야 한다 -
# 자식 컨테이너를 만드는 것은 Airflow 컨테이너가 아니라 호스트 도커 데몬이다.
HOST_RUN_ROOT = os.environ.get("MLOPS_HOST_RUN_ROOT", "")
DOCKER_NETWORK = os.environ.get("MLOPS_DOCKER_NETWORK", "prototype_default")
FALLBACK_IMAGE = os.environ.get("MLOPS_BASE_IMAGE", "prizm/runtime-base:v1")


def preflight_check(**context):
    """정식 실행 전 최소 조건만 확인한다.

    하지 않는 것: 전체 Dataset/Model 다운로드, DataLoader 실행, Mini Training.
    하는 것    : AI Hub 연결, Code Asset 존재 여부, mlops SDK 정상 로드.
    """
    conf = context["dag_run"].conf or {}
    code_key = conf.get("code_key")
    code_version = conf.get("code_version", "latest")
    if not code_key:
        raise ValueError("dag_run.conf.code_key 가 필요합니다 (등록된 Notebook Asset Key)")

    import prizm  # SDK 정상 동작 여부 자체가 Preflight의 일부

    health = prizm._client.ping()
    print(f"[preflight] AI Hub 상태: {health}")

    code_info = prizm._client.resolve("code", code_key, code_version)
    print(f"[preflight] Code Asset 확인됨: {code_info}")

    return code_info


def download_notebook(**context):
    conf = context["dag_run"].conf or {}
    code_key = conf["code_key"]
    code_version = conf.get("code_version", "latest")

    import prizm
    import yaml

    # PRIZM 백엔드가 트리거한 실행은 prizm_run_id 폴더를 쓴다 - 백엔드가 이 경로의
    # executed.ipynb를 읽어 화면에 실시간으로 보여준다. 없으면 기존 규칙(trigger.sh, 기존 포털).
    prizm_run_id = conf.get("prizm_run_id")
    run_folder = prizm_run_id or context["run_id"].replace(":", "_").replace("+", "_")
    run_dir = os.path.join(RUN_ROOT, run_folder)
    code_dir = os.path.join(run_dir, "code")
    output_dir = os.path.join(run_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    notebook_path = prizm.code.download(code_key, code_version, dest_dir=code_dir)

    # papermill 파라미터를 파일로 넘긴다. DockerOperator의 command를 Jinja 문자열로
    # 조립하는 것보다 안전하다 - 값에 공백이나 따옴표가 있어도 깨지지 않는다.
    extra_params = conf.get("extra_params") or {}
    base_params = {
        "output_dir": output_dir,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    }
    if prizm_run_id:
        base_params["prizm_run_id"] = prizm_run_id
    # base_params가 항상 우선한다 - output_dir/mlflow_tracking_uri는 사용자가
    # 실행 시점에 바꿀 수 없는 계약이다.
    parameters = {**extra_params, **base_params}
    if extra_params:
        print(f"[download] 사용자 지정 파라미터 오버라이드: {extra_params}")

    params_path = os.path.join(run_dir, "params.yaml")
    with open(params_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(parameters, handle, allow_unicode=True)

    ti = context["ti"]
    ti.xcom_push(key="notebook_path", value=notebook_path)
    ti.xcom_push(key="run_dir", value=run_dir)
    ti.xcom_push(key="params_path", value=params_path)
    ti.xcom_push(key="executed_path", value=os.path.join(run_dir, "executed.ipynb"))
    return notebook_path


default_args = {
    "owner": "mlops-core",
    "retries": 0,
}

with DAG(
    dag_id="mlops_notebook_executor",
    description=(
        "Generic Notebook Executor - AI Hub에 등록된 표준 Notebook을 "
        "code_key/code_version 파라미터로 실행한다. 과제별 DAG 없음."
    ),
    schedule=None,
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops-core", "generic-executor", "prototype"],
    params={"fallback_image": FALLBACK_IMAGE},
) as dag:
    t1 = PythonOperator(task_id="preflight_check", python_callable=preflight_check)
    t2 = PythonOperator(task_id="download_notebook", python_callable=download_notebook)
    # 노트북은 Airflow 워커가 아니라 자산의 파생 이미지 안에서 실행된다.
    # papermill은 그 이미지의 도구 venv(/opt/prizm/tools)에 있고 PATH에 올라와 있다.
    t3 = DockerOperator(
        task_id="execute_notebook",
        image="{{ dag_run.conf.get('env_image_ref') or params.fallback_image }}",
        command=[
            "papermill",
            "{{ ti.xcom_pull(task_ids='download_notebook', key='notebook_path') }}",
            "{{ ti.xcom_pull(task_ids='download_notebook', key='executed_path') }}",
            "-f", "{{ ti.xcom_pull(task_ids='download_notebook', key='params_path') }}",
            # 셀 시작·완료마다 저장 + 긴 셀 실행 중에도 3초마다 저장.
            # 백엔드가 이 파일을 읽어 실시간 노트북 화면을 만든다.
            "--request-save-on-cell-execute",
            "--autosave-cell-every", "3",
            "--log-output",
            "-k", "python3",
        ],
        mounts=[Mount(source=HOST_RUN_ROOT, target=RUN_ROOT, type="bind")],
        mount_tmp_dir=False,
        network_mode=DOCKER_NETWORK,
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_S3_ENDPOINT_URL": os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            "MLOPS_AI_HUB_URL": AI_HUB_URL,
            "MLOPS_S3_ENDPOINT": os.environ.get("MLOPS_S3_ENDPOINT", "http://minio:9000"),
            "MLOPS_S3_ACCESS_KEY": os.environ.get("MLOPS_S3_ACCESS_KEY", "minioadmin"),
            "MLOPS_S3_SECRET_KEY": os.environ.get("MLOPS_S3_SECRET_KEY", "minioadmin"),
            "MLOPS_BUCKET": os.environ.get("MLOPS_BUCKET", "mlops-assets"),
        },
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        api_version="auto",
    )

    t1 >> t2 >> t3
