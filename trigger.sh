#!/usr/bin/env bash
# Generic Notebook Executor DAG를 트리거하는 헬퍼.
#
# 사용법:
#   ./trigger.sh demo.iris_training
#   ./trigger.sh demo.wine_training
#   ./trigger.sh demo.door_defect_yolov8_training
#
# 세 개의 완전히 다른 학습 과제가 전부 같은 DAG로 실행된다는 것이
# 이 스크립트로 증명하려는 핵심 포인트다 (과제별 DAG 없음).
set -euo pipefail

CODE_KEY=${1:?"usage: ./trigger.sh <code_key> [code_version]"}
CODE_VERSION=${2:-latest}

echo "▶ mlops_notebook_executor 트리거: code_key=${CODE_KEY} code_version=${CODE_VERSION}"

docker compose exec -T airflow airflow dags unpause mlops_notebook_executor >/dev/null 2>&1 || true

docker compose exec -T airflow airflow dags trigger mlops_notebook_executor \
  --conf "{\"code_key\": \"${CODE_KEY}\", \"code_version\": \"${CODE_VERSION}\"}"

cat <<EOF

진행상황 확인:
  docker compose exec airflow airflow dags list-runs -d mlops_notebook_executor
  docker compose logs -f airflow

Web UI:
  Airflow : http://localhost:8080
            (admin 비밀번호: docker compose exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated
             또는            docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt)
  MLflow  : http://localhost:5050
  MinIO   : http://localhost:9001  (minioadmin / minioadmin)
EOF
