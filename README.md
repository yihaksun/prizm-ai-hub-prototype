# PRIZM AI Hub + SDK

`prizm-backend`가 의존하는 PRIZM 자체 구성요소 — **AI Hub**(Asset Key+Version → S3 경로
Resolve), **SDK**(`mlops.dataset/model/code`), **편집 세션 런타임 베이스 이미지**
(`prizm/runtime-base:v1`)를 담는다.

Postgres/MinIO/MLflow/Airflow 같은 오픈소스 인프라는 여기 없다 — 그건
`prizm-backend/infra/docker-compose.opensource.yml`이 책임진다. 이 저장소는 그 스택 위에
얹는 "나머지"만 담당한다.

> 원래 이 저장소는 "Notebook Only 전략"의 3가지 핵심 주장(물리경로 제거/Generic DAG/정적
> Rule Validator)을 혼자서 시연하는 독립 데모(Portal UI, 데모 Airflow/MLflow 포함)였다.
> `prizm-backend`가 오픈소스 인프라를 정식으로 갖추면서 그 역할은 옮겨갔고, 이 저장소는
> AI Hub/SDK/runtime-base 전용으로 좁혔다. 원래 데모 시나리오 전체는 git 히스토리
> (`4354961` 이전 커밋)에 남아있다.

## AI Hub가 하는 일

`ai-hub/app.py` — 81줄짜리 FastAPI. Logical Asset(type/key/version) ↔ Physical Asset(S3
bucket/object_key)을 매핑하는 아주 얇은 JSON 카탈로그.

- `POST /assets/register` — 자산 하나를 카탈로그에 등록
- `GET /assets/resolve?type=&key=&version=` — 실제 S3 경로 조회 (`version=latest` 지원)
- `GET /assets` — 전체 카탈로그 조회
- `GET /health` — 헬스체크

MinIO에 직접 접근하지 않는다 — S3 자체 업/다운로드는 SDK가 담당하고, AI Hub는 "어디 있는지"만
안다.

## Asset 저장 방식 (압축 없음)

S3에는 zip으로 뭉치지 않고 **원래 폴더 구조 그대로** 개별 오브젝트로 저장한다.
AI Hub 카탈로그가 자산마다 `kind: "file" | "dir"`를 함께 기록해서 SDK가 알아서 구분한다.

```python
# 파일 하나짜리 자산 (iris.csv, yolov8n.pt 등) - 그냥 파일 하나 업/다운로드
mlops.model.upload("demo.yolov8n_base", "./yolov8n.pt")
weight_path = mlops.model.download("demo.yolov8n_base", "v1", dest_dir="./models")

# 폴더짜리 자산 (images/ + labels/ + data.yaml) - 압축 없이 구조 그대로
mlops.dataset.upload("demo.door_defect", "./door_defect_dir")   # 폴더 전체 업로드
dataset_dir = mlops.dataset.download(
    "demo.door_defect", "v1",
    dest_dir="./datasets/door_defect",   # 원하는 로컬 경로를 직접 지정
)
```

`dest_dir`는 항상 호출부에서 지정한다 - SDK가 임의로 경로를 정하지 않는다. `sdk/prizm/_client.py`의
`upload_asset()` / `download_asset()`가 파일/폴더 여부(`os.path.isdir`)를 보고 자동으로 분기하고,
`dataset.py` / `model.py` / `code.py`는 전부 이 공통 로직에 위임하는 얇은 wrapper다.

## 실행 방법

`prizm-backend/infra/docker-compose.opensource.yml`이 Postgres 백엔드 Airflow + MLflow + MinIO를
먼저 띄운 상태여야 한다 (그쪽 `infra/README.md` 참고). 그 스택이 `name: prototype`으로 프로젝트
이름을 고정해두기 때문에, 이 저장소도 같은 이름(`name: prototype`, `docker-compose.yml` 최상단에
명시)을 쓰고 있어서 자동으로 같은 `prototype_default` 네트워크를 공유한다 — 컨테이너 이름
(`ai-hub`, `minio` 등)으로 서로를 바로 찾는다.

```bash
cd prototype
docker compose up -d --build   # ai-hub 기동
```

```bash
curl http://localhost:8000/health   # {"status":"ok"}
```

종료:

```bash
docker compose down -v
```

## 외부 컨테이너를 이 스택에 연동할 때 (PRIZM 편집 세션 등)

`prizm-backend`의 편집 세션 기능(코드 자산을 JupyterLab로 여는 기능)은 이 `docker-compose.yml`
바깥에서 `docker run`으로 JupyterLab 컨테이너를 직접 띄운다. compose가 관리하는 컨테이너가
아니므로, **compose 파일에 없어도 아래 요구사항은 그대로 적용된다**:

- `--network prototype_default` 로 이 스택의 네트워크에 join해야 한다. 이 파일과
  `prizm-backend/infra/docker-compose.opensource.yml` 둘 다 최상단에 `name: prototype`을
  명시해서 프로젝트 이름을 고정해뒀고, Compose는 그 이름 기준으로 `prototype_default` 네트워크를
  만든다 (디렉터리 이름이 뭐든 상관없다). `ai-hub`/`minio` 같은 컨테이너 DNS 이름은 이 네트워크
  안에서만 풀린다. 안 붙이면 컨테이너 안에서 `localhost:8000`/`localhost:9000`이 자기 자신을
  가리켜서 AI Hub/MinIO 연결이 조용히 실패한다 — `prizm-backend/infra`의 Airflow 서비스는
  `MLOPS_DOCKER_NETWORK: prototype_default` 환경변수로 이 값을 받아 쓰고 있지만, compose 밖에서
  별도로 `docker run`을 호출하는 쪽(PRIZM 백엔드 등)은 이 플래그를 직접 넘겨야 한다.
- 컨테이너를 root로 띄운다면(베이스 이미지에 non-root 유저가 없는 경우) JupyterLab에
  `--allow-root`가 필요하다.
- 리버스 프록시 경로 뒤에서 서빙한다면 `--ServerApp.base_url=/<prefix>/`를 반드시 붙여야 한다 —
  없으면 JupyterLab이 루트(`/`)에서 서빙된다고 가정하고 절대경로 asset/API 링크를 생성해서,
  프록시된 경로에서는 빈 화면만 뜬다.

실제 예시 (`EditSessionContainerLauncher`가 만드는 명령):

```bash
docker run -d --network prototype_default ... <이미지> \
  /opt/prizm/tools/bin/jupyter lab \
  --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --ServerApp.base_url=/edit/<sessionId>/
```

`<이미지>`(`prizm/runtime-base:v1`)는 compose가 직접 띄우진 않지만(`runtime-base` 서비스는
`profiles: ["build"]`로 기본 `up`에서 제외됨) 아래처럼 compose로 재현 가능하게 빌드된다:

```bash
docker compose --profile build build runtime-base
```

Docker Hub에도 미리 빌드해 올려뒀다 (amd64+arm64 멀티아키텍처):
`docker pull yihaksun/prizm-runtime-base:v1`

이 태그를 쓰려면 `prizm-backend/infra/.env`에 `MLOPS_BASE_IMAGE=yihaksun/prizm-runtime-base:v1`을
넣는다 (기본값은 로컬 빌드 태그인 `prizm/runtime-base:v1`).

## SDK를 Airflow 이미지에 설치하기

`prizm-backend/infra`의 Generic Notebook Executor DAG(`infra/airflow/dags/
mlops_notebook_executor.py`, 이 저장소 `airflow/dags/`에서 이관됨)는 preflight 단계에서
`import prizm`으로 AI Hub 연결을 확인한다. `infra/images/airflow.Dockerfile`이 이 저장소의
`sdk/`를 git+subdirectory로 pip install하고 있다 — SDK를 고치면 그 Dockerfile을 다시 빌드해야
반영된다 (`docker compose -f infra/docker-compose.opensource.yml build airflow`).

## 코드 구조

```text
prototype/
├── sdk/                    # mlops SDK (dataset/model/code 3종 Asset API)
├── ai-hub/                 # Asset Key+Version -> S3 경로 Resolve (FastAPI)
├── runtime/base/           # 편집 세션 런타임 베이스 이미지 (prizm/runtime-base:v1)
├── env-builds/             # 자산별 파생 빌드(FROM prizm/runtime-base:v1 + requirements.txt)
└── docker-compose.yml
```
