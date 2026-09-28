# MLOps 코어모듈 프로토타입 — Notebook Only 전략 Walking Skeleton

스크럼에서 정리한 전략을 "설명"이 아니라 "동작하는 것"으로 보여주기 위한 최소 프로토타입.
전체 아키텍처를 다 만들지 않고, 지금 팀에서 논쟁 중인 **핵심 주장 3개만** 실제로 동작시킨다.

1. **물리 경로 → Asset Key+Version으로 바꾸면 Windows→Linux 전환 문제가 구조적으로 사라진다** (AI Hub + SDK)
2. **Notebook Only + Generic DAG면 과제별 개발 없이 Scale-out 된다** (한 DAG로 완전히 다른 과제 3개 실행)
3. **Agent 없이 정적 Rule Validator만으로 충분하고, 문제를 고치는 게 실제로 쉽다** (Portal UI에서 실시간 검증)

**Portal UI (http://localhost:8090) 하나로 전체 시나리오를 클릭만으로 시연할 수 있다** — 터미널 명령이 필요 없다.

## 데모 시나리오 (Portal UI 기준)

가장 임팩트 있는 순서는 이렇다.

1. Portal 최상단 **"🎯 무엇이 사라지는가"** — before/after 코드가 나란히 보임. 이건 정적 설명.
2. 바로 아래 **"⚡ 라이브 비교"** — **여기가 핵심.** 두 버튼을 동시에 누른다.
   - 왼쪽 "그대로 실행": `before_windows_yolo_train.py`를 **한 글자도 안 고치고** Runtime Container에서
     실행 → 실제로 `FileNotFoundError: ... 'C:\Users\jwlee\Downloads\yolov8n.pt'`로 **진짜 크래시**한다
     (가짜 트레이스백 아님, Airflow 로그를 그대로 긁어옴).
   - 오른쪽 "표준 실행": 같은 코드를 표준 Notebook으로 바꾼 버전 → 파이프라인 4단계가 전부 초록으로
     채워지고 `🎉 성공 · mAP50=0.623`로 끝남.
   - **같은 화면, 같은 순간에 하나는 죽고 하나는 성공하는 대비**가 이 데모의 와우 포인트다.
3. Portal의 **① Notebook 등록** 카드에 `notebooks/broken_door_defect_yolov8_training.ipynb` 업로드
   → **Rule Validator가 문제 지점을 Cell/Line 단위로, 실제 코드 줄에 밑줄을 그어서** 지적하며 등록을
   막는다 (`BLOCKED`). 파일 선택만 해도 자동 검증됨.
4. 같은 카드에 `notebooks/demo_yolov8_training.ipynb`(고친 버전)를 다시 업로드 → 전부 ✅, `PASS` →
   "등록하기" 클릭 한 번으로 AI Hub에 등록됨. **"고치면 바로 통과한다"는 게 핵심 메시지.**
5. **② Dataset/Model 자산** 카드에서 YOLOv8n 베이스 가중치가 이미 허브에 등록되어 있는 걸 확인 (사용자가
   매번 준비할 필요 없음)
6. **📄 전체 코드 비교** 카드 — 요약이 아니라 실제 파일 전체. `Before (.py)` 탭은 146줄짜리 단일 파일
   (경로/하이퍼파라미터/증강설정이 뒤섞임), `After (Notebook)` 탭은 같은 내용을 7개 셀로 정리한 것.
   **줄어든 건 물리 경로뿐이고, 하이퍼파라미터/증강설정 38개는 양쪽에 전부 그대로 살아있다** — Notebook이
   토이 예제처럼 짧아 보이지 않도록, 실제 YOLO 학습 스크립트 수준의 파라미터를 다 넣었다.
7. **③ 실행** 카드에서 iris / wine / door_defect(YOLOv8) 를 순서대로 Run — **완전히 다른 3개 과제가
   같은 DAG(preflight → download → execute → 결과등록)로 실행**되는 걸 파이프라인 애니메이션 + 실시간
   경과시간으로 보여줌
8. **⚙️ 실행 전 파라미터 조정** — Run 버튼 위 "파라미터 조정" 토글을 열면 해당 Notebook의 papermill
   parameters 셀에서 자동 추출한 값(door_defect_yolov8_training은 38개: epochs/imgsz/batch/lr0/
   optimizer/augmentation 등)이 입력폼으로 뜬다. 값을 바꾸고 Run하면 **그 값 그대로 학습에 반영되고**,
   완료 배너에 "적용된 파라미터: imgsz=256" 처럼 실제 오버라이드 내역이 표시된다.
9. **④ 최근 실행 이력** 카드에서 accuracy / mAP50 메트릭과 등록된 Asset 버전이 자동으로 쌓이는 것 확인

> `before_asis_door_defect.ipynb`는 `before_windows_yolo_train.py`를 **정말로 한 글자도 바꾸지 않고**
> (같은 파일을 읽어와서 생성) 옮긴 것이다. Validator를 거치지 않고 seed 단계에서 직접 등록해뒀다
> (`demo._live_before_asis`, "실행" 드롭다운에는 안 보이도록 필터링됨) — "라이브 비교"의 왼쪽 버튼이
> 이걸 실행한다.

## 구성 요소 (의도적으로 최소화)

| 구성요소 | 여기서 실제로 동작하는 것 | 데모에서 생략한 것 |
|---|---|---|
| AI Hub | Key→S3 경로 매핑하는 얇은 FastAPI + MinIO | Lineage, 세밀한 ACL |
| SDK | `mlops.dataset/model/code .download()/.upload()` | — |
| **Portal** | 자산 등록/검증/실행 트리거를 위한 단일 웹 UI | 사용자별 권한, 인증 |
| **Rule Validator** | Windows/로컬 경로, git clone, shell 실행, GPU 하드코딩, SDK 미사용을 정적 규칙으로 검사 | Agent 기반 자동 수정 |
| Runtime | Docker 이미지 1개(CPU) | Python 버전별 다중 이미지, GPU/A100 |
| Airflow | Generic DAG 1개 (papermill로 notebook 실행) | 과제별 DAG, K8s/GPU 실연동, Celery |
| Preflight | AI Hub 연결 + Code Asset 존재 확인 정도의 가벼운 체크 | 별도 서비스화 |
| MLflow | 오픈소스 그대로 (Run/Metric/Artifact) | Model UI 래핑 |

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

`dest_dir`는 항상 호출부에서 지정한다 - SDK가 임의로 경로를 정하지 않는다. `sdk/mlops/_client.py`의
`upload_asset()` / `download_asset()`가 파일/폴더 여부(`os.path.isdir`)를 보고 자동으로 분기하고,
`dataset.py` / `model.py` / `code.py`는 전부 이 공통 로직에 위임하는 얇은 wrapper다.

## 아키텍처

```text
before_windows_yolo_train.py (Windows 로컬)
        │  표준 계약으로 전환
        ▼
broken_door_defect_yolov8_training.ipynb  ──▶  Portal ①Notebook 등록
        │  Rule Validator: BLOCKED (5 errors)          │
        │  고쳐서 재업로드                                │  POST /api/validate
        ▼                                                │  POST /api/register-code
demo_yolov8_training.ipynb  ──▶  PASS  ──▶  mlops.code.upload() ──▶ AI Hub ──▶ MinIO(S3)

        Portal ③실행 트리거 "Run" 클릭
                        │  POST /api/trigger (Airflow REST API, basic auth)
                        ▼
        ┌───────────────────────────────┐
        │  mlops_notebook_executor DAG  │   (모든 과제가 같은 DAG)
        │  1) preflight_check           │
        │  2) download_notebook  ───────┼──▶ mlops.code.download()
        │  3) execute_notebook   ───────┼──▶ papermill 실행
        └───────────────┬───────────────┘
                         │  (Notebook 내부에서)
                         ▼
        mlops.dataset.download() / mlops.model.download()
                         │
                         ▼
                    AI Hub ──▶ MinIO(S3)
                         │
                    학습 실행
                         │
                         ▼
        mlops.model.upload() + mlflow.log_metric/artifact
                         │
                         ▼
        Portal ④최근 실행 이력  ◀── GET /api/history ── MLflow
```

## 실행 방법 (로컬 Docker Compose)

이 저장소는 두 가지 모드로 쓸 수 있다 — **독립 데모**(이 문서의 원래 시나리오, 전체 스택을
혼자서 다 띄움)와 **prizm-backend 통합**(오픈소스 인프라는 prizm-backend의
`infra/docker-compose.opensource.yml`이 맡고, 여기는 PRIZM 고유 컴포넌트인 AI Hub만 얹음).
`profiles`로 나뉘어 있어서 기본값(`docker compose up`, profile 미지정)은 **AI Hub만** 뜬다.

### 독립 데모 (원래 3가지 핵심 주장을 혼자서 시연)

```bash
cd prototype

# 1) 이미지 빌드 + 전체 서비스(MinIO/AI Hub/MLflow/Airflow/Portal) 기동
docker compose --profile standalone up -d --build

# 2) Airflow 뜰 때까지 대기 (최초 기동은 DB 초기화 때문에 1~2분 소요)
docker compose logs -f airflow   # 로그가 잠잠해지면 Ctrl+C

# 3) 자산 시딩 (iris/wine 데이터셋, door_defect 데이터셋, yolov8n 베이스 가중치, notebook 3개 등록)
docker compose --profile seed run --rm seed

# 4) Portal 열기 — 여기서부터는 전부 클릭으로 진행
open http://localhost:8090
```

Portal 없이 터미널로 트리거하고 싶다면 `./trigger.sh demo.iris_training` 같은 식으로도 가능 (CLI 대안, `airflow dags trigger` 래핑).

결과 확인:

```bash
open http://localhost:8090   # Portal - 등록/검증/트리거/이력 전부
open http://localhost:8080   # Airflow - DAG 실행 로그 (admin/admin)
open http://localhost:5050   # MLflow - accuracy/mAP50 메트릭, 등록된 모델 아티팩트
open http://localhost:9001   # MinIO Console - 실제 저장된 오브젝트 (minioadmin/minioadmin)
```

종료/초기화:

```bash
docker compose --profile standalone down -v   # 컨테이너 + 볼륨(데이터) 전부 삭제, 처음부터 다시 데모 가능
```

### prizm-backend 통합 (오픈소스 인프라는 backend 쪽이 책임짐)

`prizm-backend/infra/docker-compose.opensource.yml`이 Postgres 백엔드 Airflow + MLflow + MinIO를
먼저 띄운 상태여야 한다 (그쪽 `infra/README.md` 참고). 그 스택이 `name: prototype`으로 프로젝트
이름을 고정해두기 때문에, 이 저장소도 같은 이름(`name: prototype`, 최상단에 명시)을 쓰고 있어서
자동으로 같은 `prototype_default` 네트워크를 공유한다 — 컨테이너 이름(`ai-hub`, `minio` 등)으로
서로를 바로 찾는다.

```bash
cd prototype
docker compose up -d --build   # profile 없음 → ai-hub만 뜬다 (minio/mlflow/airflow/portal은 안 뜸)
```

두 스택을 동시에 `standalone` profile로 전부 띄우면 Airflow(8080)/MLflow(5050)/MinIO(9000-9001)
포트가 backend 쪽과 충돌하니, 이 모드에서는 `--profile standalone`을 절대 쓰지 않는다.

## 외부 컨테이너를 이 스택에 연동할 때 (PRIZM 편집 세션 등)

`prizm-backend`의 편집 세션 기능(코드 자산을 JupyterLab로 여는 기능)은 이 `docker-compose.yml`
바깥에서 `docker run`으로 JupyterLab 컨테이너를 직접 띄운다. `airflow` 서비스처럼 compose가 관리하는
컨테이너가 아니므로, **compose 파일에 없어도 아래 요구사항은 그대로 적용된다**:

- `--network prototype_default` 로 이 스택의 네트워크에 join해야 한다. 이 파일과
  `prizm-backend/infra/docker-compose.opensource.yml` 둘 다 최상단에 `name: prototype`을
  명시해서 프로젝트 이름을 고정해뒀고, Compose는 그 이름 기준으로 `prototype_default` 네트워크를
  만든다 (디렉터리 이름이 뭐든 상관없다 - 예전엔 디렉터리명에 의존했지만 지금은 아니다).
  `ai-hub`/`minio` 같은 컨테이너 DNS 이름은 이 네트워크 안에서만 풀린다. 안 붙이면 컨테이너 안에서
  `localhost:8000`/`localhost:9000`이 자기 자신을 가리켜서 AI Hub/MinIO 연결이 조용히 실패한다 —
  이미 `airflow` 서비스는 `MLOPS_DOCKER_NETWORK: prototype_default` 환경변수로 이 값을 받아 쓰고
  있지만, compose 밖에서 별도로 `docker run`을 호출하는 쪽(PRIZM 백엔드 등)은 이 플래그를 직접
  넘겨야 한다.
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

## 코드 구조

```text
prototype/
├── sdk/                   # mlops SDK (dataset/model/code 3종 Asset API)
├── ai-hub/                 # Asset Key+Version -> S3 경로 Resolve (FastAPI)
├── mlflow/                 # MLflow 서버 (오픈소스 그대로)
├── portal/                 # ⭐ 시연용 웹 UI
│   ├── app.py               # 자산 조회/등록, Airflow REST API 트리거(+파라미터 오버라이드), 전체 코드 비교, MLflow 이력 조회
│   └── rule_validator.py    # ⭐ 정적 Rule Validator (Windows/로컬 경로, git clone, GPU 하드코딩 등)
├── airflow/
│   ├── dags/mlops_notebook_executor.py   # ⭐ Generic Notebook Executor DAG
│   └── Dockerfile                        # Runtime Catalog 프로토타입 (CPU 이미지 1개)
├── notebooks/
│   ├── before_windows_yolo_train.py                # ⭐ "이게 지금 우리 문제다" (원본 .py)
│   ├── before_asis_door_defect.ipynb               # ⭐ 위 코드를 한 글자도 안 고치고 옮긴 것 (라이브 비교 왼쪽)
│   ├── broken_door_defect_yolov8_training.ipynb    # ⭐ Validator가 막아야 하는 버전
│   ├── demo_yolov8_training.ipynb                  # ⭐ 전환된 표준 Notebook (통과 버전, 라이브 비교 오른쪽)
│   ├── demo_iris_training.ipynb                    # Generic DAG 재사용 증명용 #2
│   └── demo_wine_training.ipynb                    # Generic DAG 재사용 증명용 #3
├── seed/seed_assets.py    # 초기 자산 등록 (데모용 1회성 스크립트)
├── trigger.sh             # DAG 실행 CLI 대안
└── docker-compose.yml
```

## Rule Validator가 잡는 것

| 규칙 | 예시 | 레벨 |
|---|---|---|
| Windows 절대경로 | `r"C:\Users\..."` | error |
| 로컬 Mac/Linux 절대경로 | `/Users/...`, `/home/...` | error |
| git clone 사용 | `!git clone ...` | error |
| 임의 shell 실행 | `os.system(...)`, `subprocess.run(...)` | error |
| GPU/CUDA 하드코딩 | `CUDA_VISIBLE_DEVICES`, `device=0` | error |
| MLOps SDK 미사용 | `import mlops` 없음 | error |
| Asset Key+Version 미사용 | `mlops.dataset/model.download()` 없음 | warning |

`broken_door_defect_yolov8_training.ipynb` → `demo_yolov8_training.ipynb`로 교체 업로드하면
5개 에러가 전부 사라지는 걸 Portal에서 바로 확인할 수 있다 — **"고치기 쉬움"이 실제로 보이는 부분**.

## 이 프로토타입이 증명하는 것 / 증명하지 않는 것

**증명하는 것**
- Windows 로컬 코드의 문제 지점들이 SDK 도입만으로 실제로 사라진다 (before/broken/fixed 3단 비교)
- Rule Validator가 문제를 라인 단위로 짚어주고, 고치면 즉시 재검증→통과된다 (Agent 없이도 충분)
- 완전히 다른 3개 과제(회귀/분류/객체탐지)가 DAG 코드 변경 없이 파라미터만으로 실행된다
- Preflight를 가볍게 유지해도(전체 다운로드/미니 학습 없이) 실행 전 최소 검증이 가능하다
- MLOps 엔지니어가 코드를 손대지 않고도 등록→검증→실행→결과 등록까지 UI 클릭만으로 끝난다

**증명하지 않는 것 (다음 단계 과제로 남김)**
- 실제 A100/K8s 환경에서의 GPU 스케줄링, Runtime Catalog 이미지 다중화
- AI Hub의 Lineage/권한(ACL)/Owner 관리
- MLflow를 감싸는 자체 Model UI
- Portal의 사용자 인증/권한 분리 (지금은 데모용 단일 화면, 로그인 없음)
