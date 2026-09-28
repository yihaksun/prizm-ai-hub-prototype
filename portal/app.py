"""MLOps Portal (프로토타입).

사용자가 실제로 클릭해서 볼 수 있는 단일 화면.

- Before/After 비교 (물리 경로가 왜/어떻게 사라지는지 한눈에)
- Notebook 등록 + Rule Validator (문제 지점을 라인 단위로 안내, 고치면 바로 통과)
- Dataset / Model 자산 등록 (이미 허브에 있는 모델도 조회 가능)
- 등록된 Notebook을 Generic Airflow DAG로 실행 + 파이프라인 시각화 + 실시간 경과 시간
- MLflow 기반 최근 실행 이력
"""
import json
import os
import time

import prizm
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from prizm import _client as prizm_client
from pydantic import BaseModel
from rule_validator import has_blocking_errors, validate_notebook

AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://airflow:8080")
AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "admin")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://mlflow:5000")
DAG_ID = "mlops_notebook_executor"
NOTEBOOKS_DIR = "/app/notebooks"

app = FastAPI(title="MLOps Portal (Prototype)")


def _airflow_auth():
    return (AIRFLOW_USER, AIRFLOW_PASSWORD)


def _save_upload(content: bytes, filename: str) -> str:
    path = f"/tmp/upload_{int(time.time() * 1000)}_{filename}"
    with open(path, "wb") as f:
        f.write(content)
    return path


@app.get("/api/assets")
def list_assets(type: str = "code"):
    r = requests.get(f"{prizm_client.AI_HUB_URL}/assets", timeout=10)
    r.raise_for_status()
    data = r.json().get(type, {})
    items = []
    for key, versions in data.items():
        for v in versions.keys():
            items.append({"key": key, "version": v})
    items.sort(key=lambda x: (x["key"], x["version"]))
    return items


@app.post("/api/assets/upload")
async def upload_asset(
    type: str = Form(...),
    key: str = Form(...),
    version: str = Form(...),
    file: UploadFile = File(...),
):
    if type not in ("dataset", "model"):
        raise HTTPException(400, "type must be 'dataset' or 'model'")
    path = _save_upload(await file.read(), file.filename)
    try:
        if type == "dataset":
            result = prizm.dataset.upload(key, path, version=version)
        else:
            result = prizm.model.upload(key, path, version=version)
    finally:
        os.remove(path)
    return result


@app.post("/api/validate")
async def validate(file: UploadFile = File(...)):
    content = await file.read()
    try:
        nb = json.loads(content)
    except Exception as e:
        raise HTTPException(400, f"유효한 .ipynb(JSON) 파일이 아닙니다: {e}")
    findings = validate_notebook(nb)
    return {"findings": findings, "can_register": not has_blocking_errors(findings)}


@app.post("/api/register-code")
async def register_code(
    key: str = Form(...),
    version: str = Form(...),
    file: UploadFile = File(...),
):
    content = await file.read()
    try:
        nb = json.loads(content)
    except Exception as e:
        raise HTTPException(400, f"유효한 .ipynb(JSON) 파일이 아닙니다: {e}")
    findings = validate_notebook(nb)
    if has_blocking_errors(findings):
        raise HTTPException(422, "검증 오류가 남아있어 등록할 수 없습니다.")
    path = _save_upload(content, file.filename)
    try:
        result = prizm.code.upload(key, path, version=version)
    finally:
        os.remove(path)
    return result


class TriggerReq(BaseModel):
    code_key: str
    code_version: str = "latest"
    params: dict = {}


@app.post("/api/trigger")
def trigger(req: TriggerReq):
    dag_run_id = f"portal__{req.code_key}__{int(time.time())}"
    conf = {"code_key": req.code_key, "code_version": req.code_version}
    if req.params:
        conf["extra_params"] = req.params
    payload = {"dag_run_id": dag_run_id, "conf": conf}
    r = requests.post(
        f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns",
        json=payload,
        auth=_airflow_auth(),
        timeout=15,
    )
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=r.text)
    return r.json()


@app.get("/api/run_status")
def run_status(run_id: str):
    r = requests.get(
        f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns/{run_id}",
        auth=_airflow_auth(),
        timeout=10,
    )
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=r.text)
    state = r.json().get("state")

    tasks = []
    ti = requests.get(
        f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns/{run_id}/taskInstances",
        auth=_airflow_auth(),
        timeout=10,
    )
    if ti.status_code < 300:
        tasks = [
            {"task_id": t["task_id"], "state": t["state"]}
            for t in ti.json().get("task_instances", [])
        ]
    return {"state": state, "tasks": tasks}


@app.get("/api/run_log")
def run_log(run_id: str, task_id: str = "execute_notebook", try_number: int = 1):
    """실패한 Task의 실제 Airflow 로그를 가져온다 (가짜 트레이스백이 아니라 진짜).

    표시용으로 뒤쪽 일부만 잘라서 돌려준다.
    """
    r = requests.get(
        f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_number}",
        auth=_airflow_auth(),
        headers={"Accept": "text/plain"},
        timeout=15,
    )
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=r.text)
    lines = r.text.splitlines()
    # "Traceback"이 있으면 그 지점부터, 없으면 뒤쪽 일부만
    trace_idx = next((i for i, l in enumerate(lines) if "Traceback" in l), None)
    excerpt = lines[trace_idx:] if trace_idx is not None else lines[-40:]
    return {"log": "\n".join(excerpt[-60:])}


@app.get("/api/history")
def history():
    exp = requests.get(
        f"{MLFLOW_URL}/api/2.0/mlflow/experiments/get-by-name",
        params={"experiment_name": "mlops-core-demo"},
        timeout=10,
    )
    if exp.status_code >= 300:
        return []
    exp_id = exp.json()["experiment"]["experiment_id"]
    r = requests.post(
        f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
        json={
            "experiment_ids": [exp_id],
            "max_results": 20,
            "order_by": ["attribute.start_time DESC"],
        },
        timeout=10,
    )
    if r.status_code >= 300:
        return []
    out = []
    for run in r.json().get("runs", []):
        info = run["info"]
        data = run.get("data", {})
        tags = {t["key"]: t["value"] for t in data.get("tags", [])}
        metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
        metric_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        out.append(
            {
                "run_name": info.get("run_name"),
                "status": info.get("status"),
                "metrics": metric_str,
                "asset": f"{tags.get('mlops_asset_key', '-')}:{tags.get('mlops_asset_version', '-')}",
            }
        )
    return out


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src


def _notebook_cells(nb: dict) -> list:
    return [
        {
            "type": c.get("cell_type"),
            "source": _cell_source(c),
            "tags": c.get("metadata", {}).get("tags", []),
        }
        for c in nb.get("cells", [])
    ]


@app.get("/api/compare-code")
def compare_code():
    """Before(.py 원본 전체)와 After(Notebook 셀 전체)를 통째로 보여주기 위한 데이터."""
    before_path = os.path.join(NOTEBOOKS_DIR, "before_windows_yolo_train.py")
    after_path = os.path.join(NOTEBOOKS_DIR, "demo_yolov8_training.ipynb")
    with open(before_path, encoding="utf-8") as f:
        before_src = f.read()
    with open(after_path, encoding="utf-8") as f:
        after_nb = json.load(f)
    return {
        "before_py": before_src,
        "before_lines": len(before_src.splitlines()),
        "after_cells": _notebook_cells(after_nb),
    }


def _extract_params(source: str) -> list:
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    params = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(value, (int, float, str, bool)):
                params.append(
                    {"name": node.targets[0].id, "default": value, "type": type(value).__name__}
                )
    return params


@app.get("/api/notebook-params")
def notebook_params(code_key: str, code_version: str = "latest"):
    """등록된 Notebook의 papermill parameters 셀을 읽어 실행 전 오버라이드 폼을 만든다."""
    path = prizm.code.download(code_key, code_version, dest_dir="/tmp/param_probe")
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    target = None
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code" and "parameters" in c.get("metadata", {}).get("tags", []):
            target = c
            break
    if target is None:
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        target = code_cells[0] if code_cells else None
    if target is None:
        return []
    return _extract_params(_cell_source(target))


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLOps Portal</title>
<style>
  :root{
    --bg:#f5f6fb; --card:#ffffff; --border:#e7e8f2; --text:#151827; --muted:#8890a3;
    --accent:#6d5efc; --accent-ink:#4b3fd6; --danger:#ef4444; --warn:#f59e0b; --ok:#22c55e;
    --code-bg:#0f1420; --code-line:#1c2436;
  }
  * { box-sizing:border-box; }
  html,body { margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }
  code,.mono,.diff-before,.diff-after,.finding-code,.pipe-timer { font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }

  /* ── Hero ───────────────────────────────────────────── */
  .hero { background: radial-gradient(1200px 400px at 15% -20%, #4c3ff0 0%, #1c1440 55%, #120c2e 100%); color:#fff; padding: 44px 28px 64px; }
  .hero-inner { max-width:1080px; margin:0 auto; }
  .hero h1 { margin:0 0 8px; font-size:27px; letter-spacing:-0.02em; }
  .hero h1 .accent { background:linear-gradient(90deg,#a78bfa,#f472b6); -webkit-background-clip:text; background-clip:text; color:transparent; }
  .hero p.tag { margin:0 0 28px; color:#c9c8f0; font-size:14px; max-width:660px; line-height:1.6; }
  .stats { display:flex; gap:14px; flex-wrap:wrap; }
  .stat { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14); border-radius:14px; padding:14px 20px; min-width:180px; }
  .stat .num { font-size:24px; font-weight:800; letter-spacing:-0.02em; }
  .stat .label { font-size:11.5px; color:#c9c8f0; margin-top:3px; }

  main { max-width:1080px; margin:-34px auto 60px; padding:0 24px; display:flex; flex-direction:column; gap:22px; position:relative; }

  .card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:26px 28px; box-shadow:0 10px 30px -16px rgba(30,20,80,.18); }
  .eyebrow { font-size:11px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--accent-ink); margin:0 0 6px; }
  .card h2 { margin:0 0 4px; font-size:18px; }
  .card .desc { margin:0 0 18px; font-size:13px; color:var(--muted); line-height:1.55; }

  /* ── Before/After diff ─────────────────────────────── */
  .diff-wrap { border-radius:12px; overflow:hidden; border:1px solid var(--code-line); }
  .diff-header, .diff-row { display:grid; grid-template-columns: 1fr 40px 1fr; }
  .diff-h { font-size:11px; font-weight:800; padding:9px 16px; background:#151b2c; color:#9aa4c4; text-transform:uppercase; letter-spacing:.06em; }
  .diff-before, .diff-after { font-size:12.5px; padding:13px 16px; background:var(--code-bg); line-height:1.7; white-space:pre-wrap; word-break:break-word; border-bottom:1px solid var(--code-line); }
  .diff-before { color:#f3a5a5; border-right:1px solid var(--code-line); }
  .diff-after { color:#8fe3ac; }
  .diff-after.removed { color:#7c88a8; font-style:italic; }
  .diff-arrow { display:flex; align-items:center; justify-content:center; background:var(--code-bg); color:#525a78; border-bottom:1px solid var(--code-line); font-size:16px; }
  .hl { background:rgba(239,68,68,.28); text-decoration:underline wavy rgba(239,68,68,.9); text-underline-offset:3px; border-radius:3px; padding:0 1px; }
  .hl-ok { background:rgba(34,197,94,.22); border-radius:3px; padding:0 1px; }
  .diff-note { grid-column:1/-1; background:#151b2c; color:#9aa4c4; font-size:11.5px; padding:9px 16px; border-bottom:1px solid var(--code-line); line-height:1.6; }
  .diff-note code { background:rgba(255,255,255,.08); padding:0 4px; border-radius:3px; }

  /* ── Full code compare (tabs) ──────────────────────── */
  .code-tabs { display:flex; gap:8px; margin-bottom:0; }
  .code-tab { background:#eef0f8; color:#5b6478; border:none; padding:10px 18px; border-radius:10px 10px 0 0; font-weight:800; font-size:13px; cursor:pointer; }
  .code-tab.active { background:var(--code-bg); color:#fff; }
  .code-view { background:var(--code-bg); border-radius:0 12px 12px 12px; padding:16px 18px; max-height:480px; overflow:auto; }
  .code-view pre { margin:0; color:#c9d1e6; font-size:11.5px; line-height:1.65; white-space:pre-wrap; word-break:break-word; font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  .nb-cell { margin-bottom:8px; border-radius:8px; overflow:hidden; }
  .nb-cell.markdown { background:#1a2136; padding:10px 14px; color:#9aa4c4; font-size:12px; }
  .nb-cell.code { background:#151b2c; }
  .nb-cell-in { color:#5f6b8c; font-size:10px; padding:6px 14px 0; font-weight:700; }
  .nb-cell.code pre { padding:4px 14px 12px; }

  /* ── Parameter override form ───────────────────────── */
  .params-toggle { cursor:pointer; font-size:12.5px; font-weight:800; color:var(--accent-ink); margin:12px 0 8px; user-select:none; display:inline-block; }
  .params-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(150px,1fr)); gap:10px; background:#f8f8fd; border:1px solid var(--border); border-radius:10px; padding:14px; margin-bottom:10px; max-height:280px; overflow:auto; }
  .param-field label { display:block; font-size:10px; color:#8890a3; font-weight:800; margin-bottom:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .param-field input { width:100%; padding:6px 8px; font-size:12px; border-radius:6px; border:1px solid #d7d9e6; }
  .param-field input.changed { border-color:var(--accent); background:#f5f3ff; font-weight:700; }

  /* ── Live compare (진짜 실행) ──────────────────────── */
  .live-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:6px; }
  .live-col { border-radius:12px; border:1px solid var(--border); padding:18px; display:flex; flex-direction:column; gap:12px; }
  .live-col.before { background:#fff8f8; border-color:#fde2e2; }
  .live-col.after { background:#f6fbf8; border-color:#d6f3e2; }
  .live-col-h { font-weight:800; font-size:14px; }
  .live-col-h.before { color:#991b1b; }
  .live-col-h.after { color:#166534; }
  .live-btn { align-self:flex-start; }
  .live-btn.before-btn { background:#dc2626; }
  .mini-pipe { display:flex; align-items:flex-start; gap:0; }
  .mini-step { display:flex; flex-direction:column; align-items:center; gap:5px; width:78px; }
  .mini-dot { width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:12px; border:2px solid #dfe1ee; color:#a3a9bd; background:#fff; }
  .mini-dot.running { border-color:var(--accent); color:var(--accent); animation:pulse 1.1s infinite; }
  .mini-dot.done { border-color:var(--ok); background:var(--ok); color:#fff; }
  .mini-dot.failed { border-color:var(--danger); background:var(--danger); color:#fff; }
  .mini-label { font-size:10px; color:#5b6478; font-weight:700; text-align:center; }
  .mini-line { flex:1; height:2px; background:#e3e5f1; margin-top:14px; }
  .mini-timer { font-size:18px; font-weight:800; }
  .live-result { padding:10px 13px; border-radius:9px; font-weight:700; font-size:13px; }
  .live-result.ok { background:#f0fdf4; border:1px solid #bbf7d0; color:#166534; }
  .live-result.bad { background:#fef2f2; border:1px solid #fecaca; color:#991b1b; }
  .term { background:#0a0e18; color:#f87171; font-size:11.5px; line-height:1.6; padding:14px 16px; border-radius:10px; max-height:260px; overflow:auto; white-space:pre-wrap; word-break:break-word; margin:0; }
  .term .ok-line { color:#86efac; }

  /* ── Forms ─────────────────────────────────────────── */
  input[type=text], select { font-size:14px; padding:9px 12px; border-radius:9px; border:1px solid #d7d9e6; background:#fff; }
  input[type=file] { font-size:13px; }
  button { font-size:14px; padding:9px 16px; border-radius:9px; border:none; cursor:pointer; font-weight:700; background:var(--accent); color:#fff; transition: transform .1s ease; }
  button:hover:not(:disabled) { transform: translateY(-1px); }
  button.secondary { background:#eef0f8; color:#3a3f55; }
  button.ghost { background:transparent; color:var(--accent-ink); border:1px solid #d9d5fb; }
  button:disabled { background:#d7dae4; color:#9aa1b5; cursor:not-allowed; transform:none; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }

  .badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:11.5px; font-weight:800; letter-spacing:.02em; }
  .badge.success, .badge.finished { background:#dcfce7; color:#166534; }
  .badge.failed { background:#fee2e2; color:#991b1b; }
  .badge.running { background:#e0e7ff; color:#3730a3; }
  .badge.queued { background:#fef3c7; color:#92400e; }

  /* ── Validator verdict + findings ─────────────────── */
  .verdict { display:flex; align-items:center; gap:14px; padding:14px 18px; border-radius:12px; margin-top:4px; }
  .verdict.blocked { background:#fef2f2; border:1px solid #fecaca; }
  .verdict.pass { background:#f0fdf4; border:1px solid #bbf7d0; }
  .verdict-icon { font-size:26px; }
  .verdict-title { font-weight:800; font-size:15px; }
  .verdict.blocked .verdict-title { color:#991b1b; }
  .verdict.pass .verdict-title { color:#166534; }
  .verdict-sub { font-size:12.5px; color:var(--muted); margin-top:2px; }

  .finding { margin-bottom:12px; border-radius:10px; overflow:hidden; border:1px solid var(--border); }
  .finding-code { background:var(--code-bg); color:#c9d1e6; font-size:12.5px; padding:11px 14px; white-space:pre-wrap; word-break:break-word; }
  .finding-code mark.error { background:rgba(239,68,68,.35); color:#fecaca; text-decoration:underline wavy #ef4444; text-underline-offset:2px; border-radius:3px; padding:0 1px; }
  .finding-code mark.warning { background:rgba(245,158,11,.3); color:#fde68a; text-decoration:underline wavy #f59e0b; text-underline-offset:2px; border-radius:3px; padding:0 1px; }
  .finding-body { padding:9px 14px; font-size:12.5px; }
  .finding-body.error { background:#fef2f2; }
  .finding-body.warning { background:#fffbeb; }
  .finding-body.ok { background:#f0fdf4; padding:10px 14px; }
  .finding-loc { color:#94a3b8; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.03em; }
  .finding-msg { font-weight:700; margin-top:2px; }
  .finding-suggestion { color:#475569; margin-top:5px; }

  /* ── Pipeline visualizer ───────────────────────────── */
  .pipeline { display:flex; align-items:flex-start; margin:20px 0 4px; }
  .pipe-step { display:flex; flex-direction:column; align-items:center; gap:8px; width:120px; }
  .pipe-dot { width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:15px; border:2.5px solid #dfe1ee; color:#a3a9bd; background:#fff; transition:all .25s ease; }
  .pipe-dot.running { border-color:var(--accent); color:var(--accent); animation:pulse 1.1s infinite; }
  .pipe-dot.done { border-color:var(--ok); background:var(--ok); color:#fff; }
  .pipe-dot.failed { border-color:var(--danger); background:var(--danger); color:#fff; }
  .pipe-label { font-size:11.5px; color:#5b6478; font-weight:700; text-align:center; }
  .pipe-line { flex:1; height:3px; background:#e3e5f1; margin-top:19px; border-radius:2px; overflow:hidden; }
  .pipe-line-fill { height:100%; width:0%; background:var(--ok); transition:width .35s ease; }
  .pipe-timer { font-size:26px; font-weight:800; color:var(--accent-ink); margin-top:14px; }
  .pipe-result { margin-top:12px; padding:13px 16px; border-radius:10px; background:#f0fdf4; border:1px solid #bbf7d0; color:#166534; font-weight:700; font-size:14px; }

  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:9px 6px; border-bottom:1px solid #eef0f7; }
  th { color:#8890a3; font-weight:700; font-size:11.5px; text-transform:uppercase; letter-spacing:.03em; }
  .muted { color:#a3a9bd; }
  .modelchip { display:inline-flex; align-items:center; gap:5px; background:#f2f1fd; color:#4b3fd6; border-radius:8px; padding:4px 11px; font-size:12px; font-weight:700; margin:2px 4px 2px 0; }
</style>
</head>
<body>

<div class="hero">
  <div class="hero-inner">
    <h1>MLOps Portal <span class="accent">· Notebook Only</span></h1>
    <p class="tag">Notebook 등록 → Rule Validator 검증 → 실행. ML 개발자는 Windows에서 자유롭게 개발하고,
    MLOps에 넘기는 건 표준 계약 하나뿐입니다. 과제가 늘어도 파이프라인은 그대로입니다.</p>
    <div class="stats">
      <div class="stat"><div class="num">0회</div><div class="label">MLOps 엔지니어 코드 개입</div></div>
      <div class="stat"><div class="num">3개 과제 · 1개 DAG</div><div class="label">회귀 / 분류 / 객체탐지, 같은 파이프라인</div></div>
      <div class="stat"><div class="num" id="heroLastRun">–</div><div class="label">최근 실행 소요 시간</div></div>
    </div>
  </div>
</div>

<main>

  <div class="card">
    <div class="eyebrow">Why</div>
    <h2>🎯 무엇이 사라지는가</h2>
    <p class="desc"><code>before_windows_yolo_train.py</code> (Windows 로컬 학습 코드)와 표준 Notebook을 나란히 놓고 본 것.
    빨간 줄이 초록 줄로 바뀌는 게 아니라, <b>아예 코드에서 사라진다.</b></p>
    <div class="diff-wrap">
      <div class="diff-header">
        <div class="diff-h">😰 Before — Windows 로컬</div>
        <div></div>
        <div class="diff-h">⚡ After — 표준 Notebook</div>
      </div>
      <div class="diff-row">
        <div class="diff-before">DATA_YAML = r"<span class="hl">C:\</span>Users\jwlee\Desktop\door_project\dataset\data.yaml"</div>
        <div class="diff-arrow">→</div>
        <div class="diff-after">dataset_dir = <span class="hl-ok">prizm.dataset.download</span>(dataset_key, dataset_version, dest_dir="./datasets/door_defect")
DATA_YAML = os.path.join(dataset_dir, "data.yaml")</div>
      </div>
      <div class="diff-note">↳ S3에는 <b>압축(zip) 없이</b> images/labels/data.yaml이 그대로 올라가 있고, <code>dest_dir</code>로
      지정한 경로에 그대로 복원됩니다 (실제 Notebook에는 data.yaml의 경로 필드를 자동으로 맞춰주는 코드가 몇 줄 더 있습니다).
      어느 쪽이든 사람이 손으로 채워 넣는 물리 경로는 0개입니다.</div>
      <div class="diff-row">
        <div class="diff-before">BASE_WEIGHT = r"<span class="hl">C:\</span>Users\jwlee\Downloads\yolov8n.pt"</div>
        <div class="diff-arrow">→</div>
        <div class="diff-after">base_weight = <span class="hl-ok">prizm.model.download</span>(base_model_key, base_model_version)</div>
      </div>
      <div class="diff-row">
        <div class="diff-before">os.environ[<span class="hl">"CUDA_VISIBLE_DEVICES"</span>] = "0"&#10;<span class="hl">device = 0</span></div>
        <div class="diff-arrow">→</div>
        <div class="diff-after">device = "cpu"  # Runtime Catalog가 배정한 값 그대로</div>
      </div>
      <div class="diff-row">
        <div class="diff-before">LABEL_MAP_PATH = r"<span class="hl">Z:\공유\비전팀\door_defect\</span>label_map.json"</div>
        <div class="diff-arrow">→</div>
        <div class="diff-after removed">(삭제됨) — 라벨 정의가 Dataset Asset의 data.yaml에 포함되어 배포됨</div>
      </div>
      <div class="diff-row">
        <div class="diff-before"><span class="hl">!git clone</span> https://github.com/example/door-defect-configs.git</div>
        <div class="diff-arrow">→</div>
        <div class="diff-after removed">(삭제됨) — 필요한 설정은 AI Hub의 Dataset/Code Asset으로 등록</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="eyebrow">Live Demo · 진짜로 실행</div>
    <h2>⚡ 라이브 비교 — 그대로 vs 표준</h2>
    <p class="desc">위 비교표는 설명이고, 이건 실제 실행입니다. 왼쪽은 <code>before_windows_yolo_train.py</code>를
    <b>한 글자도 고치지 않고</b> 우리 Runtime Container에 그대로 태운 것 — Validator도 거치지 않았습니다.
    오른쪽은 같은 코드를 표준 Notebook으로 바꾼 것. 둘 다 지금 이 자리에서 실행됩니다.</p>
    <div class="live-grid">
      <div class="live-col before">
        <div class="live-col-h before">😰 Before — 수정 없이 그대로</div>
        <button id="runBeforeBtn" class="live-btn before-btn">▶ 그대로 실행</button>
        <div class="mini-pipe" id="mp-before">
          <div class="mini-step"><div class="mini-dot" id="mdot-before-preflight_check">1</div><div class="mini-label">Preflight</div></div>
          <div class="mini-line"></div>
          <div class="mini-step"><div class="mini-dot" id="mdot-before-download_notebook">2</div><div class="mini-label">Download</div></div>
          <div class="mini-line"></div>
          <div class="mini-step"><div class="mini-dot" id="mdot-before-execute_notebook">3</div><div class="mini-label">Execute</div></div>
        </div>
        <div class="mini-timer" id="mtimer-before" style="display:none">⏱ 0.0s</div>
        <div id="mresult-before"></div>
        <pre class="term" id="mlog-before" style="display:none"></pre>
      </div>
      <div class="live-col after">
        <div class="live-col-h after">⚡ After — 표준 Notebook</div>
        <button id="runAfterBtn" class="live-btn">▶ 표준 실행</button>
        <div class="mini-pipe" id="mp-after">
          <div class="mini-step"><div class="mini-dot" id="mdot-after-preflight_check">1</div><div class="mini-label">Preflight</div></div>
          <div class="mini-line"></div>
          <div class="mini-step"><div class="mini-dot" id="mdot-after-download_notebook">2</div><div class="mini-label">Download</div></div>
          <div class="mini-line"></div>
          <div class="mini-step"><div class="mini-dot" id="mdot-after-execute_notebook">3</div><div class="mini-label">Execute</div></div>
        </div>
        <div class="mini-timer" id="mtimer-after" style="display:none">⏱ 0.0s</div>
        <div id="mresult-after"></div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="eyebrow">Full Detail</div>
    <h2>📄 전체 코드 비교</h2>
    <p class="desc">위는 핵심만 뽑은 요약이고, 이건 <b>실제 파일 전체</b>입니다. Before는 경로/하이퍼파라미터/증강설정이
    한 파일에 뒤섞인 <code>.py</code> 스크립트, After는 같은 항목을 <b>Notebook 셀 단위로 정리</b>한 것 — 줄어든 건
    물리 경로뿐이고 나머지 학습 설정은 전부 그대로 살아있습니다.</p>
    <div class="code-tabs">
      <button class="code-tab active" data-tab="before">😰 Before (.py) — <span id="beforeLineCount">-</span>줄, 단일 파일</button>
      <button class="code-tab" data-tab="after">⚡ After (Notebook) — 셀 <span id="afterCellCount">-</span>개로 정리</button>
    </div>
    <div id="codeViewBefore" class="code-view"></div>
    <div id="codeViewAfter" class="code-view" style="display:none"></div>
  </div>

  <div class="card">
    <div class="eyebrow">Step 1 · Rule Validator</div>
    <h2>Notebook 등록</h2>
    <p class="desc">.ipynb 업로드 → 정적 규칙 검사. 에러가 하나라도 있으면 등록이 막힙니다. <b>고쳐서 다시 올리면 바로 통과합니다.</b></p>
    <div class="row">
      <input type="text" id="cbKey" placeholder="code key (예: demo.door_defect_yolov8_training)" style="min-width:340px">
      <input type="text" id="cbVersion" placeholder="version (예: v4)" style="width:110px">
    </div>
    <div class="row">
      <input type="file" id="cbFile" accept=".ipynb">
      <button id="validateBtn">검증하기</button>
      <button id="registerBtn" class="secondary" disabled>등록하기</button>
    </div>
    <div id="findings"></div>
  </div>

  <div class="card">
    <div class="eyebrow">Step 2</div>
    <h2>Dataset / Model 자산</h2>
    <p class="desc">Notebook과 달리 검증 없이 바로 등록됩니다 (Rule Validator는 실행 코드에만 적용).</p>
    <div class="row">
      <select id="assetType">
        <option value="dataset">dataset</option>
        <option value="model">model</option>
      </select>
      <input type="text" id="assetKey" placeholder="key">
      <input type="text" id="assetVersion" placeholder="version">
    </div>
    <div class="row">
      <input type="file" id="assetFile">
      <button id="assetUploadBtn">등록</button>
    </div>
    <div id="assetStatus" style="font-size:13px;color:var(--muted);margin-bottom:6px;"></div>
    <div>
      <div class="desc" style="margin-bottom:8px;">📦 이미 Model Hub에 등록된 자산 (예: YOLOv8 베이스 가중치는 미리 준비되어 있음)</div>
      <div id="modelHubList" class="muted">불러오는 중...</div>
    </div>
  </div>

  <div class="card">
    <div class="eyebrow">Step 3 · Generic Executor</div>
    <h2>실행</h2>
    <p class="desc">등록된 Notebook 중 하나를 골라 실행합니다. 어떤 Notebook을 골라도 같은 DAG가 돕니다.</p>
    <div class="row">
      <select id="asset" style="min-width:360px"></select>
      <button id="runBtn">▶ Run</button>
      <button id="refreshAssetsBtn" class="ghost">목록 새로고침</button>
    </div>

    <div id="paramsBox" style="display:none">
      <div class="params-toggle" id="paramsToggle">⚙️ 실행 전 파라미터 조정 (<span id="paramsCount">0</span>개, 선택사항) ▾</div>
      <div id="paramsForm" class="params-grid" style="display:none"></div>
    </div>

    <div class="pipeline" id="pipeline" style="display:none">
      <div class="pipe-step"><div class="pipe-dot" id="dot-preflight_check">1</div><div class="pipe-label">Preflight</div></div>
      <div class="pipe-line"><div class="pipe-line-fill" id="fill-1"></div></div>
      <div class="pipe-step"><div class="pipe-dot" id="dot-download_notebook">2</div><div class="pipe-label">Download</div></div>
      <div class="pipe-line"><div class="pipe-line-fill" id="fill-2"></div></div>
      <div class="pipe-step"><div class="pipe-dot" id="dot-execute_notebook">3</div><div class="pipe-label">Train / Execute</div></div>
      <div class="pipe-line"><div class="pipe-line-fill" id="fill-3"></div></div>
      <div class="pipe-step"><div class="pipe-dot" id="dot-registered">4</div><div class="pipe-label">결과 등록</div></div>
    </div>
    <div class="pipe-timer" id="pipeTimer" style="display:none">⏱ <span id="timerVal">0.0</span>s</div>
    <div id="pipeResult"></div>
  </div>

  <div class="card">
    <div class="eyebrow">Step 4</div>
    <h2>최근 실행 이력</h2>
    <table>
      <thead><tr><th>Run</th><th>Status</th><th>Metric</th><th>Asset</th></tr></thead>
      <tbody id="historyBody"><tr><td colspan="4" class="muted">불러오는 중...</td></tr></tbody>
    </table>
  </div>

</main>

<script>
function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function markSnippet(snippet, start, end, level) {
  const s = escapeHtml(snippet);
  if (typeof start !== 'number') return s;
  const pre = escapeHtml(snippet.slice(0, start));
  const mid = escapeHtml(snippet.slice(start, end));
  const post = escapeHtml(snippet.slice(end));
  return `${pre}<mark class="${level}">${mid}</mark>${post}`;
}

function renderFindings(findings) {
  const box = document.getElementById('findings');
  const errors = findings.filter(f => f.level === 'error');
  const warnings = findings.filter(f => f.level === 'warning');

  let html = '';
  if (errors.length === 0) {
    html += `<div class="verdict pass">
      <div class="verdict-icon">✅</div>
      <div><div class="verdict-title">등록 가능 — 에러 0건${warnings.length ? `, 경고 ${warnings.length}건` : ''}</div>
      <div class="verdict-sub">아래 "등록하기"를 누르면 AI Hub에 바로 등록됩니다.</div></div>
    </div>`;
  } else {
    html += `<div class="verdict blocked">
      <div class="verdict-icon">🚫</div>
      <div><div class="verdict-title">등록 불가 — 에러 ${errors.length}건</div>
      <div class="verdict-sub">아래 항목을 고쳐서 다시 업로드하면 즉시 재검증됩니다.</div></div>
    </div>`;
  }

  html += '<div style="margin-top:16px;">' + findings.map(f => {
    if (f.level === 'ok') {
      return `<div class="finding"><div class="finding-body ok">✅ ${f.message}</div></div>`;
    }
    const loc = (f.cell != null) ? `Cell ${f.cell} · Line ${f.line}` : '전체 Notebook';
    const codeHtml = f.snippet != null
      ? `<div class="finding-code">${markSnippet(f.snippet, f.match_start, f.match_end, f.level)}</div>`
      : '';
    const icon = f.level === 'error' ? '❌' : '⚠️';
    return `<div class="finding">
      ${codeHtml}
      <div class="finding-body ${f.level}">
        <div class="finding-loc">${loc}</div>
        <div class="finding-msg">${icon} ${f.message}</div>
        <div class="finding-suggestion">💡 ${f.suggestion || ''}</div>
      </div>
    </div>`;
  }).join('') + '</div>';

  box.innerHTML = html;
}

let lastFindings = null;

async function runValidate() {
  const fileInput = document.getElementById('cbFile');
  const registerBtn = document.getElementById('registerBtn');
  if (!fileInput.files.length) return;
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  document.getElementById('findings').innerHTML = '<div class="muted">검증 중...</div>';
  registerBtn.disabled = true;
  const res = await fetch('/api/validate', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) { document.getElementById('findings').innerHTML = `오류: ${data.detail || res.statusText}`; return; }
  lastFindings = data;
  renderFindings(data.findings);
  registerBtn.disabled = !data.can_register;
}

document.getElementById('validateBtn').addEventListener('click', runValidate);
document.getElementById('cbFile').addEventListener('change', runValidate);

document.getElementById('registerBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('cbFile');
  const key = document.getElementById('cbKey').value.trim();
  const version = document.getElementById('cbVersion').value.trim();
  if (!key || !version || !fileInput.files.length) { alert('key / version / 파일을 모두 입력하세요.'); return; }
  const fd = new FormData();
  fd.append('key', key); fd.append('version', version); fd.append('file', fileInput.files[0]);
  const res = await fetch('/api/register-code', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) { alert(`등록 실패: ${data.detail || res.statusText}`); return; }
  document.getElementById('findings').insertAdjacentHTML('afterbegin',
    `<div class="verdict pass" style="margin-bottom:12px;"><div class="verdict-icon">📦</div>
     <div><div class="verdict-title">등록됨 — ${data.key}:${data.version}</div>
     <div class="verdict-sub">아래 "실행" 목록에 추가됐습니다.</div></div></div>`);
  loadAssets();
});

document.getElementById('assetUploadBtn').addEventListener('click', async () => {
  const type = document.getElementById('assetType').value;
  const key = document.getElementById('assetKey').value.trim();
  const version = document.getElementById('assetVersion').value.trim();
  const fileInput = document.getElementById('assetFile');
  const st = document.getElementById('assetStatus');
  if (!key || !version || !fileInput.files.length) { st.textContent = 'type/key/version/file을 모두 입력하세요.'; return; }
  const fd = new FormData();
  fd.append('type', type); fd.append('key', key); fd.append('version', version);
  fd.append('file', fileInput.files[0]);
  st.textContent = '등록 중...';
  const res = await fetch('/api/assets/upload', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) { st.innerHTML = `<span class="badge failed">FAIL</span> ${data.detail || res.statusText}`; return; }
  st.innerHTML = `<span class="badge success">등록됨</span> ${type} ${data.key}:${data.version}`;
  loadModelHub();
});

async function loadModelHub() {
  const res = await fetch('/api/assets?type=model');
  const items = await res.json();
  const box = document.getElementById('modelHubList');
  box.innerHTML = items.length
    ? items.map(it => `<span class="modelchip">📦 ${it.key} · ${it.version}</span>`).join('')
    : '<span class="muted">등록된 Model 자산이 없습니다.</span>';
}

async function loadAssets() {
  const res = await fetch('/api/assets?type=code');
  const items = (await res.json()).filter(it => !it.key.startsWith('_') && !it.key.includes('._'));
  const sel = document.getElementById('asset');
  const prev = sel.value;
  sel.innerHTML = '';
  items.forEach(it => {
    const opt = document.createElement('option');
    opt.value = JSON.stringify(it);
    opt.textContent = `${it.key}  (${it.version})`;
    sel.appendChild(opt);
  });
  if (prev) sel.value = prev;
  await loadParamsForAsset();
}

// ── 실행 전 파라미터 조정 ──────────────────────────────────
let currentParams = [];

async function loadParamsForAsset() {
  const sel = document.getElementById('asset');
  const box = document.getElementById('paramsBox');
  const form = document.getElementById('paramsForm');
  if (!sel.value) { box.style.display = 'none'; return; }
  const { key, version } = JSON.parse(sel.value);
  let items = [];
  try {
    const res = await fetch(`/api/notebook-params?code_key=${encodeURIComponent(key)}&code_version=${encodeURIComponent(version)}`);
    if (res.ok) items = await res.json();
  } catch (e) { /* 파라미터 없이도 실행은 가능해야 함 */ }
  currentParams = items;
  document.getElementById('paramsCount').textContent = items.length;
  form.innerHTML = items.map(p => `
    <div class="param-field">
      <label title="${p.name}">${p.name}</label>
      <input type="text" data-param="${p.name}" data-type="${p.type}" data-default="${escapeHtml(String(p.default))}" value="${escapeHtml(String(p.default))}">
    </div>`).join('');
  form.querySelectorAll('input[data-param]').forEach(inp => {
    inp.addEventListener('input', () => {
      inp.classList.toggle('changed', inp.value !== inp.dataset.default);
    });
  });
  box.style.display = items.length ? 'block' : 'none';
}

function collectParamOverrides() {
  const overrides = {};
  document.querySelectorAll('#paramsForm input[data-param]').forEach(inp => {
    if (inp.value === inp.dataset.default) return;
    const type = inp.dataset.type;
    let val = inp.value;
    if (type === 'int') val = parseInt(val, 10);
    else if (type === 'float') val = parseFloat(val);
    else if (type === 'bool') val = (val === 'true' || val === 'True');
    overrides[inp.dataset.param] = val;
  });
  return overrides;
}

document.getElementById('asset').addEventListener('change', loadParamsForAsset);
document.getElementById('paramsToggle').addEventListener('click', () => {
  const form = document.getElementById('paramsForm');
  form.style.display = form.style.display === 'none' ? 'grid' : 'none';
});

// ── 전체 코드 비교 (탭) ────────────────────────────────────
async function loadCompareCode() {
  const res = await fetch('/api/compare-code');
  const data = await res.json();
  document.getElementById('beforeLineCount').textContent = data.before_lines;
  document.getElementById('afterCellCount').textContent = data.after_cells.filter(c => c.type === 'code').length;
  document.getElementById('codeViewBefore').innerHTML = `<pre>${escapeHtml(data.before_py)}</pre>`;
  document.getElementById('codeViewAfter').innerHTML = data.after_cells.map((c, i) => {
    if (c.type === 'markdown') return `<div class="nb-cell markdown">${escapeHtml(c.source)}</div>`;
    return `<div class="nb-cell code"><div class="nb-cell-in">In [${i}]:</div><pre>${escapeHtml(c.source)}</pre></div>`;
  }).join('');
}

document.querySelectorAll('.code-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.code-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('codeViewBefore').style.display = btn.dataset.tab === 'before' ? 'block' : 'none';
    document.getElementById('codeViewAfter').style.display = btn.dataset.tab === 'after' ? 'block' : 'none';
  });
});

async function loadHistory() {
  const res = await fetch('/api/history');
  const rows = await res.json();
  const body = document.getElementById('historyBody');
  if (!rows.length) { body.innerHTML = '<tr><td colspan="4" class="muted">아직 실행 이력이 없습니다.</td></tr>'; return; }
  body.innerHTML = rows.map(r => `
    <tr>
      <td>${r.run_name || '-'}</td>
      <td><span class="badge ${(r.status||'').toLowerCase()}">${r.status}</span></td>
      <td>${r.metrics || '-'}</td>
      <td>${r.asset}</td>
    </tr>`).join('');
  return rows;
}

// ── Pipeline visualizer ──────────────────────────────────
let timerInterval = null;
let startTs = null;

function setDot(id, state) {
  const el = document.getElementById('dot-' + id);
  el.className = 'pipe-dot' + (state ? ' ' + state : '');
  el.textContent = state === 'done' ? '✓' : state === 'failed' ? '✕' : el.dataset.n || el.textContent;
}

function updatePipeline(tasks, overallState) {
  const order = ['preflight_check', 'download_notebook', 'execute_notebook'];
  order.forEach((id, i) => {
    const t = tasks.find(x => x.task_id === id);
    let state = 'pending';
    if (t) {
      if (t.state === 'success') state = 'done';
      else if (t.state === 'failed' || t.state === 'upstream_failed') state = 'failed';
      else if (t.state === 'running') state = 'running';
    }
    setDot(id, state === 'pending' ? '' : state);
    document.getElementById('fill-' + (i + 1)).style.width = (state === 'done') ? '100%' : '0%';
  });
  setDot('registered', overallState === 'success' ? 'done' : (overallState === 'failed' ? 'failed' : ''));
}

function startTimer() {
  startTs = performance.now();
  document.getElementById('pipeTimer').style.display = 'block';
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    document.getElementById('timerVal').textContent = ((performance.now() - startTs) / 1000).toFixed(1);
  }, 100);
}

function stopTimer() {
  clearInterval(timerInterval);
  return startTs ? ((performance.now() - startTs) / 1000).toFixed(1) : '0.0';
}

document.getElementById('refreshAssetsBtn').addEventListener('click', loadAssets);

document.getElementById('runBtn').addEventListener('click', async () => {
  const sel = document.getElementById('asset');
  if (!sel.value) return;
  const { key, version } = JSON.parse(sel.value);
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  document.getElementById('pipeline').style.display = 'flex';
  document.getElementById('pipeResult').innerHTML = '';
  ['preflight_check','download_notebook','execute_notebook','registered'].forEach(id => setDot(id, ''));
  [1,2,3].forEach(i => document.getElementById('fill-' + i).style.width = '0%');
  startTimer();

  const overrides = collectParamOverrides();
  const trigRes = await fetch('/api/trigger', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ code_key: key, code_version: version, params: overrides })
  });
  if (!trigRes.ok) {
    stopTimer();
    document.getElementById('pipeResult').innerHTML = `<div class="pipe-result" style="background:#fef2f2;border-color:#fecaca;color:#991b1b;">트리거 실패: ${await trigRes.text()}</div>`;
    btn.disabled = false;
    return;
  }
  const trig = await trigRes.json();
  const runId = trig.dag_run_id;

  const poll = setInterval(async () => {
    const r = await fetch(`/api/run_status?run_id=${encodeURIComponent(runId)}`);
    const d = await r.json();
    updatePipeline(d.tasks || [], d.state);
    if (d.state === 'success' || d.state === 'failed') {
      clearInterval(poll);
      const elapsed = stopTimer();
      document.getElementById('heroLastRun').textContent = elapsed + 's';
      btn.disabled = false;
      const rows = await loadHistory();
      if (d.state === 'success') {
        const top = rows && rows[0];
        const overrideNote = Object.keys(overrides).length
          ? `<br><span style="font-weight:600;font-size:12px;">⚙️ 적용된 파라미터: ${Object.entries(overrides).map(([k,v]) => `${k}=${v}`).join(', ')}</span>`
          : '';
        document.getElementById('pipeResult').innerHTML =
          `<div class="pipe-result">🎉 완료 · ${elapsed}초${top ? ` · ${top.metrics} · ${top.asset}` : ''}${overrideNote}</div>`;
      } else {
        document.getElementById('pipeResult').innerHTML =
          `<div class="pipe-result" style="background:#fef2f2;border-color:#fecaca;color:#991b1b;">❌ 실패 · ${elapsed}초 — Airflow 로그를 확인하세요.</div>`;
      }
    }
  }, 1500);
});

// ── Live compare (Before 그대로 vs After 표준) ──────────────
const LIVE_CONFIG = {
  before: { codeKey: 'demo._live_before_asis', codeVersion: 'v1' },
  after: { codeKey: 'demo.door_defect_yolov8_training', codeVersion: 'v1' },
};
const liveTimers = {};

function setMiniDot(slot, taskId, state) {
  const el = document.getElementById(`mdot-${slot}-${taskId}`);
  if (!el) return;
  el.className = 'mini-dot' + (state ? ' ' + state : '');
  el.textContent = state === 'done' ? '✓' : state === 'failed' ? '✕' : el.textContent.match(/\d/) ? el.textContent : '';
}

function resetMiniPipe(slot) {
  ['preflight_check', 'download_notebook', 'execute_notebook'].forEach((t, i) => {
    const el = document.getElementById(`mdot-${slot}-${t}`);
    el.className = 'mini-dot';
    el.textContent = String(i + 1);
  });
  document.getElementById(`mresult-${slot}`).innerHTML = '';
  const logEl = document.getElementById(`mlog-${slot}`);
  if (logEl) logEl.style.display = 'none';
}

async function runLive(slot) {
  const btn = document.getElementById(slot === 'before' ? 'runBeforeBtn' : 'runAfterBtn');
  btn.disabled = true;
  resetMiniPipe(slot);
  document.getElementById(`mtimer-${slot}`).style.display = 'block';
  liveTimers[slot] = performance.now();
  const timerInt = setInterval(() => {
    document.getElementById(`mtimer-${slot}`).textContent =
      '⏱ ' + ((performance.now() - liveTimers[slot]) / 1000).toFixed(1) + 's';
  }, 100);

  const cfg = LIVE_CONFIG[slot];
  const trigRes = await fetch('/api/trigger', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ code_key: cfg.codeKey, code_version: cfg.codeVersion })
  });
  if (!trigRes.ok) {
    clearInterval(timerInt);
    document.getElementById(`mresult-${slot}`).innerHTML =
      `<div class="live-result bad">트리거 실패: ${escapeHtml(await trigRes.text())}</div>`;
    btn.disabled = false;
    return;
  }
  const trig = await trigRes.json();
  const runId = trig.dag_run_id;

  const poll = setInterval(async () => {
    const r = await fetch(`/api/run_status?run_id=${encodeURIComponent(runId)}`);
    const d = await r.json();
    (d.tasks || []).forEach(t => {
      let state = t.state === 'success' ? 'done' : (t.state === 'failed' || t.state === 'upstream_failed') ? 'failed' : (t.state === 'running' ? 'running' : '');
      setMiniDot(slot, t.task_id, state);
    });
    if (d.state === 'success' || d.state === 'failed') {
      clearInterval(poll);
      clearInterval(timerInt);
      const elapsed = ((performance.now() - liveTimers[slot]) / 1000).toFixed(1);
      btn.disabled = false;
      if (d.state === 'success') {
        const rows = await loadHistory();
        const top = rows && rows[0];
        document.getElementById(`mresult-${slot}`).innerHTML =
          `<div class="live-result ok">🎉 성공 · ${elapsed}초${top ? ` · ${top.metrics}` : ''}</div>`;
      } else {
        document.getElementById(`mresult-${slot}`).innerHTML =
          `<div class="live-result bad">❌ 실패 · ${elapsed}초 — 아래는 Airflow의 실제 에러 로그입니다.</div>`;
        const logRes = await fetch(`/api/run_log?run_id=${encodeURIComponent(runId)}&task_id=execute_notebook`);
        if (logRes.ok) {
          const logData = await logRes.json();
          const logEl = document.getElementById(`mlog-${slot}`);
          if (logEl) { logEl.textContent = logData.log; logEl.style.display = 'block'; }
        }
      }
    }
  }, 1500);
}

document.getElementById('runBeforeBtn').addEventListener('click', () => runLive('before'));
document.getElementById('runAfterBtn').addEventListener('click', () => runLive('after'));

loadAssets();
loadModelHub();
loadHistory();
loadCompareCode();
setInterval(loadHistory, 20000);
</script>
</body>
</html>
"""
