"""Rule Validator (프로토타입).

Agent 없이 정적 규칙만으로 표준 Notebook 계약 위반을 잡아낸다는
전략을 실제로 보여주기 위한 최소 구현.

전체 다운로드/미니 학습 같은 건 하지 않는다 - Notebook 텍스트만 스캔한다.
"""
import re
from typing import Any

LINE_RULES = [
    {
        "id": "windows_path",
        "level": "error",
        "regex": re.compile(r"[A-Za-z]:\\\\|[A-Za-z]:\\(?![ntr\"'])"),
        "message": "Windows 절대경로가 발견됐습니다.",
        "suggestion": "prizm.dataset.download() / prizm.model.download() 로 Asset Key+Version을 통해 받으세요.",
    },
    {
        "id": "local_unix_path",
        "level": "error",
        "regex": re.compile(r"(/Users/[^\"'\s]+|/home/[^\"'\s]+|~[\\/])"),
        "message": "로컬 절대경로(Mac/Linux)가 발견됐습니다.",
        "suggestion": "물리 경로 대신 Asset Key+Version을 사용하세요.",
    },
    {
        "id": "git_clone",
        "level": "error",
        "regex": re.compile(r"git\s+clone\b"),
        "message": "git clone 사용이 발견됐습니다.",
        "suggestion": "외부 저장소 대신 AI Hub의 Dataset/Code Asset으로 등록해서 관리하세요.",
    },
    {
        "id": "shell_exec",
        "level": "error",
        "regex": re.compile(r"os\.system\(|subprocess\.(run|Popen|call)\("),
        "message": "임의 shell 실행이 발견됐습니다.",
        "suggestion": "표준 Notebook 계약 밖의 임의 실행은 허용되지 않습니다. 필요한 패키지는 Runtime Image에 포함하세요.",
    },
    {
        "id": "hardcoded_device",
        "level": "error",
        "regex": re.compile(r"CUDA_VISIBLE_DEVICES|device\s*=\s*0\b"),
        "message": "GPU/CUDA 장치가 하드코딩되어 있습니다.",
        "suggestion": 'device는 Runtime Catalog가 배정한 값을 그대로 쓰세요 (예: device="cpu" 또는 선택한 Runtime 기준값).',
    },
]

WHOLE_NOTEBOOK_RULES = [
    {
        "id": "sdk_missing",
        "level": "error",
        "check": lambda src: "import prizm" in src,
        "message": "MLOps SDK(import prizm)가 사용되지 않았습니다.",
        "suggestion": "표준 Notebook 계약은 mlops SDK 사용을 전제로 합니다.",
    },
    {
        "id": "asset_download_missing",
        "level": "warning",
        "check": lambda src: ("prizm.dataset.download" in src) or ("prizm.model.download" in src),
        "message": "prizm.dataset.download() / prizm.model.download() 호출이 없습니다.",
        "suggestion": "데이터/모델 초기값을 로컬 파일이 아니라 Asset Key+Version으로 받고 있는지 확인하세요.",
    },
]

_OK_LABELS = {
    "sdk_missing": "MLOps SDK 사용 확인됨.",
    "asset_download_missing": "Asset Key+Version 기반 다운로드 확인됨.",
    "path_check": "로컬/Windows 절대경로 없음.",
    "git_clone": "외부 git clone 없음.",
    "shell_exec": "임의 shell 실행 없음.",
    "hardcoded_device": "GPU/CUDA 하드코딩 없음.",
}


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src or ""


def validate_notebook(nb: dict[str, Any]) -> list[dict]:
    findings: list[dict] = []
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    all_source = "\n".join(_cell_source(c) for c in code_cells)

    for idx, cell in enumerate(code_cells, start=1):
        source = _cell_source(cell)
        for lineno, raw_line in enumerate(source.splitlines(), start=1):
            line = raw_line.split("#", 1)[0]  # 설명용 주석 안 텍스트는 검사 대상에서 제외
            if not line.strip():
                continue
            lead = len(line) - len(line.lstrip())
            stripped = line.strip()
            for rule in LINE_RULES:
                m = rule["regex"].search(line)
                if m:
                    findings.append(
                        {
                            "level": rule["level"],
                            "rule": rule["id"],
                            "cell": idx,
                            "line": lineno,
                            "snippet": stripped[:200],
                            "match_start": max(0, m.start() - lead),
                            "match_end": max(0, min(len(stripped), m.end() - lead)),
                            "message": rule["message"],
                            "suggestion": rule["suggestion"],
                        }
                    )

    for rule in WHOLE_NOTEBOOK_RULES:
        if not rule["check"](all_source):
            findings.append(
                {
                    "level": rule["level"],
                    "rule": rule["id"],
                    "cell": None,
                    "line": None,
                    "snippet": None,
                    "message": rule["message"],
                    "suggestion": rule["suggestion"],
                }
            )

    # 위반이 없는 항목도 명시적으로 보여준다 - "뭘 잘 지켰는지"도 보여야
    # Validator가 계속 감시하고 있다는 게 느껴진다.
    hit_rules = {f["rule"] for f in findings}
    if "windows_path" not in hit_rules and "local_unix_path" not in hit_rules:
        findings.append(_ok("path_check"))
    for rule_id in ("sdk_missing", "asset_download_missing", "git_clone", "shell_exec", "hardcoded_device"):
        if rule_id not in hit_rules:
            findings.append(_ok(rule_id))

    order = {"error": 0, "warning": 1, "ok": 2}
    findings.sort(key=lambda f: order.get(f["level"], 9))
    return findings


def _ok(rule_id: str) -> dict:
    return {
        "level": "ok",
        "rule": rule_id,
        "cell": None,
        "line": None,
        "snippet": None,
        "message": _OK_LABELS[rule_id],
        "suggestion": None,
    }


def has_blocking_errors(findings: list[dict]) -> bool:
    return any(f["level"] == "error" for f in findings)
