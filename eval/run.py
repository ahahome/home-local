"""L1 抽取层评测运行器。

用法（在项目根目录）：
    .venv/bin/python eval/run.py                    # 全量跑一遍
    .venv/bin/python eval/run.py --tag negative-promise
    .venv/bin/python eval/run.py --limit 5 --no-cache
    .venv/bin/python eval/run.py --gate             # 低于门槛时退出码非 0（给 CI 用）

要点：
- 响应缓存：key = sha256(input + known_persons + today + prompt_hash + model)，重复跑不重复花钱
- 可复现：结果文件里记录 model + prompt hash + git sha + 时间
- 如实回报：每条用例的失败原因都会打印，不做「一切正常」式的概括
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.memory import extract  # noqa: E402
from eval import metrics  # noqa: E402

CACHE_DIR = ROOT / "eval" / ".cache"
RESULT_DIR = ROOT / "eval" / "results"

# 门槛：不变量层要求 100%，这里只约束统计类指标
GATES = {"evidence_hit_rate": 0.95, "promise_false_positive_rate": 0.05, "pass_rate": 0.70}


def load_cases(path: Path, tag: str | None, limit: int | None) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        if tag and tag not in (case.get("tags") or []):
            continue
        cases.append(case)
    return cases[:limit] if limit else cases


def prompt_hash() -> str:
    return hashlib.sha256(extract.SYSTEM.encode()).hexdigest()[:12]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def cache_key(case: dict, ph: str, model: str) -> str:
    raw = json.dumps({"input": case.get("input"), "known": case.get("known_persons") or [],
                      "today": case.get("today"), "ph": ph, "model": model},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def run_case(case: dict, ph: str, model: str, use_cache: bool) -> tuple[dict, dict, bool]:
    """返回 (case, 抽取结果, 是否命中缓存)。"""
    key = cache_key(case, ph, model)
    path = CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        return case, json.loads(path.read_text(encoding="utf-8")), True
    got = extract.extract_from_text(case.get("input") or "",
                                    case.get("known_persons") or [],
                                    case.get("today") or "")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8")
    return case, got, False


def main() -> int:
    ap = argparse.ArgumentParser(description="L1 抽取层评测")
    ap.add_argument("--cases", default=str(ROOT / "eval" / "cases" / "extract.jsonl"))
    ap.add_argument("--tag", default=None, help="只跑带该 tag 的用例")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--gate", action="store_true", help="低于门槛时返回非 0")
    ap.add_argument("--quiet", action="store_true", help="只打印汇总，不逐条列失败")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases), args.tag, args.limit)
    if not cases:
        print("没有匹配的用例", file=sys.stderr)
        return 2

    s = get_settings()
    ph = prompt_hash()
    model = s.model_mid
    t0 = time.time()
    print(f"用例 {len(cases)} 条 · 模型 {model} · prompt {ph} · sha {git_sha()}")

    results, got_by_id, cached = [], {}, 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        for case, got, hit in pool.map(
                lambda c: run_case(c, ph, model, not args.no_cache), cases):
            cached += int(hit)
            got_by_id[case["id"]] = got
            results.append(metrics.check_case(case, got))

    report = metrics.aggregate(cases, results)
    elapsed = round(time.time() - t0, 1)

    if not args.quiet:
        for r in results:
            mark = "OK  " if r["passed"] else "FAIL"
            print(f"[{mark}] {r['id']} 人物 {r['people_found']}")
            for f in r["failures"]:
                print(f"        - {f}")

    print("\n--- 汇总 ---")
    for k in ("cases", "passed", "pass_rate", "field_recall", "promise_recall",
              "evidence_hit_rate", "promise_false_positive_rate"):
        print(f"{k}: {report[k]}")
    print(f"缓存命中 {cached}/{len(cases)} · 耗时 {elapsed}s · 模型 {model} · prompt {ph}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULT_DIR / f"{stamp}-{git_sha()}.json"
    out.write_text(json.dumps({
        "meta": {"model": model, "prompt_hash": ph, "git_sha": git_sha(),
                 "ran_at": datetime.now().isoformat(timespec="seconds"),
                 "elapsed_s": elapsed, "cases_file": Path(args.cases).name,
                 "tag": args.tag, "cached": cached},
        "summary": report,
        "details": results,
        "raw": got_by_id,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"结果已写入 {out.relative_to(ROOT)}")

    if args.gate:
        failed = []
        for key, floor in GATES.items():
            val = report.get(key)
            if val is None:
                continue
            bad = val < floor if key != "promise_false_positive_rate" else val > floor
            if bad:
                failed.append(f"{key}={val}（门槛 {'≥' if key != 'promise_false_positive_rate' else '≤'} {floor}）")
        if failed:
            print("未达门槛：" + "；".join(failed))
            return 1
        print("全部达标")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
