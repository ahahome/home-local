"""L1 抽取层评测运行器。

用法（在项目根目录）：
    .venv/bin/python eval/run.py                          # 两集都跑（tune + holdout）
    .venv/bin/python eval/run.py --split holdout          # 只跑留出集
    .venv/bin/python eval/run.py --repeat 3               # 每条采样 3 次，按最差判定
    .venv/bin/python eval/run.py --repeat 3 --no-cache    # 取全新样本
    .venv/bin/python eval/run.py --cases eval/cases/x.jsonl
    .venv/bin/python eval/run.py --gate                   # 门槛未达时退出码非 0（给 CI 用）

两集的分工（这是本文件最重要的设计）：
- tune   ：提示词/代码就是对着它调的。**不卡门槛**，只作「没把老的东西改坏」的回归守卫。
- holdout：出题时不许看提示词，写完冻结。**只有它卡门槛**——它才是泛化能力的估计。
  对 tune 调参、用 holdout 验收；把两个数字分开报，不要混成一个。

重复采样（--repeat k）：
- 每条跑 k 次，**按最差的那次判定**（worst-of-k）。单次通过率会骗人：
  实测出现过「连过两次 32/32，第三次掉到 31/32」。
- 单独列出 flaky（k 次里有过有挂）——它们是最值得看的信号，说明规则没落成确定性保证。
- 缓存键含 run_index：k 次会拿到 k 个不同样本，且**可复现**（同一缓存下重跑得到同样结论）。
  想要全新样本就加 --no-cache。
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
CASES_DIR = ROOT / "eval" / "cases"

SPLITS: dict[str, Path] = {
    "tune": CASES_DIR / "tune.jsonl",
    "holdout": CASES_DIR / "holdout.jsonl",
}
# 只有留出集卡门槛。tune 是调参集，卡它等于自己给自己发奖状。
GATED_SPLITS = {"holdout"}

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


def cache_key(case: dict, ph: str, model: str, run_index: int) -> str:
    raw = json.dumps({"input": case.get("input"), "known": case.get("known_persons") or [],
                      "today": case.get("today"), "ph": ph, "model": model, "i": run_index},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def run_case(case: dict, ph: str, model: str, use_cache: bool, run_index: int):
    """返回 (case, 抽取结果, 是否命中缓存)。"""
    key = cache_key(case, ph, model, run_index)
    path = CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        return case, json.loads(path.read_text(encoding="utf-8")), True
    got = extract.extract_from_text(case.get("input") or "",
                                    case.get("known_persons") or [],
                                    case.get("today") or "")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8")
    return case, got, False


def eval_split(cases: list[dict], ph: str, model: str, repeat: int,
               use_cache: bool, concurrency: int) -> dict:
    """跑一个集，返回 {report, details, raw, flaky, cached}。

    repeat>1 时按 worst-of-k 判定：只要有任一次没通过，这条就算没通过。
    """
    tasks = [(c, i) for c in cases for i in range(repeat)]
    per_case: dict[str, dict] = {c["id"]: {"case": c, "runs": []} for c in cases}
    cached = 0

    def job(arg):
        case, idx = arg
        return case["id"], run_case(case, ph, model, use_cache, idx)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for cid, (case, got, hit) in pool.map(job, tasks):
            cached += int(hit)
            per_case[cid]["runs"].append(metrics.check_case(case, got))
            per_case[cid].setdefault("raws", []).append(got)

    details, raw, flaky = [], {}, []
    for cid, entry in per_case.items():
        runs = entry["runs"]
        failures = [r for r in runs if not r["passed"]]
        if failures and len(failures) < len(runs):
            flaky.append(cid)
        idx = runs.index(failures[0]) if failures else 0  # 取最差的那次作为代表
        rep = dict(runs[idx])
        rep["runs_total"] = len(runs)
        rep["runs_passed"] = len(runs) - len(failures)
        rep["failure_rate"] = round(len(failures) / len(runs), 4)
        details.append(rep)
        raw[cid] = entry["raws"][idx]

    report = metrics.aggregate(cases, details)
    report["repeat"] = repeat
    report["flaky"] = len(flaky)
    report["flaky_ids"] = sorted(flaky)
    report["cache_hits"] = cached
    return {"report": report, "details": details, "raw": raw, "flaky": sorted(flaky)}


def gate_failures(report: dict) -> list[str]:
    failed = []
    for key, floor in GATES.items():
        val = report.get(key)
        if val is None:
            continue
        bad = val < floor if key != "promise_false_positive_rate" else val > floor
        if bad:
            failed.append(f"{key}={val}（门槛 {'≥' if key != 'promise_false_positive_rate' else '≤'} {floor}）")
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description="L1 抽取层评测（tune / holdout 分集 + k 次重复）")
    ap.add_argument("--split", choices=["tune", "holdout", "all"], default="all")
    ap.add_argument("--cases", default=None, help="直接指定单个用例文件（临时探针用），会覆盖 --split")
    ap.add_argument("--tag", default=None, help="只跑带该 tag 的用例")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1, help="每条采样次数，按最差判定（默认 1）")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--gate", action="store_true", help="门槛未达时返回非 0（只对 holdout 生效）")
    ap.add_argument("--quiet", action="store_true", help="只打印汇总，不逐条列失败")
    args = ap.parse_args()

    if args.cases:
        targets = [(Path(args.cases).stem, Path(args.cases))]
    elif args.split == "all":
        targets = list(SPLITS.items())
    else:
        targets = [(args.split, SPLITS[args.split])]

    s = get_settings()
    ph = prompt_hash()
    model = s.model_mid
    t0 = time.time()
    print(f"模型 {model} · prompt {ph} · sha {git_sha()} · 每条采样 {args.repeat} 次"
          f"（按最差判定）· 缓存 {'关' if args.no_cache else '开'}")

    out_splits: dict[str, dict] = {}
    exit_code = 0

    for name, path in targets:
        cases = load_cases(path, args.tag, args.limit)
        if not cases:
            print(f"\n[{name}] 没有匹配的用例，跳过", file=sys.stderr)
            continue
        res = eval_split(cases, ph, model, args.repeat, not args.no_cache, args.concurrency)
        rep = res["report"]
        out_splits[name] = {"cases_file": path.name, "summary": rep,
                            "details": res["details"], "raw": res["raw"]}

        gated = name in GATED_SPLITS
        print(f"\n===== {name}（{path.name}）{'' if gated else ' —— 调参集，不卡门槛，只作回归守卫'} =====")
        if not args.quiet:
            for r in res["details"]:
                mark = "OK  " if r["passed"] else "FAIL"
                extra = "" if r["runs_total"] == 1 else f"  ({r['runs_passed']}/{r['runs_total']} 次通过)"
                print(f"[{mark}] {r['id']} 人物 {r['people_found']}{extra}")
                for f in r["failures"]:
                    print(f"        - {f}")
        print(f"用例 {rep['cases']} 条 · 通过 {rep['passed']}（worst-of-{args.repeat}）"
              f" · 抖动 {rep['flaky']} 条")
        for k in ("pass_rate", "field_recall", "promise_recall",
                  "evidence_hit_rate", "promise_false_positive_rate"):
            print(f"  {k}: {rep[k]}")
        if rep["flaky_ids"]:
            print(f"  抖动用例（有的次过、有的次挂，最值得看）：{rep['flaky_ids']}")
        if gated:
            bad = gate_failures(rep)
            print("  门槛：" + ("全部达标" if not bad else "未达门槛 —— " + "；".join(bad)))
            if bad:
                exit_code = 1
        else:
            print("  门槛：不适用（调参集）")

    elapsed = round(time.time() - t0, 1)
    print(f"\n总耗时 {elapsed}s")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULT_DIR / f"{stamp}-{git_sha()}.json"
    out.write_text(json.dumps({
        "meta": {"model": model, "prompt_hash": ph, "git_sha": git_sha(),
                 "ran_at": datetime.now().isoformat(timespec="seconds"),
                 "elapsed_s": elapsed, "repeat": args.repeat, "tag": args.tag,
                 "no_cache": args.no_cache, "gated_splits": sorted(GATED_SPLITS)},
        "splits": out_splits,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"结果已写入 {out.relative_to(ROOT)}")

    if args.gate and exit_code:
        return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
