# 评测集

四层判定，成本不同。**免费层跑在每次提交上；花钱层只在改 prompt / 改抽取逻辑时跑。**

| 层 | 位置 | 花钱 | 覆盖什么 |
|----|------|------|---------|
| **L1 抽取** | `eval/run.py` + `eval/cases/extract.jsonl` | 是 | 从文本抽人物/字段/事实/承诺/账本对不对 |
| **L2 落库** | `tests/test_invariants.py` | 否 | 三档置信度、幂等、待确认不进汇总、删除不复活 |
| **L3 检索** | 待补 | 否 | Recall@5 / MRR（需要 (query, 相关 id) 标注） |
| **L5 不变量** | `tests/test_invariants.py` | 否 | 七原则：违反即失败，没有「大部分情况对」 |
| L4 端到端 | 待补 | 最贵 | 多轮脚本 → 库状态 + 工具轨迹（PRD 里 AC-06~14、AC-17~20 属于这层） |

评估器本身的比对逻辑在 `eval/metrics.py`，由 `tests/test_eval_metrics.py` 覆盖。
**理由：评估器写错了比被测代码错了更危险**——指标会假绿，问题被掩盖。

## 怎么跑

```bash
cd /Users/emdoor/Documents/projects/integrate/home

# 免费层（不调 LLM，秒级）
.venv/bin/python -m pytest tests/ -q

# L1 全量（调真实 API，约 1 分钟 / 30 条 / 几块钱）
.venv/bin/python eval/run.py

# 只跑某一类
.venv/bin/python eval/run.py --tag negative-promise
.venv/bin/python eval/run.py --tag real

# 调试单条：跳过缓存重跑
.venv/bin/python eval/run.py --limit 5 --no-cache

# 当成门禁：指标不达标时退出码非 0（给 CI 用）
.venv/bin/python eval/run.py --gate
```

结果落在 `eval/results/<时间戳>-<git sha>.json`，里面连带记录 `model`、`prompt_hash`、
`git_sha`、耗时，任何一次分数都能复现和回溯。响应缓存放在 `eval/.cache/`
（key = `sha256(input + known_persons + today + prompt_hash + model)`），重复跑不重复花钱。

## 用例怎么写

一行一条 JSON，见 `eval/cases/extract.jsonl`。字段：

```json
{"id": "extract-p01",
 "layer": "extract",
 "tags": ["positive", "fields"],
 "today": "2026-09-30",
 "known_persons": ["老陆"],
 "input": "老陆昨天跟我说他跳槽了，现在在杭州做采购总监。",
 "expect": {"people": [{"canonical_name": "老陆",
                        "fields": [{"field": "title", "contains": "采购总监"}]}]},
 "must_not": [],
 "max_people": 1}
```

**三条规则**

1. **子集匹配**：`expect` 只写「必须命中」的。抽取多出来的合理内容不算错——否则指标全是被噪声淹没的假失败。
2. **`must_not` 比 `expect` 重要**。幻觉抓不到，是因为用例只写了「该抽出什么」。支持的形式：
   - `"promises"` / `"ledger"` / `"people"` — 全局不得出现
   - `"name:那个客户"` — 这个串不得被当成人名
   - `"field:phone"` — 不得出现该字段
   - `"value:上海"` — 不得出现该取值（用于否定句、假设句）
3. **`max_people` 是廉价的幻觉探针**：一句话只提到一个人，却抽出 3 个 → 大概率把「他老婆」「那个客户」当人名了。

**匹配写法**：字段 `{"field": "city", "contains": "杭州"}`，也可以用 `equals` 或 `any_of`
（日期格式多种写法时用 `any_of: ["03-12","3月12"]`）；事实 `{"contains": ["医院"], "scope": "any"}`
（`scope: any` 表示字段值里出现也算）；承诺 `{"contains": "报价单", "due": "2026-10-07", "who": "me"}`；
账本 `{"direction": "out", "amount": 2000}`（单位是**元**，内部自动换算成分）。

## evidence 子串校验

每条抽出的 `evidence` 必须逐字出现在原文里（比对时忽略空白与标点差异）。
这是免费、无歧义的幻觉探针——今天那个「客套话被当承诺」的 bug 就属于**内容对得上原文、
但判断错了**，所以除了 evidence 命中率，还要单独看 `promise_false_positive_rate`。

## 当前门槛

| 指标 | 门槛 | 含义 |
|------|------|------|
| `evidence_hit_rate` | ≥ 0.95 | 引用的原文片段真的在原文里 |
| `promise_false_positive_rate` | ≤ 0.05 | 标了 negative-promise 的用例里仍抽出承诺的比例 |
| `pass_rate` | ≥ 0.70 | 严格口径：一条用例所有断言都过才算过 |

L5 不变量层不设门槛——**必须 100%**。

## 用例从哪来（按性价比排序）

1. **PRD 的 20 个验收用例**（`home-PRD.md` 第 11 节）：由真实事故反推，大多是状态断言，天生可自动。
   其中 AC-01/03/04/05/15/16 已在 `tests/` 有覆盖；AC-06~14、AC-17~20 需要 L4 端到端层。
2. **真实回流**：导入自己的聊天记录后，把抽错的条目标出来 → 每条就是一个回归用例。
   增长最快、最贴真实分布，建议优先做这条。
3. **合成**：按「字段类型 × 表达方式」矩阵造（明说 / 暗示 / 转述 / 引号内 / 否定 / 假设）。
   注意别当唯一来源，会过拟合自己的生成风格。

## 已知缺口

- **L3 检索层**没做，`retrieve.hybrid_search` 目前的排序质量无量化依据。
- **L4 端到端**没做，Agent 工具调用是否越权、多轮上下文是否正确，目前没有回归。
- **无 LLM-as-judge**：`value_read` 的质量、回复是否得体这类主观项还没评。若引入 judge，
  必须先做人工双盲校准（抽 20 条算 Cohen's κ，≥0.6 才能信它的分数）。
- `tests/test_invariants.py::test_p5_delete_is_physical` 标了 `xfail`：P5 要求「删除是真删除」，
  现在 `repo.soft_delete_person` 只打 `deleted_at`，字段与别名行仍留在库里。行为层没问题（AC-02 过），
  数据层未达标，需要产品决策：物理删除（含 blob 与证据），或者 UI 上改口径为「归档」并提供彻底清除。
