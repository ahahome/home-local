# 评测集

四层判定，成本不同。**免费层跑在每次提交上；花钱层只在改 prompt / 改抽取逻辑时跑。**

| 层 | 位置 | 花钱 | 覆盖什么 |
|----|------|------|---------|
| **L1 抽取** | `eval/run.py` + `eval/cases/tune.jsonl` `holdout.jsonl` | 是 | 从文本抽人物/字段/事实/承诺/账本对不对 |
| **L2 落库** | `tests/test_invariants.py` | 否 | 三档置信度、幂等、待确认不进汇总、删除不复活 |
| **L3 检索** | 待补 | 否 | Recall@5 / MRR（需要 (query, 相关 id) 标注） |
| **L5 不变量** | `tests/test_invariants.py` | 否 | 七原则：违反即失败，没有「大部分情况对」 |
| L4 端到端 | 待补 | 最贵 | 多轮脚本 → 库状态 + 工具轨迹（PRD 里 AC-06~14、AC-17~20 属于这层） |

评估器本身的比对逻辑在 `eval/metrics.py`，由 `tests/test_eval_metrics.py` 覆盖。
**理由：评估器写错了比被测代码错了更危险**——指标会假绿，问题被掩盖。

## 分集：tune / holdout（L1 的核心纪律）

| 集 | 文件 | 用途 | 卡门槛？ |
|----|------|------|---------|
| **tune** | `eval/cases/tune.jsonl` | 提示词/代码**就是对着它调的**；只作回归守卫 | **不卡** |
| **holdout** | `eval/cases/holdout.jsonl` | 出题只看 PRD、不看提示词，写完冻结 | **只有它卡** |

**对 tune 调参、用 holdout 验收，两个数字分开报。** 混成一个数，就等于给自己发奖状。

- 两者的差 = **泛化缺口**（当前 0.9773 − 0.9667 ≈ 1.1 个百分点；建立之初是 7.7pt，
  历史峰值 8.3pt——缺口收窄是因为 tune 里掺进了两批烧下来的硬骨头，调参集本身变难了），
  也就是「tune 全绿」里有多少是对着答案改出来的。
- **holdout 会被烧掉**：一旦照着它的失败去改提示词，它就退化成 tune。
  所以每一轮修完，必须把旧 holdout 并入 tune、**新建 holdout v(k+1)**。协议见 `results/BASELINE.md` 第九节。
- **已经烧过两次**（2026-09-30 / 2026-10-01）：v1 的 26 条（`from-holdout-v1`）与 v2 的 30 条
  （`from-holdout-v2`）先后并入 `tune.jsonl`（tune 因此 32 → 58 → **88** 条）；
  当前 `holdout.jsonl` 是**按 PRD 重出的 v3**（30 条，其中 12 条 negative-promise）。
  用法例溯源自检：
  ```bash
  .venv/bin/python -c "import json;print(sum(1 for l in open('eval/cases/tune.jsonl',encoding='utf-8') if 'from-holdout' in l))"
  ```

## 每集采样 k 次（`--repeat`）

单次通过率会骗人（实测出现过「连过两次 32/32，第三次掉到 31/32」）。所以：

```bash
.venv/bin/python eval/run.py --repeat 3        # 每条采样 3 次，按最差判定（worst-of-k）
```

- 判定用**最差那次**，报告里同时给 `runs_passed/runs_total`。
- 单独列出 **flaky**（k 次里有的过、有的挂）——它们最值得看：说明规则只是概率好转，还没落成确定性保证。
- 缓存键含 `run_index`，所以 k 次会拿到 k 个不同样本，且**可复现**（同一缓存下重跑得到同一结论）。
  想要全新样本加 `--no-cache`。

## 怎么跑

```bash
cd /Users/emdoor/Documents/projects/integrate/home

# 免费层（不调 LLM，秒级）
.venv/bin/python -m pytest tests/ -q --basetemp=.pytest-tmp

# L1：两集都跑，每条采样 3 次（推荐口径）
.venv/bin/python eval/run.py --repeat 3

# 只看留出集（门槛只对它生效）
.venv/bin/python eval/run.py --split holdout --repeat 3

# 只跑某一类
.venv/bin/python eval/run.py --tag negative-promise
.venv/bin/python eval/run.py --tag real

# 调试单条：跳过缓存重跑
.venv/bin/python eval/run.py --cases eval/cases/holdout.jsonl --limit 5 --no-cache

# 当成门禁：门槛不达标时退出码非 0（只对 holdout 生效，给 CI 用）
.venv/bin/python eval/run.py --gate
```

结果落在 `eval/results/<时间戳>-<git sha>.json`，内含 `splits.{tune,holdout}` 两套汇总 + 明细，
并记录 `model`、`prompt_hash`、`git_sha`、`repeat`、`no_cache`，任何一次分数都能复现和回溯。
响应缓存放在 `eval/.cache/`
（key = `sha256(input + known_persons + today + prompt_hash + model + run_index)`），重复跑不重复花钱。

> ⚠️ **`--no-cache` 不是可选项（改后处理代码时）**：缓存存的是 **`sanitize()` 之后**的结果，
> 而缓存键只含**提示词** hash。所以只改抽取后处理、不改提示词时，缓存会返回旧代码的结果，
> 让你量出一个假的「没变化」。改 `app/memory/extract.py` 的后处理逻辑务必加 `--no-cache`。

## 用例怎么写

一行一条 JSON，见 `eval/cases/tune.jsonl` / `holdout.jsonl`。字段：

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

**门槛只对 holdout 生效**（`eval/run.py:GATED_SPLITS`）。tune 是调参集，卡它等于自己给自己发奖状。

L5 不变量层不设门槛——**必须 100%**。

## 用例从哪来（按性价比排序）

1. **PRD 的 20 个验收用例**（`home-PRD.md` 第 11 节）：由真实事故反推，大多是状态断言，天生可自动。
   其中 AC-01/03/04/05/15/16 已在 `tests/` 有覆盖；AC-06~14、AC-17~20 需要 L4 端到端层。
2. **真实回流**：导入自己的聊天记录后，把抽错的条目标出来 → 每条就是一个回归用例。
   增长最快、最贴真实分布，建议优先做这条。
3. **合成**：按「字段类型 × 表达方式」矩阵造（明说 / 暗示 / 转述 / 引号内 / 否定 / 假设）。
   注意别当唯一来源，会过拟合自己的生成风格。

## 已知缺口

- **`must_not` 没有作用域**：`field:X` / `value:X` 都是**全局**的，写不出
  「**只**检查某个人身上不该有这个字段」。attribution 类断言（PRD FR-MEM-06「不得把他人特征
  归档到当前人物名下」）因此只能靠 `max_people` 间接覆盖。待支持 `王芳.field:title` 这种带人名的写法。
- **字段名无白名单**：模型可以自造 `family` 这类 schema 外的字段并原样落库
  （`sanitize()` 只做类型归一化，不校验字段名）。要不要加白名单需先定字段表。
- **L3 检索层**没做，`retrieve.hybrid_search` 目前的排序质量无量化依据。
- **L4 端到端**没做，Agent 工具调用是否越权、多轮上下文是否正确，目前没有回归。
- **负样本仍偏少**：v1 只有 5 条 `negative-promise`（误判率分辨率 0.2 一档）→ v2 加到 11 条 →
  **v3 现为 12 条**（分辨率 0.083）。目标 20 条以上，且要覆盖**闸门词表里还没有的说法**
  （v3 已放「改日」「我妈」「回头请你喝两杯」这类探针——本轮全过，但过的是模型而不是词表，
  缺口仍在，见 `results/BASELINE.md` 第四节）。
- **无 LLM-as-judge**：`value_read` 的质量、回复是否得体这类主观项还没评。若引入 judge，
  必须先做人工双盲校准（抽 20 条算 Cohen's κ，≥0.6 才能信它的分数）。
- **P5 删除语义：已实现（两级）**。
  - `DELETE /api/persons/{id}` = **归档**：打 `deleted_at`，从列表/检索里消失、不复活，但可
    `POST /api/persons/{id}/restore` 恢复（UI 有「已归档」面板，否则「可恢复」只是口号）。
  - `POST /api/persons/{id}/purge` = **彻底清除**：物理删除字段/别名/记忆/账本/提醒/互动关联/
    关系边/向量索引/该人独占的凭证与 blob；删完调 `repo.verify_purged()` 自检，不通过就报
    「**部分删除**」而不是「已删除」（PRD FR-ID-02 的验收原文要求的）。
  - 原 `xfail` 的 `test_p5_delete_is_physical` 已拆成两条并转正：
    `test_p5_purge_removes_every_trace` + `test_p5_archive_is_reversible`。
  - ⚠️ 一处**刻意的口径修正**：schema 明确允许两个人共享同一别名（三个「小张」），
    所以验收口径是「**这个别名再也指不到被清除的那个人**」，而不是字面的「全库搜不到这个别名」——
    后者在共享别名时会误报「部分删除」。判据写在 `verify_purged()` 的 docstring 里。
