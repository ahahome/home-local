# 评测集

当前实现的验收结果见 [主 README](../README.md)、[交付说明](../docs/DELIVERY.md) 与 [模拟验收](../docs/MOCK_QA.md)。2026-10-03 对已有 30 条 holdout 样本各采样一次、关闭缓存的 [网关复测](results/20261003-130302-a2a33e1.json) 通过 29/30，字段召回 0.75；这不是全新留出集，仍存在公司字段漏抽。此前完整运行是 28/30，简单合成样本的 82/82 字段匹配另行报告。下文的历史泛化缺口和「本轮全过」描述来自原评测框架建立阶段，不应作为当前模型分数。

四层判定，成本不同。**免费层跑在每次提交上；花钱层只在改 prompt / 改抽取逻辑时跑。**

| 层 | 位置 | 花钱 | 覆盖什么 |
|----|------|------|---------|
| **L1 抽取** | `eval/run.py` + `eval/cases/tune.jsonl` `holdout.jsonl` | 是 | 从文本抽人物/字段/事实/承诺/账本对不对 |
| **L2 落库** | `tests/test_invariants.py`、`test_workflows.py`、`test_corrections.py`、`test_extraction_validation.py` | 否 | 置信度、金额、幂等、方向确认、跨入口写入、历史与纠错 |
| **L3 检索** | `eval/memory_retrieval.py`、`eval/memory_questions.py --track retrieval` | 否 | 合成生命周期回归、虚构资料自然问法的 Recall@1/3/5 与 MRR@5，来源检查单列 |
| **L5 不变量** | `tests/test_invariants.py` | 否 | 七原则：违反即失败，没有「大部分情况对」 |
| L4 任务链路 | `tests/test_workflows.py`、`scripts/mock_e2e.py`、`eval/memory_questions.py --track answers` | 真实链路需要 API | 实际 harness 多轮、抽取、来源和工具记录；回答语义需要单独人工审阅 |

评估器本身的比对逻辑在 `eval/metrics.py`，由 `tests/test_eval_metrics.py` 覆盖。
**理由：评估器写错了比被测代码错了更危险**——指标会假绿，问题被掩盖。

## 0.10 分类诊断与记忆追问

`fixtures/answer_stability_v010.json` 在真实调用前冻结，7 组、12 问，正式重复两轮。沿用 0.9 虚构资料、增加自然问法，覆盖别名与恢复原文、唯一记忆日期追问、多记忆澄清、冲突范围、简短缺资料、无问号查询、带条件尾句的明确记事及 mixed 真实更新。另回归 0.9 和 0.8 原语料。完整结果见 [0.10 验收](../docs/ANSWER_STABILITY_V010.md)，逐轮代理证据核对见 [审阅记录](../docs/ANSWER_STABILITY_V010_REVIEW.md)；语义仍待人工审阅。

```sh
.venv/bin/python -m eval.memory_questions --track answers --cases eval/fixtures/answer_stability_v010.json --repeat 2 --output-dir /tmp/home-v010-acceptance
```

报告新增 `classifier_diagnostics`（意图/状态、实际请求与重试、失败类别、结束原因、用量）、`answer_review_diagnostics`（协议完成/降级与问题摘录审计状态）、`memory_focus_diagnostics`（绑定状态、原因及来源注册）。`expect.memory_focus` 可检查确定性的绑定契约。这些统计不替代原有实际领域表变化、当前字段和来源检查；分类标签匹配也不是模型语义正确率。日期模板的零请求查询与模型分类分别统计，不能把它计作模型分类成功样本。

## 0.9 查询写入与回答可信度

`fixtures/answer_trust_v09.json` 在正式采样前冻结，含 7 组虚构开发对话、10 个回答，覆盖恢复状态与编辑日期、冲突偏好、无问号查询、假设、旧回答引文、更新与查询混合，以及带问号的明确记事。使用同一 runner 对机械导出的 0.8 基线和当前实现采样；当前实现重复两次，另用 0.8 原语料做回归。它不是独立真实用户基准。

```sh
.venv/bin/python -m eval.memory_questions --track answers --cases eval/fixtures/answer_trust_v09.json --repeat 2 --output-dir /tmp/home-v09-acceptance
```

`expect.intent` 核对保存的分类元数据；旧实现没有此功能的失败须与实际误写分开。`expect.no_memory_writes` 比较数据库变化，不只相信模型标签。快照覆盖人物、别名、档案字段、记忆、待确认、互动、互动人物、关系、承诺、账目、提醒、应用记录、纠错与证据；不覆盖所有数据库表，技能与研究的边界另有工具及集成测试。查询允许新增审计证据摘录（至多 4000 字符）；完整原文保留在会话。`expect.saved_fields` 检查当前真实字段，不能用模型的成功声明或待确认提案替代。

报告保留意图决定、草稿、修订稿、实际回执、工具轨迹和写入差异。`configured_model_tiers` 记录配置的分类与修订模型层级，与对话生成模型列表分开。`answer_review.status=completed` 只代表段落协议及来源校验，`semantic_verified=false`；评测的 `semantic_review` 继续为待人工审阅，不能将模型自己的问题标签当作语义分数。完整结果、逐轮反例与限制见 [0.9 验收](../docs/ANSWER_TRUST_V09.md)。

## 0.8 记忆检索与真实回答

`fixtures/memory_questions_v08.json` 是**虚构资料与自然语言问法**，不包含真实用户记录，也不应称为独立未见基准。它是开发回归集，包含 16 个固定检索或来源场景，其中 10 个正召回问法、5 个无答案或身份/可见性负例及 1 个档案时间状态场景。另外有 6 组对话、13 个目标回答，覆盖昵称、口语、指代、同名澄清、当前与历史信息、冲突、缺少答案，以及修改、撤销、恢复、转移与归档。

四种结果分别报告，禁止混为“模型准确率”：

| 结果 | 判定依据 | 局限 |
| --- | --- | --- |
| 固定问法排名 | 预先标注的相关记录 ID，Recall@1/3/5、MRR@5 | 不衡量模型生成；无答案、歧义和档案来源检查不进入排名分母 |
| 确定性检查 | 实际数据库行、证据原文、人物归属、字段状态和来源生命周期 | 引用真实不代表结论得到支持 |
| `limited_fact_checks` | 明确列出的字符串存在或缺失规则 | 不能判断否定、指代、时间关系和推论；例如“不能发邮件”也含有“邮件” |
| `semantic_review` | 人工审阅实际回答、当时资料、引用及工具记录 | 默认 `review_pending`、`score=null`，不使用 LLM 自评 |

固定检索只传用户问法及 fixture 中明确的姓名或范围参数，不暗中注入人物 ID。`expect.relevant` 是固定检索的相关性标签。对话中的 `expect.source_records` 只列出可能相关的证据，供人工审阅使用，不要求必须全部引用，也不用于给回答打准确率。`evidence_access` 分别列出读取和引用过的相关记录，仅用于定位漏读资料。

```bash
# 默认离线检索：禁用向量与网络，不调用模型。
.venv/bin/python -m eval.memory_questions --track retrieval --output-dir /tmp/home-v08-retrieval

# 真实回答：实际调用 harness.run_turn，保留开头的真实抽取、工具与多轮历史。
# 使用 runner 所在仓库的模型配置，会产生真实模型调用。
.venv/bin/python -m eval.memory_questions --track answers --repeat 2 --output-dir /tmp/home-v08-answers

# 按标签选择，多个 --tag 取并集；--limit 限制场景数量，完整保留选中对话的各轮。
.venv/bin/python -m eval.memory_questions --track answers --tag lifecycle --limit 1 --worker-timeout 600

# 使用同一 runner 和 fixture 对比另一份实现；app-root 在导入 app 前生效。
.venv/bin/python -m eval.memory_questions --track retrieval --app-root /absolute/path/to/baseline --output-dir /tmp/home-v07-baseline
```

`HOME_EVAL_APP_ROOT` 可代替 `--app-root`。实现目录必须包含 `app/` 和 `migrations/`；不要求包含 0.8 的评测文件。模型配置始终从 runner 仓库的设置及环境加载，不输出密钥。两轨都禁用 embedding，这份报告不能衡量向量语义召回。

每个场景及每次 repeat 使用独立子进程、临时 SQLite 和附件目录。多轮对话在自己的场景内共享历史。既不打开用户数据库，也不缓存模型结果。真实抽取可能新增或修改资料，报告会保存这些变化；新增记录不会自动加入预标相关记录集合。CLI 内部串行执行，可通过独立 CLI 进程并行运行不同标签，不共享连接和设置补丁。

JSON 和可人工审阅的 Markdown 在初始化及每完成一个场景后原子更新。`run_status=running` 表示尚未完成，`complete` 表示选中场景结束。每场景默认 600 秒硬超时，可用 `--worker-timeout` 修改；超时终止该 worker，保留错误并继续其他场景。

报告记录 fixture 与应用源码 SHA256、runner SHA256、模式、配置、repeat、Git 状态及运行期间应用源码是否变化。解压的实现目录没有自己的 Git 根时，`app_git` 为 null，不借用上层仓库的版本。比较两次运行时必须核对 fixture hash、配置、样本数量和实现 hash；运行期间源码变化的结果不可当作稳定版本对比。`scenario_errors` 统计场景级失败，`failed_turns` 统计未生成的回答轮次，`runtime_errors` 不重复叠加同一个 worker 失败及其目标回答。

退出码非零表示运行异常、确定性检查失败、有限字符串规则失败或运行期间实现发生变化。低召回按实际数值报告，不设暗含的通过门槛。人工审阅待完成不视为语义通过。评测器负控在 `tests/test_memory_questions_eval.py`，包含错误引用、历史状态伪造、错误命中、空回答及失败样本分母检查。

0.7 的 `memory_retrieval.py` 继续保留，默认 k=5、两项范围压力场景 k=1，使用它自己的 `recall_at_k` 口径。它与 0.8 的固定 Recall@1/3/5 不应直接拼成一个分数。

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
# 在项目根目录执行；下面为 macOS/Linux 命令，Windows 入口见主 README。

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
- **L3 已有开发集测量**：0.7 检索生命周期回归与 0.8 自然问法回归分别报告。缺少真实用户分布和独立未见检索集，且当前评测禁用向量，不能据此推断开放问法与 embedding 召回质量。
- **L4 已覆盖部分任务链路**：跨入口和状态回归在 `test_workflows.py`，真实对话、图片、群聊与幂等验收在 `scripts/mock_e2e.py`。0.8 增加实际 harness 多轮记录与人工审阅材料，仍不能把来源检查或字符串规则当作多轮语义准确率。
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
