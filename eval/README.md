# 评测集

当前实现的验收结果见 [主 README](../README.md)、[交付说明](../docs/DELIVERY.md) 与 [模拟验收](../docs/MOCK_QA.md)。2026-10-03 对已有 30 条 holdout 样本各采样一次、关闭缓存的 [网关复测](results/20261003-130302-a2a33e1.json) 通过 29/30，字段召回 0.75；这不是全新留出集，仍存在公司字段漏抽。此前完整运行是 28/30，简单合成样本的 82/82 字段匹配另行报告。下文的历史泛化缺口和「本轮全过」描述来自原评测框架建立阶段，不应作为当前模型分数。

0.14 写入与草稿验收、0.13 资料回答与工作流验收分别列在下文；各阶段按实现 hash 保留原始失败和修后回归，不能与上述 L1 历史抽取分数混算。

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

## 0.14 记忆写入与沟通草稿

`fixtures/write_quality_v014.json` 共 9 场景、18 步，分别检查同名别名归属、机构与显式职位、换人与旧消息重试、邮件及改稿、持久沟通准备。`write_quality_v014.py` 执行真实生产 harness、写工具与 HTTP/持久任务；每场景使用独立进程和临时 SQLite，不打开用户工作库。模型为配置的 `qwen3.7-flash` / `qwen3.7-plus`，关闭 embedding 和缓存。完整方法见 [0.14 验收](../docs/WRITE_QUALITY_V014.md)，原始报告、失败及逐步旁注见 [结果索引](results/write-quality-v014/README.md)。

| 冻结实现 | 语料范围 | 终稿步骤 / 运行异常 | 已执行机器检查 |
| --- | --- | --- | --- |
| 首次 `46bc6fea…` | 已知回归，6 场景 | 12 / 0 | 194/195 |
| 首次 `46bc6fea…` | 首次新措辞，3 场景 | 6 / 0 | 104/106 |
| 最终 `2805f5d1…` | 首次失败三场景的全部原 6 步，定向已知回归 | 6 / 0 | 94/94 |

首次两组均失败退出，保留代词城市未保存、草稿缺失及改稿人物 scope 丢失等失败。新措辞由内部验收代理在实现冻结前保留，仍是同源构造的开发资料；首次采样后即成为已知语料。派生 `fixtures/write_quality_v014_regression.json` 不改三场景的种子、原问题和预期，仅用于修后定向回归。每组只采样一次，不重采筛绿，也不相加为模型准确率。最终实现仅重新采样这 6 步；机构/职位、旧消息重试和 followup 的完整真实结果仍对应首次实现，最终相关行为由控制测试支撑。

最终冻结时自动回归为 **1752 项 Python 测试、6 项 Node 界面重试测试通过**，Web 语法检查通过。这些是无模型控制检查。真实定向回归确认贵阳实际保存到正确同名人物、city 与 applied_patch 各新增一次，写入后只读恢复没有重放写入；四份邮件/改稿均实际交付 draft，两次改稿继承人物并重新读取资料。原抽取仍可能返回空 people，本次由后续真实工具补存，不能据此称漏抽消失。

**94/94 不代表草稿语义全过。** 实际输出仍有先询价后补规格、弱化样张通过后再谈排产条件的偏差，发送前需要人工核对。`generation_task` 仅验证段落形状，所有生成内容保持 `semantic_verified=false`；原始语义评分为 null、审阅待定，代理旁注不替代独立人工评分。人物自动隔离检查覆盖指定人物表，承诺、账目、提醒和关系等差分仍需另审；准备状态写入也不能解释成全库零写入。

```sh
# 离线验证 fixture，不调用模型。
.venv/bin/python -m eval.write_quality_v014 --validate-only

# 以下会调用真实模型；输出到新目录，不覆盖已归档结果。
.venv/bin/python -m eval.write_quality_v014 --cohort known_regression --output-dir /tmp/home-v014-known
.venv/bin/python -m eval.write_quality_v014 --cohort new_wording --output-dir /tmp/home-v014-wording
.venv/bin/python -m eval.write_quality_v014 --fixture eval/fixtures/write_quality_v014_regression.json --cohort known_regression --output-dir /tmp/home-v014-targeted
```

在当前代码上执行上述命令会形成新的已知开发回归，不能复现为首次未见采样。结果记录实现、fixture、runner 与 helper hash，比较时须核对完整 hash 和实际检查范围；冻结协议与归档完整性见 [SHA256 清单](results/write-quality-v014/manifest.json)。

## 0.13 资料回答与完整关系工作流

`fixtures/answer_evidence_v013.json` 覆盖 9 场景 20 答及 2 个固定检索，`answer_followups_v013.json` 另含 4 场景 8 请求。资料事实由服务端来源快照展示；生成正文仍需独立语义核对。`relationship_workflow_v013.py --user-review` 通过真实 HTTP 路由和持久任务执行查询 → 准备沟通 → 记录实际结果 → 分析提案 → 显式用户审阅确认 → 再次查询与准备。模拟用户修订来自预声明 fixture，与原模型提案质量分开计数，不自动把预期提案写成模型结果。

结果按三个实现阶段留档：首版 20 答 **383/386**、工作流 **26/41**；第二阶段同题 20 答 **391/393**，额外 8 请求仅 7 份终稿、1 次运行异常，**72/72 仅为已执行检查**。第二阶段用户审阅闭环 **57/57**、原提案严格检查 **2/3**，总 **59/60**，失败退出。最终 `e24ebb88…` 仅复验已知 8 答，**99/99、0 运行异常**；最终用户审阅闭环 **57/57**、原提案严格检查 **3/3**，总 **60/60**。第二阶段整组 20 答和旧 0.12 回归没有改标为最终实现结果。

最终仍观察到首次邮件未实际起草、无依据身份待确认提案、称呼推成工作事实，以及沟通草稿无据声称已有附件。来源一致、协议完成、显式用户修订后的产品闭环均不证明一般语义正确。原始语义评分和审阅状态保持不变。完整阶段区分与限制见 [0.13 验收](../docs/ANSWER_EVIDENCE_V013.md) 和 [结果索引](results/answer-evidence-v013/README.md)。

```sh
# 以下问答与工作流会调用真实模型，使用独立临时库。
.venv/bin/python -m eval.memory_questions --track answers --cases eval/fixtures/answer_evidence_v013.json --repeat 1 --output-dir /tmp/home-v013-answers
.venv/bin/python -m eval.memory_questions --track retrieval --cases eval/fixtures/answer_evidence_v013.json --output-dir /tmp/home-v013-retrieval
.venv/bin/python -m eval.memory_questions --track answers --cases eval/fixtures/answer_followups_v013.json --repeat 1 --output-dir /tmp/home-v013-followups
.venv/bin/python -m eval.relationship_workflow_v013 --user-review --output-dir /tmp/home-v013-workflow
```

固定检索默认离线，不调用模型；当前代码上的复跑属于当前实现的已知开发回归，不能复现为当时的新措辞检查。原始失败、各阶段报告和独立旁注保留在结果目录，并由 [SHA256 清单](results/answer-evidence-v013/manifest.json) 索引。

## 0.12 多轮记忆集合与证据比较

`fixtures/memory_dialogue_v012.json` 在真实采样前冻结，8 组、每次 20 答，覆盖同人物集合比较及编辑、撤销和转移，普通日期问法、恢复、明确换人、同名、多人物与单数歧义，以及 record/mixed 的实际保存。使用虚构开发资料，不能称为独立未见语义基准。实现与最终结果见 [0.12 验收](../docs/MEMORY_DIALOGUE_V012.md)。

```sh
.venv/bin/python -m eval.memory_questions --track answers --cases eval/fixtures/memory_dialogue_v012.json --repeat 3 --output-dir /tmp/home-v012-acceptance
```

新增 `required_dependency_records` 与 `required_cited_records` 非空记录集合，分别检查实际已读依赖和最终引用，不能互相替代。`expect.memory_focus.records` 检查精确当前成员、原问题持久绑定、同对话最近前序终稿、旧快照逐字一致，以及完整本轮注册。日期元数据对照实际纠错审计；缺失旧来源状态明确失败，没有相应断言的范围显示 `not_exercised`。既有 `source_records` 仍只作审阅提示，不暗中升级为引用要求。负控包含缺源、重复集合、旧版本、伪造正文/锚点和日期，语义判断仍单列待人工审阅。

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
