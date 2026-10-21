# 0.14 写入与草稿结果索引

所有联系人和内容均为虚构开发语料；模型使用项目真实配置，所有执行库为独立临时 SQLite。报告不包含模型密钥，不启动 Web 服务，也不打开用户工作库。原始失败、原答案、null 语义评分和机器待审阅状态全部保留，已完成的逐步代理核对放在独立 `evidence-review.md`。

## 首次冻结采样

首次实现可由本地提交 `929cbe6` 还原：app/migrations SHA256 为 `46bc6fea5b49a26c186f618e3608dbf019d0a31cdda922a0f7d3ac431268af0e`。两组在同一实现并行采样，各场景只运行一次，前后 app hash 均未改变。

| 组别 | 场景/步骤 | 运行异常 | 已执行机器检查 | 原始报告 | 证据旁注 |
| --- | --- | --- | --- | --- | --- |
| 已知回归 | 6/12 | 0 | 194/195 | [JSON](first-known-regression/write-quality-20261021-012623-903626-2651-46bc6fea.json) | [逐步审阅](first-known-regression/evidence-review.md) |
| 首次新措辞 | 3/6 | 0 | 104/106 | [JSON](first-new-wording/write-quality-20261021-012623-903639-2650-46bc6fea.json) | [逐步审阅](first-new-wording/evidence-review.md) |

两组均以退出码 1 结束，没有失败后重采筛绿。合计 18 步终稿、0 运行异常、机器检查 298/301；该检查数不是答案或草稿的正确率。机器检查失败与逐句语义问题分别是：

- 已知同名别名 mixed 的长沙、完整“云岸策展”、显式公司/职位、换人后的两座城市均真实写到正确人物，无未处理 pending。旧用户消息重试复用原 ID/原 scope，实际域表变化为空，applied_patch 未增加；多人回查值正确。
- 新措辞的机构“森屿空间设计”和显式采购主管真实正确保存，但“顺便替我存一下，她已搬到贵阳了”虽人物/意图正确，抽取为空，工具写权限又拒绝，贵阳没有落库。不是正确值挂在 pending，也没有写到另一人。
- 两组的改稿第二步都丢失人物 scope，分别是一项机器失败。已知初次邮件由于审阅把实际邮件放在 advice 而无 draft 段，触发明确失败提示；已知改稿长度符合三十字，却有残句和条件遗漏。新措辞“起封邮件”“改成两句话”均未被展示合同识别为生成任务，原聚合草稿最终被过滤，任务失败。
- 真实 followup 准备邮件这次有问题清单/署名占位和发送前步骤，没有把不存在的附件或未完成承诺当完成事实。准备真实写入 followup/task 等状态；通用域表 `data_changes={}` 只表示被捕获事实域没有变化，不能说整个 SQLite 零写入。

固定语料 `eval/fixtures/write_quality_v014.json` 的规范化 SHA256 为 `868eb486b76775807c0ca60a85dd1080e10b5a309aa8f534390b75e8d3de99f1`，文件 SHA256 为 `c36b55e7337dbce9d88e49ad1910c890a1cd015d8099ec3de64645467f65e498`。新措辞在实现冻结前由验收代理保留、未向实现者披露，仍是与本轮开发问题同源的内部构造，非外部盲测。首次采样后即为已知语料。

两组统一 runner `eval/write_quality_v014.py` 的 SHA256 为 `218e025cf295e565aa4e425a173677d1e5bae4081fd0100e648d5e72eef87366`。复用 helpers 分别为：

- `eval/memory_questions.py`：`0cab27da2a00ae850c9f1bfd8f9429a28f8ba021945a8f3e98d05d3d7265190c`。
- `eval/relationship_workflow_v013.py`：`b6ffafb74907be7440f29a008d4100e8854b2ff4db8839689dd68858412ce5e3`。

本次 mini=`qwen3.7-flash`，mid=`qwen3.7-plus`。两组共捕获 42 个非流式 `LLM.chat` 返回对象，content hash 逐条一致；另有实际流式回答聚合稿和工具轨迹。这不包括 provider HTTP 原始字节、逐个 stream chunk 或底层传输每次尝试，42 不能叫全部网络请求数。

## 后续已知定向回归的固定输入

首次结束后，派生 `eval/fixtures/write_quality_v014_regression.json`，恰含首次失败的三场景全部 6 步：`known-mail-and-revision`、`new-selected-alias-scope`、`new-mail-before-send`。种子、问题文本、步骤和预期未改，只统一 cohort 为 `known_regression` 并补上派生元信息；原 fixture 和原报告不变。

派生 fixture 规范化 SHA256 为 `1c1f131e34916580733b2038406b201478c17057c82b29a85902441f82154560`，文件 SHA256 为 `6cc86df51c6e294aabeebbe14a1364a6fc88010296f4775a1ec16995c016c991`。修复后的定向采样必须绑定新实现 hash，与首次结果分列；它是已知开发回归，不能称新的未见验收或以它覆盖首次失败。

## 修后定向已知回归

最终补修实现 SHA256 为 `2805f5d1fa33d778376601cb321daa99b3c8e29b750f7122e56c99b43d3d8758`。使用上述派生 fixture、同一 runner，3 场景各采样一次，前后及结束后现场复核 app hash 均一致。

| 范围 | 场景/步骤 | 运行异常 | 已执行机器检查 | 原始报告 | 证据旁注 |
| --- | --- | --- | --- | --- | --- |
| 首次失败题的定向已知回归 | 3/6 | 0 | 94/94 | [JSON](targeted-regression/write-quality-20261021-015638-708776-7362-2805f5d1.json) | [逐步审阅](targeted-regression/evidence-review.md) |

本次退出码 0，原 JSON 的语义评分仍为 null、审阅状态仍为机器原样；独立旁注不把 94/94 解释成语义全过。具体证据与残余是：

- 贵阳这次真实保存到登山袁牧，city 与 applied_patch 各一行，无错人、pending 或其他业务写入。初次抽取仍为 `people=[]`，实际成功来自后续 `save_person_fact` 工具；本次分类 span 比首次更短，不能单靠模型采样隔离证明新增前缀识别是唯一因果。写入后旧资料快照失效，真实执行一次只读恢复，最终 `recovered`，没有重放写入。
- 初次邮件和两次改稿终稿均实际有 draft，review 没有再次把整封邮件错标 advice。两次改稿都从紧邻完成草稿继承人物锚点，并在本轮重新读取当前人物资料。已知短改稿 29 字、无残句；乔安改稿恰两句，发送前核心缺项另列。
- 自由生成仍未完全遵守原条件。29 字稿“数量和交期整理好发你”没有显式占位，可读成先问价以后补信息；乔安邮件允许删除克重/数量占位先问常规建议，两句稿仍说“克重和印量还没定”“样张后面再说”，未清晰保留先给规格、样张通过后再谈排产。格式、来源与 scope 通过不等于这类流程语义通过。

本组捕获 13 条非流式 LLM.chat 返回，content hash 全部一致，审计边界与首次相同；未额外重采。最终代码没有重跑首次全部 18 步，完整机构/职位、旧消息重试及 followup 准备的真实采样证据仍对应首次 `46bc6fea…`，不能拼为最终 hash 的全量模型全绿。

## 自动检查的实际范围

当前字段从 SQLite 实际行核对，pending、回执与口头声明不能代替保存结果。`write_person_scope` 自动检查 person、alias、field、memory、interaction_person 五类变化；`unchanged_persons` 比较四类直接人物行。其余 promise/ledger/reminder/relation 等域表虽在 `data_changes` 中捕获，是否多写需结合逐步旁注审阅，不能把五类人物表检查宣传为全数据库人物隔离证明。

查询无写检查排除原问题 evidence；准备状态表不属于该通用域表快照。临时数据库会清理，报告保留行快照与差异，不是完整 SQLite 副本。所有自动检查和代理证据审阅都不验证一般语义，不计算独立人工准确率，也没有使用模型自评。
