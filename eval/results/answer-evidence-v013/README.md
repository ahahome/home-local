# 0.13 资料回答与工作流验证

全部使用虚构开发资料、独立进程与临时SQLite，不接触用户工作库。分类使用配置的 `qwen3.7-flash`，回答/资料编辑使用 `qwen3.7-plus`。每个声明版本和模式仅采样一次；保留失败，不重跑筛绿。代理逐答审阅不等于独立人工评分；原始JSON中的语义评分和审阅状态不因旁注而改写。

## 三个实现阶段

| 阶段 | 应用实现SHA256前缀 | 语料与结果 | 如何理解 |
| --- | --- | --- | --- |
| 首版 | `267024bc` | 新20答：383/386，0运行异常；工作流26/41 | 新问法在首版实现时未用于调试；失败后即成为已知开发语料。工作流未实际进入确认，不能声称确认API失败。 |
| 第二阶段 | `d8d0fc28` | 同题20答391/393，0异常；额外新8请求仅7终稿、1运行异常，72/72为已执行检查；旧0.12的20答475/475 | 真实公司名抽取错误、生成任务遗漏及部分写入后来源变更异常均保留；数字不覆盖全部语义任务。 |
| 第二阶段工作流 | `d8d0fc28` | 用户审阅闭环57/57，严格提案2/3，总59/60，失败退出 | 模型原提案质量与预声明用户修订后的产品闭环分开统计；不是用预期提案替换模型输出。 |
| 最终聚焦回归 | `e24ebb88` | 已知8答99/99，0运行异常 | 实际触发只读恢复，city仅保存一次；改稿和提醒保留，但首次起草仍未完成、流程表述有张力，身份抽取仍有无据待确认提案。 |
| 最终工作流 | `e24ebb88` | 用户审阅闭环57/57，严格提案3/3，总60/60，正常退出 | 记忆evidence含归因及完整条件，用户仍编辑content补归因；两份沟通草稿仍无据声称有附件，机器检查未覆盖这些正文问题。 |

最终实现SHA256：`e24ebb886345a6a977fc7372680af1ccbb37bef88b8fae92ac58579c4ff0d716`。最终仅对额外8答作已知开发回归，并再次运行用户审阅工作流；第二阶段整组20答及旧0.12回归不改标为最终hash。应用hash覆盖 `app/`、`migrations/` 的Python/SQL文件，具体方法在 `eval.memory_questions.app_source_hash`。

## 原始报告与逐答旁注

- [首版20答与检索](development-baseline/)；[完整逐答审阅](development-baseline/acceptance-evidence-review.md)。固定检索2例10/10，Recall@1=.75、@3/@5=1、MRR@5=1，仅为词面小样本。
- [首版工作流](workflow/)；[首版工作流证据审阅](workflow/workflow-evidence-review.md)。原评测器单独保存在 `workflow/runner-original.py`。
- [首版旧0.12回归](regression-v012/)；[对应逐答审阅](regression-evidence-review.md)。20答473/473。
- [第二阶段已知20答](final-regression/)；[逐答审阅](final-regression/regression-evidence-review.md)。
- [第二阶段额外新8请求](final-followups/)；[含运行异常的完整审阅](final-followups/followups-evidence-review.md)。当时的新措辞未用于实现调试，后续复验时已经是已知开发语料。其固定检索1例4/4，仅为单题诊断。
- [第二阶段旧0.12回归](final-regression-v012/)；[逐答审阅](final-regression-v012/evidence-review.md)。该20答不含生成任务。
- [第二阶段用户审阅工作流](final-workflow-user-review/)；[逐步审阅](final-workflow-user-review/workflow-evidence-review.md)。
- [无模型来源冲突机制重放](source-change-diagnostic/README.md)。真实报告缺工具轨迹，不能把机制复现当作真实那次工具调用的证明。
- [最终已知8答聚焦回归](final-focused-regression/)；[逐答审阅](final-focused-regression/focused-evidence-review.md)。
- [最终用户审阅工作流](final-workflow/)；[逐步审阅](final-workflow/workflow-evidence-review.md)。

目录中 `final-*` 的早期命名保持不变；报告内完整应用hash决定其所属阶段。原始报告、原始来源、领域变更及失败不删除或重写。

[SHA256清单](manifest.json)逐文件索引本目录的报告、旁注、归档评测器与诊断脚本，排除清单自身；路径相对于仓库根目录，可据此核验留档材料。

## 冻结与限制

问答评测器SHA256：`0cab27da2a00ae850c9f1bfd8f9429a28f8ba021945a8f3e98d05d3d7265190c`。

首版工作流评测器SHA256：`595352328090a9926cdc11d557177849eab75c202dca8e2ad0def01a62119123`；加入显式用户审阅、真实来源回链、非流式响应审计与完整表值冲突检查后的评测器SHA256：`b6ffafb74907be7440f29a008d4100e8854b2ff4db8839689dd68858412ce5e3`。不能把两种模式的检查数直接当模型能力提升。

服务端卡片忠实展示已存快照，不能证明抽取值正确或来源与问题相关。生成模式仍采用保守语言规则；自由草稿/建议仍可能偏离原流程、无据声称附件已存在，或把“未记录”误写成“从未问过”。严格JSON与来源检查都不验证这些语义。已观察到公司名被拆错并保存，用户确认与纠错仍必要。

更多实现与复现方法见[0.13验收文档](../../../docs/ANSWER_EVIDENCE_V013.md)。
