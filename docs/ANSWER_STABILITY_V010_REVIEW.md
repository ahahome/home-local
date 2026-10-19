# v0.10 回答稳定性：代理逐轮证据审阅

本文由实现代理根据真实模型运行报告逐轮对照回答、实际来源、分类诊断和领域表变化，属于**代理证据审阅**，不是独立人工评分。正式报告继续保持 `semantic_review.status=review_pending`、`semantic_verified=false`；本文不提供语义准确率，也不把来源检查通过解释为回答语义全对。

代码、语料和 runner 在正式运行期间冻结。本文读取三份已完成的真实模型报告，已逐轮核对全部 47 答；不重跑模型，不依据已生成回答修改问题。本文中的观察只针对本轮给定的证据和数据库快照。

## 冻结与运行标识

- 应用 SHA256：`52897f982c8cdba3022029026c414b17238adbb0902384f86f845750d13bbfc7`。
- runner SHA256：`26b6bd7fb23d3816500b382e1892ceaf5d6e97aaf9fe8d61344503240ae513f8`。
- v0.10 新验收 fixture SHA256：`3639ae99cd21aa190f4d3901cff4b9ca5cba2b6adcc1bd13aa82e284a9033ba1`；7 组、12 轮，repeat 2，共 24 答。
- v0.9 回归 fixture SHA256：`a07e8822b65b84ab15abc65e7466acf1544c69428d41f64c6c2ea9ac434a70ff`；7 组、10 答。
- v0.8 回归 fixture SHA256：`488829b9d4e638b7f3f4f88839c3b56164f624ea492c27289610e913573c59d3`；6 组、13 答。
- 模型配置：mini 为 `qwen3.7-flash`，mid 为 `qwen3.7-plus`。三份报告均为 `run_status=complete`，均未记录运行期间应用源码变化。
- 可提交的新验收报告：[answers-v010-acceptance.json](../eval/results/answer-stability-v010/answers-v010-acceptance.json)。
- 可提交的 v0.9 回归报告：[answers-v09-regression.json](../eval/results/answer-stability-v010/answers-v09-regression.json)。
- 可提交的 v0.8 回归报告：[answers-v08-regression.json](../eval/results/answer-stability-v010/answers-v08-regression.json)。
- 原始运行目录为 `data/qa/answer-stability-v010-20261019/`，三份文件分别位于 `acceptance/answers-20261019-062042-294569-41055-52897f98.json`、`v09-regression/answers-20261019-062042-294585-41056-52897f98.json` 和 `v08-regression/answers-20261019-062932-620977-41533-52897f98.json`。

## 完成范围与自动检查

全部 47 答已对照最终正文、草稿/修订诊断、本轮来源、人物范围、记忆状态和领域变化。v0.8 的 13 答还由另一个子代理核对过证据；这仍属于代理审阅，没有独立人工评分。

| 项目 | 新验收 24 答 | v0.9 回归 10 答 | v0.8 回归 13 答 |
| --- | --- | --- | --- |
| 确定性检查 | 308/309 | 141/141 | 213/213 |
| 运行错误 / 场景错误 / 失败轮数 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 有限字符串规则 | 无此项 | 无此项 | 11/11 |
| 自动 query 无领域写入检查 | 20/20 | 8/8 | 旧语料未设置此项期望 |
| 已存字段检查 | 6/6 | 4/4 | 无此项 |
| 引用编号未伪造检查 | 24/24 | 10/10 | 13/13 |
| 焦点状态 / 注册 / 原因检查 | 8/8 | 未设置期望 | 未设置期望 |
| 分类器实际请求 | 22 | 11 | 15 |
| 发生分类重试的轮数 | 2 | 2 | 2 |
| 分类尝试 JSON 失败次数 | 3 | 2 | 2 |
| 分类终局失败 | 1，JSON | 0 | 0 |
| 回答修订 completed / not_needed / unavailable | 20 / 4 / 0 | 7 / 1 / 2 | 10 / 2 / 1 |
| 最终 uncited 告警轮数 | 2 | 1 | 2 |

确定性检查合计 **662/663**。唯一失败是新验收 repeat 1 的 `conflict-does-not-imply-private-channel` 第 1 轮：期望 query，分类器实际两次 JSON 失败后得到 unknown。它仍执行只读路径，不能把只读安全和分类正确混为一项。

自动无写入检查为 **28/28**。另对旧 v0.8 的 13 轮读取真实 `data_changes`：每轮只新增一条原始审计 evidence，其余快照表没有新增、修改或删除。加总 41 轮 query/unknown，在报告覆盖的 14 张表内均未观察到领域变化；v0.8 的这 13 轮是额外代理核对，不能补算成自动检查通过数。撤销、编辑、恢复、转移与归档是查询前的 fixture 操作，不算查询写入。

真实发生记忆写入的是 6 轮：新验收的 2 轮 record、2 轮 mixed，以及 v0.9 的 1 轮 record、1 轮 mixed。每轮目标当前城市均实际保存，旧当前城市相应结束；10 项已存字段检查还包括后续查询核对，不能解释成 10 次写入。

## 已发现的具体表现与反例

### 别名、恢复与日期

新验收 `alias-restored-quote-and-edit-focus` 的两次第 1 轮使用“糖糖”查询，最终均正确返回唐宁编辑后的原话：“唐宁说工作联系统一发邮件，急事也先留言，不直接打电话。”并引用该记忆。没有因资料正文只写唐宁而否认糖糖身份，也没有建议新建人物。两轮领域表均无变化。

该场景两次第 2 轮明确追问“这条记忆是什么时候被我修改的？”，焦点均为 `selected/unique_cited_memory`，重新注册为本轮 `S1`。回答分别显示记忆所标日期 `2025-09-11`、首次录入日期 `2026-10-19`、实际人工编辑日期 `2026-10-19` 和恢复操作日期 `2026-10-19`，与来源中的时间角色对应。修改和恢复碰巧同日，回答仍区分操作含义；没有把旧记忆日期当作编辑时间。v0.9 对照日期问法得到同样区分。两次先返回多条记忆后再使用同一句日期追问，均进入 `clarify/ambiguous_memory`，没有擅自选一条作答。

不过，明确日期问法进入确定性路径的结果，不能推广为普通问法的时间语义已全面解决。v0.8 `correction-withdrawal` 第 2 轮最终仍写：

> 根据记录，林舟（阿舟）在2026年10月9日纠正了之前的说法……

`2026-10-09` 是记忆所标日期，实际 `last_user_edit_at=1792391696873` 对应 **2026-10-19 14:34:56+08:00**。正文将所标日期误当纠正日期。这轮修订为 `completed` 且 `issues=[]`，没有拦住错误，是本轮明确保留的语义反例。

v0.8 `restore-move-archive` 第 2 轮最终同时显示“记忆所标日期：2026-10-09，valid_from：2026-10-09”。顶层 memory 没有这个键，但 `evidence.excerpt.user_correction.valid_from` 确实为 `2026-10-09`，且已进入 compact source；因此不能称为凭空编造字段值。问题是把人工编辑审计的内部输入键直接展示成用户时间说明，混淆自然语言时间投影与内部审计字段。

### 冲突解释由审阅删去，建议仍有轻度重复

v0.9 `conflicting-contact-records` repeat 1，第 1 轮草稿曾写：

> 也许许澄的意思是“项目事项走邮件，但紧急私事可以周三午休打电话”，也许有其他区分方式。

电话来源只写“许澄说周三午休方便接电话，其他时候先留言。”，没有“私事”或“紧急”限制。最终回答已删除该调和解释，分别引用项目邮件记录与未限定工作/私事的电话记录。这个例子说明草稿仍会产生无依据解释，一次修订在该轮进行了有效删除，不能由此推断所有轮次都能拦住。

同一轮最终建议为：

> 建议：建议直接向许澄确认这两条联系习惯分别适用的具体场景。

这不是重复的 `建议：建议：` 标签，但“建议”一词仍重复，属于文案残余。当前服务端仅规范化带冒号的开头标签，不删除正文中的普通动词。

### 新验收的冲突回答仍保留无依据的元说明和场景举例

新验收 `conflict-does-not-imply-private-channel` repeat 1，第 1 轮最终回答列出了两条记忆，但额外断言：

> 这两条均为语义记忆（提炼后的意思），并非逐字原话。

实际两条 `source.text` 都与各自 `evidence.excerpt` 完全一致。仅凭内部 `kind=semantic`，不能断言这些可见文字不是逐字记录；最终回答仍保留了这种未经证据支持的元说明，并暴露“语义记忆”标签。

该轮澄清段还保留“适用范围的区别（如‘项目事项’与‘非项目事项’）”。虽然以不确定表达呈现，没有直接宣称某条用于私事，但仍将未经记录支持的渠道分类作为候选解释带给用户。审阅只调整了段落类型，未完全删除这个具体假设。这个反例与“不能把无来源解释移入澄清中”的目标仍有差距。

repeat 2 的首轮处理较好：原话和所标日期正确，说明适用范围与变化原因不足，没有再举具体的无依据渠道划分。但两次第 2 轮“这些记录能证明工作走邮件、私事打电话吗？”的 `conversation_scope` 都变为 `none/no_person_reference`，本轮 `sources` 和 `source_dependencies` 均为空。草稿虽复述此前两条记录，审阅因缺本轮资料而全部删除，最终泛化为资料不足。这是保守避免沿用旧引用的行为，但没有完成用户期待的两条证据对照，暴露“这些记录”衔接不充分的问题；不能因为该轮检查全通过就判断交互已完整。

### 未匹配问题引文仍保留有效修订

新验收 `conflict-does-not-imply-private-channel` repeat 2 第 1 轮，编辑器的问题引文用省略号拼接为“ 两条记录日期只差一天，但内容看起来有出入...需要我帮你确认一下现在哪种方式更合适吗？ ”，不能在草稿中逐字匹配。元数据保留了 `raw_issues` 原文，同时将规范化问题设为 `quote=""`、`quote_status=unmatched`。这轮仍为 `completed`，有效的事实修订被采用，没有因问题定位引文不匹配而整份丢弃。这一结果只验证该轮降级定位的行为，不证明修订在语义上必然正确。

### 缺资料去掉了无关背景，但混合建议仍有未引用告警

新验收 `missing-blood-type-stays-brief` 两次草稿中的南京、工作联系偏好等无关背景均被删去，只保留缺资料说明和可补充血型的建议。v0.8 `no-answer` 的血型查询也删去了杭州、培训与羽毛球背景。不过这几轮缺资料加建议的最终 `answer_type=mixed`，仍出现 `uncited` 告警。三份报告合计 5 轮：新验收血型 2 轮、v0.9 花生过敏缺资料 1 轮、v0.8 血型及撤销后缺资料各 1 轮。它们没有据此编造目标事实，但“没有资料”同时显示“核对其中的人物信息”的告警，会让用户困惑。

这是现有保守引用规则的边界：只有整条纯 `no_answer` 才豁免未引用告警，混合建议不能得到该豁免。本文记录其用户体验影响，不主张放宽事实段的引用校验。

### 分类器格式失败与安全只读分别记录

新验收 `conflict-does-not-imply-private-channel` repeat 1 第 1 轮 mini 分类器调用两次，均 `finish_reason=stop`，均因 `failure_category=json` 被拒绝，最终 `intent=unknown/status=invalid`。这是格式失败，不是截断或 HTTP 错误；报告没有保存供应商原始分类文本，无法进一步断定 JSON 具体错误。该轮 `classified_turn_intent` 失败，但实际领域表无变化，且回执明确“仅查询，未保存”。安全只读有效并不意味着分类成功。

三份报告共 48 次实际分类请求，6 轮发生重试，7 次尝试因 JSON 失败；5 轮重试恢复成功，1 轮终局失败。所有尝试返回的 `finish_reason` 均为 `stop`，没有记录 HTTP 状态错误，不能解释为输出截断或网络故障。另有 5 轮确定性日期相关意图没有调用分类模型。

### 明确记事和 mixed 的实际落库

新验收 `explicit-record-prefix-with-conditional-tail` 两次均正确识别为 record，实际新增当前城市青岛，并结束天津旧字段；条件尾句没有使整条误判为只读，也没有产生“顾舟将搬到另一座城市”的虚构当前字段。原始用户句作为 note 保存，未观察到将“我会另行补充”提炼成顾舟承诺。最终去掉草稿“已记录”断言，改用固定回执指引。

新验收两次 `mixed-city-update-keeps-query-separate` 和 v0.9 `mixed-real-update-and-query` 的第 1 轮均仅将“唐宁现在搬到苏州了。”作为提取文本，真实新增苏州当前字段、搬家记忆和仅含该断言的 note；查询联系方式的半句没有写为意图或偏好。第 2 轮查询没有领域变化，来源中的当前苏州与已结束南京被分别表达。这是实际数据库检查支持的行为，不能仅由 mixed/query 标签推定。新验收 repeat 2 的修订还删去了草稿中“搬家导致习惯变化”的无依据因果，但最终有“建议：建议向……”的普通词重复。

### 一次修订仍可能协议失败

三份报告共有 37 轮 `completed`、7 轮 `not_needed`、3 轮 `unavailable`。3 轮不可用均保留草稿并告警：

- v0.9 `quoted-old-answer-is-not-a-fact`：`inline_reference_not_allowed`。最终草稿引用“不吃花生”原文，没有保存用户引用的“喜欢花生”假事实，也没有给出过敏诊断；这不代表编辑器成功完成修订。
- v0.9 `explicit-record-with-question-form`：`inconsistent_answer_type`。青岛当前字段真实保存、天津旧字段真实结束，原稿“已经记下了，顾舟现在住青岛……”有实际写入支持，不能称为虚假保存；仍未遵循统一回执指引。
- v0.8 `namesake-clarification` 第 2 轮：`invalid_review_response`。正确选中了星河科技林舟，但原稿在“根据林舟自己的偏好”的框架中加入“日常沟通尽量简短/其他轻量方式”；实际偏好只支持急事电话、平时别发长邮件。编辑器曾试图删除这类外推，协议失败后未采用。

修订可用性和最终资料正确性必须分别报告。`completed` 只说明修订协议和引用编号检查完成；`not_needed` 包含确定性日期回答、固定澄清等路径，二者都不构成语义证明。

另一个文案残余出现在 v0.9 `hypothesis-does-not-update`：审阅删除 `save_person_fact` 函数名后，最终仍使用“city 字段”。该轮没有错误更新假设北京，但产品语言仍可进一步自然化。

## 逐轮证据核对

下表共 47 行，不设准确率或“语义通过”分数。“领域无变化”依据真实 `data_changes`，允许原始审计 evidence，不以分类器的 query 标签代替数据库检查。所有 v0.8 查询均符合这一快照观察，下表仅在有关风险处重复注明。

| 报告 | repeat | 场景 | 轮 | 代理证据观察 |
| --- | --- | --- | --- | --- |
| v0.10 | 1 | alias-restored-quote-and-edit-focus | 1 | 返回编辑后的唐宁原话，糖糖身份未被否认；有真实记忆引用，领域无变化。 |
| v0.10 | 1 | alias-restored-quote-and-edit-focus | 2 | 唯一记忆绑定，区分 2025 所标日期与 2026 编辑/恢复日期；只读模板，无模型修订。 |
| v0.10 | 1 | conflict-does-not-imply-private-channel | 1 | 列出两源，仍断言“并非逐字原话”并在澄清中举项目/非项目划分；分类两次 JSON 失败，安全只读。 |
| v0.10 | 1 | conflict-does-not-imply-private-channel | 2 | “这些记录”未衔接到人物，本轮无来源，最终泛化 no_answer；没有编造，但证据对照未完成。 |
| v0.10 | 1 | missing-blood-type-stays-brief | 1 | 删除南京/联系偏好等无关背景，未编造血型；缺资料加建议仍显示 uncited。 |
| v0.10 | 1 | query-without-punctuation-remains-readonly | 1 | 无问号仍只读，顾工解析为顾舟；正确引用不吃花生、清淡口味及所标日期。 |
| v0.10 | 1 | query-without-punctuation-remains-readonly | 2 | 不吃花生不足以认定过敏；最终删除草稿过敏/口味等原因猜测，直接建议向本人核实。 |
| v0.10 | 1 | explicit-record-prefix-with-conditional-tail | 1 | record 真实落青岛、结束天津；未生成未来搬家城市，固定回执指引。 |
| v0.10 | 1 | mixed-city-update-keeps-query-separate | 1 | 真实保存苏州字段、搬家记忆与仅含断言的 note；只答短信/约通话，有 S4，未存查询半句。 |
| v0.10 | 1 | mixed-city-update-keeps-query-separate | 2 | 重新读取当前苏州 S1、南京历史 S2；query 无领域变化。 |
| v0.10 | 1 | multiple-memory-date-followup-needs-selection | 1 | 小许身份对应许澄，两条原话分别引用 S2/S3；分类首次 JSON 失败后重试成功。 |
| v0.10 | 1 | multiple-memory-date-followup-needs-selection | 2 | 多条记忆触发 ambiguous_memory 固定澄清，不猜编辑日期，不写入。 |
| v0.10 | 2 | alias-restored-quote-and-edit-focus | 1 | 正确返回编辑后唐宁邮件原话，有引用；未误否认糖糖身份，领域无变化。 |
| v0.10 | 2 | alias-restored-quote-and-edit-focus | 2 | 唯一记忆确定性绑定，区分 2025 所标日期与 2026 编辑/恢复日期，领域无变化。 |
| v0.10 | 2 | conflict-does-not-imply-private-channel | 1 | 原话/所标日期正确，未加入具体渠道调和解释；问题引文 unmatched 被规范化，raw_issues 保留，修订有效采用。 |
| v0.10 | 2 | conflict-does-not-imply-private-channel | 2 | 再次 scope none、本轮无来源，最终泛化 no_answer；保守但未完成两条记录的证据对照。 |
| v0.10 | 2 | missing-blood-type-stays-brief | 1 | 删除无关城市/联系背景，未编造血型；缺资料加建议仍出现 mixed uncited。 |
| v0.10 | 2 | query-without-punctuation-remains-readonly | 1 | 顾工对应顾舟，正确引用不吃花生、清淡和所标日期；无问号仍只读。 |
| v0.10 | 2 | query-without-punctuation-remains-readonly | 2 | 删除过敏/不喜欢/上火等原因猜测，只按“不吃花生”原文说明不能认定过敏，有 S2。 |
| v0.10 | 2 | explicit-record-prefix-with-conditional-tail | 1 | 青岛真实落库、天津结束；条件尾句未导致意图误判，最终采用固定回执指引。 |
| v0.10 | 2 | mixed-city-update-keeps-query-separate | 1 | 仅保存苏州断言；修订删除日期角色含混和搬家导致习惯变化的因果，最终仍有“建议：建议向”词重复。 |
| v0.10 | 2 | mixed-city-update-keeps-query-separate | 2 | 重新读取苏州当前字段 S1，未被旧南京字段覆盖；query 无领域变化。 |
| v0.10 | 2 | multiple-memory-date-followup-needs-selection | 1 | 两条原话分别引用，未虚构冲突范围；“记录日期”仍不及“所标日期”清晰。 |
| v0.10 | 2 | multiple-memory-date-followup-needs-selection | 2 | 多条记忆固定澄清，无编辑日期猜测，无领域写入。 |
| v0.9 | 1 | restored-record-and-edit-date | 1 | 明确可查到、已恢复未撤销，引用编辑后的原话；领域无变化。 |
| v0.9 | 1 | restored-record-and-edit-date | 2 | 正确区分所标日期、首次录入、人工编辑与恢复日期；唯一记忆绑定，领域无变化。 |
| v0.9 | 1 | conflicting-contact-records | 1 | 草稿虚构紧急私事划分，最终修订已删除；保留两源差异，但建议中重复“建议”一词。 |
| v0.9 | 1 | query-without-question-mark | 1 | 无问号仍 query；唐宁/糖糖身份对应，联系方式与所标日期有来源，领域无变化。 |
| v0.9 | 1 | query-without-question-mark | 2 | 不借顾舟资料推唐宁过敏，未写入；一次 JSON 失败重试成功，缺资料加建议仍显示 uncited。 |
| v0.9 | 1 | hypothesis-does-not-update | 1 | 假设北京未写入，当前天津有来源；删除内部函数名，但最终仍出现 city 字段用语。 |
| v0.9 | 1 | quoted-old-answer-is-not-a-fact | 1 | 饮食原文正确、未记假引文；审阅因自行写引用协议失败，保留原稿并告警。 |
| v0.9 | 1 | mixed-real-update-and-query | 1 | 只提取搬苏州断言，当前字段和搬家记忆真实落库；联系方式仍引用原有短信记录。 |
| v0.9 | 1 | mixed-real-update-and-query | 2 | 重新读取苏州当前/南京历史字段，查询无领域变化。 |
| v0.9 | 1 | explicit-record-with-question-form | 1 | 问号仍判 record，青岛真实保存、天津结束；分类 JSON 重试成功，修订因 inconsistent_answer_type 不可用，保留有实际保存支持但不统一的回执。 |
| v0.8 | 1 | nickname-context | 1 | 陈言公司星桥、当前上海与历史杭州均正确引用，身份未串到其他人物。 |
| v0.8 | 1 | nickname-context | 2 | 继承陈言，不吃香菜有 S4；删去午餐细节。午餐在 fixture/expected_material/profile_context 中实际存在，仅缺独立 S 注册，不能据删除认定原稿虚构。 |
| v0.8 | 1 | namesake-clarification | 1 | 固定澄清重名人物，正文未泄漏偏好；结构化候选含两份林舟，具体界面呈现需另行验证。 |
| v0.8 | 1 | namesake-clarification | 2 | 选中星河科技林舟，未使用学校同名者偏好；修订不可用，额外沟通建议仍混入本人偏好的表述框架。 |
| v0.8 | 1 | temporal-conflict | 1 | 上海当前、杭州历史有效期有正确来源；“2025 年 9 月起记录至今”仍有生效/录入措辞含混，城市结论正确。 |
| v0.8 | 1 | temporal-conflict | 2 | 开会不证明住杭州，当前上海有来源；删除“即使从杭州出发”假设，仍重复邀请更新资料。 |
| v0.8 | 1 | no-answer | 1 | 血型资料不足，删除杭州/培训/羽毛球背景；建议使最终为 mixed，仍有 uncited 告警。 |
| v0.8 | 1 | correction-withdrawal | 1 | 急事电话/平时别发长邮件及 10 月 9 日所标日期正确；事实时效说明被放入无引用 advice 段。 |
| v0.8 | 1 | correction-withdrawal | 2 | 编辑后邮件原话正确，但误称 10 月 9 日纠正，实际人工编辑在 10 月 19 日；completed 且 issues=[] 未拦截，时效说明也混入建议。 |
| v0.8 | 1 | correction-withdrawal | 3 | 撤销后不再引用旧偏好，回答资料不足；旧依赖为 withdrawn，缺资料加建议仍有 uncited。 |
| v0.8 | 1 | restore-move-archive | 1 | 恢复编辑后邮件原话；10 月 9 日所标日期和 10 月 19 日恢复操作分开表达。 |
| v0.8 | 1 | restore-move-archive | 2 | 转移后归为何川，保留电话/邮件差异，未按新旧选边或发明具体原因；暴露编辑审计中的 valid_from 内部键，其值确有来源。 |
| v0.8 | 1 | restore-move-archive | 3 | 人物归档后资料不可用，无来源内容暴露或重新建人；旧依赖为 unavailable。 |

## 诊断与边界

焦点诊断合计 3 轮 `selected/registered`：新验收两次唯一记忆日期追问、v0.9 一次对应追问；2 轮 `clarify/ambiguous_memory`，均来自新验收多记忆日期追问；其余 42 轮 `not_applicable`。不能将“不适用”视为对所有自然追问的解析成功，“这些记录”衔接失败就是已观察到的反例。

47/47 的“引用编号未伪造”仅说明最终引用没有使用本轮未注册编号，其他来源检查也只验证对应快照、人物和状态约束；它们不证明正文每个事实被引用完整支持。普通问法中的纠正日期错误、`kind=semantic` 导出的非原话断言、将沟通建议放在本人偏好框架下，均能在引用编号有效时发生。

最终正文未发现 `建议：建议：` 的重复标签；v0.9 冲突回答与新验收 repeat 2 mixed 回答仍各有一处“建议：建议……”的普通词重复。产品语言还残留 `city 字段`、`valid_from`、`语义记忆` 等内部用语。v0.8 同名澄清的结构化候选已在报告中核对，界面是否完整呈现候选没有在本次只读报告审阅中验证。

本轮已验证明确日期焦点、别名恢复查询、只读边界、真实城市保存以及若干有效草稿修订；同时明确保留普通问法日期误归因、冲突追问失去来源、无依据元说明/候选解释、3 次修订协议失败、1 次分类终局失败、混合缺资料告警和内部用语等问题。正式 JSON 不因本文改分，继续保持 `semantic_review.status=review_pending`、`semantic_verified=false`。全部 47 答的代理核对已完成，独立人工语义评分尚未完成，不提供准确率或“语义全对”的结论。
