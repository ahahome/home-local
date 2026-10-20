# 0.12 多轮记忆问答验证材料

采样日期：2026-10-20。全部资料为虚构开发语料。完整解释、复现命令与已知反例见 [验收报告](../../../docs/MEMORY_DIALOGUE_V012.md)。

## 最终实现

- 应用源码 SHA256：`03a19327124586cfb40c73ca246084b8a3ff91716b2437e170928186a5d3baf3`
- 评测器 SHA256：`0cab27da2a00ae850c9f1bfd8f9429a28f8ba021945a8f3e98d05d3d7265190c`
- 0.12规范化语料 SHA256：`31156596f125bbf920f1d347ce8b3a374c492d6f3952ca13e6862142922891d5`
- 每个回答报告单独执行 `repeat=1`，临时SQLite、独立场景进程，embedding/cache关闭。
- 分类模型配置qwen3.7-flash，回答/修订qwen3.7-plus。报告保留各自实际诊断和配置。

| 文件 | 内容 |
| --- | --- |
| [acceptance-run-1.json](acceptance-run-1.json) / [核对](acceptance-run-1-evidence-review.md) | 新语料20答，483/483检查 |
| [acceptance-run-2.json](acceptance-run-2.json) / [核对](acceptance-run-2-evidence-review.md) | 新语料20答，482/483；一次分类JSON失败，未越过只读边界 |
| [acceptance-run-3.json](acceptance-run-3.json) / [核对](acceptance-run-3-evidence-review.md) | 新语料20答，470/470检查 |
| [regression-v010.json](regression-v010.json) | 旧0.10语料12答，223/223检查 |
| [regression-v08.json](regression-v08.json) | 旧0.8语料13答，283/283检查，11条有限字面探针通过 |
| [regression-evidence-review.md](regression-evidence-review.md) | 主代理逐答核对上述25答及具体残余问题 |
| [retrieval.json](retrieval.json) | 1个固定FTS查询，6/6检查；不涉及模型回答 |

检查分母随实际来源、执行路径和断言数量变化，不是固定题数或语义准确率。没有指定的新契约显示not_exercised；不能把空分母当作验证通过。原始JSON保持runner生成内容，其中人工语义审阅仍pending；旁边Markdown是代理证据核对，不冒充独立人工评分。

## 保留的开发首版

[development-baseline/](development-baseline/)保留前一版实现`cf2a6a85239e883adcd33e8c921e0e3bfcc0c73799a62411363669f3737c4b7f`的三份完整报告及各自20答核对，检查484/485、470/472、493/493。其取源正确但范围/动机解释越界，以及分类失败后未保存状态被修订抹去等反例，促成最终确定性文字对照和稳定失败通知。

语料在首次模型采样前冻结，此后没有改问法；最终实现已看过同一开发语料的反例，所以最终三次运行属于开发复验，不是独立未见基准。更早因开发修改而中止的不完整本机运行不计入这里的完整结果。未筛选掉失败样本，没有将不同源码的运行合并为同一轮结果。
