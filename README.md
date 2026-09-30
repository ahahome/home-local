# home

> 以「人」为索引的私人关系数据库与 AI 助理 —— [marsdata.ai](https://marsdata.ai)（Mars Agent）的本地自托管复刻版。

## 这是什么产品

home 是一个**完全跑在你自己机器上**的「关系维护 Agent」。它不是日历、不是待办清单、也不是传统 CRM——它按**人**而不是按时间/主题组织信息：

你平时在聊天、截图、文件、网页里遇到的关于「你在乎的人」的一切（换了工作、孩子上几年级、答应过什么、随了多少份子钱、上次聊到什么时候），都被自动抽取成**结构化档案**，挂上时间线，并能在你需要时回答问题、到期提醒、代拟消息。

一句话：**替你的记忆打工，把人情往来变成可查询、可回溯、可执行的数据。**

与云端 Mars Agent 的本质区别：所有数据（SQLite + blob）都在本地，LLM 调用走你自己的 API key，随时可以导出或彻底删除。

## 主要做什么

### 1. 自动建档（输入 → 结构化档案）
- **对话直说**：聊天中随口一提「王工刚升了技术总监」，自动抽取成字段，进入人物档案。
- **截图识别**：发微信聊天截图，vision 模型转写后走同一抽取管线。
- **文档/网页**：导入文件或粘贴 URL，抓取正文后抽取。
- **批量导入**：CSV / vCard 联系人一键导入。

### 2. 身份消歧（绝不认错人）
同名「老王」到底是不是同一个人？四路召回 + 加权打分 + 三档决策：
- **≥0.85 且领先差值 ≥0.10** → 直接归并；
- **0.50–0.85** → 进待确认队列，问你；
- **<0.50** → 建新人物。
即使分数高、但两个候选分差不够，也**必须回问**，绝不静默合并。

### 3. 双时态记忆（时间是一等公民）
每条字段带 `valid_from`（事实何时成立）、`valid_to`、`recorded_at`（系统何时录入）。支持「2024 年的时候他做什么工作？」这类**历史回溯查询**，而不是只保留最新值。

### 4. 五层记忆
语义（属性字段）、情景（事件）、反思（周结）、日志（每日 journal）、技能（沉淀下来的话术/流程）——五层各有独立写入路径和检索策略。

### 5. 问答与代笔（代笔不代发）
- 「老王们最近谁有变动？」「我答应过孩子什么？」——混合检索（SQL + FTS5 分词 + 向量余弦 + 时间衰减）给出带来源的答案。
- 可以帮你**起草**祝福语/开场白，但**绝不替你发送**。

### 6. 承诺与提醒
「下周五前给张总方案」自动建承诺 + 定时提醒（APScheduler，每分钟轮询，本地时区 Asia/Shanghai）。

### 7. 人情账本
随礼、借贷按人记账；待确认条目不进入任何汇总，杜绝「没确认的钱也算了总数」。

### 8. 待确认队列
所有不确定的抽取结果进 Pending 面板，人工确认/丢弃后才生效——假设优于断言，但不等于断言。

### 9. 合并与拆分
误建的两个「老陆」可合并，合并时写快照（tombstone），可**反向拆分**恢复原状；删除是真删除，不会复活。

### 10. Deep Research
需要深度调研时：预检索 → 出研究提案 → 3 个边界互斥的子 Agent 并行调研 → lead agent 审校（丢弃无来源的论断）→ 出带引用来源的报告。

## 产品原则（七条不变量，写进代码而非口号）

| # | 原则 | 代码体现 |
|---|------|---------|
| P1 | 永不静默合并 | `disambiguate.py` 三档决策 + 分差门槛 |
| P2 | 写入必有回执 | 全部 15 个工具返回结构化 receipt |
| P3 | 时间是一等公民 | 双时态字段 + 间隔感知 + 全本地时区 |
| P4 | 假设优于断言 | 抽取带 confidence + evidence，低置信进 pending |
| P5 | 删除是真删除 | 物理删除；合并用 tombstone 可拆分 |
| P6 | 代笔不代发 | 系统提示硬约束 + 无发送类工具 |
| P7 | 不假装知道 | 「严禁模拟工具」硬规则；无来源论断被审校丢弃 |

## 当前开发进度

**阶段：MVP 已完成并实测通过。** 单机全链路可用，启动后访问：`http://127.0.0.1:8000`。

### ✅ 已完成且实测验证（用真实 DashScope API 跑通）

| 模块 | 验证结果 |
|------|---------|
| 对话抽取建档 | 随口陈述 → 字段/事实自动入库，带回执 |
| 档案问答 | 「他做什么工作」类问题答对，带来源 |
| 承诺 + 定时提醒 | 自动识别日期并创建提醒（已验证 2026-10-07 到期项） |
| 人情账本去重 | 同一笔重复描述不重复计入汇总 |
| 微信截图识别 | 模拟截图 → vision 转写 5 条消息 → 抽出人物属性 + 2 条承诺 |
| 待确认队列 | 低置信条目正确入队，不进汇总 |
| 人物合并/拆分 | 「陆/老陆」合并后成功拆分还原 |
| 每日 journal | 本地时区 23:50 生成，跨天不错位 |
| Deep Research | 并行子 Agent 出报告，附来源链接 |
| 单元测试 | **12/12 通过**（消歧、幂等写入、账本去重） |

### ⚠️ 未做 / 未做满（PRD 中的后续项）

- **IM 直连**：微信/抖音消息目前靠截图/粘贴中转，没有自动同步通道。
- **云浏览器代操作**：PRD 中「替你登录网页操作」未实现（且与 P6 原则有张力，需重新设计边界）。
- **SSE 流式输出**：当前为整段响应，对话体验可升级为流式。
- **3D 关系星图**：Atlas 面板目前是列表视图，3D 可视化未做。
- **语音 ASR**：语音消息未接。
- **多租户隔离 CI**：单用户设计，多人共享部署需补隔离测试。

### 技术栈与规模

Python 3.13 venv · FastAPI · SQLite(WAL) · jieba FTS5 · numpy 向量 · APScheduler · 单文件 SPA 前端。
共 **31 个 Python 文件 + 1 个 SPA**，核心代码约 **3954 行**，20+ 张表。
模型走 DashScope OpenAI 兼容接口，按任务分级：qwen-turbo(轻) / qwen-plus(中) / deepseek-v4-pro(思考) / qwen3.8-max(重) / qwen3-vl-plus(视觉) / text-embedding-v4(向量)。

## 快速开始

```bash
cd home
source .venv/bin/activate

# .env 已含配置模板，填入 HOME_LLM_API_KEY

# 启动
.venv/bin/python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000 —— 单页应用内含：对话 / 人物 / 待确认 / 星图 / 提醒 / 日志 / 技能 / 通知 八个面板。

## 测试

```bash
.venv/bin/python -m pytest tests/ -q --basetemp=/tmp/home-pytest-tmp
```

（沙箱环境下 pytest 默认 tmp 目录会报 PermissionError，需显式 `--basetemp`。）

## 架构

```
app/
  config.py      .env 配置、模型分级、阈值（消歧 0.85/置信 0.85）
  llm.py         OpenAI 兼容客户端：chat/chat_json/vision/embed，重试与宽松 JSON 修复
  store/
    schema.sql   20+ 表；双时态字段、applied_patch、merge_log、tombstone
    db.py        WAL、外键、busy_timeout、user_version 迁移、savepoint
    blob.py      SHA256 内容寻址存储，原子写（tmp+fsync+os.replace）
    repo.py      全部 CRUD；upsert_field 同毫秒冲突自增；账本排除 pending/void
  memory/
    extract.py       LLM 抽取 → 结构化 patch（confidence + evidence，金额归一化到分）
    disambiguate.py  四路召回 + 加权打分 + 三档决策；别名精确命中基线 0.9
    apply.py         幂等写入（savepoint/patch + 内容指纹 + applied_patch 表）；合并/拆分
    retrieve.py      混合检索：SQL + FTS5 + 向量 + 时间衰减与重要性融合
    compile_card.py  dossier card 预编译（≤900 字符，控上下文预算）
    fts.py / vector.py  jieba 分词（带正则兜底）/ numpy 内存余弦索引
  agent/
    prompts.py   系统提示：七原则 + 三种回答形态 + 「严禁模拟工具」硬规则
    tools.py     15 个 OpenAI function tools，全部返回回执
    harness.py   工具循环、上下文预算、时间间隔感知、写后抽取、交互链路
    research.py  Deep Research：提案 → 并行子 Agent → 审校 → 带来源报告
  ingest/        截图(vision) / 文档 / 网页(Bing) / CSV/vCard 导入 / 路由归一化
  jobs/          APScheduler：提醒(1min)、journal(23:50)、周反思(周日)
  api/           FastAPI，各模块路由
web/             单文件 SPA
tests/           12 个单元测试（消歧 / 幂等写入 / 账本）
```

## 设计取舍备忘（踩过的坑）

- `person_alias` 主键是 `(person_id, alias, source)` 而非 `(alias, source)`——后者会让共用别名的第二个人被静默丢弃。
- 别名精确命中基线给 0.9，否则永远到不了绑定阈值。
- `upsert_field` 同毫秒两写会撞 `valid_from` 主键，冲突时 +1ms。
- 模型可能「角色扮演」调用工具（输出"正在调用天眼查API"），系统提示里必须有「严禁模拟工具」硬约束。
- DuckDuckGo 在代理下 502，网页检索走 Bing；其真实 DOM 结构是 `<h2 ...><a target=_blank href=...>`，正则要宽松。

## 安全提示

⚠️ 开发期间用于验证的 DashScope API key 曾以明文出现在对话中，**建议尽快到阿里云控制台轮换/作废该 key**，并把新 key 只写入本地 `.env`（已在 `.gitignore` 中，不要提交）。

## 相关文档

- PRD：`/Users/emdoor/WorkBuddy/2026-09-29-16-25-48/home-PRD.md`（13 模块 / 60+ 需求 / 20 验收用例，已剥离商业模式）
