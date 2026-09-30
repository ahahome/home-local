"""信息抽取：把任意文本（消息/截图转写/文档/网页）抽成结构化 patch（FR-MEM-02）。

输出统一为带 confidence 与 evidence 的 patch 结构，交由 apply.py 分档落库。

SL1 评测暴露的缺陷，处理方式不一样——记在这里免得后人以为都靠 prompt：
- A 承诺误判：SYSTEM 第 5 条给正/反例降概率，_promise_is_pleasantry() 在代码里兜底拦截。
  （只改 prompt 不够：实测仍会低置信地猜出「以后常见面」。）
  判定方式刻意【不用动词白名单】——白名单会漏（「回头再说吧」「请你搓一顿」），
  改用「命中客套触发词 且 全句无具体动作/时间」。
  闸门的【输入】也不能只判 content：模型会改写措辞（「回头我加你微信」→「加你微信」），
  触发词被改没了就漏（hold2-n06），所以用 _locate_span() 把 content 摁回原文、
  找到它所在的那句一起判；摁不回去（改写太多）才退化为只判 content。
- B evidence 拼接：_evidence() 强制 evidence 必须是原文中的连续片段。纯代码保证。
- C 字段归位（薪资误入账本、城市误入 company）：只靠 SYSTEM 的归位规则，没有代码兜底。
  归位是语义问题，硬规则容易误伤，所以留在 prompt 层，靠评测监控。
- E 角色词建档（「王芳的老公」被建成独立人物，违反 FR-ID-01）：is_role_person() 在代码里拦掉，
  并把该亲属的信息收拢到本人名下（家属信息），避免「不建档」变成「丢信息」。
- F 相对日期算错：第一层是缺「星期几」（_with_weekday 补上）；第二层是缺「本周/下周的边界」
  ——星期几给了，模型仍把「下周五」算成最近的周五（hold2-p02），由 _week_bounds()
  把周界与锚点日期一起喂进去。凡是靠推算的字段，把结果当权威值喂给它，不让它心算。
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from app.llm import get_llm
from app.util import flatten_for_match

SYSTEM = """你是一个信息抽取器。从文本中抽取关于「人」的事实，只输出 JSON。

硬规则：
1. 只有能确定指向一个真实人物的名字才抽取。"我对象""那个客户""前台"这类角色词不算人名；无法确定是谁的称呼不算。
   「王芳的老公」「李娜的儿子」这种「某人 + 亲属称谓」也不是独立的人，不要为它单独建人物；
   把关于他的信息记到本人名下（facts 或 family 字段）：例如「王芳她老公是主任医师」→ 记成王芳的一条事实/家属信息。
2. 每条事实必须带 evidence：从原文【逐字、连续】截取的依据片段。不要改写；
   不要用「；」把不相邻的两处原文拼成一条；如果找不到一整段连续的原文，就把 evidence 留空。
3. confidence 校准：明确陈述=0.9以上；合理推断=0.6~0.8；猜测=0.6以下。涉及金钱从严，不确定给 0.75 以下。
4. 相对日期（"下周三""这周五""下个月5号""前两天"）基于给出的今天日期换算成 YYYY-MM-DD。
   给出的今天日期已注明星期几，并给出了本月/下个月、本周/下周的范围和「本周五/下周五」的锚点，
   这些是权威值，直接据此推算，不要自己重算月份、星期或周界。无法确定填 null。
5. 承诺 = 具体事项 + 可执行动作（通常还带时间）。下列一律不算承诺，无论它出现在单句、
   多轮对话还是聊天记录片段里：「改天吃饭」「回头联系」「以后常见面」「以后常联系」
   「有机会聚聚」「保持联系」「有空一起…」。这些是社交客套，只是寒暄。
   - 不算：「改天一起吃个饭」「以后咱们能常见面」「回头我联系你」
          「那个事回头再说吧」「改天请你搓一顿」「哪天得空喝一杯」（没有具体事项和时间的寒暄）
   - 算：  「我答应下周三之前把报价单发给他」「9/8 之前给你确认」「明天上午把合同扫描件发他邮箱」
   判断依据是【有没有具体的、可执行的事】，不是用词——不要因为对话轮次变多、语气变正式就放宽标准。
6. 关于"用户自己"的目标、偏好、状态，不要放进 people，放进 my_notes。
7. 只输出一个 JSON 对象。不要解释、不要 markdown 代码围栏。

字段归位规则（这三类是最常见的放错位置）：
- ledger 只记「你和某人之间真实发生的收付款」：转账、借款、还款、份子钱、代付、报销。
  判断依据是「钱有没有在你和这个人之间移动」。某人的薪资/年薪/奖金/收入、公司规模/营收/
  融资额、房价、报价、商品单价——钱没在这个人和你之间移动，都不是 ledger，应记为 facts 或对应字段。
- 地点：「搬到/住在/去了/从…来 + 城市名」→ city 字段，不要放进 company。
  特别注意「公司搬到杭州了」这种情况：说的是公司在哪，但「杭州」是 city，不是 company。
  只有公司/机构名（如「恒远进出口」「宁德时代」）才放 company。
- 「他是宁波人」是 city；「他女儿在加拿大读书」是家属信息，不是他的 city。

别把「事实」也一起丢掉（上面否掉的是客套和角色词，不是事实本身）：
- 某人单位裁员/搬迁/出事、某人换工作/升职/结婚/生病/住院/生孩子/搬家——这些是明确发生过的事，
  即使不涉及金钱、也不是承诺，也必须抽成一条 facts。
- 每抽到一个人，至少给他留下「为什么值得记」的一条：一个字段、一条 facts 或一条 value_read 都可以；
  但不要为了凑条数硬造内容。

同一字段只给一条值：
- 一次抽取里，同一个 field 只出现一次。如果同时出现城市和更细的区/街道
  （如「杭州」与「滨江」「钱江新城」「浦东」），city 只取城市层级（杭州），区/街道不要再单独给一条 city。

输出结构（没有的项给空数组/空对象）：
{
  "people": [
    {
      "name": "原文中的称呼",
      "canonical_name": "判断出的规范名（真名或最常用称呼），不确定就同 name",
      "fields": [{"field":"company|title|city|birthday|phone|email|trait.特点|need.需求","value":"...","confidence":0.9,"evidence":"原文片段"}],
      "aliases": [{"alias":"此人的其他称呼","confidence":0.8}],
      "relations": [{"to":"另一个人名","kind":"colleague|family|referred_by|friend|know","confidence":0.7,"evidence":"原文片段"}],
      "facts": [{"text":"客观发生的事，一句话，含时间与人物","confidence":0.9,"evidence":"原文片段"}],
      "promises": [{"who":"me|them","content":"承诺内容","due":"YYYY-MM-DD或null","confidence":0.9}],
      "ledger": [{"direction":"in|out","amount":123.45,"note":"说明","occurred":"YYYY-MM-DD","confidence":0.85}],
      "value_read": {"text":"对此人特点/价值的判断","confidence":0.7,"evidence":"原文中的一段连续片段"}
    }
  ],
  "my_notes": [{"text":"关于用户自己的事实","kind":"semantic|episodic","confidence":0.9}]
}"""

# 切分 evidence 用：模型最常见的拼接分隔符
_EV_SPLIT = re.compile(r"[，。！？；：、\n\r]+")

# 社交客套的触发词（缺陷 A 的代码兜底）。
# 提示词第 5 条已经写了「客套不是承诺」，但实测模型偶尔仍会低置信地猜出一条
# （例如 0.4 分的「以后在杭州能常见面」）。统计上它从「高置信断言」降成了
# 「待确认」，但客套就是客套，跟置信度无关——所以在代码里直接拦掉，不交给模型自觉。
#
# 【刻意不做动词白名单】：早期版本要求触发词后面 8 字内必须跟着
# 「吃饭|聚|聊|见面|喝|约|坐|玩|联系」，holdout 一测就漏了一半——
# 「那个事回头再说吧」（说）、「改天请你搓一顿」（搓）都不在白名单里，
# 那两条只是「模型这次听话」才过的，不是代码兜住的。
# 白名单天生会漏；反过来判「命中触发词 且 全句没有具体动作/时间」更稳。
_PLEASANTRY = re.compile(
    r"改天|哪天|有空|得空|抽空|择日|有机会|回头|下次|以后|日后|有时间|找个?时间"
    r"|保持联系|经常联系|常联系|常常联系|多联系|再联系"
)
# 反向豁免：带具体动作/时间的不是客套——「下次见面把合同签了」「以后每周五发报表」是真承诺。
# 注意别把数字写得太宽：「一起」的一不能算具体量，所以只认「数词+单位」。
_CONCRETE = re.compile(
    r"把|将|[0-9０-９]"
    r"|[一二三四五六七八九十百千万]+(?:块|元|钱|万|亿|个|天|周|月|日|号|点|次|份|张|件|条)"
    r"|之前|以前|周[一二三四五六日天]|月|号|点前|以后.{0,3}(?:每|第)"
)

# 角色词（缺陷 E 的代码兜底，FR-ID-01）。
# 「我对象」「那个客户」「前台」这类不是人名，提示词第 1 条已经写了，但实测模型仍会
# 把「王芳的老公」建成一个独立人物（3 次里 2 次），而且 apply.py 的关系边也会顺手建档
# （relations[].to 找不到人时 create_person），所以必须在 sanitize 这个所有摄入路径的
# 公共收口处拦掉，并且连关系边一起过滤。
#
# 为什么不能只靠提示词：这是「语义事实」而不是「概率倾向」——见 llm-extract-eval-tuning 规则 3。
_KIN = ("老公|老婆|丈夫|妻子|媳妇|太太|先生|对象|男友|女友|男朋友|女朋友|"
        "儿子|女儿|孩子|娃|孙子|孙女|外孙|外孙女|儿媳|女婿|"
        "爸爸|妈妈|父亲|母亲|爹|娘|公公|婆婆|岳父|岳母|"
        "哥哥|姐姐|弟弟|妹妹|表哥|表姐|表弟|表妹|堂哥|堂姐|堂弟|堂妹|"
        "叔叔|阿姨|舅舅|姑姑|伯伯|爷爷|奶奶|姥姥|姥爷")
# 纯角色词：只做**全等**匹配（「老师」是角色词，但「张老师」不是——那是个称呼/名字）。
_ROLE_ONLY = ("我对象|我的对象|那个客户|这个客户|客户|前台|老板|领导|同事|朋友|同学|室友|"
              "房东|司机|师傅|老师|医生|护士|邻居|甲方|乙方|供应商|下属|助理|秘书|"
              "快递员|外卖员|有个人|一个人|某人")
_ROLE_PERSON = re.compile(rf"^(?:.+的(?:{_KIN}|{_ROLE_ONLY})|(?:{_ROLE_ONLY}))$")
# 「王芳的老公」→ base=王芳。用来把家属信息收拢回本人名下。
_KIN_OF = re.compile(rf"^(?P<base>.+?)的(?:{_KIN})$")

# 中文星期，索引对齐 datetime.weekday()（周一=0）
_WEEKDAY_CN = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _f(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def _s(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _is_pleasantry(content: str) -> bool:
    """这句话是不是社交客套（「改天吃饭」「以后常见面」这类寒暄）。

    判定式 = 命中客套触发词 **且** 全句没有任何具体动作/时间标记。
    带「把…给…」「周X」「号」「点前」「数词+量词」这种可执行细节的，
    即使含「以后」「下次」「回头」也照常当承诺保留。

    刻意【不是】「触发词 + 动词白名单」：白名单会漏（实测「回头再说吧」「请你搓一顿」
    都漏掉了），而漏掉的表现是「靠模型这次听话」，看起来绿其实是脆的。
    """
    text = _s(content)
    if not text or not _PLEASANTRY.search(text):
        return False
    return not _CONCRETE.search(text)


def _locate_span(content: str, source: str, src_flat: str) -> str:
    """把 content 摁回原文，返回包含它的那个标点片段；摁不回去返回空串。

    为什么需要：模型抽承诺时会改写措辞（「回头我加你微信」→ content=「加你微信」），
    改写会把客套触发词弄丢——只判 content 就漏（hold2-n06，3 次全漏）。
    content 通常是原文某个片段的子串（或逐字保留），摁回去就能让闸门
    看到被改掉的触发词。改写太多摁不回去时返回空，由调用方退化为只判 content。
    """
    if not source or not src_flat:
        return ""
    key = flatten_for_match(content)
    if not key or key not in src_flat:
        return ""
    for seg in _EV_SPLIT.split(source):
        seg = seg.strip()
        if seg and key in flatten_for_match(seg):
            return seg
    return source


def _promise_is_pleasantry(content: str, span: str) -> bool:
    """判客套：content 与它所在的原句（span）一起看。

    触发词在【任一】版本里出现、且【所有】版本都没有具体动作/时间 → 客套。
    不对称是刻意的：触发词可能被改写弄丢（所以任一命中就要查），
    但「具体」只要在一个版本里出现就当真承诺放行（宁可放过，不误杀）。
    span 为空时等价于 _is_pleasantry(content)。
    """
    texts = [t for t in (content, span) if t]
    if not texts:
        return False
    if not any(_PLEASANTRY.search(t) for t in texts):
        return False
    return not any(_CONCRETE.search(t) for t in texts)


def is_role_person(name: str) -> bool:
    """这个名字是角色词/亲属称谓，还是真人名（FR-ID-01）。

    「我对象」「那个客户」「前台」→ True；
    「王芳的老公」「李娜的儿子」「她的同事」→ True（不是独立的人）；
    「张老师」「老王」「陈工」→ False（这是称呼/名字，不是角色词）。

    公开给 apply.py 用：落库层要再拦一道，不能因为「上游 sanitize 应该已经过滤了」
    就放行——apply.py 的 relations 分支会 create_person，那是一条真实的旁路。
    """
    n = _s(name)
    return bool(n) and bool(_ROLE_PERSON.match(n))


def _with_weekday(today: str) -> str:
    """把 '2026-09-30' 补成 '2026-09-30（周三）'，算不出就原样返回。

    为什么在代码里算：LLM 自己做星期运算是不可靠的——实测「这周五」（今天周三）
    被算成了周六。把星期几直接喂给它，而不是让它推。这是一行代码的事，
    但它决定的是一条承诺提醒建在哪一天（FR-TIM-02）。
    """
    s = _s(today)[:10]
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return _s(today)
    return f"{d.strftime('%Y-%m-%d')}（{_WEEKDAY_CN[d.weekday()]}）"


def _week_bounds(today: str) -> str:
    """给出「本月/下个月」「本周/下周」的日期范围与锚点；解析失败返回空串。

    为什么需要：只给「今天星期几」还不够——模型不知道「这周/下周」以哪天为界，
    实测今天周三时把「下周五」算成了最近的周五（hold2-p02，3 次全错）。
    而只给周界又会带出新问题：周界里全是下个月的日期（如本周 09-28~10-04），
    模型会把「现在」锚到 10 月，于是「下个月 5 号」被算成 11-05（hold-p11）。
    所以月界和周界都要作为权威值直接给它，让它「从给定值里选」，
    而不是自己推算（FR-TIM-02：提醒建在哪一天全看这个）。
    """
    s = _s(today)[:10]
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return ""
    mon = d - timedelta(days=d.weekday())
    nmon = mon + timedelta(days=7)
    nmonth = (d.replace(day=28) + timedelta(days=8)).replace(day=1)  # 下个月 1 号
    f = lambda x: x.strftime("%Y-%m-%d")  # noqa: E731
    return (
        f"本月是 {d.strftime('%Y-%m')}，下个月是 {nmonth.strftime('%Y-%m')}。"
        f"本周（周一到周日）是 {f(mon)} 至 {f(mon + timedelta(days=6))}；"
        f"下周是 {f(nmon)} 至 {f(nmon + timedelta(days=6))}。"
        f"「这/本周X」指本周的星期X（如本周五={f(mon + timedelta(days=4))}），"
        f"「下周X」指下周的星期X（如下周五={f(nmon + timedelta(days=4))}）。"
    )


def _inherit_family_facts(ref: dict, pseudo: dict) -> None:
    """把「X 的 <亲属>」的信息收拢到 X 名下（家属信息），不再为它建人物。

    只用**原文里逐字出现过的片段**作为 fact 文本（evidence 已经是校验过的连续片段），
    不自己造句——否则「为了不丢信息」反而会造出一条幻觉。
    """
    seen = {flatten_for_match(f["text"]) for f in ref["facts"]}
    for fa in pseudo["facts"]:
        key = flatten_for_match(fa["text"])
        if key and key not in seen:
            seen.add(key)
            ref["facts"].append(dict(fa))
    for f in pseudo["fields"]:
        text = _s(f.get("evidence"))
        key = flatten_for_match(text)
        if not text or key in seen:
            continue
        seen.add(key)
        ref["facts"].append({"text": text, "confidence": f["confidence"], "evidence": text})


def _drop_role_persons(people: list[dict]) -> list[dict]:
    """剔除角色词「人物」，并把它们的内容收拢回本人（FR-ID-01 / FR-MEM-06）。

    三条规则：
    1. 名字是角色词/亲属称谓的条目 → 不作为人物保留。
    2. 「X 的 <亲属>」且 X 也在本批里 → 把内容并到 X 名下（家属信息）；
       **只有**这种指名道姓的参照才做归属——「前台」「那个客户」这种没有明确参照人的，
       直接丢弃而不是塞给本批唯一的那个人（那恰恰是 FR-MEM-06 禁止的
       「把他人特征归档到当前人物名下」）。
    3. 关系边/别名指向角色词的 → 一并过滤。apply.py 在 relations[].to 找不到人时
       会 `create_person`，不过滤等于绕开这道闸门又建了一个人物。
    """
    real: list[dict] = []
    pseudo: list[dict] = []
    for p in people:
        if is_role_person(p.get("canonical_name")) or is_role_person(p.get("name")):
            pseudo.append(p)
        else:
            real.append(p)

    by_name = {p["canonical_name"]: p for p in real}
    for ps in pseudo:
        m = _KIN_OF.match(_s(ps.get("canonical_name")))
        ref = by_name.get(m.group("base")) if m else None
        if ref is not None and ref is not ps:
            _inherit_family_facts(ref, ps)

    for p in real:
        p["relations"] = [r for r in p["relations"] if not is_role_person(r.get("to"))]
        p["aliases"] = [a for a in p["aliases"] if not is_role_person(a.get("alias"))]
    return real


def _evidence(ev: Any, src_flat: str) -> str:
    """把模型给的 evidence 收敛成「原文中的一段连续片段」。

    模型偶尔会把不相邻的两处原文用「；」拼成一条 evidence，这违反「逐字截取」，
    评测里会表现为 evidence 不在原文（缺陷 B）。这里做两级兜底：
    1. 整条能对上原文 → 原样保留；
    2. 否则按标点切开，取第一段确实逐字出现的 → 保留该段（仍是连续片段）；
    3. 都对不上 → 返回空串：宁可这条没有凭证，也不留一条伪造的凭证。

    注意这里【只】处理凭证，不动 confidence。凭证不可用 ≠ 事实不可信，
    悄悄改 confidence 会掩盖模型自身的校准误差，那是另一条独立规则该管的事。
    """
    ev = _s(ev)
    if not ev or not src_flat:
        return ev
    if flatten_for_match(ev) in src_flat:
        return ev
    for part in _EV_SPLIT.split(ev):
        part = part.strip()
        if part and flatten_for_match(part) in src_flat:
            return part
    return ""


def sanitize(obj: Any, source: str = "") -> dict:
    """把模型输出规整成稳定的 patch 结构，金额统一为分。

    source：原文。给定时会强制 evidence 必须是原文里的连续片段（缺陷 B 的兜底）。
    """
    out: dict[str, Any] = {"people": [], "my_notes": []}
    built: list[dict] = []
    if not isinstance(obj, dict):
        return out
    src_flat = flatten_for_match(source) if source else ""
    for p in obj.get("people") or []:
        if not isinstance(p, dict):
            continue
        name = _s(p.get("name"))
        if not name:
            continue
        person = {
            "name": name,
            "canonical_name": _s(p.get("canonical_name")) or name,
            "fields": [], "aliases": [], "relations": [], "facts": [],
            "promises": [], "ledger": [], "value_read": None,
        }
        for f in p.get("fields") or []:
            if isinstance(f, dict) and _s(f.get("field")) and _s(f.get("value")):
                person["fields"].append({
                    "field": _s(f["field"]), "value": _s(f["value"]),
                    "confidence": _f(f.get("confidence")),
                    "evidence": _evidence(f.get("evidence"), src_flat),
                })
        for a in p.get("aliases") or []:
            if isinstance(a, dict) and _s(a.get("alias")):
                person["aliases"].append({"alias": _s(a["alias"]), "confidence": _f(a.get("confidence"))})
        for r in p.get("relations") or []:
            if isinstance(r, dict) and _s(r.get("to")):
                person["relations"].append({
                    "to": _s(r["to"]), "kind": _s(r.get("kind")) or "know",
                    "confidence": _f(r.get("confidence")),
                    "evidence": _evidence(r.get("evidence"), src_flat),
                })
        for fa in p.get("facts") or []:
            if isinstance(fa, dict) and _s(fa.get("text")):
                person["facts"].append({
                    "text": _s(fa["text"]), "confidence": _f(fa.get("confidence"), 0.85),
                    "evidence": _evidence(fa.get("evidence"), src_flat),
                })
        for pr in p.get("promises") or []:
            if not isinstance(pr, dict) or not _s(pr.get("content")):
                continue
            content = _s(pr["content"])
            # 客套不算承诺（缺陷 A 的代码兜底）。闸门要连 content 所在的原句一起判：
            # 模型改写措辞会把触发词弄丢（「回头我加你微信」→「加你微信」，hold2-n06）。
            span = _locate_span(content, source, src_flat) if source else ""
            if _promise_is_pleasantry(content, span):
                continue
            person["promises"].append({
                "who": pr.get("who") if pr.get("who") in ("me", "them") else "me",
                "content": content, "due": _s(pr.get("due")) or None,
                "confidence": _f(pr.get("confidence"), 0.85),
            })
        for le in p.get("ledger") or []:
            if isinstance(le, dict) and le.get("amount") is not None:
                try:
                    cents = int(round(float(le["amount"]) * 100))
                except (TypeError, ValueError):
                    continue
                if cents <= 0:
                    continue
                person["ledger"].append({
                    "direction": le.get("direction") if le.get("direction") in ("in", "out") else "out",
                    "amount_cents": cents, "note": _s(le.get("note")),
                    "occurred": _s(le.get("occurred")) or None,
                    "confidence": _f(le.get("confidence"), 0.7),
                })
        vr = p.get("value_read")
        if isinstance(vr, dict) and _s(vr.get("text")):
            person["value_read"] = {
                "text": _s(vr["text"]), "confidence": _f(vr.get("confidence"), 0.7),
                "evidence": _evidence(vr.get("evidence"), src_flat),
            }
        built.append(person)

    out["people"] = _drop_role_persons(built)

    for n in obj.get("my_notes") or []:
        if isinstance(n, dict) and _s(n.get("text")):
            out["my_notes"].append({
                "text": _s(n["text"]),
                "kind": n.get("kind") if n.get("kind") in ("semantic", "episodic") else "semantic",
                "confidence": _f(n.get("confidence"), 0.9),
            })
    return out


def extract_from_text(text: str, known_names: list[str], today: str) -> dict:
    llm = get_llm()
    names = "、".join(known_names) if known_names else "（暂无）"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"今天是 {_with_weekday(today)}。{_week_bounds(today)}已知人物：{names}\n\n待抽取文本：\n{text}"},
    ]
    return sanitize(llm.chat_json(messages, tier="mid", max_tokens=4000), source=text)
