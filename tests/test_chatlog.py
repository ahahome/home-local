"""聊天记录导入器：多格式容错解析 + 误判防护（FR-ING-01/05 的延伸）。

只测纯解析层（不触网、不需要 LLM），这是格式兼容的回归底线。
"""
from app.ingest import chatlog

# --- 格式 1：留痕 / WeChatExporter 风格：时间戳 + 名字，正文在下一行 ---
LIUCHEN = """2024-05-12 14:23:11 张伟
在忙吗
2024-05-12 14:23:20 我
刚开完会，怎么了
2024-05-12 14:24:02 张伟
我们公司下个月要搬到杭州了，以后见面方便点
"""

# --- 格式 2：QQ 消息管理器导出 ---
QQ = """================ 消息记录 ================
消息分组:我的好友
消息对象:李娜
================ 消息记录 ================
2024-06-01 09:10:00 李娜
早
2024-06-01 09:11:30 我
早啊
2024-06-01 09:12:00 李娜(12345678)
孩子今年上初二了，作业多得要命
"""

# --- 格式 3：方括号单行 ---
BRACKET = """[2024-06-03 20:01:05] 王强: 周五那个方案我改完了
[2024-06-03 20:02:11] 我: 收到，我看看
[2024-06-03 20:03:00] 王强: 顺便说下，我换了手机号 13800001111
"""

# --- 格式 4：仅日期行 + 名字: 内容 ---
DATE_INLINE = """2024-06-05 星期三
赵敏: 我妈住院了，这周可能来不了
我: 严重吗
赵敏: 不严重，就是血压高
2024-06-06 星期四
赵敏: 出院了，谢谢关心
"""

# --- 格式 5：名字在前，时间戳在后 ---
NAME_FIRST = """刘洋 2024-06-08 10:00:00
下周三来北京出差
我 2024-06-08 10:01:00
到时候一起吃饭
"""


def test_parse_liuchen_format():
    r = chatlog.parse(LIUCHEN)
    assert r["stats"]["parsed"] == 3
    assert set(r["speakers"]) == {"张伟", "我"}
    assert r["me"] == "我" and r["me_guessed"] is False
    assert r["messages"][0]["speaker"] == "张伟"
    assert r["messages"][0]["text"] == "在忙吗"
    assert r["span"] == ["2024-05-12", "2024-05-12"]


def test_parse_qq_format_strips_qq_number_and_noise():
    r = chatlog.parse(QQ)
    assert r["stats"]["parsed"] == 3
    # 分隔线与「消息分组」被跳过，不带进消息
    assert all("=====" not in m["text"] for m in r["messages"])
    assert "李娜" in r["speakers"] and "12345678" not in r["speakers"]
    assert r["partner_hint"] == "李娜"
    assert r["me"] == "我"


def test_parse_bracket_single_line():
    r = chatlog.parse(BRACKET)
    assert r["stats"]["parsed"] == 3
    assert r["messages"][2]["text"].startswith("顺便说下")
    assert r["me"] == "我"


def test_parse_date_only_with_inline_speaker():
    r = chatlog.parse(DATE_INLINE)
    assert r["stats"]["parsed"] == 4
    assert set(r["speakers"]) == {"赵敏", "我"}
    assert r["messages"][0]["date"] == "2024-06-05"
    # 跨天：最后一条落在 06-06
    assert r["messages"][-1]["date"] == "2024-06-06"
    assert r["span"] == ["2024-06-05", "2024-06-06"]


def test_parse_name_first_format():
    r = chatlog.parse(NAME_FIRST)
    assert r["stats"]["parsed"] == 2
    assert r["messages"][0]["speaker"] == "刘洋"
    assert r["messages"][0]["date"] == "2024-06-08"


def test_system_and_media_lines_skipped():
    text = """2024-06-09 10:00:00 张伟
[图片]
2024-06-09 10:00:30 张伟
"张伟" 撤回了一条消息
2024-06-09 10:01:00 张伟
刚发错了
"""
    r = chatlog.parse(text)
    assert r["stats"]["system"] == 1
    assert r["stats"]["parsed"] == 2
    assert r["messages"][1]["text"] == "刚发错了"


def test_me_guessed_when_no_self_marker():
    text = """2024-06-10 10:00:00 甲
你好
2024-06-10 10:00:05 甲
在吗
2024-06-10 10:00:10 乙
在
"""
    r = chatlog.parse(text)
    assert r["me"] == "甲" and r["me_guessed"] is True


def test_explicit_me_overrides_guess():
    r = chatlog.parse(LIUCHEN, me="张伟")
    assert r["me"] == "张伟" and r["me_guessed"] is False


def test_multi_line_body_merged():
    text = """2024-06-11 10:00:00 张伟
第一行
第二行
2024-06-11 10:01:00 我
收到
"""
    r = chatlog.parse(text)
    assert r["messages"][0]["text"] == "第一行\n第二行"


def test_looks_like_chatlog_true_and_false():
    assert chatlog.looks_like_chatlog(LIUCHEN) is True
    assert chatlog.looks_like_chatlog(QQ) is True
    assert chatlog.looks_like_chatlog(BRACKET) is True
    doc = """# 产品说明
本系统用于管理客户关系。
字段说明：company 表示公司，title 表示职位。
注意事项：请勿随意删除数据。
"""
    assert chatlog.looks_like_chatlog(doc) is False
    assert chatlog.looks_like_chatlog("") is False


def test_decode_text_gbk_fallback():
    raw = LIUCHEN.encode("gb18030")
    assert "张伟" in chatlog.decode_text(raw)
    assert "张伟" in chatlog.decode_text(LIUCHEN.encode("utf-8-sig"))


def test_clean_name_drops_tail_parenthesis():
    assert chatlog._clean_name("李娜(12345678)") == "李娜"
    assert chatlog._clean_name("张三：") == "张三"
