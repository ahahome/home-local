"""截图转写：vision 模型把聊天/转账/名片截图转成纯文本（FR-ING-01）。"""
from __future__ import annotations

from app.llm import get_llm

PROMPT = """把这张截图里的内容转写成纯文本。

要求：
1. 如果是聊天截图：按「说话人: 内容」每行一条，保留顺序；有时间则保留时间。
2. 如果是转账/账单截图：转写出 金额、方向（转入/转出）、对方、时间、备注。
3. 如果是名片/资料截图：转写出所有字段。
4. 不要总结、不要评论，只转写原文内容。emoji 保留。
5. 看不清的部分用 [?] 标注，不要猜。"""


def transcribe_screenshot(image_bytes: bytes) -> str:
    return get_llm().vision(PROMPT, [image_bytes], max_tokens=3000)
