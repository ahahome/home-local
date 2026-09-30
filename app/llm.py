"""LLM 客户端：封装 chat / vision / embed，含模型分层、重试、JSON 解析。

模型分层（可用性已实测）：
  mini  = qwen-turbo        便宜：标题、日志、轻量归纳
  mid   = qwen-plus         主力：抽取、harness 工具循环（工具调用稳定）
  think = deepseek-v4-pro   分析：消歧复核、决策
  pro   = qwen3.8-max       深度：Deep Research 合成
  vision= qwen3-vl-plus     截图转写
  embed = text-embedding-v4 1024 维
"""
from __future__ import annotations

import base64
import time
from typing import Any, Iterable

from openai import OpenAI

from app.config import get_settings
from app.util import parse_json_loose


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self):
        s = get_settings()
        if not s.llm_api_key:
            raise LLMError("HOME_LLM_API_KEY 未配置")
        self.s = s
        self.client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key, timeout=120.0)

    def _model(self, tier: str) -> str:
        return {
            "mini": self.s.model_mini,
            "mid": self.s.model_mid,
            "think": self.s.model_think,
            "pro": self.s.model_pro,
            "vision": self.s.model_vision,
        }.get(tier, tier)

    def _with_retry(self, fn, what: str):
        delay = 1.0
        for attempt in range(4):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "AccessDenied" in msg or "Access denied" in msg:
                    raise LLMError(f"{what}: 模型无权限访问 — {msg}") from e
                if attempt == 3:
                    raise LLMError(f"{what}: 重试耗尽 — {msg}") from e
                time.sleep(delay)
                delay = min(delay * 2, 8.0)

    # ---------- 文本对话 ----------
    def chat(
        self,
        messages: list[dict],
        tier: str = "mid",
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 3000,
    ) -> dict:
        """返回 message dict：{role, content, tool_calls?}"""
        kwargs: dict[str, Any] = dict(
            model=self._model(tier),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        def call():
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message

        msg = self._with_retry(call, f"chat[{tier}]")
        out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
                for tc in msg.tool_calls
            ]
        return out

    def chat_json(self, messages: list[dict], tier: str = "mid", max_tokens: int = 3000, retries: int = 2) -> Any:
        """要求模型输出 JSON 并解析；解析失败时自我修正重试。"""
        last_err = None
        conv = list(messages)
        for _ in range(retries + 1):
            msg = self.chat(conv, tier=tier, max_tokens=max_tokens)
            raw = msg.get("content", "")
            try:
                return parse_json_loose(raw)
            except Exception as e:  # noqa: BLE001
                last_err = e
                conv.append({"role": "assistant", "content": raw})
                conv.append({"role": "user", "content": "上面的输出不是合法 JSON。请只输出一个合法的 JSON（不要代码围栏、不要解释）。"})
        raise LLMError(f"chat_json 解析失败: {last_err}")

    # ---------- 视觉 ----------
    def vision(self, prompt: str, images: list[bytes], max_tokens: int = 3000) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(img).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        def call():
            resp = self.client.chat.completions.create(
                model=self.s.model_vision,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        return self._with_retry(call, "vision")

    # ---------- 向量 ----------
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def call():
            resp = self.client.embeddings.create(model=self.s.model_embed, input=texts)
            return [d.embedding for d in resp.data]

        return self._with_retry(call, "embed")


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm
