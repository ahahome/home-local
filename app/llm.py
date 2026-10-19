"""LLM 客户端：封装 chat / vision / embed，含模型分层、重试、JSON 解析。

具体模型从 Settings 读取。非流式调用显式传 stream=False，兼容默认返回 SSE 的网关。
"""
from __future__ import annotations

import base64
import time
from typing import Any, Iterable

import httpx
from openai import OpenAI

from app.config import get_settings
from app.util import parse_json_loose


class LLMError(RuntimeError):
    """Safe error text plus bounded diagnostics; provider bodies stay out of metadata."""

    def __init__(self, message, *, category='request', retryable=False, status_code=None):
        super().__init__(message)
        self.category = category if category in {'request', 'config', 'timeout'} else 'request'
        self.retryable = retryable is True and self.category != 'config'
        self.status_code = status_code if type(status_code) is int and 100 <= status_code <= 599 else None


def failure_metadata(error):
    """Classify without serializing exception text, headers, request IDs, or bodies."""
    if isinstance(error, LLMError):
        return {'category': error.category, 'retryable': error.retryable,
                'status_code': error.status_code}
    status = getattr(error, 'status_code', None)
    status = status if type(status) is int and 100 <= status <= 599 else None
    if status in {400, 401, 403, 404, 405, 415, 422}:
        return {'category': 'config', 'retryable': False, 'status_code': status}
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return {'category': 'timeout', 'retryable': True, 'status_code': status}
    # OpenAI wraps httpx errors, but its public exception classes are stable.
    from openai import APIConnectionError, APITimeoutError
    if isinstance(error, APITimeoutError):
        return {'category': 'timeout', 'retryable': True, 'status_code': status}
    if status is not None:
        return {'category': 'request', 'retryable': status in {408, 409, 425, 429} or status >= 500,
                'status_code': status}
    if isinstance(error, (ConnectionError, httpx.TransportError, APIConnectionError)):
        return {'category': 'request', 'retryable': True, 'status_code': None}
    return {'category': 'request', 'retryable': False, 'status_code': None}


class LLM:
    def __init__(self):
        try:
            s = get_settings()
        except Exception as error:
            raise LLMError('模型配置无效', category='config') from error
        if not s.llm_api_key:
            raise LLMError("HOME_LLM_API_KEY 未配置", category='config')
        self.s = s
        try:
            self.client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key, timeout=120.0, max_retries=0)
        except Exception as error:
            raise LLMError('模型客户端配置无效', category='config') from error

    def _model(self, tier: str) -> str:
        return {
            "mini": self.s.model_mini,
            "mid": self.s.model_mid,
            "think": self.s.model_think,
            "pro": self.s.model_pro,
            "vision": self.s.model_vision,
        }.get(tier, tier)

    def _with_retry(self, fn, what: str, *, max_attempts=4):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 4:
            raise ValueError('max_attempts must be between 1 and 4')
        delay = 1.0
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                info = failure_metadata(e)
                if not info['retryable'] or attempt == max_attempts - 1:
                    label = {'config': '请求配置或权限错误', 'timeout': '请求超时',
                             'request': '模型请求失败'}[info['category']]
                    raise LLMError(f'{what}: {label}', category=info['category'],
                                   retryable=info['retryable'], status_code=info['status_code']) from e
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
        *,
        max_attempts: int = 4,
        response_metadata: bool = False,
    ) -> dict:
        """返回 message dict：{role, content, tool_calls?}"""
        kwargs: dict[str, Any] = dict(
            model=self._model(tier),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        def call():
            return self.client.chat.completions.create(**kwargs)

        resp = self._with_retry(call, f"chat[{tier}]", max_attempts=max_attempts)
        choice = resp.choices[0] if resp.choices else None
        msg = getattr(choice, 'message', None)
        out: dict[str, Any] = {"role": "assistant", "content": getattr(msg, 'content', None) or ""}
        if getattr(msg, "tool_calls", None):
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
                for tc in msg.tool_calls
            ]
        if response_metadata:
            finish = getattr(choice, 'finish_reason', None)
            usage, provider_usage = {}, getattr(resp, 'usage', None)
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                count = getattr(provider_usage, key, None)
                if type(count) is int and 0 <= count <= 100_000_000:
                    usage[key] = count
            reasoning = getattr(getattr(provider_usage, 'completion_tokens_details', None), 'reasoning_tokens', None)
            if type(reasoning) is int and 0 <= reasoning <= 100_000_000:
                usage['reasoning_tokens'] = reasoning
            out['response_metadata'] = {
                'finish_reason': finish if isinstance(finish, str) and finish in {
                    'stop', 'length', 'tool_calls', 'content_filter', 'function_call'} else None,
                'usage': usage,
                'model': kwargs['model'],
            }
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

    def chat_stream(self, messages, tier="mid", tools=None, max_tokens=3000, on_delta=None):
        kwargs = {"model": self._model(tier), "messages": messages,
                  "max_tokens": max_tokens, "temperature": 0.3, "stream": True}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        content, calls = [], {}
        with self.client.chat.completions.create(**kwargs) as stream:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    content.append(delta.content)
                    if on_delta:
                        on_delta(delta.content)
                for tc in delta.tool_calls or []:
                    call = calls.setdefault(tc.index, {"id": "", "type": "function",
                        "function": {"name": "", "arguments": ""}})
                    if tc.id:
                        call["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            call["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            call["function"]["arguments"] += tc.function.arguments
        result = {"role": "assistant", "content": "".join(content)}
        if calls:
            result["tool_calls"] = [calls[k] for k in sorted(calls)]
        return result

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
                stream=False,
            )
            return resp.choices[0].message.content or ""

        return self._with_retry(call, "vision")

    # ---------- 向量 ----------
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.s.model_embed:
            raise LLMError("向量模型未配置，当前仅使用结构化与全文检索")

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
