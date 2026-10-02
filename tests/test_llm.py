from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.llm import LLM, LLMError


def client_stub():
    llm = LLM.__new__(LLM)
    llm.s = SimpleNamespace(
        model_mini="mini", model_mid="mid", model_think="think",
        model_pro="pro", model_vision="vision", model_embed="",
    )
    create = Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))]
    ))
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        embeddings=SimpleNamespace(create=Mock()),
    )
    return llm, create


def test_chat_explicitly_requests_non_streaming_gateway_response():
    llm, create = client_stub()
    assert llm.chat([{"role": "user", "content": "hello"}])["content"] == "ok"
    assert create.call_args.kwargs["stream"] is False


def test_vision_explicitly_requests_non_streaming_gateway_response():
    llm, create = client_stub()
    assert llm.vision("color?", [b"image"]) == "ok"
    assert create.call_args.kwargs["stream"] is False


def test_disabled_embeddings_fail_without_gateway_request():
    llm, _ = client_stub()
    with pytest.raises(LLMError, match="向量模型未配置"):
        llm.embed(["hello"])
    llm.client.embeddings.create.assert_not_called()


def test_empty_embeddings_do_not_require_a_model():
    llm, _ = client_stub()
    assert llm.embed([]) == []
