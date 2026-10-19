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


def test_chat_response_metadata_is_opt_in_and_contains_only_finish_usage_and_model():
    llm, create = client_stub()
    create.return_value.choices[0].finish_reason = 'length'
    create.return_value.usage = SimpleNamespace(prompt_tokens=20, completion_tokens=1800,
        total_tokens=1820, completion_tokens_details=SimpleNamespace(reasoning_tokens=1800),
        private_provider_field='sk-private')
    create.return_value.request_id = 'sk-private'
    assert llm.chat([{'role': 'user', 'content': 'hello'}]) == {'role': 'assistant', 'content': 'ok'}
    result = llm.chat([{'role': 'user', 'content': 'hello'}], tier='mini',
                      max_attempts=1, response_metadata=True)
    assert result['response_metadata'] == {'finish_reason': 'length', 'model': 'mini',
        'usage': {'prompt_tokens': 20, 'completion_tokens': 1800, 'total_tokens': 1820,
                  'reasoning_tokens': 1800}}
    assert 'sk-private' not in str(result)
    assert 'max_attempts' not in create.call_args.kwargs
    assert 'response_metadata' not in create.call_args.kwargs


def test_empty_choices_are_reported_as_empty_content_with_no_hidden_request():
    llm, create = client_stub()
    create.return_value.choices = []
    result = llm.chat([{'role': 'user', 'content': 'hello'}], max_attempts=1, response_metadata=True)
    assert result['content'] == '' and result['response_metadata']['finish_reason'] is None
    assert create.call_count == 1


def test_malformed_finish_reason_is_not_exposed_or_raised_after_request():
    llm, create = client_stub()
    create.return_value.choices[0].finish_reason = {'private': 'sk-private'}
    result = llm.chat([{'role': 'user', 'content': 'hello'}], response_metadata=True)
    assert result['response_metadata']['finish_reason'] is None
    assert 'sk-private' not in str(result)
    assert create.call_count == 1


def test_single_attempt_timeout_has_safe_retryable_metadata(monkeypatch):
    llm, create = client_stub()
    create.side_effect = TimeoutError('sk-private provider headers and body')
    monkeypatch.setattr('app.llm.time.sleep', lambda _: pytest.fail('must not retry internally'))
    with pytest.raises(LLMError) as failure:
        llm.chat([{'role': 'user', 'content': 'hello'}], max_attempts=1)
    assert create.call_count == 1
    assert failure.value.category == 'timeout' and failure.value.retryable
    assert 'sk-private' not in str(failure.value)


@pytest.mark.parametrize('status', [400, 401, 403, 404, 405, 415, 422])
def test_hard_configuration_failure_never_retries_or_exposes_body(monkeypatch, status):
    class ProviderError(RuntimeError):
        status_code = status
    llm, create = client_stub()
    create.side_effect = ProviderError('sk-private provider body')
    monkeypatch.setattr('app.llm.time.sleep', lambda _: pytest.fail('configuration must not retry'))
    with pytest.raises(LLMError) as failure:
        llm.chat([{'role': 'user', 'content': 'hello'}])
    assert create.call_count == 1
    assert failure.value.category == 'config' and not failure.value.retryable
    assert failure.value.status_code == status and 'sk-private' not in str(failure.value)


def test_default_chat_keeps_four_attempt_budget_for_recoverable_requests(monkeypatch):
    llm, create = client_stub()
    success = create.return_value
    create.side_effect = [ConnectionError('first'), ConnectionError('second'), ConnectionError('third'), success]
    delays = []
    monkeypatch.setattr('app.llm.time.sleep', delays.append)
    assert llm.chat([{'role': 'user', 'content': 'hello'}])['content'] == 'ok'
    assert create.call_count == 4 and delays == [1, 2, 4]


def test_missing_key_is_a_nonretryable_configuration_error(monkeypatch):
    monkeypatch.setattr('app.llm.get_settings', lambda: SimpleNamespace(llm_api_key=''))
    with pytest.raises(LLMError) as failure:
        LLM()
    assert failure.value.category == 'config' and not failure.value.retryable
