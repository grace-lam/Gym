from unittest.mock import AsyncMock, patch

import httpx
import pytest

from harbor.llms.base import ContextLengthExceededError
from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(**kwargs) -> NemoGymLLM:
    defaults = dict(model_name="test-model", api_base="http://localhost:8000/v1")
    defaults.update(kwargs)
    return NemoGymLLM(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nemo_gym_llm_extracts_openai_shape():
    """Standard vLLM-shaped response: prompt_token_ids at top-level, generation_token_ids in message."""
    llm = _make_llm(collect_rollout_details=True)

    mock_json = {
        "choices": [
            {
                "message": {
                    "content": "hello",
                    "generation_token_ids": [7, 8],
                },
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                "finish_reason": "stop",
            }
        ],
        "prompt_token_ids": [1, 2, 3],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }

    with patch.object(llm, "_post_chat_completions", new_callable=AsyncMock, return_value=mock_json):
        response = await llm.call(prompt="hello")

    assert response.content == "hello"
    assert response.prompt_token_ids == [1, 2, 3]
    assert response.completion_token_ids == [7, 8]
    assert response.logprobs == [-0.1, -0.2]
    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.cache_tokens == 4


@pytest.mark.asyncio
async def test_nemo_gym_llm_extracts_nemo_proxy_shape():
    """NeMo proxy shape: token IDs embedded in the message dict."""
    llm = _make_llm(collect_rollout_details=True)

    mock_json = {
        "choices": [
            {
                "message": {
                    "content": "proxy output",
                    "prompt_token_ids": [11, 12],
                    "generation_token_ids": ["token_id:13", "token_id:14"],
                    "generation_log_probs": [-0.3, -0.4],
                },
                "finish_reason": "stop",
            }
        ],
    }

    with patch.object(llm, "_post_chat_completions", new_callable=AsyncMock, return_value=mock_json):
        response = await llm.call(prompt="hello")

    assert response.content == "proxy output"
    assert response.prompt_token_ids == [11, 12]
    assert response.completion_token_ids == [13, 14]
    assert response.logprobs == [-0.3, -0.4]


@pytest.mark.asyncio
async def test_nemo_gym_llm_context_error_translation():
    """HTTP 400 with context-length message raises ContextLengthExceededError."""
    llm = _make_llm()

    error_response = httpx.Response(
        status_code=400,
        text="maximum context length exceeded",
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )

    async def _raise_context_error(payload, timeout_sec=None):
        llm._raise_for_status(error_response)

    with patch.object(llm, "_post_chat_completions", side_effect=_raise_context_error):
        with pytest.raises(ContextLengthExceededError):
            await llm.call(prompt="hello")


@pytest.mark.asyncio
async def test_nemo_gym_llm_no_rollout_details_for_openai_model():
    """When response has no token IDs / logprobs, those fields are None."""
    llm = _make_llm(collect_rollout_details=True)

    mock_json = {
        "choices": [
            {
                "message": {"content": "plain output"},
                "finish_reason": "stop",
            }
        ],
    }

    with patch.object(llm, "_post_chat_completions", new_callable=AsyncMock, return_value=mock_json):
        response = await llm.call(prompt="hello")

    assert response.prompt_token_ids is None
    assert response.completion_token_ids is None
    assert response.logprobs is None


@pytest.mark.asyncio
async def test_nemo_gym_llm_extra_chat_params_forwarded():
    """Extra chat params from responses_create_params are included in the payload."""
    llm = _make_llm(
        responses_create_params={"temperature": 0.5, "top_p": 0.9, "input": []},
    )

    captured_payload = {}

    async def _capture_post(payload, timeout_sec=None):
        nonlocal captured_payload
        captured_payload = payload
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    with patch.object(llm, "_post_chat_completions", side_effect=_capture_post):
        response = await llm.call(prompt="hello")

    assert response.content == "ok"
    # Temperature and top_p should appear in payload via extra_chat_params
    assert captured_payload.get("temperature") == 0.5
    assert captured_payload.get("top_p") == 0.9


@pytest.mark.parametrize(
    ("api_base", "expected_endpoint"),
    [
        ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
    ],
)
def test_nemo_gym_llm_chat_completions_endpoint(api_base, expected_endpoint):
    llm = _make_llm(api_base=api_base)
    assert llm._chat_completions_endpoint() == expected_endpoint
