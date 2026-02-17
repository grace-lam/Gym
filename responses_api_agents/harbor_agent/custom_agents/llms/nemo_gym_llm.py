import re
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.models.metric import UsageInfo
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

# Phrases in vLLM / OpenAI error bodies that signal context-length overflow.
_CONTEXT_LENGTH_ERROR_PHRASES = (
    "context length exceeded",
    "context_length_exceeded",
    "maximum context length",
    "`inputs` tokens + `max_new_tokens`",
)


class NemoGymLLM(BaseLLM):
    """LLM backend that calls NeMo Gym model servers via chat completions."""

    def __init__(
        self,
        model_name: str,
        api_base: str,
        collect_rollout_details: bool = False,
        model_info: dict[str, Any] | None = None,
        responses_create_params: dict[str, Any] | None = None,
        timeout_sec: float = 600.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name
        self._api_base = api_base.rstrip("/")
        self._collect_rollout_details = collect_rollout_details
        self._model_info = model_info or {}
        self._timeout_sec = timeout_sec

        # Pre-compute extra chat params from responses_create_params once,
        # since they don't change between calls.
        self._extra_chat_params = self._build_extra_chat_params(
            responses_create_params or {}
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=15),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type((
                ContextLengthExceededError,
                OutputLengthExceededError,
            ))
        ),
        reraise=True,
    )
    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if message_history is None:
            message_history = []
        messages = message_history + [{"role": "user", "content": prompt}]

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }
        payload.update(self._extra_chat_params)

        response_dict = await self._post_chat_completions(payload)

        choices = response_dict.get("choices", [])
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        if content is None:
            content = ""
        reasoning_content = (
            message.get("reasoning_content") if isinstance(message, dict) else None
        )

        # vLLM model server with uses_reasoning_parser merges reasoning into content
        # as <think>...</think> and does not return reasoning_content. Extract it so the
        # trajectory gets reasoning_content and content is the remainder.
        if reasoning_content is None and isinstance(content, str) and "<think>" in content:
            reasoning_matches, content = self._extract_reasoning_from_content(content)
            if reasoning_matches:
                reasoning_content = "\n".join(reasoning_matches)

        if isinstance(choice, dict) and choice.get("finish_reason") == "length":
            raise OutputLengthExceededError(
                f"Model {self._model_name} hit max_tokens limit. "
                "Response was truncated. Consider increasing max_tokens if possible.",
                truncated_response=content,
            )

        usage = self._extract_usage_info(response_dict)
        prompt_token_ids = None
        completion_token_ids = None
        logprobs = None
        if self._collect_rollout_details:
            prompt_token_ids, completion_token_ids = self._extract_token_ids(response_dict)
            logprobs = self._extract_logprobs(response_dict)

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            usage=usage,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            logprobs=logprobs,
        )

    def get_model_context_limit(self) -> int:
        """Get the context limit (max input tokens) for the current model.

        Returns:
            int: The maximum input tokens the model can accept, or a fallback value if unavailable.
        """
        fallback_context_limit = 1000000

        try:
            max_input_tokens = self._model_info.get("max_input_tokens")

            # Fallback to max_tokens if max_input_tokens not available
            if max_input_tokens is None:
                max_input_tokens = self._model_info.get("max_tokens")

            if isinstance(max_input_tokens, int) and max_input_tokens > 0:
                return max_input_tokens

            # Model info exists but doesn't have context limit info
            self._logger.warning(
                f"Model '{self._model_name}' info found but missing context limit fields. "
                f"Using fallback context limit: {fallback_context_limit}"
            )
        except Exception as e:
            self._logger.warning(
                f"Failed to retrieve model info for '{self._model_name}': {e}. "
                f"Using fallback context limit: {fallback_context_limit}"
            )

        return fallback_context_limit

    def get_model_output_limit(self) -> int | None:
        """Get the output limit (max output tokens) for the current model.

        Returns:
            int | None: The maximum output tokens the model can generate, or None if unavailable.
        """
        try:
            max_output_tokens = self._model_info.get("max_output_tokens")

            if max_output_tokens is None:
                # Model info exists but doesn't have max_output_tokens
                self._logger.debug(
                    f"Model '{self._model_name}' info found but missing max_output_tokens field."
                )

            if isinstance(max_output_tokens, int) and max_output_tokens > 0:
                return max_output_tokens

            return None
        except Exception as e:
            self._logger.debug(
                f"Failed to retrieve model info for '{self._model_name}': {e}."
            )
            return None

    async def _post_chat_completions(
        self, payload: dict[str, Any], timeout_sec: float | None = None
    ) -> dict[str, Any]:
        endpoint = self._chat_completions_endpoint()
        timeout = timeout_sec if timeout_sec is not None else self._timeout_sec
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=payload)

        if response.status_code >= 400:
            self._raise_for_status(response)

        return response.json()

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Inspect HTTP error responses and raise appropriate harbor errors."""
        error_text = response.text.lower()

        if any(phrase in error_text for phrase in _CONTEXT_LENGTH_ERROR_PHRASES):
            raise ContextLengthExceededError(
                f"Model {self._model_name} context length exceeded: {response.text}"
            )

        response.raise_for_status()

    def _chat_completions_endpoint(self) -> str:
        """Build a chat completions endpoint that tolerates base URLs with/without /v1."""
        if self._api_base.endswith("/v1"):
            return f"{self._api_base}/chat/completions"
        return f"{self._api_base}/v1/chat/completions"

    def _extract_token_ids(self, response: dict[str, Any]) -> tuple[list[int] | None, list[int] | None]:
        choices = response.get("choices", [])
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}

        # vllm_model/app.py writes token-id details into choice.message.
        prompt_token_ids = (
            message.get("prompt_token_ids") if isinstance(message, dict) else None
        )
        # Keep a top-level prompt fallback for compatibility with OpenAI-style response shapes.
        if prompt_token_ids is None:
            prompt_token_ids = response.get("prompt_token_ids")

        completion_token_ids = (
            message.get("generation_token_ids") if isinstance(message, dict) else None
        )

        return (
            self._normalize_token_ids(prompt_token_ids),
            self._normalize_token_ids(completion_token_ids),
        )

    def _build_extra_chat_params(self, responses_create_params: dict[str, Any]) -> dict[str, Any]:
        """Convert responses_create_params to chat completion params (called once at init)."""
        if not responses_create_params:
            return {}

        from responses_api_models.vllm_model.app import VLLMConverter

        params_for_conversion = {
            key: value for key, value in responses_create_params.items() if key != "input"
        }
        params_for_conversion["input"] = []
        responses_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            params_for_conversion
        )

        converter = VLLMConverter(
            return_token_id_information=self._collect_rollout_details,
        )
        chat_params = converter.responses_to_chat_completion_create_params(
            responses_params
        ).model_dump(exclude_unset=True)

        # Harbor constructs chat history itself; keep only non-message params.
        chat_params.pop("messages", None)
        return chat_params

    def _extract_logprobs(self, response: dict[str, Any]) -> list[float] | None:
        choices = response.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return None

        choice = choices[0]
        if not isinstance(choice, dict):
            return None

        # Primary schema from responses_api_models/vllm_model/app.py
        message = choice.get("message", {})
        if isinstance(message, dict):
            generation_log_probs = message.get("generation_log_probs")
            if isinstance(generation_log_probs, list):
                return [
                    float(lp) for lp in generation_log_probs if isinstance(lp, (int, float))
                ] or None

        # Fallback schema used by OpenAI-style responses
        logprobs_data = choice.get("logprobs")
        if isinstance(logprobs_data, dict):
            content = logprobs_data.get("content", [])
            extracted = [
                token_data["logprob"]
                for token_data in content
                if isinstance(token_data, dict) and "logprob" in token_data
            ]
            if extracted:
                return extracted

        return None

    def _extract_usage_info(self, response: dict[str, Any]) -> UsageInfo | None:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return None

        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        prompt_tokens_details = usage.get("prompt_tokens_details") or {}
        cache_tokens = (
            prompt_tokens_details.get("cached_tokens", 0)
            if isinstance(prompt_tokens_details, dict)
            else 0
        ) or 0

        return UsageInfo(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            cache_tokens=int(cache_tokens),
            cost_usd=0.0,
        )

    def _normalize_token_ids(self, token_ids: Any) -> list[int] | None:
        if not isinstance(token_ids, list):
            return None

        normalized: list[int] = []
        for token_id in token_ids:
            if isinstance(token_id, int):
                normalized.append(token_id)
                continue
            if isinstance(token_id, str):
                stripped = token_id.removeprefix("token_id:")
                if stripped.isdigit():
                    normalized.append(int(stripped))
                    continue
            return None

        return normalized or None

    def _extract_reasoning_from_content(self, content: str) -> tuple[list[str], str]:
        """Extract reasoning from <think></think> tags; return (matches, cleaned_content)."""
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(content)
        cleaned = pattern.sub("", content).strip()
        return matches, cleaned

