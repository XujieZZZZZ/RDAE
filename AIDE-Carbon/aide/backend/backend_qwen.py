"""
Backend for DASHSCOPE API using OpenAI-compatible interface.
"""

import json
import logging
import os
import time

import openai
from funcy import notnone, once, select_values
from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

QWEN_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_qwen_client():
    global _client
    api_key ="sk-1711bc211b1d45a69d402e7857e6ad36"
    #api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    _client = openai.OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_retries=0,
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query DeepSeek via OpenAI-compatible API.
    """

    _setup_qwen_client()
    filtered_kwargs = select_values(notnone, model_kwargs)

    messages = opt_messages_to_list(system_message, user_message)

    if len(messages) == 1 and messages[0]["role"] == "system":
        messages.append({
            "role": "user",
            "content": "Please respond according to the above instructions."
        })

    # If function calling is requested, attach the function spec
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

   # logger.info(f"DeepSeek request: system={system_message}, user={user_message}")

    completion = None
    t0 = time.time()

    try:
        completion = backoff_create(
            _client.chat.completions.create,
            QWEN_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
    except openai.BadRequestError as e:
        # Check whether the error indicates that function calling is not supported
        if "function calling" in str(e).lower() or "tool_choice" in str(e).lower():
            logger.warning(
                "Function calling was attempted but is not supported by this model. "
                "Falling back to plain text generation."
            )
            # Remove function-calling parameters and retry
            filtered_kwargs.pop("tools", None)
            filtered_kwargs.pop("tool_choice", None)

            # Retry without function calling
            completion = backoff_create(
                _client.chat.completions.create,
                QWEN_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
        else:
            # If it's some other error, re-raise
            raise


    req_time = time.time() - t0
    choice = completion.choices[0]
    output = choice.message.content

    # Decide how to parse the response
    if func_spec is None or "tools" not in filtered_kwargs:
        # No function calling was ultimately used
        output = choice.message.content
    else:
        # Attempt to extract tool calls
        tool_calls = getattr(choice.message, "tool_calls", None)
        if not tool_calls:
            logger.warning(
                "No function call was used despite function spec. Fallback to text.\n"
                f"Message content: {choice.message.content}"
            )
            output = choice.message.content
        else:
            first_call = tool_calls[0]
            # Optional: verify that the function name matches
            if first_call.function.name != func_spec.name:
                logger.warning(
                    f"Function name mismatch: expected {func_spec.name}, "
                    f"got {first_call.function.name}. Fallback to text."
                )
                output = choice.message.content
            else:
                try:
                    output = json.loads(first_call.function.arguments)
                except json.JSONDecodeError as ex:
                    logger.error(
                        "Error decoding function arguments:\n"
                        f"{first_call.function.arguments}"
                    )
                    raise ex


    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
        "model": completion.model,
        "created": getattr(completion, "created", None),
    }

    logger.info(
        f"Qwen API call completed - {completion.model} - "
        f"{req_time:.2f}s - {in_tokens + out_tokens} tokens"
    )
    logger.info(f"Qwen API response: {output}")

    return output, req_time, in_tokens, out_tokens, info
