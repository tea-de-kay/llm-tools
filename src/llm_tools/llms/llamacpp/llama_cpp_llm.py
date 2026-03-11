from __future__ import annotations

from asyncio import get_running_loop
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, cast

from llama_cpp import (
    ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPart,
    ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestResponseFormat,
    CreateChatCompletionResponse,
    Llama,
    LlamaRAMCache,
)
from pydantic import BaseModel

from llm_tools.models.errors import ErrorCode, ErrorInfo
from llm_tools.models.llm import (
    DEFAULT_LLM_GENERATION_CONFIG,
    LLM,
    BasePrompt,
    LlmGenerationConfig,
    LlmMedium,
    LlmMessage,
    LlmMessageChunk,
    LlmSpec,
    LlmUsage,
)
from llm_tools.models.settings import LlmApiSettings, LlmSettings
from llm_tools.models.types import LlmMessageRole
from llm_tools.utils.log import LogFactory


_log = LogFactory.get_logger(__name__)


class LlamaCppLlmMessage(BaseModel):
    type: Literal["message"] = "message"
    role: LlmMessageRole
    content: str | list[ChatCompletionRequestMessageContentPart]

    @classmethod
    def from_llm_message(cls, msg: LlmMessage) -> LlamaCppLlmMessage:
        if msg.role is LlmMessageRole.USER:
            content: str | list[ChatCompletionRequestMessageContentPart] = []
            if msg.content:
                content.append(
                    ChatCompletionRequestMessageContentPartText(
                        type="text", text=msg.content
                    )
                )
            for medium in msg.media or []:
                content.append(cls.medium_to_content(medium))
        else:
            content = msg.content or ""

        return LlamaCppLlmMessage(role=msg.role, content=content)

    @staticmethod
    def medium_to_content(
        medium: LlmMedium,
    ) -> ChatCompletionRequestMessageContentPartImage:
        return ChatCompletionRequestMessageContentPartImage(
            type="image_url",
            image_url=f"data:{medium.mime_type};base64,{medium.content_b64}",
        )


class LlamaCppLlmSettings(LlmSettings):
    n_gpu_layers: int = 0
    n_ctx: int = 4096
    verbose: bool = False
    n_threads: int | None = None
    use_cache: bool = True
    cache_size_in_bytes: int = 100 * 1024 * 1024


class LlamaCppLLM(LLM):
    _LLMS: dict[tuple[str, str, str], Llama] = {}
    _SINGLE_THREAD_EXECUTOR = ThreadPoolExecutor(1, "llama-cpp-llm")

    def __init__(
        self, api_settings: LlmApiSettings, settings: LlamaCppLlmSettings, spec: LlmSpec
    ) -> None:
        self._api_settings = api_settings
        self._settings = settings
        self._llm_spec = spec

        self._llm = self._load_llm()

    def _load_llm(self) -> Llama:
        repo_id = self._api_settings.hf_repo_id
        model_filename = self._api_settings.hf_model_filename
        path = self._api_settings.model_path
        cache_key = (repo_id, model_filename, path)
        llm = self._LLMS.get(cache_key)
        if llm is None:
            if path:
                llm = Llama(model_path=path, **self._settings.model_dump())
            else:
                llm = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=model_filename,
                    **self._settings.model_dump(),
                )

            if self._settings.use_cache:
                llm.set_cache(
                    LlamaRAMCache(capacity_bytes=self._settings.cache_size_in_bytes)
                )

            self._LLMS[cache_key] = llm

        return llm

    async def generate(
        self,
        prompt: BasePrompt,
        config: LlmGenerationConfig = DEFAULT_LLM_GENERATION_CONFIG,
    ) -> LlmMessage:
        messages = prompt.to_llm_messages()
        _log.debug("Prompt [messages='{}']", messages)

        try:
            result = await self._generate(messages, config)
        except Exception as e:
            _log.exception("Exception for LLM request [exception='{}']", e)
            result = self._get_error_response(e)

        return result

    def generate_stream(
        self,
        prompt: BasePrompt,
        config: LlmGenerationConfig = DEFAULT_LLM_GENERATION_CONFIG,
    ) -> AsyncIterator[LlmMessageChunk | LlmMessage]:
        raise NotImplementedError()

    async def _generate(
        self,
        messages: Sequence[LlmMessage],
        config: LlmGenerationConfig,
    ) -> LlmMessage:
        loop = get_running_loop()
        response: CreateChatCompletionResponse = await loop.run_in_executor(
            self._SINGLE_THREAD_EXECUTOR,
            self._call_to_api,
            self._get_messages(messages),
            config,
        )

        _log.trace("LLM response [response='{}']", response)

        return self._get_llm_message(response)

    async def _generate_chunks(
        self,
        messages: Sequence[LlmMessage],
        config: LlmGenerationConfig,
        exclude_stream_prefixes: Sequence[str] = (),
    ) -> AsyncIterator[LlmMessageChunk | LlmMessage]:
        raise NotImplementedError()
        yield

    def _call_to_api(
        self,
        messages: list[ChatCompletionRequestMessage],
        config: LlmGenerationConfig,
    ) -> CreateChatCompletionResponse:
        response = self._llm.create_chat_completion(
            messages=messages,
            temperature=self._get_temperature(config.temperature),
            response_format=self._get_response_format(config),
            max_tokens=self._get_max_output_tokens(config),
            stream=False,
        )

        return cast(CreateChatCompletionResponse, response)

    def _get_messages(
        self, messages: Sequence[LlmMessage]
    ) -> list[ChatCompletionRequestMessage]:
        converted = [LlamaCppLlmMessage.from_llm_message(msg) for msg in messages]

        return [cast(ChatCompletionRequestMessage, m.model_dump()) for m in converted]

    def _get_response_format(
        self, config: LlmGenerationConfig
    ) -> ChatCompletionRequestResponseFormat:
        type = "text"
        schema = None
        if config.json_schema is not None:
            schema = config.json_schema
            if not isinstance(schema, dict):
                schema = schema.model_json_schema()
            type = "json_object"
        elif config.json_output:
            type = "json_object"

        _log.debug("Response format [type='{}', schema='{}']", type, schema)

        return ChatCompletionRequestResponseFormat(type=type, schema=schema)

    def _get_max_output_tokens(self, config: LlmGenerationConfig) -> int | None:
        return config.max_output_tokens or self._llm_spec.max_output_tokens

    def _get_max_prefix_len(self, prefixes: Sequence[str]) -> int:
        if not prefixes:
            return 0

        return max(len(prefix) for prefix in prefixes)

    def _get_error_response(self, e: Exception) -> LlmMessage:
        match e:
            case _:
                code = ErrorCode.LLM_API_ERROR
                detail = str(type(e))

        return LlmMessage(
            role=LlmMessageRole.ASSISTANT,
            content=None,
            usage=None,
            error=ErrorInfo(code=code, detail=detail),
        )

    def _get_temperature(self, temperature_: float) -> float:
        temperature = 0
        if self._llm_spec.supports_temperature:
            temperature = min(
                max(temperature_, self._llm_spec.min_temperature),
                self._llm_spec.max_temperature,
            )

        return temperature

    def _get_llm_message(self, response: CreateChatCompletionResponse) -> LlmMessage:
        content = response["choices"][0]["message"]["content"] or ""
        usage = response["usage"]
        return LlmMessage(
            content=content,
            role=LlmMessageRole.ASSISTANT,
            usage=[
                LlmUsage(
                    input_tokens=usage["prompt_tokens"],
                    output_tokens=usage["completion_tokens"],
                    model=self._llm_spec.base_model_name,
                    llm_name=self._llm_spec.base_model_name,
                ),
            ],
        )
