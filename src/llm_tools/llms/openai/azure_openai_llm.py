from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Literal, cast

from openai import (
    APITimeoutError,
    AsyncAzureOpenAI,
    AsyncStream,
    BadRequestError,
    Omit,
    RateLimitError,
    omit,
)
from openai.types.responses import (
    Response,
    ResponseFormatTextConfigParam,
    ResponseInputItemParam,
    ResponseStreamEvent,
    ResponseTextConfigParam,
)
from openai.types.responses.response_input_param import Message as ResponseInputMessage
from openai.types.shared_params import Reasoning
from pydantic import BaseModel, Field

from llm_tools.llms.openai.const import (
    BAD_REQUEST_CONTENT_FILTER_CODE,
    BAD_REQUEST_CONTEXT_LENGTH_CODE,
)
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
from llm_tools.models.settings import LlmApiSettings
from llm_tools.models.types import LlmMessageRole, LlmReasoningEffort
from llm_tools.utils.log import LogFactory


_log = LogFactory.get_logger(__name__)


class LlmImageUrl(BaseModel):
    url: str
    detail: str


class OpenAiMessageTextContent(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


class OpenAiMessageImageContent(BaseModel):
    type: Literal["input_image"] = "input_image"
    image_url: str = Field(repr=False)
    detail: str

    @classmethod
    def from_medium(cls, medium: LlmMedium) -> OpenAiMessageImageContent:
        return OpenAiMessageImageContent(
            image_url=f"data:{medium.mime_type};base64,{medium.content_b64}",
            detail=medium.detail_level,
        )


class OpenAiLlmMessage(BaseModel):
    type: Literal["message"] = "message"
    role: LlmMessageRole
    content: str | list[OpenAiMessageTextContent | OpenAiMessageImageContent]

    @classmethod
    def from_llm_message(cls, msg: LlmMessage) -> OpenAiLlmMessage:
        if msg.role is LlmMessageRole.USER:
            content = []
            if msg.content:
                content.append(OpenAiMessageTextContent(text=msg.content))
            for medium in msg.media or []:
                content.append(OpenAiMessageImageContent.from_medium(medium))
        else:
            content = msg.content or ""

        return OpenAiLlmMessage(role=msg.role, content=content)


class AzureOpenAiLLM(LLM):
    _CLIENTS: dict[tuple[str, str, str], AsyncAzureOpenAI] = {}

    def __init__(self, api_settings: LlmApiSettings, spec: LlmSpec) -> None:
        super().__init__(api_settings, spec)

        self._client = self._get_client()
        self._model_deployment: str = cast(str, self._api_settings.deployment)

    def _get_client(self) -> AsyncAzureOpenAI:
        url = self._api_settings.url
        version = self._api_settings.version
        api_key = self._api_settings.key
        cache_key = (url, version, api_key)
        client = self._CLIENTS.get(cache_key)
        if client is None:
            client = AsyncAzureOpenAI(
                azure_endpoint=url,
                api_version=version,
                api_key=api_key,
            )
            self._CLIENTS[cache_key] = client

        return client

    async def generate(
        self,
        prompt: BasePrompt,
        config: LlmGenerationConfig = DEFAULT_LLM_GENERATION_CONFIG,
    ) -> LlmMessage:
        messages = prompt.to_llm_messages()
        _log.debug("Prompt [messages='{}']", messages)

        try:
            gen = await self._generate(messages, config)
        except Exception as e:
            _log.exception("Exception for LLM request [exception='{}']", e)
            gen = self._get_error_response(e)

        return gen

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
        response = await self._call_to_api(
            messages=self._get_messages(messages), config=config
        )
        _log.trace("LLM response [response='{}']", response)

        return await self._get_llm_message(response)

    async def _generate_chunks(
        self,
        messages: Sequence[LlmMessage],
        config: LlmGenerationConfig,
        exclude_stream_prefixes: Sequence[str] = (),
    ) -> AsyncIterator[LlmMessageChunk | LlmMessage]:
        raise NotImplementedError()
        yield

    async def _call_to_api(
        self,
        messages: list[ResponseInputItemParam],
        config: LlmGenerationConfig,
    ) -> Response:
        response = await self._client.responses.create(
            model=self._model_deployment,
            input=messages,
            temperature=self._get_temperature(config.temperature),
            reasoning=self._get_reasoning(config.reasoning_effort),
            text=self._get_response_format(config),
            max_output_tokens=self._get_max_output_tokens(config),
        )
        return response

    async def _call_to_api_stream(
        self,
        messages: list[ResponseInputItemParam],
        config: LlmGenerationConfig,
    ) -> AsyncStream[ResponseStreamEvent]:
        events = await self._client.responses.create(
            model=self._model_deployment,
            input=messages,
            stream=True,
            temperature=self._get_temperature(config.temperature),
            reasoning=self._get_reasoning(config.reasoning_effort),
            text=self._get_response_format(config),
            max_output_tokens=self._get_max_output_tokens(config),
        )
        return events

    def _get_messages(
        self, messages: Sequence[LlmMessage]
    ) -> list[ResponseInputItemParam]:
        converted = [OpenAiLlmMessage.from_llm_message(msg) for msg in messages]

        return [cast(ResponseInputMessage, m.model_dump()) for m in converted]

    def _get_response_format(
        self, config: LlmGenerationConfig
    ) -> ResponseTextConfigParam:
        response_format = {"type": "text"}
        if config.json_schema is not None:
            schema = config.json_schema
            if not isinstance(schema, dict):
                schema = schema.model_json_schema()
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "Response", "schema": schema},
            }
        elif config.json_output:
            response_format = {"type": "json_object"}

        _log.debug("Response format [format='{}']", response_format)

        return ResponseTextConfigParam(
            format=cast(ResponseFormatTextConfigParam, response_format)
        )

    def _get_max_output_tokens(self, config: LlmGenerationConfig) -> int | None:
        return config.max_output_tokens or self._llm_spec.max_output_tokens

    def _get_max_prefix_len(self, prefixes: Sequence[str]) -> int:
        if not prefixes:
            return 0

        return max(len(prefix) for prefix in prefixes)

    def _get_error_response(self, e: Exception) -> LlmMessage:
        code = ErrorCode.LLM_API_ERROR
        detail = ""
        match e:
            case BadRequestError(message=message):
                if e.code == BAD_REQUEST_CONTEXT_LENGTH_CODE:
                    code = ErrorCode.LLM_CONTEXT_LENGTH_EXCEEDED
                elif e.code == BAD_REQUEST_CONTENT_FILTER_CODE:
                    code = self._get_filter_code(e)
                else:
                    code = ErrorCode.LLM_INVALID_REQUEST
                detail = message
            case RateLimitError(message=message):
                code = ErrorCode.LLM_RATE_LIMIT
                detail = message
            case APITimeoutError(message=message):
                code = ErrorCode.LLM_TIMEOUT
                detail = message
            case _:
                code = ErrorCode.LLM_API_ERROR
                detail = str(type(e))

        return LlmMessage(
            role=LlmMessageRole.ASSISTANT,
            content=None,
            usage=None,
            error=ErrorInfo(code=code, detail=detail),
        )

    def _get_filter_code(self, exception: BadRequestError) -> ErrorCode:
        code = ErrorCode.LLM_CONTENT_FILTER
        if isinstance(exception.body, dict):
            content_filter_result = exception.body["innererror"][
                "content_filter_result"
            ]
            is_filtered_hate = content_filter_result["hate"]["filtered"]
            is_filtered_self_harm = content_filter_result["self_harm"]["filtered"]
            is_filtered_sexual = content_filter_result["sexual"]["filtered"]
            is_filtered_violence = content_filter_result["violence"]["filtered"]

            if is_filtered_hate:
                code = ErrorCode.LLM_CONTENT_FILTER_HATE
            elif is_filtered_self_harm:
                code = ErrorCode.LLM_CONTENT_FILTER_SELF_HARM
            elif is_filtered_sexual:
                code = ErrorCode.LLM_CONTENT_FILTER_SEXUAL
            elif is_filtered_violence:
                code = ErrorCode.LLM_CONTENT_FILTER_VIOLENCE

        return code

    def _get_temperature(self, temperature_: float) -> float | Omit:
        temperature = omit
        if self._llm_spec.supports_temperature:
            temperature = min(
                max(temperature_, self._llm_spec.min_temperature),
                self._llm_spec.max_temperature,
            )

        return temperature

    def _get_reasoning(
        self, reasoning_effort: LlmReasoningEffort | None
    ) -> Reasoning | Omit:
        if reasoning_effort is None:
            reasoning_effort = self._llm_spec.default_reasoning_effort

        if (
            self._llm_spec.supports_reasoning
            and reasoning_effort in self._llm_spec.supports_reasoning_effort_levels
        ):
            return Reasoning(effort=reasoning_effort)

        return omit

    async def _get_llm_message(self, response: Response) -> LlmMessage:
        assert response.usage is not None
        return LlmMessage(
            content=response.output_text,
            role=LlmMessageRole.ASSISTANT,
            usage=[
                LlmUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=response.model,
                    llm_name=self._api_settings.base_model_name,
                ),
            ],
        )
