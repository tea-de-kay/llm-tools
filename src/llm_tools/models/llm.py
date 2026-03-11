from __future__ import annotations

from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel, Field

from llm_tools.models.errors import ErrorInfo
from llm_tools.models.settings import LlmApiSettings
from llm_tools.models.types import LlmMessageRole, LlmReasoningEffort, Usage


class LlmUsage(Usage):
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    llm_name: str | None = None

    def __add__(self, other: LlmUsage) -> LlmUsage:
        if (
            self.llm_name is not None and other.llm_name is not None
        ) and self.llm_name != other.llm_name:
            raise ValueError("Usages from different LLMs cannot be added.")

        return LlmUsage(
            input_tokens=((self.input_tokens or 0) + (other.input_tokens or 0)),
            output_tokens=((self.output_tokens or 0) + (other.output_tokens or 0)),
            model=self.model or other.model,
            llm_name=self.llm_name or other.llm_name,
        )


class LlmMedium(BaseModel):
    id: str
    content: bytes = Field(repr=False)
    mime_type: str
    detail_level: Literal["auto", "low", "hight"] = "auto"

    @property
    def content_b64(self) -> str:
        return b64encode(self.content).decode("utf-8")


class LlmMessage(BaseModel):
    role: LlmMessageRole
    content: str | None = None
    usage: list[Usage] | None = None
    media: list[LlmMedium] | None = None
    error: ErrorInfo = ErrorInfo()


class LlmMessageChunk(BaseModel):
    role: LlmMessageRole
    content: str | None = None


class LlmGenerationConfig(BaseModel):
    temperature: float = 0
    reasoning_effort: LlmReasoningEffort | None = None
    json_output: bool = False
    json_schema: dict | type[BaseModel] | None = None
    chunk_buffer_n: int = 10
    max_output_tokens: int | None = None


DEFAULT_LLM_GENERATION_CONFIG = LlmGenerationConfig()


class LlmSpec(BaseModel):
    base_model_name: str
    max_input_tokens: int
    max_output_tokens: int | None = None
    vision: bool = False
    streaming: bool = False
    system_message_role: Literal["system", "developer", "user"] = "system"
    supports_temperature: bool = True
    supports_reasoning: bool = False
    supports_reasoning_effort_levels: list[LlmReasoningEffort] = [
        "low",
        "medium",
        "high",
    ]
    default_reasoning_effort: LlmReasoningEffort = "medium"
    min_temperature: float = 0
    max_temperature: float = 2
    supports_tools: bool = False


class BasePrompt(BaseModel):
    def to_llm_messages(self) -> list[LlmMessage]:
        raise NotImplementedError()


class LLM(ABC):
    def __init__(self, api_settings: LlmApiSettings, spec: LlmSpec) -> None:
        self._api_settings = api_settings
        self._llm_spec = spec

    @abstractmethod
    async def generate(
        self,
        prompt: BasePrompt,
        config: LlmGenerationConfig = DEFAULT_LLM_GENERATION_CONFIG,
    ) -> LlmMessage:
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: BasePrompt,
        config: LlmGenerationConfig = DEFAULT_LLM_GENERATION_CONFIG,
    ) -> AsyncIterator[LlmMessageChunk | LlmMessage]:
        pass
