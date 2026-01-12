from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import AsyncIterator, Sequence
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


class LLM(ABC):
    def __init__(self, api_settings: LlmApiSettings, spec: LlmSpec) -> None:
        self._api_settings = api_settings
        self._llm_spec = spec

    @abstractmethod
    def generate(
        self,
        messages: Sequence[LlmMessage],
        config: LlmGenerationConfig = DEFAULT_LLM_GENERATION_CONFIG,
        stream: bool = False,
    ) -> AsyncIterator[LlmMessageChunk | LlmMessage]:
        pass
