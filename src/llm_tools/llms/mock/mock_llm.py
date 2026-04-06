from collections.abc import AsyncIterator, Iterable
from itertools import cycle

from llm_tools.models.llm import (
    DEFAULT_LLM_GENERATION_CONFIG,
    LLM,
    BasePrompt,
    LlmGenerationConfig,
    LlmMessage,
    LlmMessageChunk,
    LlmMessageRole,
)


class MockLlm(LLM):
    def __init__(self, responses: Iterable[str]) -> None:
        self._responses = cycle(responses)

    async def generate(
        self,
        prompt: BasePrompt,
        config: LlmGenerationConfig = DEFAULT_LLM_GENERATION_CONFIG,
    ) -> LlmMessage:
        return LlmMessage(role=LlmMessageRole.ASSISTANT, content=next(self._responses))

    def generate_stream(
        self,
        prompt: BasePrompt,
        config: LlmGenerationConfig = DEFAULT_LLM_GENERATION_CONFIG,
    ) -> AsyncIterator[LlmMessageChunk | LlmMessage]:
        raise NotImplementedError()
