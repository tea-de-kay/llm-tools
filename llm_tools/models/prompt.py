from pydantic import BaseModel

from llm_tools.models.llm import LlmMessage


class ConversationHistory(BaseModel):
    messages: list[LlmMessage] = []


class StructuredLlmOutput(BaseModel):
    """Base class for structured LLM outputs"""


class PromptTemplateVariables(BaseModel):
    """Base class for variables resolved in Jinja templates for LLM prompts"""
