from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel


class Usage(BaseModel):
    """Base class for usages"""


class LlmMessageRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    ASSISTANT = "assistant"
    USER = "user"


type LlmReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]
