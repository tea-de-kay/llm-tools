from enum import StrEnum

from pydantic import BaseModel


class ErrorCode(StrEnum):
    NO_ERROR = "no_error"
    GENERAL_ERROR = "general_error"
    LLM_CONTENT_FILTER = "llm_content_filter"
    LLM_CONTENT_FILTER_HATE = "llm_content_filter_hate"
    LLM_CONTENT_FILTER_SELF_HARM = "llm_content_filter_self_harm"
    LLM_CONTENT_FILTER_SEXUAL = "llm_content_filter_sexual"
    LLM_CONTENT_FILTER_VIOLENCE = "llm_content_filter_violence"
    LLM_CONTEXT_LENGTH_EXCEEDED = "llm_context_length_exceeded"
    LLM_INVALID_REQUEST = "llm_invalid_request"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_TIMEOUT = "llm_timeout"
    LLM_API_ERROR = "llm_api_error"


class ErrorInfo(BaseModel):
    code: ErrorCode = ErrorCode.NO_ERROR
    detail: str = ""

    def __bool__(self) -> bool:
        return self.code is not ErrorCode.NO_ERROR
