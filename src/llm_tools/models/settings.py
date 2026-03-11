from typing import Literal

from pydantic import BaseModel


class LlmApiSettings(BaseModel):
    provider: Literal["openai"] = "openai"
    """"A supported LLM provider."""

    version: str
    """The version of the LLM API."""

    url: str
    """The URL of the LLM API."""

    deployment: str
    """The name of the LLM API deployment."""

    key: str
    """The LLM API key."""

    base_model_name: str
    """The base LLM model name."""
