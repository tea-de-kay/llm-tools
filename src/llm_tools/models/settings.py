from typing import Literal

from llama_cpp import Any
from pydantic import BaseModel


class LlmSettings(BaseModel):
    pass


class LlmApiSettings(BaseModel):
    provider: Literal["openai", "llama_cpp"] = "openai"
    """"A supported LLM provider."""

    version: str = ""
    """The version of the LLM API."""

    url: str = ""
    """The URL of the LLM API."""

    deployment: str = ""
    """The name of the LLM API deployment."""

    key: str = ""
    """The LLM API key."""

    base_model_name: str = ""
    """The base LLM model name."""

    hf_repo_id: str = ""
    """The hugging face repo ID."""

    hf_model_filename: str = ""
    """The hugging face model file name."""

    model_path: str = ""
    """The local file path of the model."""
