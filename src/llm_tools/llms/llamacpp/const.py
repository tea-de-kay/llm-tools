from llama_cpp import Any


DEFAULT_LLM_SETTINGS: dict[str, Any] = {
    "n_gpu_layers": 0,
    "n_ctx": 4096,
}
