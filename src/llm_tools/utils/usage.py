from collections.abc import Sequence

from llm_tools.models.llm import LlmUsage
from llm_tools.models.types import Usage


def sum_llm_usages(usages: Sequence[Usage] | None) -> LlmUsage:
    usage = LlmUsage()
    for u in usages or []:
        if isinstance(u, LlmUsage):
            usage += u

    return usage
