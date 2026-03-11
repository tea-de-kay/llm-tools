from jinja2 import Environment
from jinja2 import Template as JinjaTemplate
from pydantic import ConfigDict, field_validator

from llm_tools.models.llm import BasePrompt, LlmMedium, LlmMessage
from llm_tools.models.prompt import ConversationHistory, PromptTemplateVariables
from llm_tools.models.types import LlmMessageRole


JINJA_ENV = Environment(trim_blocks=True, autoescape=False)


def create_template(template: str, strip: bool = True) -> JinjaTemplate:
    if strip:
        template = template.strip()
    return JINJA_ENV.from_string(template)


class LlmPrompt(BasePrompt):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_prompt_template: str | JinjaTemplate | None
    user_prompt_template: str | JinjaTemplate | None
    prompt_inputs: PromptTemplateVariables
    history: ConversationHistory | None = None
    media: list[LlmMedium] | None = None

    @field_validator("system_prompt_template", "user_prompt_template")
    @classmethod
    def generate_template(
        cls,
        value: str | JinjaTemplate | None,
    ) -> JinjaTemplate | None:
        if isinstance(value, str):
            value = create_template(value, strip=True)

        return value

    def to_llm_messages(self) -> list[LlmMessage]:
        prompt_inputs = dict(self.prompt_inputs)
        messages: list[LlmMessage] = []
        if isinstance(self.system_prompt_template, JinjaTemplate):
            msg = LlmMessage(
                role=LlmMessageRole.SYSTEM,
                content=self.system_prompt_template.render(prompt_inputs),
            )
            messages.append(msg)

        if self.history is not None:
            messages.extend(self.history.messages)

        if isinstance(self.user_prompt_template, JinjaTemplate):
            user_msg = LlmMessage(
                role=LlmMessageRole.USER,
                content=self.user_prompt_template.render(prompt_inputs),
                media=self.media,
            )
            messages.append(user_msg)

        return messages
