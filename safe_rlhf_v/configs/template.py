from typing import Any, Callable

from transformers import AutoProcessor, AutoTokenizer

from safe_rlhf_v.configs.format_model import ModelFormatter
from safe_rlhf_v.utils import template_registry


class ChatTemplate:

    def __init__(
        self,
        formatter: AutoTokenizer | AutoProcessor,
        template: str | None = None,
        custom_formatter: Callable | None = None,
    ) -> None:
        self.dataset_formatter = None
        if template:
            self.dataset_formatter = template_registry.get_template_class(template)
        self.model_formatter = ModelFormatter(formatter, custom_formatter)

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> tuple[str, str, Any]:
        raw_conversation, multi_modal_info = self.dataset_formatter.format_supervised_sample(
            raw_sample
        )
        raw_prompt = raw_conversation[:-1]
        return (
            self.model_formatter(raw_prompt),
            self.model_formatter(raw_conversation),
            multi_modal_info,
        )

    def format_diffusion_supervised_sample(self, raw_sample: dict[str, Any]) -> tuple[str, Any]:
        raw_prompt, multi_modal_info = self.dataset_formatter.format_diffusion_supervised_sample(
            raw_sample
        )
        return raw_prompt, multi_modal_info

    def format_diffusion_preference_sample(self, raw_sample: dict[str, Any]) -> tuple[str, Any]:
        raw_prompt, multi_modal_info = self.dataset_formatter.format_diffusion_preference_sample(
            raw_sample
        )
        return raw_prompt, multi_modal_info

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> tuple[str, str, Any]:
        better_conversation, worse_conversation, multi_modal_info = (
            self.dataset_formatter.format_preference_sample(raw_sample)
        )
        return (
            self.model_formatter(better_conversation),
            self.model_formatter(worse_conversation),
            multi_modal_info,
        )

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> tuple[str, Any]:
        raw_prompt, multi_modal_info = self.dataset_formatter.format_prompt_only_sample(raw_sample)
        return self.model_formatter(raw_prompt, add_generation_prompt=True), multi_modal_info

    def format_unmatched_supervised_sample(
        self, raw_sample_for_prompt: dict[str, Any], raw_sample_for_response: dict[str, Any]
    ) -> tuple[str, Any]:
        raw_conversation, multi_modal_info = (
            self.dataset_formatter.format_unmatched_supervised_sample(
                raw_sample_for_prompt, raw_sample_for_response
            )
        )
        return self.model_formatter(raw_conversation), multi_modal_info

    def format_chat_sample(self, raw_conversation: list[dict[str, Any]]) -> tuple[str, Any]:
        return self.model_formatter(raw_conversation), {}

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        if hasattr(self.dataset_formatter, 'check_equal'):
            return self.dataset_formatter.check_equal(raw_sample)
        better_conversation, worse_conversation, _ = (
            self.dataset_formatter.format_preference_sample(raw_sample)
        )
        return better_conversation == worse_conversation

    def check_validation(self, raw_sample: dict[str, Any]) -> bool:
        if hasattr(self.dataset_formatter, 'check_validation'):
            return self.dataset_formatter.check_validation(raw_sample)
        return True
