import importlib
from collections import OrderedDict
from typing import Any

from transformers import AutoConfig
from transformers.models.auto.auto_factory import (
    _BaseAutoModelClass,
    _LazyAutoMapping,
    getattribute_from_module,
)
from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING_NAMES,
    model_type_to_module_name,
)


class _LazyAutoMappingInAlignAnything(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type: str, attr: str) -> Any:
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            try:
                self._modules[module_name] = importlib.import_module(
                    f'.{module_name}',
                    'safe_rlhf_v.models',
                )
            except ImportError:
                self._modules[module_name] = importlib.import_module(
                    f'.{module_name}', 'transformers.models'
                )
        return getattribute_from_module(self._modules[module_name], attr)


def get_model_class_for_trust_remote_code(model_type, model_mapping_names):
    model_class_name = model_mapping_names[model_type]
    try:
        model_class = getattr(
            importlib.import_module(f'.{model_type}', 'safe_rlhf_v.models'), model_class_name
        )
    except ImportError:
        model_class = getattr(
            importlib.import_module(f'.{model_type}', 'transformers.models'), model_class_name
        )
    return model_class


MODEL_FOR_SCORE_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        # Score model mapping
        ('llava', 'AccustomedLlavaRewardModel'),
        ('llava_next', 'AccustomedLlavaNextRewardModel'),
        ('qwen2_vl', 'AccustomedQwen2VLRewardModel'),
    ],
)

MODEL_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        # Score model mapping
        ('llava', 'AccustomedLlavaModel'),
        ('llava_next', 'AccustomedLlavaNextModel'),
        ('qwen2_vl', 'AccustomedQwen2VLModel'),
    ],
)

MODEL_FOR_SCORE_MAPPING: OrderedDict[str, Any] = _LazyAutoMappingInAlignAnything(
    CONFIG_MAPPING_NAMES,
    MODEL_FOR_SCORE_MAPPING_NAMES,
)

MODEL_MAPPING: OrderedDict[str, Any] = _LazyAutoMappingInAlignAnything(
    CONFIG_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

class AnyModelForScore(_BaseAutoModelClass):
    _model_mapping: OrderedDict[str, Any] = MODEL_FOR_SCORE_MAPPING


class AnyModel(_BaseAutoModelClass):
    _model_mapping: OrderedDict[str, Any] = MODEL_MAPPING

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        trust_remote_code=False,
        code_revision=None,
        commit_hash=None,
        **kwargs,
    ):
        config, kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            return_unused_kwargs=True,
            trust_remote_code=trust_remote_code,
            code_revision=code_revision,
            _commit_hash=commit_hash,
            **kwargs,
        )
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
