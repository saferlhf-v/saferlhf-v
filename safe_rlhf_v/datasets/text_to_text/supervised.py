from typing import Any, Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch
import transformers
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from safe_rlhf_v.utils.multi_process import get_current_device
from safe_rlhf_v.utils.tools import right_padding
from datasets import load_dataset


IGNORE_INDEX = -100

__all__ = [
    'SupervisedDataset',
    'SupervisedCollator',
    'SupervisedSample',
    'SupervisedBatch',
]


class SupervisedSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)


class SupervisedBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)


class SupervisedDataset(Dataset):

    def __init__(
        self,
        path: str,
        template: str,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin | transforms.Compose | None = None,
        name: str | None = None,
        size: int | None = None,
        split: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__()
        assert path, f'You must set the valid datasets path! Here is {path}'
        assert template, f'You must set the valid template path! Here is {template}'
        self.tokenizer = tokenizer
        self.processor = processor
        self.raw_data = load_dataset(
            path,
            name=name,
            split=split,
            data_files=data_files,
            *optional_args,
            trust_remote_code=True,
        )
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))
        self.template = template

    def preprocess(self, raw_sample: dict[str, Any]) -> SupervisedSample:
        return_dict = {}
        formatted_prompt, formatted_text, _ = self.template.format_supervised_sample(raw_sample)
        return_dict['input_ids'] = self.tokenize(formatted_text)
        labels = return_dict['input_ids'].clone()
        # mask non-assistant input
        labels[: len(self.tokenize(formatted_prompt))] = IGNORE_INDEX
        return_dict['labels'] = labels

        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id)

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        if self.tokenizer.eos_token not in text:
            text += self.tokenizer.eos_token
        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors='pt',
        )['input_ids'][0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[index]
        data = self.preprocess(raw_sample.copy())
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)


class SupervisedCollator:

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[SupervisedSample]) -> SupervisedBatch:
        return_dict = {}
        current_device = get_current_device()

        return_dict['input_ids'] = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        ).to(current_device)

        return_dict['labels'] = right_padding(
            [sample['labels'] for sample in samples],
            padding_value=IGNORE_INDEX,
        ).to(current_device)

        return_dict['attention_mask'] = (
            return_dict['input_ids'].ne(self.pad_token_id).to(current_device)
        )

        return return_dict


class UnmatchedSupervisedDataset(SupervisedDataset):

    def preprocess(
        self, raw_sample_for_prompt: dict[str, Any], raw_sample_for_response: dict[str, Any]
    ) -> SupervisedSample:
        return_dict = {}
        formatted_text, _ = self.template.format_unmatched_supervised_sample(
            raw_sample_for_prompt, raw_sample_for_response
        )
        return_dict['input_ids'] = self.tokenize(formatted_text)

        return return_dict

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample_for_prompt = self.raw_data[index]
        if index == 0:
            raw_sample_for_response = self.raw_data[-1]
        else:
            raw_sample_for_response = self.raw_data[index - 1]
        data = self.preprocess(raw_sample_for_prompt, raw_sample_for_response)
        return data

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return UnmatchedSupervisedCollator(self.tokenizer.pad_token_id)


class UnmatchedSupervisedCollator:

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[SupervisedSample]) -> SupervisedBatch:
        return_dict = {}
        current_device = get_current_device()

        return_dict['input_ids'] = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        ).to(current_device)

        return_dict['labels'] = None

        return_dict['attention_mask'] = (
            return_dict['input_ids'].ne(self.pad_token_id).to(current_device)
        )

        return return_dict
