from dataclasses import dataclass

import torch
from transformers.utils.generic import ModelOutput


@dataclass
class ScoreModelOutput(ModelOutput):
    """Output of the score model."""

    scores: torch.FloatTensor | None = None  # size = (B, L, D)
    clipped_scores: torch.FloatTensor | None = None  # size = (B, L-I, D)
    end_scores: torch.FloatTensor | None = None  # size = (B, D)
    last_hidden_state: torch.FloatTensor | None = None  # size = (B, L, E)
    clipped_states: torch.FloatTensor | None = None  # size = (B, L-I, D)
    end_last_hidden_state: torch.FloatTensor | None = None  # size = (B, E)
    end_index: torch.LongTensor | None = None  # size = (B,)
