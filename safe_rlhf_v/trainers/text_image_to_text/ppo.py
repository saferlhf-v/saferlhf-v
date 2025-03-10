import argparse
import copy
import os
import sys
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from transformers import GenerationConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf_v.datasets.text_image_to_text import (
    PromptOnlyBatch,
    PromptOnlyDataset,
    SupervisedDataset,
)
from safe_rlhf_v.models.pretrained_model import load_pretrained_models
from safe_rlhf_v.trainers.text_to_text.ppo import PPOTrainer as PPOTextTrainer
from safe_rlhf_v.utils.multi_process import (
    get_all_reduce_max,
    get_all_reduce_mean,
    get_current_device,
)
from safe_rlhf_v.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    gather_log_probabilities,
    is_same_tokenizer,
    masked_mean,
    read_cfgs,
    remove_pad_tokens,
    seed_everything,
    update_dict,
    move_padding_left,
)

class PPOTrainer(PPOTextTrainer):
    """Trainer base class for PPO training."""

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        # load training datasets
        self.prompt_only_dataloader, self.eval_dataloader, self.ptx_dataloader = (
            self.get_dataloaders(PromptOnlyDataset, PromptOnlyDataset, SupervisedDataset)
        )

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_cfgs)
        if self.ds_eval_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf_eval = HfDeepSpeedConfig(self.ds_eval_cfgs)
        # loading actor model
        self.actor_model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.actor_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            freeze_mm_proj=self.cfgs.train_cfgs.freeze_mm_proj,
            freeze_vision_tower=self.cfgs.train_cfgs.freeze_vision_tower,
            freeze_language_model=self.cfgs.train_cfgs.freeze_language_model,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        self.tokenizer.model_max_length = self.cfgs.model_cfgs.model_max_length
        # loading actor reference model
        self.actor_reference_model, _, _ = load_pretrained_models(
            self.cfgs.model_cfgs.actor_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        # loading reward model
        self.reward_model, self.reward_tokenizer, _ = load_pretrained_models(
            self.cfgs.model_cfgs.reward_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            is_reward_model=True,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        # loading reward critic model
        self.reward_critic_model, self.reward_critic_tokenizer, _ = load_pretrained_models(
            self.cfgs.model_cfgs.reward_critic_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            is_reward_model=True,
            freeze_mm_proj=self.cfgs.train_cfgs.freeze_mm_proj,
            freeze_vision_tower=self.cfgs.train_cfgs.freeze_vision_tower,
            freeze_language_model=self.cfgs.train_cfgs.freeze_language_model,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        # initial checking
        if is_same_tokenizer(self.tokenizer, self.reward_tokenizer):
            self.reward_tokenizer = self.tokenizer
        if not is_same_tokenizer(self.tokenizer, self.reward_critic_tokenizer):
            raise ValueError(
                (
                    'Reward critic tokenizer must be the same as actor tokenizer. '
                    'Expected {0.__module__}.{0.__qualname__}(vocab_size={1}), '
                    'but got {2.__module__}.{2.__qualname__}(vocab_size={3}). '
                    'Please consider pass `--reward_critic_model_name_or_path` from the command line.'
                ).format(
                    type(self.tokenizer),
                    len(self.tokenizer),
                    type(self.reward_critic_tokenizer),
                    len(self.reward_critic_tokenizer),
                ),
            )

        # training setup
        self.generation_config = GenerationConfig(
            max_new_tokens=self.cfgs.model_cfgs.max_new_tokens,
            temperature=self.cfgs.model_cfgs.temperature,
            top_p=self.cfgs.model_cfgs.top_p,
            repetition_penalty=self.cfgs.model_cfgs.repetition_penalty,
            do_sample=True,
        )

    def actor_step(
        self, mini_prompt_only_batch: PromptOnlyBatch
    ) -> list[dict[str, Any], list[int]]:
        infer_batch = self.infer_batch(mini_prompt_only_batch)
        actor_batch = copy.deepcopy(infer_batch)
        sequences = self.actor_model.module.generate(
            **infer_batch,
            generation_config=self.generation_config,
            synced_gpus=True,
            do_sample=True,
        )
        sequences = move_padding_left(sequences.contiguous(), self.tokenizer.pad_token_id)
        attention_mask = sequences.not_equal(self.tokenizer.pad_token_id)
        actor_batch['input_ids'] = sequences
        actor_batch['attention_mask'] = attention_mask

        response_lens = []
        batch_size = sequences.size(0)
        for idx in range(batch_size):
            prompt_length = len(
                remove_pad_tokens(
                    mini_prompt_only_batch['input_ids'][idx].squeeze().tolist(),
                    self.tokenizer.pad_token_id,
                )
            )
            sequence_wo_pad = remove_pad_tokens(
                sequences[idx].squeeze().tolist(), self.tokenizer.pad_token_id
            )
            response = sequence_wo_pad[prompt_length:]
            response_lens.append(len(response))
        return actor_batch, response_lens

    @torch.no_grad()
    def rollout(self, prompt_only_batch: PromptOnlyBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences."""
        # freeze the model for rolling out
        self.set_train(mode=False)

        micro_inference_batches = []
        micro_training_batches = []
        mini_batch = prompt_only_batch.copy()
        # actor generation
        actor_batch, response_lens = self.actor_step(mini_batch)
        # reward model and reward critic model scoring
        reward_batch = self.reward_model_step(actor_batch)
        # calculate the log probabilities
        logits = self.actor_model(**actor_batch).logits
        ref_logits = self.actor_reference_model(**actor_batch).logits

        logprob_list = []
        ref_logprob_list = []
        reward_value_list = []

        batch_size = logits.size(0)

        for idx in range(batch_size):
            response_length = response_lens[idx]
            input_id = actor_batch['input_ids'][idx, 1:][-response_length:].unsqueeze(0)

            logit = logits[idx, :-1][-response_length:].unsqueeze(0)
            ref_logit = ref_logits[idx, :-1][-response_length:].unsqueeze(0)
            reward_value = reward_batch['reward_values'][idx][-response_length:].unsqueeze(0)

            logprob = gather_log_probabilities(logit, input_id).squeeze()
            if logprob.shape == torch.Size([]):
                logprob = torch.full((3,), 1e-9)
            logprob_list.append(logprob)
            
            ref_logprob = gather_log_probabilities(ref_logit, input_id).squeeze()
            if ref_logprob.shape == torch.Size([]):
                ref_logprob = torch.full((3,), 1e-9)
            ref_logprob_list.append(ref_logprob)
            
            reward_value = reward_value.squeeze()
            if reward_value.shape == torch.Size([]):
                reward_value = torch.full((3,), 1e-9)
            reward_value_list.append(reward_value)

        log_probs = torch.nn.utils.rnn.pad_sequence(
            logprob_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        ref_log_probs = torch.nn.utils.rnn.pad_sequence(
            ref_logprob_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        reward_values = torch.nn.utils.rnn.pad_sequence(
            reward_value_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        response_mask = (log_probs != 0).bool().to(logits.device)

        micro_training_batch = {}
        micro_training_batch['response_lens'] = response_lens
        micro_training_batch['log_probs'] = log_probs
        micro_training_batch['ref_log_probs'] = ref_log_probs
        micro_training_batch['reward'] = reward_batch['reward']
        micro_training_batch['reward_values'] = reward_values
        micro_training_batch['response_mask'] = response_mask

        mini_batch['input_ids'] = reward_batch['input_ids']
        mini_batch['attention_mask'] = actor_batch['attention_mask']
        # add rollout results to the batches
        micro_inference_batches.append(mini_batch)
        micro_training_batches.append(micro_training_batch)

        # unfreeze the model for training
        self.set_train()

        return micro_inference_batches, micro_training_batches

    def rl_step(
        self, inference_batch: dict[str, torch.Tensor], training_batch: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Perform a single update step with RL loss."""
        response_lens = training_batch['response_lens']
        old_log_probs = training_batch['log_probs']
        ref_log_probs = training_batch['ref_log_probs']
        reward = training_batch['reward']
        old_reward_values = training_batch['reward_values']

        input_ids = inference_batch['input_ids']
        batch_size = input_ids.size(0)
        sequence_mask = training_batch['response_mask']

        with torch.no_grad():
            old_rewards = self.add_kl_divergence_regularization(
                reward,
                old_log_probs,
                ref_log_probs,
                sequence_mask,
            )
            reward_advantages, reward_returns = self.get_advantages_and_returns(
                old_reward_values,
                old_rewards,
                sequence_mask,
                start=0,
            )
        logits = self.actor_model(**self.infer_batch(inference_batch), use_cache=False).logits
        logprob_list = []

        for idx in range(batch_size):
            response_length = response_lens[idx]
            input_id = input_ids[idx, 1:][-response_length:].unsqueeze(0)
            logit = logits[idx, :-1][-response_length:].unsqueeze(0)
            
            logprob = gather_log_probabilities(logit, input_id).squeeze()
            if logprob.shape == torch.Size([]):
                logprob = torch.full((3,), 1e-9)
            logprob_list.append(logprob)

        log_probs = torch.nn.utils.rnn.pad_sequence(
            logprob_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        actor_loss = self.actor_loss_fn(
            log_probs,
            old_log_probs,
            reward_advantages,
            sequence_mask,
        )
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        raw_reward_values = self.reward_critic_model(**self.infer_batch(inference_batch)).scores
        raw_reward_values = raw_reward_values.squeeze(dim=-1)[:, :-1]

        reward_value_list = []

        for idx in range(batch_size):
            response_length = response_lens[idx]
            reward_value = raw_reward_values[idx][-response_length:].unsqueeze(0)
            reward_value = reward_value.squeeze()
            if reward_value.shape == torch.Size([]):
                reward_value = torch.full((3,), 1e-9)
            reward_value_list.append(reward_value)
        reward_values = torch.nn.utils.rnn.pad_sequence(
            reward_value_list, batch_first=True, padding_value=0.0
        ).to(logits.device)

        reward_critic_loss = self.critic_loss_fn(
            reward_values,
            old_reward_values,
            reward_returns,
            sequence_mask,
        )
        self.reward_critic_model.backward(reward_critic_loss)
        self.reward_critic_model.step()

        with torch.no_grad():
            mask = sequence_mask
            kl_divergence = ((old_log_probs - ref_log_probs) * mask).sum(dim=-1).mean()
            mean_generated_length = mask.sum(dim=-1).float().mean()
            max_generated_length = mask.sum(dim=-1).float().max()

            reward = reward.mean()
            reward_with_kl_penalty = (old_rewards * mask).sum(dim=-1).mean()
            reward_advantage = masked_mean(reward_advantages, mask)
            reward_return = masked_mean(reward_returns, mask)
            reward_value = masked_mean(reward_values, mask)

            actor_loss = get_all_reduce_mean(actor_loss)
            reward_critic_loss = get_all_reduce_mean(reward_critic_loss)
            reward = get_all_reduce_mean(reward)
            reward_with_kl_penalty = get_all_reduce_mean(reward_with_kl_penalty)
            reward_advantage = get_all_reduce_mean(reward_advantage)
            reward_return = get_all_reduce_mean(reward_return)
            reward_value = get_all_reduce_mean(reward_value)
            kl_divergence = get_all_reduce_mean(kl_divergence)
            mean_generated_length = get_all_reduce_mean(mean_generated_length)
            max_generated_length = get_all_reduce_max(max_generated_length)

        dist.barrier()

        return {
            'train/actor_loss': actor_loss.item(),
            'train/reward_critic_loss': reward_critic_loss.item(),
            'train/reward': reward.item(),
            'train/reward_with_kl_penalty': reward_with_kl_penalty.item(),
            'train/reward_advantage': reward_advantage.item(),
            'train/reward_return': reward_return.item(),
            'train/reward_value': reward_value.item(),
            'train/kl_divergence': kl_divergence.item(),
            'train/actor_lr': self.actor_model.optimizer.param_groups[0]['lr'],
            'train/reward_critic_lr': self.reward_critic_model.optimizer.param_groups[0]['lr'],
            'train/mean_generated_length': mean_generated_length.item(),
            'train/max_generated_length': max_generated_length.item(),
        }

    def add_kl_divergence_regularization(
        self,
        reward: torch.Tensor,  # size = (B,)
        log_probs: torch.Tensor,  # size = (B, L)
        ref_log_probs: torch.Tensor,  # size = (B, L)
        sequence_mask: torch.BoolTensor,  # size = (B, L)
    ) -> torch.Tensor:  # size = (B, L)
        """Add KL divergence regularization on scalar rewards."""
        end_index = torch.cat([m.nonzero()[-1] for m in sequence_mask])  # size = (B,)

        # size = (B, L)
        kl_divergence_estimate = log_probs - ref_log_probs
        kl_penalty_rewards = -self.kl_coeff * kl_divergence_estimate
        rewards = torch.scatter_add(
            kl_penalty_rewards,
            dim=-1,
            index=end_index.unsqueeze(dim=-1),
            src=reward.to(kl_penalty_rewards.dtype).unsqueeze(dim=-1),
        )
        return torch.clamp(rewards, min=-self.clip_range_score, max=self.clip_range_score)

    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        sequence_mask: torch.BoolTensor,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using Generalized Advantage Estimation (GAE)."""
        # Modified from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        last_gae_lambda = 0.0
        advantages_reversed = []
        values = values * sequence_mask
        rewards = rewards * sequence_mask
        length = rewards.size(-1)
        for t in reversed(range(start, length)):  # pylint: disable=invalid-name
            next_values = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
            advantages_reversed.append(last_gae_lambda)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_image_to_text', 'ppo')
    dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)

    # get custom configs from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    # finetune the model
    trainer = PPOTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
