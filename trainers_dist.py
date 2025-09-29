import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy,size_based_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator 
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple


from scheduler import  train_NN_batch,batch_EE_forward,EE_forward
from bandit_utils import (bandit_data_preprocess, 
                          process_bandit_train_data, 
                          sampling_based_on_scores, 
                          add_bandit_metrics,
                          get_observed_rewards,
                          detach_unselected_samples,
                          bandit_scoring)
from transformers import AutoConfig




def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing   
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def preference_loss_with_more_info(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0, 
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # 整个logit

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
    
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing 
        sig_losses = -F.sigmoid(beta * logits) * (1 - label_smoothing) - F.sigmoid(-beta * logits) * label_smoothing 
        
         
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards,sig_losses



def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)  模型在每个位置预测结果的概率分布
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length) 每个位置是句子的词的idx
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens. 

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor): 
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated') 
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value) 
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0) 
    return concatenated_batch 

class BasicTrainer(object):
    def __init__(   self, 
                    policy: nn.Module, 
                    config: DictConfig, 
                    seed: int, 
                    run_dir: str, 
                    reference_model: Optional[nn.Module] = None, 
                    rank: int = 0, 
                    world_size: int = 1,
                    net1:Optional[nn.Module]=None,
                    net2:Optional[nn.Module]=None
                    ):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model
        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

        
        self.last_hiddenstate_only=config.bandit.last_hiddenstate_only
        if self.config.bandit.layer_encoder:
            self.last_hiddenstate_only=False

        if rank ==0 and self.config.bandit.enable_bandit:
            self.net1=net1.to(f"cuda:{rank}")
            self.net2=net2.to(f"cuda:{rank}")
           
            self.f1_only=config.bandit.f1_only
            # maintain the avg batch level reward
            self.avg_reward=0
            self.reward_counter=0
            self.forward_only=config.bandit.forward_only

            self.use_pool=self.config.bandit.use_pool
            # bandit batch pool
            if self.use_pool:
                self.pool_size=self.config.bandit.pool_size
                minibatch_size=config.batch_size//config.gradient_accumulation_steps
                self.x1_pool=torch.zeros(self.pool_size,
                                         minibatch_size,
                                         32,
                                         self.net1.input_size
                                         ).cpu()
                self.x2_pool=torch.zeros(self.pool_size,
                                         minibatch_size,
                                         self.net2.input_size
                                         ).cpu()
                self.y1_pool=torch.zeros(self.pool_size,minibatch_size).cpu()
                self.y2_pool=torch.zeros(self.pool_size,minibatch_size).cpu()
                print("pool init success!")
                self.batch_idx=0 
                self.pool_use_num=self.config.bandit.pool_use_num

            if self.config.bandit.init_weight_constraint:
                self.net1_ref = self.net1.state_dict()
                self.net2_ref = self.net2.state_dict()
            else:
                self.net1_ref = None
                self.net2_ref = None
       

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch) 

        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32) #2*batch_size,seq_len,vocab_size
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False) # 2*batch_size
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]] 
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps
    
    def concatenated_forward_with_hidden_state(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch) 
        output = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask'],output_hidden_states=True) #2*batch_size,seq_len,vocab_size
        all_logits=output.logits.to(torch.float32) 

        if self.last_hiddenstate_only:  
            last_hidden_state = output.hidden_states[-1].to(torch.float32) # 32,467,2560
        else:
            # last_hidden_state = torch.cat(output.hidden_states[1:], dim=0).to(torch.float32) # 32,467,2560*32
            last_hidden_state=torch.stack(list(output.hidden_states[1:]),dim=0).to(torch.float32) # l,2b,s,d
            last_hidden_state=last_hidden_state.permute(1,0,2,3) # 2b,l,s,d
        
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False) # 2*batch_size
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]  
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]

        last_hidden_state_chosen=last_hidden_state[:batch['chosen_input_ids'].shape[0]]  # b,l,s,d
        last_hidden_state_reject=last_hidden_state[batch['chosen_input_ids'].shape[0]:] # b,l,s,d
        

        return chosen_logps, rejected_logps, last_hidden_state_chosen,last_hidden_state_reject

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True,selected_ratio=0.75,theta=0.1,pre_bandit_train_data=None):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        start=time.time()

        metrics = {}
        my_metrics={}
        train_test = 'train' if train else 'eval'

        if loss_config.name in {'dpo', 'ipo'}:
 
            t=time.time()
            policy_chosen_logps, policy_rejected_logps, h_chosen,h_reject = self.concatenated_forward_with_hidden_state(self.policy, batch)

            my_metrics['policy_forward_time']=time.time()-t

            with torch.no_grad():
                t=time.time()
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)
                my_metrics['ref_forward_time']=time.time()-t
            
            if loss_config.name == 'dpo':
                loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
            elif loss_config.name == 'ipo':
                loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
            else:
                raise ValueError(f'unknown loss {loss_config.name}')       

            if not self.config.bandit.use_sig_loss:
                losses, chosen_rewards, rejected_rewards = preference_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
            else:
                losses, chosen_rewards, rejected_rewards,sig_losses = preference_loss_with_more_info(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
            
            reward_accuracies = (chosen_rewards > rejected_rewards).float() 

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size) 
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size) 
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size) 
            
            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

            end=time.time()-start
            rank0_print(f"dpo forward time:{end} s")

            #### Get data that bandit needs ####
            # if use encoder, attention mask is not used  (both sft and dpo)
            if train: # use encoder use sigloss
                h_chosen=h_chosen.detach().clone()
                h_reject=h_reject.detach().clone()
                
                if not self.config.bandit.concat_wl: # mean wl
                    h_chosen_cls=(h_chosen+h_reject)/2 
                else:# concat wl
                    h_chosen_cls = torch.cat((h_chosen, h_reject), dim=-1)
                
                if not self.config.bandit.use_encoder: # mean seq
                    h_chosen_cls=torch.mean(h_chosen_cls,dim=1)

                # when dpo, not use attention mask
                # _attention_map=torch.cat((batch['chosen_attention_mask'], batch['rejected_attention_mask']), dim=1) # batch,seqlen*2
                _attention_map=torch.ones_like(batch['chosen_attention_mask'])  
                # _attention_map=torch.ones(h_chosen_cls.shape[0],500,dtype=torch.long)

                if self.config.bandit.layer_encoder: #construct h alone

                    # b,l,s,d-> l,b,s,d
                    h_chosen=h_chosen.permute(1,0,2,3)
                    h_reject=h_reject.permute(1,0,2,3)
                    h_chosen_cls=(h_chosen+h_reject)/2
                    # l,b,s,d-> l,b,d
                    h_chosen_cls=torch.mean(h_chosen_cls,dim=2) # l,b,d
                    h_chosen_cls=h_chosen_cls.permute(1,0,2).contiguous() # b,l,d as input of seq
                    _attention_map=torch.ones(h_chosen_cls.shape[0],h_chosen_cls.shape[1],dtype=torch.long).to(h_chosen_cls.device) # create a tensor should be moved to device immediately


        elif loss_config.name == 'sft': 

            output = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask'],output_hidden_states=True)
            policy_chosen_logits=output.logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)
            losses = -policy_chosen_logps

            if self.last_hiddenstate_only: 
                last_hidden_state = output.hidden_states[-1].to(torch.float32) # 32,467,2560
            else:
                last_hidden_state = torch.cat(output.hidden_states[1:], dim=-1).to(torch.float32) # 32,467,2560*32

            if not self.config.bandit.use_encoder:
                h_chosen_cls=torch.mean(last_hidden_state,dim=1).detach().clone()
            else:
                h_chosen_cls=last_hidden_state.detach().clone()

            _attention_map=batch['chosen_attention_mask'] 

            if self.config.bandit.layer_encoder:
                h_chosen_cls = torch.stack(list(output.hidden_states[1:]), dim=0).to(torch.float32) # l,b,s,d
                # l,b,s,d-> l,b,d
                h_chosen_cls=torch.mean(h_chosen_cls,dim=2) # l,b,d
                h_chosen_cls=h_chosen_cls.permute(1,0,2).contiguous() # b,l,d as input of seq
                _attention_map=torch.ones(h_chosen_cls.shape[0],h_chosen_cls.shape[1],dtype=torch.long).to(h_chosen_cls.device) # create a tensor should be moved to device immediately

            end=time.time()-start
            rank0_print(f"dpo forward time:{end} s")
            
        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        
        if train:
            if loss_config.name in {'dpo', 'ipo'}:
                return losses.mean(), metrics,my_metrics,(chosen_rewards-rejected_rewards,policy_chosen_logps,chosen_rewards,h_chosen_cls,_attention_map,losses,sig_losses)  # h,rm,p,r
            elif loss_config.name in {'sft'}:
                return losses.mean(), metrics,my_metrics,(policy_chosen_logps,h_chosen_cls,_attention_map,losses) 
        else:
            return losses.mean(), metrics,my_metrics
        


    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        ref_forward_times=[]
        policy_forward_times=[]
        policy_backward_times=[]

        pre_bandit_train_data=None
        for batch in self.train_iterator: # ['prompt', 'chosen', 'rejected', 'chosen_response_only', 'rejected_response_only', 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 'prompt_input_ids', 'prompt_attention_mask']
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in {'dpo', 'ipo'}:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics, my_metrics= self.get_batch_metrics(local_eval_batch, 
                                                                            self.config.loss, 
                                                                            selected_ratio=self.config.bandit.selected_ratio,
                                                                            theta=self.config.bandit.theta,
                                                                            train=False)

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name in {'dpo', 'ipo'}:
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                if self.config.sample_during_eval:                    
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in {'dpo', 'ipo'}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name in {'dpo', 'ipo'}:
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

            #### SAVE MODEL ####
            if self.example_counter>0 and self.example_counter % self.config.save_every == 0:
               
                output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                rank0_print(f'creating checkpoint to write to {output_dir}...')
                self.save(output_dir, mean_eval_metrics)
                
            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            batch_my_metrics = defaultdict(list)
            mini_pre_bandit_train_data=pre_bandit_train_data
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)  
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank) 
               

                loss,metrics,my_metrics,local_bandit_inputs = self.get_batch_metrics(local_microbatch, 
                                                                                       self.config.loss, 
                                                                                       train=True,
                                                                                       selected_ratio=self.config.bandit.selected_ratio,
                                                                                       theta=self.config.bandit.theta,
                                                                                       pre_bandit_train_data=None) 
                

                if self.config.loss.name in {'dpo', 'ipo'}:
                    (_rm,_logp,_r,local_losses,
                     h_chosen_cls,_attention_mask,
                     global_losses,global_sig_losses)=bandit_data_preprocess(local_bandit_inputs,self.rank,self.world_size,self.config.loss.name)
                elif self.config.loss.name in {'sft'}:
                    (_logp,
                     h_chosen_cls,_attention_mask,
                     local_losses,global_losses) = bandit_data_preprocess(local_bandit_inputs,self.rank,self.world_size,self.config.loss.name)

                train_start=time.time()
                if self.rank ==0  and mini_pre_bandit_train_data is not None :

                    f1_loss_list=[]
                    f2_loss_list=[]

                    cur_metrics= global_losses.mean().item() 
                    if self.config.bandit.use_sig_loss and self.config.loss.name in {'dpo', 'ipo'}:
                        cur_metrics= global_sig_losses.mean().item()

                    # if sft then r_rm r_rewards r_logps is None 
                    X1_train,train_attention_mask,X2_train,pre_metrics,f1_output,f2_output,final_output,r_rm,r_rewards,r_logps=mini_pre_bandit_train_data
                    # construct f1 labels
                    y1=[] 


                    if self.config.loss.name in {'dpo', 'ipo'}:
                        y1,avg_reward,reward_counter=get_observed_rewards(cur_metrics,pre_metrics,
                                                r_rm=r_rm,r_rewards=r_rewards,r_logps=r_logps,
                                                loss_name=self.config.loss.name,
                                                construct_reward=self.config.bandit.construct_bandit_reward,
                                                reward_type=self.config.bandit.f1_reward_type,
                                                norm_metrics=self.config.bandit.norm_metrics,
                                                avg_reward=self.avg_reward,
                                                counter=self.reward_counter,
                                                avg_weight=0.5,
                                                enable_avg=self.config.bandit.enable_avg_reward
                                                )
                    elif self.config.loss.name in {'sft'}: 
                        y1,avg_reward,reward_counter=get_observed_rewards(cur_metrics,pre_metrics,
                                                r_rm=r_rm,r_logps=r_logps,
                                                loss_name=self.config.loss.name,
                                                construct_reward=self.config.bandit.construct_bandit_reward,
                                                reward_type=self.config.bandit.f1_reward_type,
                                                norm_metrics=self.config.bandit.norm_metrics,
                                                avg_reward=self.avg_reward,
                                                counter=self.reward_counter,
                                                avg_weight=0.5,
                                                enable_avg=self.config.bandit.enable_avg_reward
                                                )
                    # update avg_reward
                    
                    self.reward_counter=reward_counter
                    self.avg_reward=avg_reward                    

                    # construct f2 labels     
                    y2=[]
                    for f1,label in zip(f1_output,y1):
                        y2.append(label-f1)


                    # todo: directly to tensor,not list at all
                    X1_train=torch.stack(X1_train).unsqueeze(0).float().detach()
                    X2_train=torch.stack(X2_train).unsqueeze(0).float().detach()
                    y1 = torch.stack(y1).unsqueeze(0).float().detach() #[1,batch]
                    y2=torch.stack(y2).unsqueeze(0).float().detach() # [1,batch]

                    if self.use_pool: # save X1、X2 is ok
                        start=time.time()
                        num_batch=self.batch_idx if self.batch_idx<self.pool_use_num else self.pool_use_num
                        select_batch_idx =random.sample(range(0,self.batch_idx), num_batch)
                        # load data to gpu
                        if self.batch_idx==0:
                            X1_used=X1_train
                            X2_used=X2_train
                            y1_used=y1
                            y2_used=y2
                        else:
                            # num_batch,batch_size,...
                            offline_X1=self.x1_pool[select_batch_idx].to(self.device)
                            offline_X2=self.x2_pool[select_batch_idx].to(self.device)
                            offline_y1=self.y1_pool[select_batch_idx].to(self.device)
                            offline_y2=self.y2_pool[select_batch_idx].to(self.device)
                            X1_used=torch.cat((X1_train,offline_X1),dim=0)
                            X2_used=torch.cat((X2_train,offline_X2),dim=0)
                            y1_used=torch.cat((y1,offline_y1),dim=0)
                            y2_used=torch.cat((y2,offline_y2),dim=0) # num_used,batch

                        # stack data 
                        X1_used=X1_used.reshape(-1,X1_used.shape[-2],X1_used.shape[-1]) # num_samples,seq,dim
                        X2_used=X2_used.reshape(-1,X2_used.shape[-1]) #num_samples, dim
                        y1_used=y1_used.reshape(-1) # num_used*batch
                        y2_used=y2_used.reshape(-1) # num_used *batch

                        print(f"prepare bandit training data  {time.time()-start} ")

                                           
                    if not self.forward_only:  # enable bandit train
                        print("enable bandit training!")
                        start=time.time()
                        f1_loss=train_NN_batch(
                            self.net1, 
                            X1_used, 
                            y1_used, 
                            lr=self.config.bandit.lr,
                            num_epochs=self.config.bandit.f1_num_epochs,
                            num_batch=self.config.bandit.f1_num_batch,
                            threshold=self.config.bandit.f1_loss_threshold,
                            is_f1=True,
                            weight_decay=self.config.bandit.weight_decay,
                            use_combine_loss=self.config.bandit.use_combine_loss,
                            device=self.device,
                            ref_weight=self.net1_ref,
                            use_encoder= self.config.bandit.use_encoder,
                            attention_mask=None
                        )
                        print(f"f1 update time:{time.time()-start}")

                        if not self.f1_only:
                            start=time.time()
                            f2_loss=train_NN_batch(
                                self.net2, 
                                X2_used, 
                                y2_used, 
                                lr=self.config.bandit.lr,
                                num_epochs=self.config.bandit.f2_num_epochs,
                                num_batch=self.config.bandit.f2_num_batch,
                                threshold=self.config.bandit.f1_loss_threshold,
                                is_f1=False,
                                weight_decay=self.config.bandit.weight_decay,
                                use_combine_loss=self.config.bandit.use_combine_loss,
                                device=self.device,
                                ref_weight=self.net2_ref
                                )
                            print(f"f2 update time:{time.time()-start}")
                        else:
                            f2_loss=torch.tensor(0.0)
                        
                        f1_loss_list.append(f1_loss)
                        f2_loss_list.append(f2_loss)

                    else: # fake loss
                        print("skipping bandit training!")
                        f1_loss_list.append(torch.tensor(0.0))
                        f2_loss_list.append(torch.tensor(0.0))


                    if self.use_pool:
                        start=time.time()
                        # update pool
                        X1_train=X1_train.detach().cpu()
                        X2_train=X2_train.detach().cpu()
                        y1=y1.detach().cpu()
                        y2=y2.detach().cpu()
                        # not full,add
                        if self.batch_idx<self.pool_size:
                            r =self.batch_idx
                            self.batch_idx+=1 # if not full,add
                        # pool full, then random replace
                        else:
                            r = random.randint(0, self.batch_idx-1)

                        self.x1_pool[r]=X1_train.squeeze()
                        self.x2_pool[r]=X2_train.squeeze()
                        self.y1_pool[r]=y1.squeeze()
                        self.y2_pool[r]=y2.squeeze()

                       

                        print(f"update pool data  {time.time()-start} ")


                    # report
                    add_bandit_metrics(batch_metrics,
                                           f1_loss_list,f2_loss_list,
                                           list(y1.squeeze()),list(y2.squeeze()),
                                           f1_output,f2_output,final_output)
                
                rank0_print(f"bandit update time:{time.time()-train_start} s")
                
                dist.barrier()

                # bandit forward and pass bandit training data
                forward_start=time.time()
                if self.rank ==0 and self.config.bandit.enable_bandit:
                    net1 = self.net1
                    net2 = self.net2   

                    if not self.config.bandit.use_encoder:
                        _attention_mask=None
                    # batch bandit forward 
                    _f1,_f2,_dc=batch_EE_forward(net1, net2, h_chosen_cls,_attention_mask)  # batch,layer,h  batch,layer
                    scores,indices,f1_output,f2_output=bandit_scoring(_f1,_f2,h_chosen_cls,f1_only=self.f1_only,f2_weight=self.config.bandit.f2_weight) 

                    X2_train=[]
                    for i in range(_dc.shape[0]):
                        X2_train.append(torch.reshape(_dc[i],(1,len(_dc[i]))))

                    # caculate the distribution of samples
                    selected_sample_ids=sampling_based_on_scores(scores,indices,selected_ratio=self.config.bandit.selected_ratio)

                    if self.config.loss.name in {'dpo', 'ipo'}:
                        (X1_train,x1_attention_mask,X2_train,
                        f1_output,f2_output,final_output,
                        r_rewards,r_logps,reward_margins)=process_bandit_train_data(selected_sample_ids,
                                                                                    h_chosen_cls,_attention_mask,X2_train,
                                                                                    f1_output,f2_output,scores,
                                                                                    _r=_r,_logp=_logp,_rm=_rm,
                                                                                    loss_name=self.config.loss.name,
                                                                                    use_all_samples=self.config.bandit.use_all_samples)
                        
                    elif self.config.loss.name in {'sft'}: # higher loss,higher weight
                        (X1_train,x1_attention_mask,X2_train,
                        f1_output,f2_output,final_output,
                        r_logps,reward_margins)=process_bandit_train_data(selected_sample_ids,
                                                                                    h_chosen_cls,_attention_mask,X2_train,
                                                                                    f1_output,f2_output,scores,
                                                                                    _rm=-1*global_losses,
                                                                                    _logp=_logp,
                                                                                    loss_name=self.config.loss.name,
                                                                                    use_all_samples=self.config.bandit.use_all_samples)
                        r_rewards=None

                    cur_metrics=global_losses.mean().item() 
                    if self.config.bandit.use_sig_loss and self.config.loss.name in {'dpo', 'ipo'}:
                        cur_metrics=global_sig_losses.mean().item()
                    
                    cur_bandit_train_data=(X1_train,
                    x1_attention_mask,
                    X2_train,
                    cur_metrics,
                    f1_output,
                    f2_output,
                    final_output,
                    reward_margins,
                    r_rewards,
                    r_logps)

                    mini_pre_bandit_train_data=cur_bandit_train_data  

               
                rank0_print(f"bandit forward time:{time.time()-forward_start} s")
                dist.barrier()

                local_num=int(self.config.batch_size * self.config.bandit.selected_ratio/self.config.gradient_accumulation_steps)

                if self.rank==0 and self.config.bandit.enable_bandit: 
                    selected_sample_ids = selected_sample_ids.cuda(self.rank)
                if self.rank!=0 and self.config.bandit.enable_bandit:
                    selected_sample_ids=torch.zeros(local_num,dtype=torch.int64).cuda(self.rank)

                if self.config.bandit.enable_bandit:
                    selected_sample_ids=all_gather_if_needed(selected_sample_ids.detach(), self.rank, self.world_size)
                    selected_sample_ids=selected_sample_ids[:local_num]

                if not self.config.bandit.train_only and self.config.bandit.enable_bandit:
                    # detach_unselected_samples
                    detach_unselected_samples(local_losses,selected_sample_ids,self.rank)

                loss=local_losses.mean()

                if self.config.loss.name in {'dpo', 'ipo'}:
                    policy_backward_times.append(time.time()-t)
                    policy_forward_times.append(my_metrics["policy_forward_time"])
                    ref_forward_times.append(my_metrics["ref_forward_time"])
                

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)
            
            pre_bandit_train_data=mini_pre_bandit_train_data


            t=time.time()
            grad_norm = self.clip_gradient() 
            if not self.config.bandit.pretrain:       
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            rank0_print(f"dpo step time :{time.time()-t} s")

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)  
            batch_metrics['grad_norm'].append(grad_norm)


            self.batch_counter += 1 
            self.example_counter += self.config.batch_size  

            self.forward_only=True
            if self.batch_counter % self.config.bandit.train_step==0:
                self.forward_only=False

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:                
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()} 
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    mean_train_metrics.update(batch_my_metrics)
                    wandb.log(mean_train_metrics, step=self.example_counter)
                    
                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')


        if self.config.loss.name in {'dpo', 'ipo'}:
            def get_avg(a):
                return sum(a) / len(a) 
            rank0_print(f"avg ref forward time:{get_avg(ref_forward_times)}")
            rank0_print(f"avg policy forward time:{get_avg(policy_forward_times)}")
            rank0_print(f"avg policy backward time:{get_avg(policy_backward_times)}")






    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict


        net1_state_dict=self.net1.state_dict()
        net2_state_dict=self.net2.state_dict()
        self.write_state_dict(self.example_counter,net1_state_dict,metrics,'net1.pt',output_dir)
        self.write_state_dict(self.example_counter,net2_state_dict,metrics,'net2.pt',output_dir)
        del net1_state_dict
        del net2_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1,net1: Optional[nn.Module]=None, net2: Optional[nn.Module]=None):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size,net1,net2)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'
        
        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None

        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)
        
        print('Loaded model on rank', rank)
        dist.barrier()

        self.device=f"cuda:{rank}"  

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
    
    def save(self, output_dir=None, metrics=None):

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        if self.rank==0 and self.config.bandit.enable_bandit:
            net1_state_dict=self.net1.state_dict()
            net2_state_dict=self.net2.state_dict()
            self.write_state_dict(self.example_counter,net1_state_dict,metrics,'net1.pt',output_dir)
            self.write_state_dict(self.example_counter,net2_state_dict,metrics,'net2.pt',output_dir)
            del net1_state_dict
            del net2_state_dict
        dist.barrier()
        
        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()
        

class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        
        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
    
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        