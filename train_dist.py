import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig


import trainers_dist as  trainers


import wandb
import json
import socket
from typing import Optional, Set
import resource

from transformers import AutoConfig

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None,net1:Optional[nn.Module] = None,net2:Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)
    
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.login(key='e4f8c913f81e45fa7de7157fb270810a0a049d3a')
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size,net1=net1,net2=net2)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)

    if config.loss.name in {'dpo', 'ipo'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name in {'dpo', 'ipo'}:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')


    model_config=AutoConfig.from_pretrained(config.model.name_or_path)
    f1_hidden_size=config.bandit.f1_hidden_size
    f1_num_layers=config.bandit.f1_num_layers
    f2_num_layers=config.bandit.f2_num_layers
    block_size=config.bandit.block_size
    net1_norm=config.bandit.net1_norm
    net2_norm=config.bandit.net2_norm
    net1_activate=config.bandit.net1_activate
    net2_activate=config.bandit.net2_activate
    train_encoder=True if config.loss.name in {'sft'} else False

    if config.bandit.last_hiddenstate_only:
        f1_input_size=model_config.hidden_size
        if config.bandit.concat_wl and config.loss.name in {'dpo', 'ipo'}:
            f1_input_size=f1_input_size*2
    else:
        f1_input_size=model_config.hidden_size*model_config.num_hidden_layers


    explore_size= f1_hidden_size*(f1_num_layers-2) // block_size

    if not config.bandit.use_encoder:
        net1 = Residual_Network_exploitation(
            None, 
            dim=f1_input_size, 
            hidden_size=f1_hidden_size,
            k=1,
            num_layers=f1_num_layers, 
            use_residual=True,
            activate=net1_activate,
            norm=net1_norm,
            use_dropout=config.bandit.f1_dropout,
            drop_rate=config.bandit.f1_drop_rate
            )
    else:
        net1 = Residual_Network_exploitation_with_Encoder(
            None, 
            dim=f1_input_size, 
            hidden_size=f1_hidden_size,
            k=1,
            num_layers=f1_num_layers, 
            use_residual=True,
            activate=net1_activate,
            norm=net1_norm,
            use_dropout=config.bandit.f1_dropout,
            drop_rate=config.bandit.f1_drop_rate,
            connector_hidden_dim=config.bandit.connector_hidden_dim,
            encoder_name_or_path=config.bandit.encoder_name_or_path,
            train_encoder=train_encoder
            )

    
    net2 = Residual_Network_exploration(
        None, 
        explore_size, 
        hidden_size=explore_size,
        k=1,
        num_layers=f2_num_layers, 
        block_size=block_size, 
        use_residual=True,
        activate=net2_activate,
        norm=net2_norm,
        use_dropout=config.bandit.f2_dropout,
        drop_rate=config.bandit.f2_drop_rate
        )

    if config.bandit.load_bandit:
        bandit_dir=config.bandit.bandit_dir
        net1_path=os.path.join(bandit_dir,"net1.pt")
        state_dict=torch.load(net1_path,map_location='cpu')
        net1.load_state_dict(state_dict['state'])

        net2_path=os.path.join(bandit_dir,"net2.pt")
        state_dict=torch.load(net2_path,map_location='cpu')
        net2.load_state_dict(state_dict['state'])

        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loaded pre-trained bandit weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')

    
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model,net1,net2), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model,net1,net2)


if __name__ == '__main__':
    main()