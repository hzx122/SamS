#!/bin/bash

python -u train_dist.py \
    seed=0 \
    model=llama3_8b \
    model.name_or_path=/path/to/Meta-Llama-3-8B-Instruct \
    lr=1e-6 \
    warmup_steps=6000 \
    optimizer=AdamW \
    datasets=[ultrafeedback] \
    do_first_eval=true \
    wandb.enabled=true \
    wandb.project=xxx \
    exp_name=xxx \
    loss=dpo \
    loss.beta=0.01 \
    loss.label_smoothing=0 \
    gradient_accumulation_steps=2 \
    batch_size=16 \
    eval_batch_size=32 \
    trainer=FSDPTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16 \
    bandit.load_bandit=true \
    bandit.bandit_dir=xxx \
    bandit.use_pool=true \
    bandit.pool_size=40000 \
    bandit.pool_use_num=32 \
    bandit.norm_metrics=true \
    bandit.train_only=false \
    bandit.concat_wl=false \
    bandit.use_encoder=true \
    bandit.enable_bandit=true \
    bandit.layer_encoder=true \
    bandit.forward_only=true \
    bandit.selected_ratio=0.5 \
    bandit.enable_avg_reward=true \
    bandit.f2_weight=0.01 \
    bandit.f1_only=false \
    bandit.train_step=1 \
    bandit.f1_num_layers=16 \
    bandit.f2_num_layers=16 \
    bandit.f1_num_epochs=2 \
    bandit.f1_num_batch=2 \
    bandit.f2_num_epochs=2 \
    bandit.f2_num_batch=2 \
    bandit.lr=0.0001 \
    bandit.encoder_name_or_path=/path/to/all-MiniLM-L6-v2 \
    bandit.f1_reward_type=add_rm_logp \
    save_every=30000  \
    n_epochs=1 \
    max_length=1024 \
    max_prompt_length=512 \
    n_examples=null \
    debug=false
    




