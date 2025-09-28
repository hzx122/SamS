#!/bin/bash


# 单卡
python -u train_dist.py \
    model=llama3_8b \
    model.name_or_path=/path/to/models/Meta-Llama-3-8B-Instruct \
    datasets=[ultrafeedback] \
    loss=sft \
    lr=1e-6 \
    warmup_steps=60000 \
    optimizer=AdamW \
    wandb.project=ultrafeedback_llama3_bandit_pretrain \
    exp_name=ultrafeedback_llama3_bandit_pretrain_layer_encoder_offered_data \
    gradient_accumulation_steps=2 \
    batch_size=16 \
    eval_batch_size=32 \
    trainer=FSDPTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16 \
    bandit.train_only=true \
    bandit.norm_metrics=true \
    bandit.use_pool=true \
    bandit.pool_size=40000 \
    bandit.pool_use_num=32 \
    bandit.lr=0.0001 \
    bandit.concat_wl=false \
    bandit.f1_reward_type=add_rm_logp \
    bandit.encoder_name_or_path=/path/to/models/all-MiniLM-L6-v2 \
    bandit.enable_bandit=true \
    bandit.layer_encoder=true \
    bandit.use_encoder=true \
    bandit.load_bandit=false \
    bandit.forward_only=false \
    bandit.enable_avg_reward=true \
    bandit.train_step=1 \
    bandit.f1_only=false \
    bandit.f1_num_layers=16 \
    bandit.f2_num_layers=16 \
    bandit.f1_num_epochs=2 \
    bandit.f1_num_batch=2 \
    bandit.f2_num_epochs=2 \
    bandit.f2_num_batch=2 \
    bandit.f2_weight=0.01 \
    bandit.pretrain=true \
    save_every=20000000  \
    max_length=2048 \
    max_prompt_length=1800 \
    n_epochs=1 \
    n_examples=null \
    debug=false
