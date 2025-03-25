#!/bin/bash

export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

RUN_NAME="TEST"

export WANDB_API_KEY=""
export WANDB_ENTITY=""
export WANDB_PROJECT=""
export WANDB_NAME=$RUN_NAME

WANDB_DIR="$CHECKPOINT_SAVE/wandb/$RUN_NAME"
CHECKPOINT_DIR="$CHECKPOINT_SAVE/checkpoints/$RUN_NAME"

if [ ! -d $WANDB_DIR ]; then
    mkdir -p $WANDB_DIR
fi
if [ ! -d $CHECKPOINT_DIR ]; then
    mkdir -p $CHECKPOINT_DIR
fi

python -m verl.trainer.main_ppo \
    data.train_files=$path_to_your_train_parquet \
    data.val_files=$path_to_your_test_parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=7168 \
    +data.apply_chat_template=False \
    actor_rollout_ref.model.path=$path_to_your_model/Qwen2.5-0.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.n=64 \
    actor_rollout_ref.rollout.temperature=1.0 \
    critic.optim.lr=1e-5 \
    critic.model.path=$path_to_your_model/Qwen2.5-0.5B \
    critic.ppo_mini_batch_size=32 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    critic.ppo_max_token_len_per_gpu=16384 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.lam=1.0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    critic.cliprange_value=0.2 \
    +trainer.val_before_train=False \
    reward_model.reward_manager=prime_save \
    +reward_model.my_reward_verify_url=$your_deployed_model_url \
    +reward_model.my_reward_verify_model=qwen25-14b \
    +reward_model.my_reward_verify_key=EMPTY \
    +reward_model.my_reward_verify_max_concurrency=64 \
    +reward_model.my_reward_verify_max_tokens=0 \
    +reward_model.my_reward_train_save_path=$CHECKPOINT_SAVE/reward/$RUN_NAME/train \
    +reward_model.my_reward_val_save_path=$CHECKPOINT_SAVE/reward/$RUN_NAME/val \
    trainer.critic_warmup=0 \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.test_freq=-1 \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=20 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.wandb_dir=$WANDB_DIR 2>&1 | tee $CHECKPOINT_SAVE/$RUN_NAME-$(date +%m-%d-%H).log