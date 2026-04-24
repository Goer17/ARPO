SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "Switched to parent directory: $PARENT_DIR"

export ARPO_PATH="$PARENT_DIR"
export CKPT_PATH="/mnt/workspace/lyy/ckpts"

# ============================ Environment Setup ============================
# Set basic environment variables
export PYTHONUNBUFFERED=1            
export HYDRA_FULL_ERROR=1           
export VLLM_ATTENTION_BACKEND=XFORMERS 
export VERL_LOGGING_LEVEL=DEBUG
export MKL_SERVICE_FORCE_INTEL=1    
export MKL_THREADING_LAYER=GNU       
export RAY_memory_usage_threshold=0.8  
export RAY_memory_monitor_refresh_ms=0 
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0


# Set Python path
export PYTHONPATH=$ARPO_PATH/verl_arpo_entropy:$PYTHONPATH

# ============================ Basic Configuration ============================
# Experiment name and project
PROJECT_NAME="deep_research"
EXPERIMENT_NAME="qwen3.grpo"

# Configuration file path
CONFIG_PATH="$ARPO_PATH/scripts/config" # Modify the absolute path of the config folder, relative path is not recommended
CONFIG_NAME="ppo_trainer_dr.yaml"

# Optional switches
ENABLE_MULTI_TURN=${ENABLE_MULTI_TURN:-True}

# Distributed training settings
NNODES=1                            
N_GPUS_PER_NODE=8                   

# ============================ Data Configuration ============================
# Data parameters
PROMPT_KEY="prompt"                 # Prompt field name
TRAIN_BATCH_SIZE=128                # Training batch size
PPO_MINI_BATCH_SIZE=16              # PPO mini-batch size
MAX_PROMPT_LENGTH=2000              # Maximum prompt length
MAX_RESPONSE_LENGTH=10000           # Maximum response length

# Data file paths
TRAIN_FILES="$ARPO_PATH/rl_datasets/only_search/hard_search_1k.parquet"
VALID_FILES=["$ARPO_PATH/rl_datasets/only_search/gaia_test.parquet","$ARPO_PATH/rl_datasets/only_search/hle_test.parquet"]

# ============================ Model Configuration ============================
# Actor model path
ACTOR_MODEL_PATH="/mnt/workspace/conversational_ai/OpenModels/Qwen3/Qwen3-8B" # Modify training model path

# ============================ Rollout Configuration ==========================
# Rollout settings
ROLLOUT_NAME="sglang"               # Use sglang engine for GRPO + tool
ROLLOUT_MODE="sync"                 # Default GRPO rollout mode
ROLLOUT_N=16                         # Number of responses generated per sample
INITIAL_ROLLOUTS=8                 # Initial rollout number
BEAM_SIZE=2                        # Beam size
BRANCH_PROBABILITY=0.5             # Branch probability
Entropy_weight=0.2
TOOL_CONFIG_PATH="$ARPO_PATH/scripts/config/tool_config/google_search_tool_config.yaml"
# ============================ Reward Model Configuration ==========================
# Reward model settings
REWARD_MANAGER="naive"              # Reward manager type
CUSTOM_REWARD_FUNCTION_PATH="$ARPO_PATH/verl_arpo_entropy/verl/utils/reward_score/deep_research.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============================ Training Configuration ============================
# Training parameters
TOTAL_EPOCHS=5                      # Total training epochs
SAVE_FREQ=5                        # Save frequency
TEST_FREQ=5                        # Test frequency

# ============================ Path Configuration ============================
# Save path
SAVE_PATH="$CKPT_PATH/rl/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="$CKPT_PATH/rollout"

# ============================ WandB Configuration ============================
# WandB settings
export WANDB_API_KEY="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" # Modify your wandb key

# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    # wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

# Create save directory
if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

# Create rollout save directory
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi


# ============================ Start Training ============================
python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.multi_turn.enable=${ENABLE_MULTI_TURN} \
    actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log 
