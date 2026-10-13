#!/bin/bash
# Bailing-MoE-V3 DPO with torchturbo via ms-swift
#
# Usage:
#   bash scripts/runs/bailing_v3/install_swift_deps.sh
#   bash scripts/runs/bailing_v3/swift_dpo.sh

set -o pipefail
set -eux

# model name or path
MODEL=${MODEL:-/models/Ling-3.0-flash}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/tbsf-dpo/$(basename "$MODEL")}
# sequence length
SEQ_LEN=${SEQ_LEN:-65536}
# tuner type: 'full' | 'lora'
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
# preference dataset in ms-swift messages/rejected_messages format
DATASET=${DATASET:-/datasets/orca_dpo_pairs/swift_orca_rlhf.jsonl}
# turbo engine flags live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/dpo.yaml"}

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
# AI_CPU is used with multi-comm-domain parallelism.
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export MULTI_STREAM_MEMORY_REUSE=1

export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error
# distributed-launch env (swift's BaseArguments also reads NPROC_PER_NODE /
# RANK / WORLD_SIZE for its own dist setup)
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE}}
export WORLD_SIZE=$((${NPUS_PER_NODE} * ${NNODES}))
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/ascend_log

CUR_TIME=$(date "+%Y%m%d-%H%M%S")
LOG_FILE=${LOG_FILE_PREFIX:-$OUTPUT_DIR/rank$NODE_RANK}-$CUR_TIME.log
# python interpreter (defaults to python3)
PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"
export PYTHONPATH=$REPO_ROOT/src:$PYTHONPATH

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPROC_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT}
"

mkdir -p $OUTPUT_DIR
echo $CUR_TIME >> $LOG_FILE
echo $OUTPUT_DIR >> $LOG_FILE

"${PYTHON}" -m torch.distributed.run $DISTRIBUTED_ARGS "${RUN_SWIFT}" \
    dpo \
    --rlhf_type dpo \
    --tt_config ${TT_CONFIG} \
    --model_type turbo_bailing_hybrid \
    --model $MODEL \
    --check_model false \
    --template ling_v3 \
    --add_non_thinking_prefix true \
    --dataset $DATASET \
    --load_from_cache_file true \
    --max_length ${SEQ_LEN} \
    --attn_impl flash_attn \
    --dataset_num_proc 96 \
    --tuner_type ${FINETUNING_TYPE} \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --beta 0.1 \
    --loss_type sigmoid \
    --learning_rate 1.25e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 2 \
    --bf16 true \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --output_dir $OUTPUT_DIR \
    --ddp_timeout 1800000000 \
    2>&1 | tee $LOG_FILE
