#!/bin/bash
# This example will start serving the Llama3-8B model
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr 0.0.0.0 \
                  --master_port 6000"

# Ensure CHECKPOINT and TOKENIZER_MODEL are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: You must provide CHECKPOINT and TOKENIZER_MODEL as command-line arguments."
  echo "Usage: $0 /path/to/checkpoint /path/to/tokenizer_model"
  exit 1
fi

# Assign command-line arguments to variables
CHECKPOINT=$1
TOKENIZER_MODEL=$2

pip install flask-restful

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
      --use-checkpoint-args \
      --disable-bias-linear \
      --load ${CHECKPOINT}  \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model ${TOKENIZER_MODEL} \
      --transformer-impl transformer_engine \
      --normalization RMSNorm \
      --group-query-attention \
      --use-rotary-position-embeddings \
      --untie-embeddings-and-output-weights \
      --no-masked-softmax-fusion \
      --attention-softmax-in-fp32 \
      --bf16  \
      --micro-batch-size 1  \
      --seq-length 8192 \
      --tensor-model-parallel-size 1  \
      --pipeline-model-parallel-size 1  \
      --attn-logit-softcapping 0 \
      --final-logit-softcapping 0 \
      --gemma-post-attention-layernorm \
      --use-alternating-window-size \
      --post-mlp-layernorm \
      --gated-linear-unit \
      --norm-epsilon 1e-6 \
      --attention-dropout 0.0 \
      --hidden-dropout 0.0 \