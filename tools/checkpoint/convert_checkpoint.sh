#!/bin/bash
# This script is used to convert a HF checkpoint to Megatron-LM format
# Usage: bash convert_checkpoint.sh
# Modify the HF_FORMAT_DIR to the directory of the HF checkpoint
# Modify the TP and PP to the target tensor parallel size and target pipeline parallel size
# Modify the SHARED_FS to the shared file system
# Change whether gemma or llama at the bottom

set -euov pipefail

HF_FORMAT_DIR="meta-llama/Llama-3.1-70B"
# get the model_name from the base directory
# todo: maybe we should use awk to replace / with __
# but what about local models?
# users can manually set the model_name
model_name=$(basename $HF_FORMAT_DIR)

TP=8
PP=1
export SHARED_FS="/shared/aisingapore"

export MEGATRON_DIR="${SHARED_FS}/source_files/Megatron-LM/"
TOKENIZER_MODEL=$HF_FORMAT_DIR
MEGATRON_FORMAT_FOLDER=${SHARED_FS}/checkpoints/megatron
MEGATRON_FORMAT_DIR="${MEGATRON_FORMAT_FOLDER}/${model_name}-PP${PP}-TP${TP}-mcore"

mkdir -p $MEGATRON_FORMAT_DIR

# patch to install accelerate, should be installed in container already, but this just checks
pip install accelerate > /dev/null

args=(
    --bf16
    --model-type GPT
    --target-tensor-parallel-size ${TP}
    --target-pipeline-parallel-size ${PP}
    --checkpoint-type hf
    --load-dir ${HF_FORMAT_DIR}
    --save-dir ${MEGATRON_FORMAT_DIR}
    --tokenizer-model ${TOKENIZER_MODEL}
    # Honestly this is not used, but it's required
    # TODO
    --model-size llama3-8B
)

gemma_args=(
    --loader gemma
    --saver mcore_gemma
    --loader-transformer-impl transformer_engine
)

llama_args=(
    --loader llama_mistral
    --saver mcore
)

# change the model here
args+=(${llama_args[@]})
#args+=(${gemma_args[@]})

mkdir -p $MEGATRON_FORMAT_DIR
# for this echo, split the args into multiple lines
rm -rf $MEGATRON_FORMAT_DIR/convert_script.sh || true
cp $0 $MEGATRON_FORMAT_DIR/convert_script.sh
chmod 440 $MEGATRON_FORMAT_DIR/convert_script.sh # to keep as reference

cd $MEGATRON_DIR

# this is a quick fix for gemma
cp gemma_attention.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py

python tools/checkpoint/convert.py ${args[@]}
