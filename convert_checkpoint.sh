HF_FORMAT_DIR="/shared/aisingapore/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"
MEGATRON_FORMAT_DIR="/shared/aisingapore/checkpoints/megatron/megatron-pp1"
TP=1
PP=1
TOKENIZER_MODEL=$HF_FORMAT_DIR

python tools/checkpoint/convert.py \
    --bf16 \
    --model-type GPT \
    --loader llama_mistral \
    --saver mcore \
    --target-tensor-parallel-size ${TP} \
    --target-pipeline-parallel-size ${PP} \
    --checkpoint-type hf \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --model-size llama3-8B \