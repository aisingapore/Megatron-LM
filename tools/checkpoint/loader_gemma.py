# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
try:
    import transformers
except ImportError:
    raise ImportError("The 'transformers' package is not installed.")
import gc
import shutil
from tqdm import tqdm
import types


def add_arguments(parser):
    group = parser.add_argument_group(title='Llama/Mistral loader.')

    # silence warnings
    parser.add_argument('--model-size', type=str, required=False, default = 'llama',
                        choices=['llama2-7B', 'llama2-13B', 'llama2-70B', 'llama2-7Bf', 'llama2-13Bf', 'llama2-70Bf', 'llama3-8B', 'llama3-70B', 'llama3-8Bf', 'llama3-70Bf', 'mistral-7B', 'mistral-7Bf', 'yi-34B'],
                        help='Model size can be `llama2-7B`, `llama2-13B`, `llama2-70B`, `llama3-8B`, `llama3-70B`, `mistral-7B` (for pretrained models), '
                        'and `llama2-7Bf`, `llama2-13Bf`, `llama2-70Bf`, `llama3-8Bf`, `llama3-70bf` and `mistral-7Bf` (for chat-finetuned models).')
    parser.add_argument('--checkpoint-type', type=str, required=True,
                        help='Type of checkpoint to convert, options are "meta" or "hf"')
    parser.add_argument('--bf16', action='store_true', help='Whether to load weights in bf16.')
    parser.add_argument('--fp16', action='store_true', help='Whether to load weights in fp16.')
    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Tokenizer model file.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--loader-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')


def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split('.'))
    assert major >= 4 and minor >= 31


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

def load_args_from_checkpoint(args):

    # Read Llama args.
    model_args_path = os.path.join(args.load, "config.json")
    with open(model_args_path) as f:
        model_args = json.load(f)
    # Update Megatron args.
    args.seq_length = 8192
    args.max_position_embeddings = model_args["max_position_embeddings"]
    args.hidden_size = model_args["hidden_size"]
    args.num_attention_heads = model_args["num_attention_heads"]
    args.num_layers = model_args["num_hidden_layers"]
    args.global_batch_size = 1024
    args.norm_epsilon = model_args["rms_norm_eps"]
    args.iteration = 1 # '0', 'release' don't work
    args.position_embedding_type = "rope"
    args.normalization = "RMSNorm"
    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = True
    args.vocab_size = model_args["vocab_size"]
    args.padded_vocab_size = model_args["vocab_size"]
    args.ffn_hidden_size = model_args["intermediate_size"]
    args.final_logit_softcapping = model_args['final_logit_softcapping']
    args.attn_logit_softcapping = model_args['attn_logit_softcapping']
    args.rotary_base = model_args['rope_theta']
    args.window_size = model_args['sliding_window']
    args.query_pre_attn_scalar = model_args['query_pre_attn_scalar']
    args.kv_channels = model_args['head_dim']
    args.gated_linear_unit = True

    if "num_key_value_heads" in model_args:
        args.group_query_attention = True
        args.num_query_groups = model_args["num_key_value_heads"]


def set_preprocess_state(args, model, hf_model):
    '''Set embedding params.'''
    model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model):
    '''Set output layer & norm params.'''
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def set_attn_state(args, layer, hf_layer):
    '''Set self-attention params.'''

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    nh = args.num_attention_heads // tp
    ng = (args.num_query_groups if args.group_query_attention \
        else args.num_attention_heads) // tp
    dim = args.kv_channels
    assert nh % ng == 0

    # Copy weights (re-order dimensions for Megatron).
    fused = torch.cat([
        hf_attn.q_proj.weight.reshape((ng, dim*nh//ng, -1)),
        hf_attn.k_proj.weight.reshape((ng, dim, -1)),
        hf_attn.v_proj.weight.reshape((ng, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size))
    print(fused.shape)
    attn.linear_qkv.weight.data.copy_(fused)
    attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)


def set_mlp_state(args, layer, hf_layer):
    '''Set MLP params.'''

    mlp = layer.mlp
    hf_mlp = hf_layer.mlp

    mlp.linear_fc1.weight.data.copy_(torch.cat([
        hf_mlp.gate_proj.weight,
        hf_mlp.up_proj.weight,
    ], dim=0))
    mlp.linear_fc2.weight.data.copy_(hf_mlp.down_proj.weight)


def set_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)
    layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(hf_layer.input_layernorm.weight)
    layer.post_self_attn_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)
    layer.mlp.linear_fc1.layer_norm_weight.data.copy_(hf_layer.pre_feedforward_layernorm.weight)
    layer.post_mlp_layernorm.weight.data.copy_(hf_layer.post_feedforward_layernorm.weight)


def load_checkpoint_to_model(args):
    '''Set model params.'''

    from pretrain_gpt import model_provider
    from transformers import AutoModelForCausalLM as ModelForCausalLM

    # Init Megatron model.
    model = model_provider(True, True).to(args.params_dtype)
    # Load Huggingface model.
    print(model)

    for name, param in model.named_parameters():
        print(name, param.shape)

    hf_model = ModelForCausalLM.from_pretrained(args.load, torch_dtype=args.params_dtype, low_cpu_mem_usage=True, device_map="cpu")

    # Set model state.
    set_preprocess_state(args, model, hf_model)
    set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model, hf_model, layer_idx)

    return model


def _load_checkpoint(queue, args):

    verify_transformers_version()

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--mock-data', # To pass the "blend data checks" in arguments.py
                '--no-initialization',
                '--use-alternating-window-size',
                '--gemma-post-attention-layernorm',
                '--post-mlp-layernorm',
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint(margs)

    if "llama2" in args.model_size or "yi" in args.model_size:
        margs.tokenizer_type = "Llama2Tokenizer"
    elif "llama3" in args.model_size:
        margs.tokenizer_type = "HuggingFaceTokenizer"
    elif "mistral" in args.model_size:
        margs.tokenizer_type = "HuggingFaceTokenizer"
    else:
        margs.tokenizer_type = "HuggingFaceTokenizer"

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    # margs.use_legacy_models = False
    margs.transformer_impl = args.loader_transformer_impl

    margs.position_embedding_type = "rope"

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)

    # Determine how to make our models.
    assert args.model_type == 'GPT', 'Llama-2, Llama-3 and Mistral are GPT models.'
    margs.model_type = ModelType.encoder_or_decoder
    margs.params_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    fused_kernels.load(margs)

    # Short aliases.
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.make_vocab_size_divisible_by = None
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0

    margs.model_size = args.model_size

    # Get true (non-padded) vocab size
    tokenizer = transformers.AutoTokenizer.from_pretrained(margs.tokenizer_model)
    md.true_vocab_size = tokenizer._tokenizer.get_vocab_size(with_added_tokens=True)

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {
        "word embeddings": model.embedding.word_embeddings.weight.data
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(model.embedding, 'position_embeddings')

    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.decoder.layers[layer_num]
        message["input norm weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data
        message["post attn norm weight"] = layer.post_self_attn_layernorm.weight.data
        message["pre mlp norm"] = layer.mlp.linear_fc1.layer_norm_weight.data
        message["post mlp norm"] = layer.post_mlp_layernorm.weight

        # Grab all parallel tensors for this layer.
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        mlp_l0_weight = []
        mlp_l0_bias = []
        mlp_l1_weight = []
        layer = model.decoder.layers[layer_num]
        qkv_weight.append(layer.self_attention.linear_qkv.weight.data)
        dense_weight.append(layer.self_attention.linear_proj.weight.data)
        mlp_l0_weight.append(layer.mlp.linear_fc1.weight.data)
        mlp_l1_weight.append(layer.mlp.linear_fc2.weight.data)

        # Handle gated linear units.
        if md.swiglu:
            # Concat all the first halves ('W's) and all the second halves ('V's).
            for tp_rank in range(tp_size):
                mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
            message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
            message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

        # Simple concat of the rest.
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)

        queue_put(f"transformer layer {layer_num}", message)

    # Send final norm from tp_rank 0.
    message = {
        "weight": model.decoder.final_layernorm.weight.data,
    }
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": model.output_layer.weight.data
        }
        queue_put("output layer", message)

    queue.put("done")

    if args.checkpoint_type == "meta":
        shutil.rmtree(os.path.join(args.save_dir, 'tmp'))


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except Exception:
        queue.put("exit")
        raise
