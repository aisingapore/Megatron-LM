
from typing import Union
import os
from transformers import PretrainedConfig, AutoModelForCausalLM
import torch
import warnings
import time

import argparse

def fused_to_qkv(fused, nh, ng, dim):
    """
    split fused qkv into q, k, v

    Args:
        fused: [b, s, dim*3*nh]
        nh: number of heads
        ng: number of groups
        dim: kv channels

    Returns:
        q,k,v
    """
    hidden_size = dim * nh
    reshaped = fused.reshape(ng, dim*nh//ng + 2 *dim, -1)
    q,k,v = torch.split(reshaped, [dim*nh//ng, dim, dim], dim=1)
    return q.reshape(-1, hidden_size), k.reshape(-1, hidden_size), v.reshape(-1, hidden_size)

def fused_mlp_to_gate_up(fused_tensor,ffn_hidden_size):
    """
    Spit into gate and up_proj

    Returns:
        gate, up_proj
    """
    gate, up_proj = torch.split(fused_tensor, [ffn_hidden_size, ffn_hidden_size], dim=0)
    return gate, up_proj

class CheckpointLoader:

    def __init__(self, base_folder : Union[str,os.PathLike],
                pipeline_parallel_size: int,
                tensor_parallel_size: int,
                reference_config: PretrainedConfig=None):
        self.base_folder = base_folder
        self.pipeline_n = pipeline_parallel_size
        self.tensor_n = tensor_parallel_size
        self._state = list()
        if self.pipeline_n == 1:
            self.FILE_FORMAT = "mp_rank_{tensor:02d}/model_optim_rng.pt"
        else:
            self.FILE_FORMAT = "mp_rank_{tensor:02d}_{pipeline:03d}/model_optim_rng.pt"

        for pp in range(self.pipeline_n):
            self._state.append([])
            for tp in range(self.tensor_n):
                if not os.path.exists(f"{self.base_folder}/{self.FILE_FORMAT.format(tensor = tp, pipeline = pp)}"):
                    raise FileNotFoundError(f"File {self.base_folder}/{self.FILE_FORMAT.format(tensor = tp, pipeline = pp)} not found")
                else:
                    print(f"Found file {self.base_folder}/{self.FILE_FORMAT.format(tensor = tp, pipeline = pp)}")
                    state  = torch.load(f"{self.base_folder}/{self.FILE_FORMAT.format(tensor = tp, pipeline = pp)}", map_location="cpu")['model']
                    # remove the extra states to save memory
                    state_keys = list(state.keys())
                    for key in state_keys:
                        if "_extra_state" in key:
                            del state[key]
                    self._state[pp].append(torch.load(f"{self.base_folder}/{self.FILE_FORMAT.format(tensor = tp, pipeline = pp)}", map_location="cpu")['model'])

        if isinstance(reference_config, str):
            self.reference_config = PretrainedConfig.from_pretrained(reference_config)
        else:
            self.reference_config = reference_config
            assert isinstance(reference_config, PretrainedConfig), "reference_model should be an instance of PretrainedConfig or str"
        self.model = None
        
    def load_embedding(self) -> torch.Tensor:
        print("Loading embedding", end="...")
        embedding = []
        for i in range(self.tensor_n):
            to_add = self._state[0][i]['embedding.word_embeddings.weight']
            embedding.append(to_add)

        embedding = torch.cat(embedding, dim=0)
        x = self.reference_config.vocab_size
        embedding = embedding[:x,:]
        print("Done")
        return embedding
    
    def load_output_layer(self) -> torch.Tensor:
        print("Loading output layer", end="...")
        output_layer = []
        for i in range(self.tensor_n):
            to_add = self._state[-1][i]['output_layer.weight']
            output_layer.append(to_add)

        output_layer = torch.cat(output_layer, dim=0)
        x = self.reference_config.vocab_size
        output_layer = output_layer[:x,:]
        print("Done")
        return output_layer
    
    def load_final_layernorm(self) -> torch.Tensor:
        print("Loading final layernorm", end="...")
        final_layernorm = self._state[-1][0]['decoder.final_layernorm.weight']
        print("Done")
        return final_layernorm
    

    def load_layer(self, layer_idx: int) -> dict:
        """
        Function to load all the weights of a given layer, comprises of q,k,v, o_proj, gate_proj, up_proj, down_proj, input_layernorm, post_attention_layernorm
        """
        hidden_size = self.reference_config.hidden_size
        num_heads = self.reference_config.num_attention_heads
        n_layers = self.reference_config.num_hidden_layers
        gqa_head = self.reference_config.num_key_value_heads
        ffn_size = self.reference_config.intermediate_size
        dim = hidden_size // num_heads
        state_dict = {
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": None,
            f"model.layers.{layer_idx}.self_attn.k_proj.weight": None,
            f"model.layers.{layer_idx}.self_attn.v_proj.weight": None,
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": None,
            f"model.layers.{layer_idx}.input_layernorm.weight": None,
            f"model.layers.{layer_idx}.mlp.up_proj.weight": None,
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": None,
            f"model.layers.{layer_idx}.mlp.down_proj.weight": None,
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": None
        }
        # for gemma
        state_dict[f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight"] = None
        state_dict[f"model.layers.{layer_idx}.post_feedforward_layernorm.weight"] = None

        _pipeline_number, _layer_number = divmod(layer_idx, self.reference_config.num_hidden_layers // self.pipeline_n)
        print(f"Loading layer {layer_idx} from pipeline {_pipeline_number} tensor {_layer_number}", end="...")
        attn = []
        o_proj = []
        gate_proj = []
        up_proj = []
        down_proj = []
        for i in range(self.tensor_n):
            _state = self._state[_pipeline_number][i]
            if i == 0:
                # remember that norms are duplicated
                state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = \
                    _state[f"decoder.layers.{_layer_number}.self_attention.linear_qkv.layer_norm_weight"]
                state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = \
                    _state[f"decoder.layers.{_layer_number}.post_self_attn_layernorm.weight"]
                state_dict[f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight"] = \
                    _state[f"decoder.layers.{_layer_number}.mlp.linear_fc1.layer_norm_weight"]
                state_dict[f"model.layers.{layer_idx}.post_feedforward_layernorm.weight"] = \
                    _state[f"decoder.layers.{_layer_number}.post_mlp_layernorm.weight"]
            
            attn.append(_state[f"decoder.layers.{_layer_number}.self_attention.linear_qkv.weight"])
            o_proj.append(_state[f"decoder.layers.{_layer_number}.self_attention.linear_proj.weight"])
            fused = _state[f"decoder.layers.{_layer_number}.mlp.linear_fc1.weight"]
            _gate_proj, _up_proj = fused_mlp_to_gate_up(fused, ffn_size // self.tensor_n)
            gate_proj.append(_gate_proj)
            up_proj.append(_up_proj)
            down_proj.append(_state[f"decoder.layers.{_layer_number}.mlp.linear_fc2.weight"])

        # cat the tensors
        o_proj = torch.cat(o_proj, dim=1)
        gate_proj = torch.cat(gate_proj, dim=0)
        up_proj = torch.cat(up_proj, dim=0)
        down_proj = torch.cat(down_proj, dim=1)

        q, k, v = fused_to_qkv(torch.cat(attn, dim=0), num_heads, gqa_head, dim)

        state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q
        state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k
        state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v
        state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = o_proj
        state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_proj
        state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_proj
        state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = down_proj

        # check if all tensors are loaded
        for key, value in state_dict.items():
            if value is None:
                warnings.warn(f"{key} is not loaded")
        
        return state_dict

    # test run by loading when needed vs loading all at once
    def load_model(self, from_model = None, from_pretrained= None, dtype = "bfloat16") -> AutoModelForCausalLM:
        print("Loading model")
        if from_model is not None:
            self.model = from_model
        elif from_pretrained is not None:
            self.model = AutoModelForCausalLM.from_pretrained(from_pretrained, torch_dtype = dtype)
        else:
            print("Loading model from scratch, this will take longer than from pretrained")
            print("Consider using `load_model(from_pretrained= )` or `load_model(from_model= )`")
            self.model = AutoModelForCausalLM.from_config(self.reference_config)

        self.model.load_state_dict({
            "model.encoder.embed_tokens.weight": self.load_embedding(),
            "model.lm_head.weight": self.load_output_layer(),
            "model.model.norm.weight": self.load_final_layernorm()
        }
        , strict=False)

        for i in range(self.reference_config.num_hidden_layers):
            layer_dict = self.load_layer(i)
            self.model.load_state_dict(layer_dict, strict=False)
            print("Layer loaded")

    def save_model(self, save_path: Union[str, os.PathLike]):
        
        if self.model is None:
            raise ValueError("Model is not loaded yet")
        self.model.save_pretrained(save_path)
        


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Gemma2 checkpoint to Megatron-LM checkpoint")
    parser.add_argument("--base-folder","-i", type=str, required=True,
                        help="Path to the folder containing the Gemma2 checkpoint")
    parser.add_argument("--pp-size", "-p", type=int, required=True,
                        help="Number of pipeline parallel groups")
    parser.add_argument("--tp-size","-t", type=int, required=True,
                        help="Number of tensor parallel groups")
    parser.add_argument("--reference-hf-model", type=str, required=True,
                        help="Path to the reference hf model, should contain config.json etc")
    parser.add_argument("--save-path","-o", type=str, required=True,
                        help="Path to save the converted checkpoint")
    parser.add_argument("--megatron-dir", type=str,
                        help="Path to the Megatron-LM directory, either use this or use PYTHONPATH")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type to use for the model, defaults to bfloat16")
    return parser.parse_args()

def main():

    args = parse_args()

    if args.megatron_dir is not None:
        import sys
        sys.path.append(args.megatron_dir)
    else:
        # check if megatron can be imported
        try:
            import megatron
        except ImportError:
            raise ImportError("Megatron-LM is not installed, please provide the path to the Megatron-LM directory")
        
    loader = CheckpointLoader(args.base_folder, args.pp_size, args.tp_size, args.reference_hf_model)

    loader.load_model(from_pretrained=args.reference_hf_model, dtype=args.dtype)

    print("Saving model", end="...")
    loader.save_model(args.save_path)
    print("Done")

if __name__ == "__main__":
    main()
