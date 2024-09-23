# validate_conversion.py
# Usage
# python validate_conversion.py <path_to_left_model> <path_to_right_model>
# used to test post conversion from megatron lm
import sys
from transformers import AutoModelForCausalLM
import torch
from tqdm import tqdm
left = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype = "bfloat16")
right = AutoModelForCausalLM.from_pretrained(sys.argv[2], torch_dtype = "bfloat16")

failed_layers = set()
iteration = tqdm(zip(left.named_parameters(), right.named_parameters()), total=len(list(left.parameters())))
for (left_name, left_param), (right_name, right_param) in iteration:
    if left_name != right_name:
        print(f"Parameter names do not match: {left_name} != {right_name}")
        failed_layers.add(left_name)
    if left_param.size() != right_param.size():
        print(f"Parameter sizes do not match: {left_param.size()} != {right_param.size()}")
        failed_layers.add(left_name)
    if not torch.equal(left_param, right_param):
        print(f"Parameter values are not close: {left_name} != {right_name}")
        failed_layers.add(left_name)

if failed_layers:
    print("Conversion failed")
    print(f"Failed layers: {failed_layers}")

else:
    print("Conversion successful")