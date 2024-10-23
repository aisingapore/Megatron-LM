# validate_conversion.py
# Usage
# python validate_conversion.py <path_to_left_model> <path_to_right_model>
# Can accept huggingface format `python validate_conversion.py ./converted-checkpoint/ meta-llama/Meta-Llama-3.1-8B`
# used to test post conversion from megatron lm
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
left = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype = "bfloat16")
right = AutoModelForCausalLM.from_pretrained(sys.argv[2], torch_dtype = "bfloat16")
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])

failed_layers = set()
total_weight_shift = 0.0
len_params = len(list(left.parameters()))
iteration = tqdm(zip(left.named_parameters(), right.named_parameters()), total=len_params)
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
        total_weight_shift += torch.sum(torch.abs(left_param - right_param)).item()

if failed_layers:
    print("Conversion failed")
    print(f"Failed layers: {failed_layers}")
    print(f"Average layer weight shift: {total_weight_shift / len(failed_layers)}")
    print("This is expected post training")
    

else:
    print("Conversion successful")

while True:
    prompt = input("Enter a prompt to generate text or type 'exit' to quit: ")
    if prompt.lower() == "exit":
        break
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")
    left = left.to("cuda")
    right = right.to("cuda")
    with torch.inference_mode():
        left_output = left.generate(input_ids, do_sample=False, max_length=50)
        right_output = right.generate(input_ids, do_sample=False, max_length=50)

    left_text = tokenizer.decode(left_output[0], skip_special_tokens=True)
    right_text = tokenizer.decode(right_output[0], skip_special_tokens=True)

    print(f"Left model generated: {left_text}")
    print(f"Right model generated: {right_text}")
    print(f"Output is same: {left_text == right_text}")