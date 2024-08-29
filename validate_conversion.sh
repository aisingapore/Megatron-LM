#!/bin/bash

wait_for_flask() {
    echo "Waiting for Flask app to start..."
    while ! curl -s http://localhost:5000 > /dev/null; do
        sleep 1
    done
    echo "Flask app is ready!"
}

HF_FORMAT_DIR="/shared/aisingapore/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"
MEGATRON_FORMAT_DIR="/shared/aisingapore/checkpoints/megatron/megatron-pp1"

bash examples/inference/llama_mistral/run_text_generation_llama3.sh $MEGATRON_FORMAT_DIR $HF_FORMAT_DIR > /dev/null 2>&1 &
SERVER_PID=$!

wait_for_flask
echo "Sending curl request..."
RESPONSE=$(curl 'http://localhost:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8' -d '{"prompts":["Today is a great day to"], "tokens_to_generate":100, "top_k":1}')

# Step 4: Kill the Flask server
echo "Killing Flask server..."
kill $SERVER_PID

# Step 5: Run the Python process


echo "Running Python process..."
python examples/inference/llama_mistral/huggingface_reference.py --prompt "Today is a great day to" --model-path $HF_FORMAT_DIR 

echo "Response text:"
echo $RESPONSE | jq -r '.text'