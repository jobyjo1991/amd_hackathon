run models
----------
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --dtype float16


run project
-------------
