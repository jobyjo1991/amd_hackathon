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
# 1. Step directly into your project root repository folder
cd /workspace/shared/FinServe/amd_hackathon

python3 -m pip install -r requirements.txt --ignore-installed

# 2. Launch your Streamlit dashboard with cluster network bypass configurations
python3 -m streamlit run app.py \
    --server.port 8501 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false