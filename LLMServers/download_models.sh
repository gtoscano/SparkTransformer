#!/bin/bash
source .env

docker run --rm -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN} \
  -e HF_TOKEN=${HUGGINGFACE_HUB_TOKEN} \
  -e NGC_API_KEY={NGC_API_KEY} \
  nvcr.io/nvidia/vllm:25.09-py3 \
  bash -c "hf download $1"

# chmod +x download-models.sh

# Use it to download any model:
# ./download-models.sh deepseek-ai/deepseek-coder-V2-Lite-Instruct
# ./download-models.sh meta-llama/Llama-3.1-70B-Instruct
#       # Use Case   | 16GB NVIDIA (TRT)                       | 16GB AMD/vLLM
# Coding  #1 | NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4  | Qwen/Qwen2.5-Coder-7B-Instruct
# Coding  #2 | nvidia/Qwen3-14B-NVFP4                  | deepseek-ai/deepseek-coder-6.7b-instruct
# General #1 | nvidia/Qwen3-14B-NVFP4                  | Qwen/Qwen2.5-14B-Instruct
# General #2 | nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4 | meta-llama/Llama-3.1-8B-Instruct

# Use Case   | 32GB NVIDIA (TRT)                              | 32GB AMD/vLLM
# Coding  #1 | NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4         | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
# Coding  #2 | nvidia/Qwen3-32B-NVFP4                         | Qwen/Qwen2.5-Coder-14B-Instruct
# General #1 | nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4 | Qwen/Qwen3-32B-Instruct, Qwen/QwQ-32B, allenai/OLMo-3-32B-Think
# General #2 | nvidia/NVIDIA-Nemotron-Nano-12B-v2             | meta-llama/Llama-3.1-8B-Instruct

# Use Case   | 128GB NVIDIA (TRT)                             | 128GB AMD/vLLM
# Coding  #1 | NVFP4/Qwen3-Coder-480B-A35B-Instruct-FP4       | deepseek-ai/DeepSeek-Coder-V2-Instruct
# Coding  #2 | NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4         | Qwen/Qwen3-Coder-30B-A3B-Instruct, Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
# General #1 | nvidia/Llama-3.3-70B-Instruct-NVFP4            | meta-llama/Llama-3.3-70B-Instruct
# General #2 | nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4 | Qwen/Qwen2.5-72B-Instruct

# deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# deepseek-ai/DeepSeek-V3.1

# Start with Qwen3-30B-A3B-NVFP4 for the best all-around experience
# Use gpt-oss-20b when memory is tight or for pure math/reasoning
# Deploy Nemotron-Super-49B-v1.5-FP8 for production coding and advanced reasoning tasks
# Llama-4-Scout-17B-16E (NVFP4) (multimodal)
