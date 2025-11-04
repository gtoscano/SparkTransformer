# ⚡️ SparkTransformer

A lightweight transformer environment for the new **NVIDIA DGX Spark™** — enabling fast experimentation, inference, and fine-tuning with Hugging Face models on GPU.*

---

## 🚀 Overview

**SparkTransformer** provides a reproducible **Docker + Compose** setup built on the official **NVIDIA PyTorch 25.09** container.
It’s designed for **DGX Spark** systems and GPU-accelerated research workflows, supporting everything from model inference and evaluation to large-scale fine-tuning.

Use it as a clean foundation for transformer research, classroom projects, or production experiments.

---

## ✨ Features

✅ Optimized for **NVIDIA DGX Spark™** (CUDA 13 / PyTorch 2.5)
✅ Preinstalled: `transformers`, `peft`, `datasets`, `trl`, `bitsandbytes`, `accelerate`
✅ Hugging Face authentication (token or mounted cache)
✅ User-mapped UID/GID for clean file ownership
✅ Extensible workspace for notebooks and experiments

---

## 🧱 Project Structure

```
SparkTransformer/
├── Dockerfile
├── docker-compose.yml
├── .env
└── workspace/
    ├── notebooks/
    ├── examples/
    └── dgx-spark-playbooks/
```

---

## ⚙️ 1. Environment Setup

Create a `.env` file in the project root:

```bash
USERNAME=gtoscano
UID=1000
GID=1000
HUGGINGFACE_HUB_TOKEN=hf_xxx_your_token_here
```

> 🔐 Keep `.env` out of version control.
> Your Hugging Face token must have access to gated models (e.g., Meta Llama 3).

---

## 🐋 2. Build & Start the DGX Spark Container

```bash
docker compose build
docker compose up -d
```

---

## 🔑 3. Verify Authentication

```bash
docker compose exec trainer bash -lc 'python - <<PY
import os
from huggingface_hub import HfApi
api = HfApi()
print("Token present:", bool(os.getenv("HUGGINGFACE_HUB_TOKEN")))
info = api.model_info("bert-base-uncased", token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
print("Model loaded:", info.modelId)
PY'
```

Expected:

```
Token present: True
Model loaded: bert-base-uncased
```

---

## 🧠 4. Run Transformer Workloads

### ▶ Text Generation (Inference)

```bash
docker compose exec trainer bash -lc 'python - <<PY
from transformers import pipeline
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
print(pipe("The future of AI research is")[0]["generated_text"])
PY'
```

### 🧩 Fine-Tuning (Example with Llama 3)

Clone NVIDIA’s fine-tuning recipes inside your workspace:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks ./workspace/dgx-spark-playbooks
```

Run any recipe:

```bash
docker compose exec trainer \
  python workspace/dgx-spark-playbooks/nvidia/pytorch-fine-tune/assets/Llama3_8B_LoRA_finetuning.py
```

---

## 🧪 5. Your Own Experiments

Place notebooks or scripts in `workspace/`:

```
workspace/
└── notebooks/
    └── summarization.ipynb
```

Run them inside the DGX Spark container:

```bash
docker compose exec trainer python notebooks/summarization.py
```

---

## 🧹 6. Shut Down

```bash
docker compose down
```

---

## 🧰 Utilities

| Purpose                | Command                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- |
| Test CUDA availability | `docker compose exec trainer python -c "import torch; print(torch.cuda.is_available())"`                      |
| Update packages        | `docker compose exec trainer pip install -U transformers peft datasets trl bitsandbytes accelerate`           |
| Remove pynvml warning  | `docker compose exec trainer pip uninstall -y pynvml && docker compose exec trainer pip install nvidia-ml-py` |

---

## 🧩 Quickstart Recap

| Step | Command                                    | Description           |
| ---- | ------------------------------------------ | --------------------- |
| 1    | `echo "HUGGINGFACE_HUB_TOKEN=..." >> .env` | Add your HF token     |
| 2    | `docker compose build`                     | Build DGX Spark image |
| 3    | `docker compose up -d`                     | Start container       |
| 4    | `git clone ...workspace/...`               | Get examples          |
| 5    | `docker compose exec trainer python ...`   | Run transformers      |

---

## 💡 Tips

* For large-scale training, expand `docker-compose.yml` to multiple GPU services.
* Store datasets and checkpoints in mounted volumes under `workspace/`.
* Keep `.env` private and regenerate tokens if you rotate HF keys.

---

## 🏁 Example Quick Run

```bash
git clone https://github.com/gtoscano/SparkTransformer.git
cd SparkTransformer
cp .env.example .env  # add UID/GID/token
docker compose up -d
docker compose exec trainer python workspace/examples/text_generation.py
```

---

**Project:** [gtoscano/SparkTransformer](https://github.com/gtoscano/SparkTransformer)
**Platform:** NVIDIA DGX Spark™
**Base Image:** `nvcr.io/nvidia/pytorch:25.09-py3`
**License:** MIT
**Author:** Gregorio Toscano

