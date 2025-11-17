# ⚡️ SparkTransformer

A lightweight transformer environment for the **NVIDIA DGX Spark™** — enabling fast experimentation, inference, and fine-tuning with Hugging Face models on GPU.

This environment uses Docker + Compose with pipenv-based Python 3.12 dependency management (auto-installed on container start).

---

## 🚀 Overview

**SparkTransformer** provides a reproducible **GPU-accelerated training stack** built on the official:

```
nvcr.io/nvidia/pytorch:25.09-py3
```

It is designed for:

* DGX Spark GPU systems
* Transformer research & classroom use
* LoRA / QLoRA fine-tuning
* Fast prototyping for Hugging Face models
* Python 3.12 development in a clean containerized workspace

---

## ✨ Features

✅ NVIDIA PyTorch 25.09 (CUDA 12.4, PyTorch 2.5, Python 3.12)
✅ Preinstalled: `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `bitsandbytes`
✅ **pipenv auto-installation**: on every container start, if `/workspace/Pipfile` exists it installs dependencies
✅ UID/GID passthrough to keep file permissions clean
✅ Hugging Face token support for gated models
✅ Supports LoRA & QLoRA training on Qwen 2.5, LLaMA, Mistral, etc.
✅ Extensible layout for notebooks, scripts, and training modules

---

## 🧱 Project Structure

```
SparkTransformer/
├── Dockerfile
├── docker-compose.yml
├── .env
└── workspace/
    ├── Pipfile
    ├── train.py
    ├── train_lora.py
    ├── notebooks/
    └── examples/
```

> Everything inside `workspace/` is mapped to `/workspace` in the container —
> this is where training and development occur.

---

# ⚙️ 1. Environment Setup

Create a `.env` file in the project root:

```bash
USERNAME=gtoscano
UID=$(id -u)
GID=$(id -g)
HUGGINGFACE_HUB_TOKEN=hf_xxx_your_token_here
```

✔ Keeps HF keys out of your environment
✔ UID/GID ensure correct file permissions

---

# 🐋 2. Build & Start the DGX Spark Environment

```bash
docker compose build
docker compose up -d
```

On startup, the container will:

* enter `/workspace`
* detect your `Pipfile`
* run:

```bash
pipenv install --deploy --system
```

You will see a message like:

```
[entrypoint] Pipfile detected in /workspace. Installing dependencies via pipenv…
```

---

# 🧠 3. Verify the Environment

Test CUDA + HuggingFace:

```bash
docker compose exec trainer bash -lc 'python - <<PY
import torch; from huggingface_hub import HfApi; import os
print("CUDA:", torch.cuda.is_available())
api = HfApi()
info = api.model_info("bert-base-uncased", token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
print("HF OK:", info.modelId)
PY'
```

---

# 🧪 4. Training Examples (Python 3.12)

Two training scripts are included in **workspace/**:

---

## ▶ `train.py` — Supervised Fine-Tuning (IMDB)

```bash
docker compose exec trainer python train.py
```

This script loads IMDB, tokenizes with DistilBERT, and fine-tunes for classification.

---

## 🧩 `train_lora.py` — LoRA Fine-Tuning (Qwen 2.5)

```bash
docker compose exec trainer python train_lora.py
```

This script:

* loads **Qwen2.5-0.5B**
* applies **LoRA adapters** via PEFT
* trains on IMDB sentiment data

Works out-of-the-box on DGX Spark GPU systems.

---

# 🧠 5. Run Arbitrary Transformers Workloads

### Text Generation Example

```bash
docker compose exec trainer bash -lc 'python - <<PY
from transformers import pipeline
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
print(pipe("The future of GPU AI is")[0]["generated_text"])
PY'
```

---

# 📓 6. Notebooks and Scripts

Place your notebooks or scripts inside:

```
workspace/notebooks/
```

Inside the container:

```bash
docker compose exec trainer python notebooks/my_notebook.py
```

---

# 🧹 7. Shut Down

```bash
docker compose down
```

---

# 🧰 Utilities

| Purpose                     | Command                                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------------------- |
| Test CUDA availability      | `docker compose exec trainer python -c "import torch; print(torch.cuda.is_available())"`        |
| Run default training script | `docker compose exec trainer python train.py`                                                   |
| Run LoRA training           | `docker compose exec trainer python train_lora.py`                                              |
| Update installed libraries  | `docker compose exec trainer pip install -U transformers datasets accelerate peft bitsandbytes` |
| Verify pipenv was applied   | `docker compose exec trainer pip list`                                                          |

---

# 🧩 Quickstart Recap

| Step | Command                            | Description                |
| ---- | ---------------------------------- | -------------------------- |
| 1    | Add HF token to `.env`             | Gives access to HF models  |
| 2    | `docker compose build`             | Build DGX Spark image      |
| 3    | `docker compose up -d`             | Start environment          |
| 4    | Add `Pipfile` to `workspace/`      | Auto-installs dependencies |
| 5    | `docker compose exec trainer bash` | Enter environment          |
| 6    | `python train.py`                  | Run a training example     |

---

# 💡 Tips

* Store datasets and checkpoints inside `workspace/` (this persists on the host).
* Use pipenv in the host only to generate `Pipfile` + `Pipfile.lock`; installation happens in the container.
* For multi-GPU training, scale docker-compose services or use `torchrun`.

---

# 🏁 Example Quick Run

```bash
git clone https://github.com/gtoscano/SparkTransformer.git
cd SparkTransformer

# Create .env with UID/GID and HF token
cp .env.example .env

docker compose up -d

# Run LoRA training
docker compose exec trainer python train_lora.py
```

---

**Project:** gtoscano/SparkTransformer
**Platform:** NVIDIA DGX Spark™
**Base Image:** `nvcr.io/nvidia/pytorch:25.09-py3`
**License:** MIT
**Author:** Gregorio Toscano
