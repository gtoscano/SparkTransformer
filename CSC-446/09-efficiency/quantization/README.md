# 📘 **README.md — Quantization, LoRA, QLoRA & GGUF**

### *Efficient LLMs for Teaching, Research, and Deployment*

This repository demonstrates **quantization**, **LoRA**, **QLoRA**, and **GGUF-based inference** using Llama 3.2 and other small LLMs.
It includes:

* Python demos of **4-bit quantization**
* A full **QLoRA fine-tuning pipeline**
* **Inference using trained adapters**
* Running quantized **GGUF models** using `llama.cpp`

---

# 📁 File Structure

```
quantization/
│
├── quant_llama32_4bit_lora.py     # Demo: load Llama 3.2 quantized + LoRA, no training
├── qlora_finetune.py              # Full QLoRA training script (4-bit + LoRA)
├── qlora_inference.py             # Inference using trained QLoRA adapters
│
├── models/                        # GGUF models downloaded from Hugging Face
├── llama.cpp/                     # llama.cpp source and build directory
│
└── README.md                      # This documentation
```

---

# 🧠 Introduction to Quantization, LoRA, QLoRA & GGUF

Large Language Models (LLMs) are powerful—but **computationally expensive**.
Quantization, LoRA, and QLoRA allow us to make them:

* Smaller
* Faster
* Cheaper
* Trainable on consumer GPUs

This repository demonstrates each step in a hands-on way.

---

# 📦 1. What Is Quantization?

Quantization reduces model weight precision:

* FP16 → INT8 / INT4
* Example: **4-bit NF4 (QLoRA)**

### ✔️ Benefits

| Benefit                 | Explanation                                |
| ----------------------- | ------------------------------------------ |
| **Lower VRAM usage**    | 4-bit weights use ~75% less VRAM than FP16 |
| **Faster inference**    | Less data → faster forward pass            |
| **Cheaper deployments** | Run LLMs on small GPUs or CPUs             |
| **On-device LLMs**      | Laptops, mobile, Raspberry Pi              |

### ⚠️ Drawbacks

| Drawback                        | Notes                                       |
| ------------------------------- | ------------------------------------------- |
| Accuracy drop                   | Low-bit quantization may degrade some tasks |
| Method matters                  | NF4, AWQ, GPTQ, GGUF all differ in fidelity |
| Some layers resist quantization | Embeddings/output heads often kept in FP16  |

---

# ⚙️ 2. What Is LoRA?

**LoRA (Low-Rank Adaptation)** adds trainable low-rank matrices to a frozen model.

✔ Fine-tune huge models cheaply
✔ Only trains adapter matrices (millions of params, not billions)
✔ Compatible with quantized models → **QLoRA**

---

# ⚙️ 3. What Is QLoRA?

**QLoRA = Quantized LoRA**

This allows full fine-tuning of large LLMs using **4-bit quantization** + **LoRA adapters**.

Pipeline:

1. Load base model quantized in **4-bit NF4**
2. Freeze base model
3. Insert LoRA adapters (`q_proj`, `v_proj`, etc.)
4. Train only the adapters
5. Save adapters for easy reuse (`./qlora-out/`)

This makes it possible to fine-tune:

* 7B models on a single 3090
* 33B models on consumer GPUs
* 70B models on a single A100

---

# 🐍 4. Python Scripts in This Repository

## 🔹 `quant_llama32_4bit_lora.py` — **Demo (No Training)**

This script is a **teaching/demo script only**.

### What it DOES:

* Loads Llama 3.2 in **4-bit NF4** (bitsandbytes)
* Wraps the model with **LoRA adapters** (untrained)
* Demonstrates **sampling** with temperature + top-p
* Shows how quantization and LoRA *attach* to a model

### What it does NOT do:

❌ No dataset
❌ No training
❌ LoRA weights remain randomly initialized

### Use case:

✔ Demonstrate quantization + sampling
✔ Show LoRA structure without fine-tuning
✔ Very fast / no memory requirements

---

## 🔹 `qlora_finetune.py` — **Actual QLoRA Fine-Tuning**

This is the **full training script**.

### What it does:

* Loads model in **4-bit NF4**
* Adds LoRA adapters
* Loads the IMDB dataset
* Adds `labels = input_ids` for CausalLM training
* Runs HuggingFace `Trainer` to fine-tune adapters
* Saves trained adapters into `./qlora-out/`

### Use case:

✔ Learn how QLoRA fine-tuning works
✔ Fine-tune Llama/Qwen/Mistral on small GPU
✔ Hands-on training with real data

---

## 🔹 `qlora_inference.py` — **Inference Using Trained Adapters**

This file:

* Loads the base model (4-bit or full precision)
* Loads the trained LoRA/QLoRA adapters from `./qlora-out`
* Runs generation using the improved model

### Use case:

✔ Compare base model vs. fine-tuned model
✔ Deploy a QLoRA model for inference
✔ Evaluate the new behavior on custom prompts

---

# 🔍 Summary of Differences Between Scripts

| Feature               | `quant_llama32_4bit_lora.py`    | `qlora_finetune.py`      | `qlora_inference.py`       |
| --------------------- | ------------------------------- | ------------------------ | -------------------------- |
| Quantized model       | ✅ Yes                           | ✅ Yes                    | Optional (Yes recommended) |
| LoRA adapters added   | ✅ Yes                           | ✅ Yes                    | Loads trained adapters     |
| LoRA adapters trained | ❌ No (random)                   | ✅ Yes                    | —                          |
| Dataset               | ❌ None                          | ✅ IMDB                   | ❌ None                     |
| Loss calculation      | ❌ No                            | ✅ Yes                    | ❌ No                       |
| Trainer used          | ❌ No                            | ✅ Yes                    | ❌ No                       |
| Saves adapters        | ❌ No                            | ✅ Saves to `./qlora-out` | ❌ No                       |
| Purpose               | Demo of quantization + sampling | Actual QLoRA fine-tuning | Use trained adapters       |

---

# 🦙 5. GGUF / GGML + `llama.cpp`

While Transformers + QLoRA excel at training, **GGUF + llama.cpp** shine for fast inference.

### Why GGUF?

* Extremely fast CPU/GPU inference
* Supports many quantization formats
* Works without Python
* Portable, production-friendly

### Download a GGUF model

```bash
huggingface-cli login

huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  --local-dir models/Llama32_3B
```

---

# ⚒️ 6. Build & Run `llama.cpp`

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
cd build
```

### Run a downloaded GGUF model

```bash
bin/llama-cli \
  -m /workspace/quantization/models/Llama32_3B/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  -p "Explain quantization in simple terms."
```

### Download and run a model automatically

```bash
bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

---

# 🎓 7. When to Use What?

| Task                              | Best Tool            |
| --------------------------------- | -------------------- |
| Fine-tuning on limited GPU        | **QLoRA**            |
| Teaching quantization             | QLoRA + GGUF         |
| Python-based inference            | bitsandbytes models  |
| CPU-only or lightweight inference | **GGUF + llama.cpp** |
| Deploying small/fast models       | GGUF                 |

---

# 🎉 Final Notes

This repository helps students and practitioners understand:

* Quantization trade-offs
* How LoRA and QLoRA work
* Training adapters on limited hardware
* Deploying compact models with GGUF
* How to compare pipelines in practice (demo → training → inference → deployment)
