---
marp: true
size: 16:9
paginate: true
theme: default
style: |
  section {
    font-size: 28px;
  }
  h1, h2, h3 {
    color: #C41E3A;
  }
  code {
    background: #f5f5f5;
    padding: 4px;
    border-radius: 6px;
  }
---

<!-- TITLE SLIDE -->
# 🧠 Quantization, LoRA, QLoRA & GGUF  
## With Llama 3.2 / Qwen / Mistral  
### *Efficient LLMs for Teaching & Research*

---

# 🚀 Why We Need Quantization

- LLMs are huge  
- Heavy GPU memory requirements  
- Slow on consumer hardware  
- Expensive to deploy  
- Hard for students to run locally

### Quantization makes models:
- ✔ Smaller  
- ✔ Faster  
- ✔ Cheaper  
- ✔ Runnable on CPUs, laptops, small GPUs

---

# ⚙️ What Is Quantization?

Reducing precision of weights/activations:  
FP32 → FP16 → INT8 → INT4

### Benefits
- Lower memory
- Faster inference
- Reduced energy / cost
- Enables on-device LLMs

### Drawbacks
- Some accuracy loss
- Sensitive layers must remain FP16
- Method matters (NF4 vs GPTQ vs AWQ vs GGUF)

---

# 🧩 Quantization Types

### **Bitsandbytes NF4 (QLoRA)**
- 4-bit Normalized Float  
- Great quality retention  
- Used for fine-tuning  

### **GPTQ / AWQ**
- Post-training quantization  
- Weight-only  
- Good for deployment  

### **GGUF**
- Runtime quantization format  
- Perfect for CPU inference  
- Used in llama.cpp

---

# 🔧 LoRA — Low-Rank Adaptation

LoRA adds small low-rank matrices to a **frozen** base model.

### Why LoRA?
- Fine-tune big models cheaply  
- Train only a few million parameters  
- Very fast  
- Extensible to QLoRA

---

# 🔧 QLoRA — Quantized LoRA

QLoRA combines:

### 1. **4-bit NF4 quantization** (bitsandbytes)  
### 2. **LoRA adapters**  
### 3. **Fine-tuning only the adapters**  

Result:  
Fine-tune 7B–70B models on **consumer GPUs**.

---

# 🐍 Scripts Included in This Repository

```

quant_llama32_4bit_lora.py
qlora_finetune.py
qlora_inference.py
models/
llama.cpp/
README.md

````

---

# 📂 File Purpose Overview

### **`quant_llama32_4bit_lora.py`**
Demo only  
- Loads Llama in 4-bit  
- Adds LoRA adapters (UNTRAINED)  
- Shows sampling (temperature / top_p)  
- No dataset  
- No training  
- LoRA weights remain random

Useful for teaching **quantization + sampling**.

---

### **`qlora_finetune.py`**  
Actual training  
- Loads 4-bit model  
- Adds LoRA adapters  
- Loads IMDB dataset  
- Creates labels  
- Fine-tunes LoRA using HF Trainer  
- Saves adapters to `./qlora-out`

This is **real QLoRA training**.

---

### **`qlora_inference.py`**  
Uses trained adapters  
- Loads base model (4-bit or full precision)  
- Loads LoRA adapters from `./qlora-out`  
- Generates text with improved behavior

This is **deployment/inference**.

---

# 🧠 Comparison of Scripts

| Feature | quant_llama32_4bit_lora.py | qlora_finetune.py | qlora_inference.py |
|--------|-----------------------------|-------------------|--------------------|
| 4-bit quantization | ✔ | ✔ | ✔ (recommended) |
| LoRA adapters added | ✔ | ✔ | loads trained |
| LoRA adapters trained | ❌ random | ✔ yes | — |
| Uses IMDB dataset | ❌ | ✔ | ❌ |
| Computes loss | ❌ | ✔ | ❌ |
| Uses Trainer | ❌ | ✔ | ❌ |
| Saves adapters | ❌ | ✔ `./qlora-out` | ❌ |
| Use case | demo | fine-tune | generation |

---

# 🧪 quant_llama32_4bit_lora.py  
## (Demo Only)

```python
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb)

cfg = LoraConfig(...)

model = get_peft_model(base, cfg)

output = model.generate(..., temperature=0.7)
````

* Runs **sampling**
* Shows **quantization effect**
* LoRA has **no learned behavior**

---

# 🏋️ qlora_finetune.py

## (Actual Fine-Tuning)

```python
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_cfg)

lora_cfg = LoraConfig(...)
model = get_peft_model(model, lora_cfg)

enc["labels"] = enc["input_ids"]

trainer.train()
model.save_pretrained("./qlora-out")
```

* Computes loss
* Trains LoRA adapters
* Saves them for inference

---

# 🧾 qlora_inference.py

## (Use the Trained LoRA)

```python
base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb)
model = PeftModel.from_pretrained(base_model, "./qlora-out")

outputs = model.generate(...)
```

* Loads fine-tuned adapters
* Uses them to generate improved text

---

# 🦙 GGUF + llama.cpp for Deployment

### Why GGUF?

* Extremely fast CPU inference
* Supports many quantization levels
* Perfect for production or laptops
* Loads without Python

---

# 🔽 Download a GGUF Model

```bash
huggingface-cli login

huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  --local-dir models/Llama32_3B
```

---

# ⚒️ Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
cd build
```

---

# ▶️ Run GGUF Model

```bash
bin/llama-cli \
  -m models/Llama32_3B/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  -p "Explain quantization to a beginner."
```

---

# ▶️ Auto-Download & Run

```bash
bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

---

# 📌 When to Use What?

### QLoRA (training)

* Fine-tune on small GPUs
* Update only LoRA adapters
* Great for research/education

### GGUF + llama.cpp (inference)

* Fast
* Portable
* CPU-friendly
* Perfect for deployment

### quant_llama32_4bit_lora.py (demo)

* Show quantization
* Show sampling
* No training required

---

# 🏁 Summary

* Quantization makes LLMs practical
* LoRA enables efficient fine-tuning
* QLoRA makes it possible on **any GPU**
* GGUF brings fast inference to **any device**
* The three scripts show **demo → training → inference**

---

# 🎉 Thank You!

