# 🧠 Pruning LLMs — A Practical Guide

*A quick primer for students learning model compression and efficiency techniques.*

---

## 📌 Overview

Modern language models are large, expensive to run, and energy-intensive.
**Pruning** is one of the oldest and most intuitive tools to make them smaller and cheaper.

But here is the key lesson:

> **Pruning only helps when the hardware or runtime can actually exploit the sparsity.**

This README explains:

* What pruning is
* Different types of pruning
* What *actually* gives you speedups
* Where pruning fits into a modern LLM efficiency pipeline
* How each of the provided scripts demonstrates a specific pruning strategy

---

# 🚀 What Is Pruning?

Pruning removes parameters from a neural network to make it:

* Smaller
* Potentially faster
* Cheaper to store or deploy

The classic idea is simple:

> “Remove weights that don’t matter much.”

But for large LLMs (like TinyLlama, Qwen, LLaMA, GPT-2), the effectiveness depends heavily on **HOW** you prune.

---

# 🧩 Types of Pruning

We study 3 major categories.

---

## 1️⃣ Unstructured Pruning

*(fastest to apply, but almost never speeds up GPU inference)*

Examples:

* Set individual weights to zero
* Zero out attention head slices without reducing tensor shape
* SparseGPT (unstructured mode)

Your scripts:

* `pruning_unstructured_heads_qwen.py`
* `pruning_unstructured_heads_tinyllama.py`
* `pruning_sparsegpt_tinyllama.py` (SparseGPT 50% unstructured sparsity)

### ❗ Important:

Unstructured pruning **changes the values** but **does not change the shape** of weight matrices.

So PyTorch + CUDA still does:

```
Dense matrix × Dense matrix
```

Even if most entries are zero.

**Result:**
⚠️ Almost no speedup.
Sometimes even *slower* due to cache behavior.

This is why your TinyLlama numbers looked like:

* Dense: ~29 ms
* SparseGPT 50%: ~29 ms
* Unstructured head zeroing: ~349 ms → ~349 ms

GPU kernels simply ignore sparsity.

---

## 2️⃣ Structured Pruning

*(removes entire heads / neurons / channels; CAN give real speedups)*

Examples:

* Remove whole **attention heads**
* Remove whole **MLP neurons**
* Shrink weight matrices by slicing them

Your scripts:

* `pruning_structured_head_pruning_example_gpt2.py`
* `pruning_structured_mlp_gpt2.py`

Here, the architecture actually changes.
Example:

```
Linear(in, out) → Linear(in, out * 0.7)
```

Now matrix multiplications become smaller:

* Fewer rows
* Fewer columns
* Fewer FLOPs
* Less memory bandwidth

**Result:**
💡 Real speedup on GPU *because the math changed*.

This is why GPT-2 pruning demos show lower latency after structured pruning.

---

## 3️⃣ SparseGPT (Advanced Pruning)

*(high-quality sparsity; useful when combined with sparse kernels)*

Your script:

* `pruning_sparsegpt_tinyllama.py`

SparseGPT performs **one-shot pruning** with input/output reconstruction.
It can prune models by 50–90% with minimal accuracy loss **IF**:

* you use proper calibration data
* AND your runtime actually supports sparse kernels

SparseGPT is powerful, but **PyTorch no longer speeds up** unstructured sparsity.

To benefit from SparseGPT:

🚀 Export to ONNX
🚀 Use ONNX Runtime or TensorRT-LLM with sparse-aware operators
🚀 Or use structured patterns like **2:4 sparsity** that Tensor Cores accelerate

---

# 🟩 When Pruning Is Actually Useful

Here’s the key conceptual slide.

## ✔️ A. Structured pruning that changes model shape

Best for speedups.

* Remove entire attention heads
* Remove MLP neurons
* Shrink linear layer dimensions

You saw this work with GPT-2.

## ✔️ B. On hardware or runtimes that understand sparsity

Unstructured sparsity helps **only** when:

* ❄️ CPU sparse BLAS kernels
* ⚡ TensorRT-LLM with 2:4 sparsity
* ⚡ ONNX Runtime sparse operators
* ⚡ Specialized accelerators (TPUs, NPUs, Graphcore, Habana, etc.)

Your current PyTorch + CUDA setup does:

> “Cool, lots of zeros. I’m still running a full dense GEMM.” 🙂

That’s why sparse TinyLlama was still ~29 ms.

## ✔️ C. As a compression or regularization tool

Even without speedups, pruning can:

* Reduce model **disk size** (after compression formats)
* Improve **generalization**
* Serve as a precursor for:

  * Distillation
  * Quantization
  * Structured model design

A common real-world pipeline:

```
Dense model
→ Pruning
→ Distillation into smaller student
→ Quantization (INT8/INT4)
→ Deployment with efficient runtime
```

---

# 🟥 When Pruning Is **NOT** Useful

Pruning will NOT give you speed improvements when:

* You use **unstructured** sparsity on GPUs
* You use **PyTorch dense kernels**
* You remove weights but **do not change tensor shapes**
* Your runtime does not include sparse operators
* You do not export to ONNX/ORT/TensorRT

This is exactly what you observed.

---

# 🧰 What Each Provided Script Demonstrates

## 1. `pruning_unstructured_heads_qwen.py`

Zeroes head slices without reducing dimensions.

* ✔ Demonstrates head masking
* ❌ No speedup (dense kernels)

---

## 2. `pruning_unstructured_heads_tinyllama.py`

Same technique on TinyLlama.

* ✔ Easy to understand
* ❌ No latency improvement

---

## 3. `pruning_structured_head_pruning_example_gpt2.py`

Uses GPT-2’s built-in `prune_heads`.

* ✔ True structured pruning
* ✔ Matrix shapes shrink
* ✔ Measurable speedup
* Best educational example

---

## 4. `pruning_structured_mlp_gpt2.py`

Removes MLP neurons → shrinks FC layers.

* ✔ Real FLOP reduction
* ✔ Shows where LLMs spend compute
* ✔ Produces visible speed gain

---

## 5. `pruning_sparsegpt_tinyllama.py`

50% unstructured sparsity with activation-based reconstruction.

* ✔ Shows what state-of-the-art pruning looks like
* ✔ Prepares for ONNX sparse workflows
* ❌ PyTorch shows no speedup
* ⚡ Export to ORT/TensorRT for *actual* sparse acceleration

---

# 🌉 How Pruning Fits in the LLM Efficiency Toolbox

Modern LLM efficiency = **Multiple layers working together**:

1. 🔹 **Quantization** → INT8 / INT4 → biggest win on GPU
2. 🔹 **Distillation** → smaller dense student model
3. 🔹 **Structured pruning** → real FLOP reduction
4. 🔹 **Graph optimization** (ONNX/ORT/TensorRT)
5. 🔹 **Efficient serving** (vLLM, TensorRT-LLM)

Pruning is not the first tool to reach for, but it is:

* A key concept
* A useful compression method
* A stepping stone to more advanced pipelines

---

# 📈 Final Takeaways for Students

### 🔥 Pruning ≠ Automatic Speedup

Removing parameters does **not** guarantee faster kernels.

### 🔥 Structured pruning > unstructured pruning (on GPUs)

You need fewer rows/columns, not more zeros.

### 🔥 SparseGPT is powerful but needs sparse-aware runtimes

PyTorch will not accelerate sparse weights by default.

### 🔥 Combine pruning with quantization + ONNX/TensorRT for best results

Pruning becomes most effective when deployed as part of a larger LLM optimization workflow, not in isolation.
